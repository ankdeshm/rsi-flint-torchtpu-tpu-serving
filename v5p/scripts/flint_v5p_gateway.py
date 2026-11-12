#!/usr/bin/env python3
"""Async Cache-Aware & Least-Inflight Multi-Replica Router for Flint on TPU v5p.

Load-balances `/v1/completions` and `/v1/chat/completions` across local host replicas
(`DP=2` for `TP=2`, `DP=4` for `TP=1`) or across all 4 hosts of the `v5p-16` slice
(`DP=4` for `TP=4`, `DP=8` for `TP=2`, `DP=16` for `TP=1`).

Routing policies:
  - When a request carries header `X-Prefix-Trunk: <int>` (or prompt prefix hash
    when `policy=cache_aware`), routes deterministically to the replica group owning
    that warm prefix trunk (with least-inflight tie-breaking if multiple replicas
    hold that trunk) so RadixCache hits are preserved without cross-replica eviction!
  - Otherwise routes to the backend with minimum currently in-flight requests
    (round-robin tie-breaker).
  - `/flush_cache` broadcasts cache resets to ALL backend replicas concurrently.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import itertools
import json
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import httpx
import uvicorn

GLOBAL_ARGS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None, keepalive_expiry=30.0)
    app.state.backends = []
    for i, url in enumerate(GLOBAL_ARGS.backends):
        u = url.rstrip("/")
        app.state.backends.append(
            {
                "id": i,
                "url": u,
                "client": httpx.AsyncClient(timeout=None, base_url=u, limits=limits),
                "inflight": 0,
            }
        )
    app.state.rr = itertools.count()
    yield
    for b in app.state.backends:
        await b["client"].aclose()


app = FastAPI(lifespan=lifespan)


def _pick_backend(app: FastAPI, req_data: dict, headers) -> dict:
    backends = app.state.backends
    n = len(backends)
    rr = next(app.state.rr)

    trunk_hdr = headers.get("x-prefix-trunk")
    if trunk_hdr is not None:
        t_id = int(trunk_hdr)
        # If there are fewer trunks than backends, spread each trunk across a subset
        # of backends that all have that trunk pre-warmed!
        num_trunks = int(headers.get("x-num-trunks", "0") or "0")
        if num_trunks > 0 and num_trunks < n:
            candidates = [idx for idx in range(n) if (idx % num_trunks) == (t_id % num_trunks)]
            chosen = min(candidates, key=lambda idx: (backends[idx]["inflight"], (idx - rr) % n))
            return backends[chosen]
        return backends[t_id % n]

    if GLOBAL_ARGS.policy == "cache_aware":
        prompt = req_data.get("prompt")
        if isinstance(prompt, list) and len(prompt) >= 64:
            h = int.from_bytes(hashlib.blake2b( json.dumps(prompt[:64]).encode(), digest_size=4).digest(), "little")
            return backends[h % n]

    chosen = min(range(n), key=lambda idx: (backends[idx]["inflight"], (idx - rr) % n))
    return backends[chosen]


async def _proxy_post(api: str, request: Request):
    raw_body = await request.body()
    req_data = json.loads(raw_body)
    backend = _pick_backend(request.app, req_data, request.headers)
    fwd_headers = {"Content-Type": "application/json"}
    is_stream = bool(req_data.get("stream", False))
    backend["inflight"] += 1
    if not is_stream:
        try:
            resp = await backend["client"].post(api, content=raw_body, headers=fwd_headers)
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        finally:
            backend["inflight"] -= 1

    async def _gen():
        try:
            async with backend["client"].stream("POST", api, content=raw_body, headers=fwd_headers) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
        finally:
            backend["inflight"] -= 1

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    return await _proxy_post("/v1/completions", request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _proxy_post("/v1/chat/completions", request)


@app.post("/flush_cache")
async def flush_cache(request: Request):
    async def _one(b):
        try:
            r = await b["client"].post("/flush_cache", timeout=30.0)
            return {"id": b["id"], "status": r.status_code}
        except Exception as e:
            return {"id": b["id"], "error": str(e)}

    res = await asyncio.gather(*(_one(b) for b in request.app.state.backends))
    return {"status": "ok", "backends": res}


@app.get("/health")
@app.get("/get_model_info")
async def health(request: Request):
    for b in request.app.state.backends:
        try:
            r = await b["client"].get("/health", timeout=5.0)
            if r.status_code != 200:
                return Response(content=json.dumps({"status": "unhealthy", "id": b["id"]}), status_code=503)
        except Exception as e:
            return Response(content=json.dumps({"status": "unhealthy", "id": b["id"], "err": str(e)}), status_code=503)
    return {"status": "ok", "num_backends": len(request.app.state.backends)}


def main() -> None:
    global GLOBAL_ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--policy", choices=["least_inflight", "cache_aware"], default="least_inflight")
    ap.add_argument("--backends", nargs="+", required=True)
    GLOBAL_ARGS = ap.parse_args()
    uvicorn.run(app, host=GLOBAL_ARGS.host, port=GLOBAL_ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
