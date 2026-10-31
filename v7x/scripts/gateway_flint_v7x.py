#!/usr/bin/env python3
"""Cache-aware Prefix-Affinity + Least-Inflight Async HTTP Gateway for Flint on TPU v7x-16.

Routes /v1/completions and /v1/chat/completions across N SGLang replicas (4, 8, 16, or 32)
spanning all 4 hosts of the v7x-16 slice, with:
  - Fan-out /flush_cache and /get_server_info across all N workers
  - Disjoint trunk-affinity worker pools + parallel pre-warm fan-out for 80%/95% RadixCache benchmarks
  - Pure least-inflight + round-robin tie-breaking for 0% cache benchmarks and balanced decode loads
  - Zero-buffering SSE streaming passthrough via aiohttp
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import zlib
from typing import Sequence

import aiohttp
from aiohttp import web


class ReplicaGateway:
    def __init__(self, workers: Sequence[str]) -> None:
        self.workers = [w.rstrip("/") for w in workers]
        self.n = len(self.workers)
        self.inflight = [0] * self.n
        self.served = [0] * self.n
        self.rr_cursor = 0
        self.session: aiohttp.ClientSession | None = None

    async def start(self, _app: web.Application) -> None:
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, tcp_nodelay=True, keepalive_timeout=120)
        timeout = aiohttp.ClientTimeout(total=1800, connect=30, sock_read=1800)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

    async def cleanup(self, _app: web.Application) -> None:
        if self.session is not None:
            await self.session.close()

    def _least_inflight(self, indices: Sequence[int]) -> int:
        best_idx = indices[0]
        best_load = self.inflight[best_idx]
        start = self.rr_cursor
        self.rr_cursor = (self.rr_cursor + 1) % max(len(indices), 1)
        m = len(indices)
        for off in range(m):
            idx = indices[(start + off) % m]
            load = self.inflight[idx]
            if load < best_load:
                best_load = load
                best_idx = idx
        return best_idx

    def _trunk_candidates(self, trunk_idx: int, num_trunks: int, conc: int) -> list[int]:
        eff_workers = min(self.n, max(num_trunks, conc))
        fanout = max(1, eff_workers // max(num_trunks, 1))
        # Ensure disjoint contiguous slices when num_trunks * fanout <= self.n
        start = (trunk_idx * fanout) % self.n
        return [(start + k) % self.n for k in range(fanout)]

    def _prefix_hash(self, body: dict) -> int:
        prompt = body.get("prompt")
        if isinstance(prompt, list) and prompt:
            head = prompt[:64] if isinstance(prompt[0], int) else (prompt[0][:64] if isinstance(prompt[0], list) else prompt[:4])
            raw = repr(head).encode()
        elif isinstance(prompt, str):
            raw = prompt[:256].encode()
        else:
            msgs = body.get("messages")
            raw = repr(msgs[:2] if isinstance(msgs, list) else "").encode()
        return zlib.crc32(raw)

    def choose_worker(self, headers: web.BaseRequest.headers.__class__, body: dict) -> int:
        if self.n == 1:
            return 0
        if "X-Trunk-Index" in headers:
            t_idx = int(headers["X-Trunk-Index"])
            n_trunks = int(headers.get("X-Num-Trunks", "1"))
            conc = int(headers.get("X-Concurrency", str(self.n)))
            cands = self._trunk_candidates(t_idx, n_trunks, conc)
            return self._least_inflight(cands)
        policy = headers.get("X-Policy", "prefix_aware")
        all_idx = list(range(self.n))
        if policy == "least_inflight":
            return self._least_inflight(all_idx)
        home = self._prefix_hash(body) % self.n
        min_load = min(self.inflight)
        if self.inflight[home] <= min_load + 2:
            return home
        return self._least_inflight(all_idx)

    async def handle_health(self, _request: web.Request) -> web.Response:
        assert self.session is not None

        async def one(w: str) -> bool:
            try:
                async with self.session.get(f"{w}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
            except Exception:
                return False

        results = await asyncio.gather(*(one(w) for w in self.workers))
        if all(results):
            return web.json_response({"status": "ok", "workers": self.n})
        return web.json_response({"status": "degraded", "healthy": sum(results), "total": self.n}, status=503)

    async def handle_stats(self, _request: web.Request) -> web.Response:
        return web.json_response(
            {
                "num_workers": self.n,
                "workers": self.workers,
                "inflight": self.inflight,
                "served": self.served,
            }
        )

    async def handle_flush_cache(self, _request: web.Request) -> web.Response:
        assert self.session is not None

        async def flush_one(w: str) -> int:
            for attempt in range(5):
                try:
                    async with self.session.post(f"{w}/flush_cache", timeout=aiohttp.ClientTimeout(total=15)) as r:
                        if r.status == 200:
                            return 1
                except Exception:
                    pass
                await asyncio.sleep(0.2 * (attempt + 1))
            return 0

        res = await asyncio.gather(*(flush_one(w) for w in self.workers))
        return web.json_response({"flushed": sum(res), "total": self.n})

    async def handle_server_info(self, _request: web.Request) -> web.Response:
        assert self.session is not None

        async def fetch_one(w: str) -> dict:
            try:
                async with self.session.get(f"{w}/get_server_info", timeout=aiohttp.ClientTimeout(total=10)) as r:
                    if r.status == 200:
                        return await r.json()
            except Exception:
                pass
            return {}

        infos = await asyncio.gather(*(fetch_one(w) for w in self.workers))
        max_tokens_list = [i.get("max_total_num_tokens") for i in infos if i.get("max_total_num_tokens")]
        return web.json_response(
            {
                "num_workers": self.n,
                "per_replica_max_total_num_tokens": max_tokens_list[0] if max_tokens_list else None,
                "slice_total_kv_tokens": sum(max_tokens_list) if max_tokens_list else None,
                "workers_info": infos,
            }
        )

    async def handle_models(self, _request: web.Request) -> web.Response:
        assert self.session is not None
        async with self.session.get(f"{self.workers[0]}/v1/models") as r:
            raw = await r.read()
            return web.Response(body=raw, status=r.status, content_type="application/json")

    async def _forward_prewarm(self, path: str, raw_body: bytes, candidates: list[int]) -> web.Response:
        assert self.session is not None

        async def warm_one(idx: int) -> dict:
            w = self.workers[idx]
            self.inflight[idx] += 1
            try:
                async with self.session.post(
                    f"{w}{path}",
                    data=raw_body,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    text = await resp.text()
                    return {"worker": w, "status": resp.status, "len": len(text)}
            finally:
                self.inflight[idx] -= 1
                self.served[idx] += 1

        results = await asyncio.gather(*(warm_one(i) for i in candidates))
        return web.json_response({"prewarmed": len(results), "results": results})

    async def handle_proxy_post(self, request: web.Request) -> web.StreamResponse:
        assert self.session is not None
        raw_body = await request.read()
        try:
            body = json.loads(raw_body)
        except Exception:
            body = {}

        path = request.path_qs
        if request.headers.get("X-Prewarm") == "1" and "X-Trunk-Index" in request.headers:
            t_idx = int(request.headers["X-Trunk-Index"])
            n_trunks = int(request.headers.get("X-Num-Trunks", "1"))
            conc = int(request.headers.get("X-Concurrency", str(self.n)))
            cands = self._trunk_candidates(t_idx, n_trunks, conc)
            return await self._forward_prewarm(path, raw_body, cands)

        idx = self.choose_worker(request.headers, body)
        target_url = f"{self.workers[idx]}{path}"
        is_stream = bool(body.get("stream", False))

        self.inflight[idx] += 1
        try:
            async with self.session.post(
                target_url,
                data=raw_body,
                headers={"Content-Type": "application/json"},
            ) as upstream:
                if not is_stream or upstream.status != 200:
                    payload = await upstream.read()
                    return web.Response(
                        body=payload,
                        status=upstream.status,
                        content_type=upstream.content_type or "application/json",
                    )
                resp = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Worker-Index": str(idx),
                    },
                )
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    if chunk:
                        await resp.write(chunk)
                await resp.write_eof()
                return resp
        finally:
            self.inflight[idx] -= 1
            self.served[idx] += 1


def kill_existing_gateway(port: int) -> None:
    my_pid = os.getpid()
    ppid = os.getppid()
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        cmd = parts[1]
        if pid in (1, my_pid, ppid):
            continue
        if "gateway_flint_v7x.py" in cmd and f"--port {port}" in cmd:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def build_app(workers: list[str]) -> web.Application:
    gw = ReplicaGateway(workers)
    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.on_startup.append(gw.start)
    app.on_cleanup.append(gw.cleanup)
    app.router.add_get("/health", gw.handle_health)
    app.router.add_get("/gateway_stats", gw.handle_stats)
    app.router.add_get("/get_server_info", gw.handle_server_info)
    app.router.add_post("/flush_cache", gw.handle_flush_cache)
    app.router.add_get("/v1/models", gw.handle_models)
    app.router.add_post("/v1/completions", gw.handle_proxy_post)
    app.router.add_post("/v1/chat/completions", gw.handle_proxy_post)
    app.router.add_post("/generate", gw.handle_proxy_post)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", required=True, help="Comma-separated worker base URLs")
    ap.add_argument("--daemon", action="store_true")
    ap.add_argument("--log-file", default="/workspace/logs/gateway.log")
    args = ap.parse_args()

    kill_existing_gateway(args.port)
    workers = [w.strip() for w in args.workers.split(",") if w.strip()]
    assert workers, "No workers provided"

    if args.daemon:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--workers",
            ",".join(workers),
        ]
        with open(args.log_file, "wb") as lf:
            subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"status": "daemon_started", "port": args.port, "num_workers": len(workers)}))
        return

    app = build_app(workers)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
