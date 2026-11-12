#!/usr/bin/env python3
"""Matched Matched-Workload Serving Benchmark & Quality Verifier for RecursiveSI/flint on TPU v7x-16.

Aligned 1:1 with peer v5p-16 benchmark protocol:
  - /v1/completions with deterministic token-ID arrays in [1000, 32999]
  - max_tokens=OSL, min_tokens=OSL, ignore_eos=True, temperature=0.0, stream=True
  - POST /flush_cache before every benchmark point + deterministic 64-token-aligned trunk pre-warm for 80%/95%
  - Immediate checkpointing after EVERY point to local disk AND GCS (/models/flint-v7x-serving-results/)
  - Automatic --resume support for zero-loss Spot preemption recovery
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


def generate_prompts(isl: int, conc: int, num_requests: int, hit_rate: float) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Generate deterministic token ID prompts and optional shared prefix trunks."""
    if hit_rate <= 0.0:
        prompts: list[list[int]] = []
        for i in range(num_requests):
            rng = random.Random(100000 + isl * 1000 + conc * 10 + i)
            prompts.append([rng.randint(1000, 32999) for _ in range(isl)])
        return [], prompts, []

    shared_len = (int(isl * hit_rate) // 64) * 64
    unique_len = max(1, isl - shared_len)
    shared_len = isl - unique_len
    num_trunks = max(1, min(8, conc // 4))

    trunks: list[list[int]] = []
    for t in range(num_trunks):
        rng_t = random.Random(50000 + isl * 100 + t)
        trunks.append([rng_t.randint(1000, 32999) for _ in range(shared_len)])

    prompts = []
    trunk_ids: list[int] = []
    for i in range(num_requests):
        t_id = i % num_trunks
        rng_u = random.Random(900000 + isl * 1000 + conc * 10 + i)
        suffix = [rng_u.randint(1000, 32999) for _ in range(unique_len)]
        prompts.append(trunks[t_id] + suffix)
        trunk_ids.append(t_id)
    return trunks, prompts, trunk_ids


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


async def run_quality_checks(base_url: str, model_name: str) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=600)
    results: dict[str, Any] = {}
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1) Coding correctness: merge overlapping intervals + unit test execution
        t0 = time.perf_counter()
        payload_code = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a concise Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                        "that merges all overlapping intervals and returns them sorted by start time. Return code in a ```python block."
                    ),
                }
            ],
            "max_tokens": 768,
            "temperature": 0.0,
        }
        async with session.post(f"{base_url}/v1/chat/completions", json=payload_code) as resp:
            data = await resp.json()
        dt_code = time.perf_counter() - t0
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        m = re.search(r"```python\s*(.*?)```", content, re.DOTALL)
        extracted_code = m.group(1).strip() if m else content.strip()
        unit_test_ok = False
        unit_err = ""
        test_harness = (
            extracted_code
            + "\n\nassert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\n"
            + "assert merge_intervals([[1,4],[4,5]]) == [[1,5]]\n"
            + "assert merge_intervals([]) == []\n"
            + "print('UNIT_TEST_PASS')\n"
        )
        try:
            out = subprocess.check_output([sys.executable, "-c", test_harness], stderr=subprocess.STDOUT, text=True, timeout=10)
            unit_test_ok = "UNIT_TEST_PASS" in out
        except Exception as e:
            unit_err = getattr(e, "output", str(e))[-300:]

        results["coding_merge_intervals"] = {
            "unit_test_passed": unit_test_ok,
            "latency_s": round(dt_code, 3),
            "has_reasoning_content": bool(reasoning),
            "reasoning_chars": len(reasoning),
            "content_preview": content[:400],
            "error": unit_err,
        }

        # 2) Long-context 32K needle-in-a-haystack passkey test (verifies SWA-513 + 14 Full NoPE + Attention Sinks!)
        passkey = "FLINT-TPUV7X-947261"
        filler_block = "def helper_noop(x: int) -> int:\n    return (x * 31 + 7) % 10007\n\n"
        # ~16 tokens per filler_block -> ~2000 blocks ~ 32K tokens, needle placed at 25% depth (far outside 513 SWA window!)
        blocks = [filler_block] * 500 + [f"# CRITICAL_DEPLOYMENT_PASSKEY = '{passkey}'\n\n"] + [filler_block] * 1400
        long_prompt = (
            "Below is a large Python module. Find the exact string assigned to `CRITICAL_DEPLOYMENT_PASSKEY` "
            "and reply with ONLY that exact string.\n\n"
            + "".join(blocks)
            + "\nWhat is the exact value of `CRITICAL_DEPLOYMENT_PASSKEY`?"
        )
        t1 = time.perf_counter()
        payload_needle = {
            "model": model_name,
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 256,
            "temperature": 0.0,
        }
        async with session.post(f"{base_url}/v1/chat/completions", json=payload_needle) as resp2:
            data2 = await resp2.json()
        dt_needle = time.perf_counter() - t1
        msg2 = data2.get("choices", [{}])[0].get("message", {})
        ans2 = (msg2.get("content") or "") + " " + (msg2.get("reasoning_content") or "")
        usage2 = data2.get("usage", {})
        results["long_context_32k_needle"] = {
            "needle_passed": passkey in ans2,
            "expected_passkey": passkey,
            "prompt_tokens": usage2.get("prompt_tokens"),
            "completion_tokens": usage2.get("completion_tokens"),
            "latency_s": round(dt_needle, 3),
            "response_preview": (msg2.get("content") or "")[:200],
        }
    return results


async def run_single_point(
    session: aiohttp.ClientSession,
    base_url: str,
    model_name: str,
    isl: int,
    osl: int,
    hit_rate: float,
    conc: int,
    num_requests: int,
    num_replicas: int,
) -> dict[str, Any]:
    # 1. Flush RadixCache on all replicas
    async with session.post(f"{base_url}/flush_cache") as r_flush:
        flush_info = await r_flush.json()

    trunks, prompts, trunk_ids = generate_prompts(isl, conc, num_requests, hit_rate)
    num_trunks = len(trunks)

    # 2. Pre-warm shared trunks if hit_rate > 0
    prewarm_sec = 0.0
    if num_trunks > 0:
        tw0 = time.perf_counter()
        for t_idx, trunk_tokens in enumerate(trunks):
            headers = {
                "X-Prewarm": "1",
                "X-Trunk-Index": str(t_idx),
                "X-Num-Trunks": str(num_trunks),
                "X-Concurrency": str(conc),
            }
            warm_body = {
                "model": model_name,
                "prompt": trunk_tokens,
                "max_tokens": 1,
                "min_tokens": 1,
                "ignore_eos": True,
                "temperature": 0.0,
                "stream": False,
            }
            async with session.post(f"{base_url}/v1/completions", json=warm_body, headers=headers) as rw:
                await rw.read()
        prewarm_sec = time.perf_counter() - tw0

    sem = asyncio.Semaphore(conc)
    ttfts: list[float] = []
    tpots: list[float] = []
    e2es: list[float] = []
    itls: list[float] = []
    out_tok_counts: list[int] = []
    in_tok_counts: list[int] = []
    cached_tok_counts: list[int] = []
    errors: list[str] = []

    async def one_req(req_idx: int) -> None:
        headers: dict[str, str] = {}
        if num_trunks > 0:
            headers["X-Trunk-Index"] = str(trunk_ids[req_idx])
            headers["X-Num-Trunks"] = str(num_trunks)
            headers["X-Concurrency"] = str(conc)
        else:
            headers["X-Policy"] = "least_inflight"

        body = {
            "model": model_name,
            "prompt": prompts[req_idx],
            "max_tokens": osl,
            "min_tokens": osl,
            "ignore_eos": True,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async with sem:
            t_start = time.perf_counter()
            t_first: float | None = None
            t_prev: float | None = None
            chunk_events = 0
            usage_prompt = isl
            usage_comp = 0
            usage_cached = 0
            try:
                async with session.post(f"{base_url}/v1/completions", json=body, headers=headers) as resp:
                    if resp.status != 200:
                        err_txt = (await resp.text())[:300]
                        errors.append(f"HTTP {resp.status}: {err_txt}")
                        return
                    buf = b""
                    async for raw_chunk in resp.content.iter_any():
                        now = time.perf_counter()
                        buf += raw_chunk
                        while b"\n" in buf:
                            line_b, buf = buf.split(b"\n", 1)
                            line = line_b.strip()
                            if not line or not line.startswith(b"data:"):
                                continue
                            payload_b = line[5:].strip()
                            if payload_b == b"[DONE]":
                                continue
                            try:
                                obj = json.loads(payload_b)
                            except Exception:
                                continue
                            choices = obj.get("choices") or []
                            if choices:
                                txt = choices[0].get("text")
                                if txt is not None and txt != "":
                                    if t_first is None:
                                        t_first = now
                                    elif t_prev is not None:
                                        itls.append((now - t_prev) * 1000.0)
                                    t_prev = now
                                    chunk_events += 1
                            u = obj.get("usage")
                            if u:
                                usage_prompt = int(u.get("prompt_tokens") or isl)
                                usage_comp = int(u.get("completion_tokens") or 0)
                                details = u.get("prompt_tokens_details") or {}
                                if details.get("cached_tokens") is not None:
                                    usage_cached = int(details["cached_tokens"])
                t_end = time.perf_counter()
                out_n = usage_comp if usage_comp > 0 else max(chunk_events, osl)
                if t_first is None:
                    t_first = t_end
                ttft_ms = (t_first - t_start) * 1000.0
                e2e_s = t_end - t_start
                tpot_ms = ((t_end - t_first) * 1000.0) / max(out_n - 1, 1)
                ttfts.append(ttft_ms)
                e2es.append(e2e_s)
                tpots.append(tpot_ms)
                out_tok_counts.append(out_n)
                in_tok_counts.append(usage_prompt)
                cached_tok_counts.append(usage_cached)
            except Exception as e:
                errors.append(repr(e))

    t_bench0 = time.perf_counter()
    await asyncio.gather(*(one_req(i) for i in range(num_requests)))
    wall_s = max(time.perf_counter() - t_bench0, 1e-6)

    total_out = sum(out_tok_counts)
    total_in = sum(in_tok_counts)
    total_cached = sum(cached_tok_counts)
    slice_out_tps = total_out / wall_s
    slice_tot_tps = (total_in + total_out) / wall_s
    tpot_mean = mean(tpots)
    obs_hit = (total_cached / max(total_in, 1)) * 100.0 if total_cached > 0 else (hit_rate * 100.0 if hit_rate == 0 else None)

    # Also query /get_server_info post-point for SGLang internal cache hit rate & KV pool metrics
    srv_hit_rates: list[float] = []
    per_rep_kv = None
    slice_kv = None
    try:
        async with session.get(f"{base_url}/get_server_info") as rs:
            if rs.status == 200:
                sinfo = await rs.json()
                per_rep_kv = sinfo.get("per_replica_max_total_num_tokens")
                slice_kv = sinfo.get("slice_total_kv_tokens")
                for wi in sinfo.get("workers_info") or []:
                    for st in wi.get("internal_states") or []:
                        hr = st.get("last_gen_throughput")  # keep raw if needed
                        chr_val = st.get("cache_hit_rate") or st.get("radix_cache_hit_rate")
                        if isinstance(chr_val, (int, float)):
                            srv_hit_rates.append(float(chr_val) * 100.0 if chr_val <= 1.0 else float(chr_val))
    except Exception:
        pass

    if obs_hit is None and srv_hit_rates:
        obs_hit = round(mean(srv_hit_rates), 2)
    elif obs_hit is not None:
        obs_hit = round(obs_hit, 2)

    return {
        "isl": isl,
        "osl": osl,
        "target_hit_rate": hit_rate,
        "concurrency": conc,
        "num_requests": num_requests,
        "completed_requests": len(out_tok_counts),
        "failed_requests": len(errors),
        "errors_sample": errors[:3],
        "wall_sec": round(wall_s, 3),
        "prewarm_sec": round(prewarm_sec, 3),
        "flush_info": flush_info,
        "total_prompt_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cached_tokens": total_cached,
        "observed_cache_hit_pct": obs_hit,
        "slice_output_tok_s": round(slice_out_tps, 2),
        "per_host_output_tok_s": round(slice_out_tps / 4.0, 2),
        "per_chip_output_tok_s": round(slice_out_tps / 16.0, 2),
        "per_replica_output_tok_s": round(slice_out_tps / max(num_replicas, 1), 2),
        "slice_total_tok_s": round(slice_tot_tps, 2),
        "single_stream_tok_s": round(1000.0 / tpot_mean, 2) if tpot_mean > 0 else 0.0,
        "ttft_ms_mean": round(mean(ttfts), 2),
        "ttft_ms_p50": round(pct(ttfts, 50), 2),
        "ttft_ms_p99": round(pct(ttfts, 99), 2),
        "tpot_ms_mean": round(tpot_mean, 3),
        "tpot_ms_p50": round(pct(tpots, 50), 3),
        "tpot_ms_p99": round(pct(tpots, 99), 3),
        "itl_ms_mean": round(mean(itls), 3),
        "itl_ms_p50": round(pct(itls, 50), 3),
        "itl_ms_p99": round(pct(itls, 99), 3),
        "e2e_s_mean": round(mean(e2es), 3),
        "e2e_s_p50": round(pct(e2es, 50), 3),
        "e2e_s_p99": round(pct(e2es, 99), 3),
        "per_replica_kv_tokens": per_rep_kv,
        "slice_kv_tokens": slice_kv,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def choose_num_requests(isl: int, conc: int, req_mult: int, max_reqs: int) -> int:
    if isl >= 65536:
        # Long-context: keep runtime bounded while still exercising full concurrency
        if conc <= 2:
            return max(2, conc * 2)
        return min(max(conc, 8), 64)
    if conc == 1:
        return 4
    if conc == 2:
        return 6
    return min(max(conc * req_mult, 16), max_reqs)


def persist_bundle(bundle: dict[str, Any], local_out: Path, gcs_out: Path | None) -> None:
    local_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_local = local_out.with_suffix(".json.tmp")
    payload = json.dumps(bundle, indent=2) + "\n"
    tmp_local.write_text(payload)
    tmp_local.replace(local_out)
    if gcs_out is not None:
        try:
            gcs_out.parent.mkdir(parents=True, exist_ok=True)
            gcs_out.write_text(payload)
        except Exception as e:
            print(f"[warn] GCS write to {gcs_out} failed: {e}", file=sys.stderr, flush=True)


async def async_main(args: argparse.Namespace) -> None:
    local_out = Path(args.output)
    gcs_out = Path(args.gcs_output) if args.gcs_output else None

    bundle: dict[str, Any] = {
        "run_name": args.run_name,
        "hardware": "TPU v7x-16 ( 4 hosts x 4 chips = 16 chips / 32 TensorCores, 2x2x4)",
        "config": {
            "model_name": args.model_name,
            "weights_dtype": args.weights_dtype,
            "kv_cache_dtype": args.kv_cache_dtype,
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "dp_replicas": args.num_replicas,
            "attention_backend": args.attention_backend,
            "notes": args.notes,
        },
        "quality_checks": {},
        "points": [],
    }

    done_keys: set[tuple[int, int, float, int]] = set()
    if args.resume:
        for candidate in ([gcs_out, local_out] if gcs_out else [local_out]):
            if candidate and candidate.exists():
                try:
                    prev = json.loads(candidate.read_text())
                    bundle["points"] = prev.get("points", [])
                    bundle["quality_checks"] = prev.get("quality_checks", {})
                    for p in bundle["points"]:
                        if p.get("failed_requests", 0) == 0 and p.get("completed_requests", 0) > 0:
                            done_keys.add((int(p["isl"]), int(p["osl"]), float(p["target_hit_rate"]), int(p["concurrency"])))
                    print(f"[resume] Loaded {len(done_keys)} completed points from {candidate}", flush=True)
                    break
                except Exception:
                    pass

    if args.verify_quality and not bundle.get("quality_checks"):
        print(f"[{args.run_name}] Running functional & 32K needle quality verification...", flush=True)
        bundle["quality_checks"] = await run_quality_checks(args.base_url, args.model_name)
        print(f"[{args.run_name}] Quality results: {json.dumps(bundle['quality_checks'])}", flush=True)
        persist_bundle(bundle, local_out, gcs_out)

    shapes = []
    for item in args.shapes.split(","):
        item = item.strip()
        if not item:
            continue
        i_s, o_s = item.split(":")
        shapes.append((int(i_s), int(o_s)))

    hit_rates = [float(x.strip()) for x in args.hit_rates.split(",") if x.strip()]
    concs = [int(x.strip()) for x in args.concurrencies.split(",") if x.strip()]

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=1800, connect=30, sock_read=1800)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for isl, osl in shapes:
            for hr in hit_rates:
                for conc in concs:
                    if args.max_total_seq_tokens > 0 and isl * min(conc, max(1, conc // max(args.num_replicas, 1))) > args.max_total_seq_tokens:
                        print(f"[skip] {isl}:{osl} hit={hr} c={conc} exceeds per-replica token cap", flush=True)
                        continue
                    key = (isl, osl, hr, conc)
                    if key in done_keys:
                        print(f"[skip-done] {isl}:{osl} hit={hr} c={conc}", flush=True)
                        continue
                    n_reqs = args.fixed_num_requests if args.fixed_num_requests > 0 else choose_num_requests(isl, conc, args.req_mult, args.max_reqs)
                    print(
                        f"[{args.run_name}] START isl={isl} osl={osl} hit={int(hr*100)}% c={conc} n_reqs={n_reqs} ...",
                        flush=True,
                    )
                    pt = await run_single_point(
                        session=session,
                        base_url=args.base_url.rstrip("/"),
                        model_name=args.model_name,
                        isl=isl,
                        osl=osl,
                        hit_rate=hr,
                        conc=conc,
                        num_requests=n_reqs,
                        num_replicas=args.num_replicas,
                    )
                    bundle["points"].append(pt)
                    persist_bundle(bundle, local_out, gcs_out)
                    print(
                        f"[{args.run_name}] DONE  isl={isl} osl={osl} hit={int(hr*100)}% c={conc} | "
                        f"out={pt['slice_output_tok_s']} tok/s (rep={pt['per_replica_output_tok_s']}) | "
                        f"TTFT p50={pt['ttft_ms_p50']}ms p99={pt['ttft_ms_p99']}ms | "
                        f"TPOT p50={pt['tpot_ms_p50']}ms (1-stream={pt['single_stream_tok_s']} tok/s) | "
                        f"E2E p50={pt['e2e_s_p50']}s | errs={pt['failed_requests']}",
                        flush=True,
                    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model-name", default="/dev/shm/flint")
    ap.add_argument("--weights-dtype", default="bf16")
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--num-replicas", type=int, default=16)
    ap.add_argument("--attention-backend", default="pallas_tpu_rpa_v3")
    ap.add_argument("--notes", default="")
    ap.add_argument("--shapes", default="4096:1024,8192:1024,16384:1024")
    ap.add_argument("--hit-rates", default="0.0,0.8,0.95")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--req-mult", type=int, default=2)
    ap.add_argument("--max-reqs", type=int, default=512)
    ap.add_argument("--fixed-num-requests", type=int, default=0)
    ap.add_argument("--max-total-seq-tokens", type=int, default=0)
    ap.add_argument("--verify-quality", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--gcs-output", default="")
    args = ap.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
