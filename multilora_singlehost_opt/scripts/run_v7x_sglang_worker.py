#!/usr/bin/env python3
"""Distributed/Single-Host SGLang-TorchTPU Optimization Matrix Runner for RecursiveSI/flint on TPU v7x (8 TensorCores).

Supports concurrent execution across multiple independent 1-host v7x pods (e.g. `worker-0` forward + `worker-1` reverse)
via atomic claim files + per-point JSON checkpointing on GCS (`/models/benchmark-results/flint_v7x_singlehost_opt/`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np


if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

GCS_ROOT = Path("/models/benchmark-results/flint_v7x_singlehost_opt")
RUNS_DIR = GCS_ROOT / "runs"
CLAIMS_DIR = GCS_ROOT / "claims"
LOCAL_RESULTS = Path("/workspace/results")


@dataclass
class SweepPointGroup:
    shapes: str
    hit_rates: str
    concurrencies: str
    req_mult: int = 1
    max_reqs: int = 128
    max_total_seq_tokens: int = 0


@dataclass
class MatrixConfig:
    config_id: str
    axis: str
    description: str
    model_path: str = "/dev/shm/flint-fp8"
    weights_dtype: str = "fp8_block128"
    kv_cache_dtype: str = "fp8_e4m3"
    tp_size: int = 2
    dp_size: int = 1
    ep_size: int = 1
    num_replicas: int = 4
    moe_runner_backend: str = "auto"
    swa_full_tokens_ratio: float = 0.10
    chunked_prefill_size: int = 8192
    max_prefill_tokens: int = 16384
    max_running_requests: int = 64
    torch_compile_max_bs: int = 64
    context_length: int = 135168
    mem_fraction_static: float = 0.85
    disable_torch_compile: bool = False
    extra_args: str = ""
    fallback_disable_compile: bool = False
    verify_quality: bool = False
    run_realistic_spec_suite: bool = False
    wait_timeout: int = 900
    groups: list[SweepPointGroup] = field(default_factory=list)


MATRIX: list[MatrixConfig] = [
    # ---------------- AXIS A: Parallelism (TP x DP x EP across 8 TensorCores) ----------------
    MatrixConfig(
        config_id="A1_fp8_tp2_dp4",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, TP=2 x DP=4 (4 intra-package chip replicas, zero inter-chip ICI), SWA ratio 0.10",
        verify_quality=True,
        run_realistic_spec_suite=True,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.5,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=2200000),
        ],
    ),
    MatrixConfig(
        config_id="A2_fp8_tp1_dp8",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, TP=1 x DP=8 (4 servers x --tp-size 1 --dp-size 2 = 8 independent TensorCore workers)",
        tp_size=1,
        dp_size=2,
        num_replicas=4,
        mem_fraction_static=0.82,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.0,0.95", "1,16,32", req_mult=1, max_reqs=32, max_total_seq_tokens=900000),
        ],
    ),
    MatrixConfig(
        config_id="A3_fp8_tp4_dp2",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, TP=4 x DP=2 (2 replicas x 2 v7x chips / 4 TensorCores each)",
        tp_size=4,
        dp_size=1,
        num_replicas=2,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=4200000),
        ],
    ),
    MatrixConfig(
        config_id="A4_fp8_tp8_dp1",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, TP=8 x DP=1 (single 8-TensorCore replica across whole host for lowest TTFT/TPOT)",
        tp_size=8,
        dp_size=1,
        num_replicas=1,
        run_realistic_spec_suite=True,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64),
        ],
    ),
    MatrixConfig(
        config_id="A5_fp8_tp8_ep8",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, TP=8 + EP=8 (32 routed experts per TensorCore via expert parallelism)",
        tp_size=8,
        dp_size=1,
        ep_size=8,
        num_replicas=1,
        wait_timeout=900,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.0,0.95", "1,16,32", req_mult=1, max_reqs=32),
        ],
    ),
    MatrixConfig(
        config_id="A6_fp8_dpattn8_ep8",
        axis="A_parallelism",
        description="Block-FP8 weights + FP8 KV, DP-Attention (DP=4, attn_tp=2, TP=8) + EP=8 (--enable-dp-attention --enable-dp-lm-head)",
        tp_size=8,
        dp_size=4,
        ep_size=8,
        num_replicas=1,
        extra_args="--enable-dp-attention --enable-dp-lm-head",
        fallback_disable_compile=True,
        wait_timeout=900,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
        ],
    ),
    # ---------------- AXIS B: Speculative Decoding (NGRAM Draft/Verify) ----------------
    MatrixConfig(
        config_id="B1_fp8_tp2_dp4_ngram4",
        axis="B_speculative_decoding",
        description="N-Gram Speculative Decoding (draft_tokens=4, bfs=[1,4]) on Block-FP8 TP=2 x DP=4",
        tp_size=2,
        dp_size=1,
        num_replicas=4,
        extra_args="--speculative-algorithm NGRAM --speculative-num-draft-tokens 4 --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 4",
        fallback_disable_compile=True,
        run_realistic_spec_suite=True,
        wait_timeout=750,
        groups=[
            SweepPointGroup("1024:512,4096:512", "0.0,0.95", "1,4,16,64", req_mult=1, max_reqs=64),
        ],
    ),
    MatrixConfig(
        config_id="B2_fp8_tp8_ep8_ngram6",
        axis="B_speculative_decoding",
        description="N-Gram Speculative Decoding (draft_tokens=6, bfs=[1,4]) on Block-FP8 TP=8 + EP=8 ultra-low-latency config",
        tp_size=8,
        dp_size=1,
        ep_size=8,
        num_replicas=1,
        extra_args="--speculative-algorithm NGRAM --speculative-num-draft-tokens 6 --speculative-ngram-min-bfs-breadth 1 --speculative-ngram-max-bfs-breadth 4",
        fallback_disable_compile=True,
        run_realistic_spec_suite=True,
        wait_timeout=750,
        groups=[
            SweepPointGroup("1024:512,4096:512", "0.0,0.95", "1,4,16,64", req_mult=1, max_reqs=64),
        ],
    ),
    # ---------------- AXIS C: Precision & MoE Kernels ----------------
    MatrixConfig(
        config_id="C1_bf16_tp2_dp4_fp8kv",
        axis="C_precision_kernels",
        description="BF16 weights (79.6 GB) + FP8_e4m3 KV cache, TP=2 x DP=4, SWA ratio 0.10",
        model_path="/dev/shm/flint",
        weights_dtype="bf16",
        kv_cache_dtype="fp8_e4m3",
        verify_quality=True,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=1600000),
        ],
    ),
    MatrixConfig(
        config_id="C2_bf16_tp2_dp4_bf16kv",
        axis="C_precision_kernels",
        description="BF16 weights + BF16 KV cache (full unquantized baseline), TP=2 x DP=4, SWA ratio 0.10",
        model_path="/dev/shm/flint",
        weights_dtype="bf16",
        kv_cache_dtype="bfloat16",
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.0,0.95", "1,16,32", req_mult=1, max_reqs=32, max_total_seq_tokens=900000),
        ],
    ),
    MatrixConfig(
        config_id="C3_fp8_tp2_dp4_bf16kv",
        axis="C_precision_kernels",
        description="Block-FP8 weights + BF16 KV cache (completes 2x2 weight x KV precision matrix), TP=2 x DP=4",
        model_path="/dev/shm/flint-fp8",
        weights_dtype="fp8_block128",
        kv_cache_dtype="bfloat16",
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=1400000),
        ],
    ),
    MatrixConfig(
        config_id="C4_fp8_tp2_dp4_fused_rs",
        axis="C_precision_kernels",
        description="Block-FP8 weights + FP8 KV with moe_runner_backend=pallas_tpu_fused_rs (fused MoE + ReduceScatter)",
        moe_runner_backend="pallas_tpu_fused_rs",
        wait_timeout=750,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.95", "1,32", req_mult=1, max_reqs=32),
        ],
    ),
    MatrixConfig(
        config_id="C5_w8a8_int8_tp2_dp4",
        axis="C_precision_kernels",
        description="Online W8A8 INT8 quantization (--quantization w8a8_int8) from BF16 checkpoint, TP=2 x DP=4",
        model_path="/dev/shm/flint",
        weights_dtype="w8a8_int8",
        extra_args="--quantization w8a8_int8",
        wait_timeout=600,
        groups=[
            SweepPointGroup("1024:1024,4096:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
        ],
    ),
    # ---------------- AXIS D: RadixCache, SWA Memory Ratio, Chunked Prefill & Batching ----------------
    MatrixConfig(
        config_id="D1_fp8_tp2_dp4_noradix",
        axis="D_caching_memory_scheduling",
        description="RadixCache DISABLED (--disable-radix-cache) on Block-FP8 TP=2 x DP=4 across 0%, 50%, 95% shared trunks",
        extra_args="--disable-radix-cache",
        groups=[
            SweepPointGroup("4096:1024", "0.0,0.5,0.95", "1,16,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.95", "1,16", req_mult=1, max_reqs=16),
        ],
    ),
    MatrixConfig(
        config_id="D2_fp8_tp2_dp4_swa005",
        axis="D_caching_memory_scheduling",
        description="Hybrid SWA KV pool ratio=0.05 (maximizes 14-layer Full-Attention NoPE KV budget for 64K/128K concurrency)",
        swa_full_tokens_ratio=0.05,
        groups=[
            SweepPointGroup("4096:1024", "0.95", "1,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64,128", req_mult=1, max_reqs=64, max_total_seq_tokens=2600000),
        ],
    ),
    MatrixConfig(
        config_id="D3_fp8_tp2_dp4_swa020",
        axis="D_caching_memory_scheduling",
        description="Hybrid SWA KV pool ratio=0.20 (larger 35-layer SWA-513 pool vs smaller Full-Attention pool)",
        swa_full_tokens_ratio=0.20,
        groups=[
            SweepPointGroup("4096:1024", "0.95", "1,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=2000000),
        ],
    ),
    MatrixConfig(
        config_id="D4_fp8_tp2_dp4_cp16k_mrr128",
        axis="D_caching_memory_scheduling",
        description="Aggressive batching & prefill chunking: chunked_prefill_size=16384, max_prefill_tokens=32768, max_running_requests=128",
        chunked_prefill_size=16384,
        max_prefill_tokens=32768,
        max_running_requests=128,
        torch_compile_max_bs=128,
        groups=[
            SweepPointGroup("4096:1024", "0.0,0.95", "1,64,128,256", req_mult=1, max_reqs=256),
            SweepPointGroup("65536:1024,131072:1024", "0.0,0.95", "1,16,64", req_mult=1, max_reqs=64, max_total_seq_tokens=2200000),
        ],
    ),
    MatrixConfig(
        config_id="D5_fp8_tp2_dp4_cp4k_mrr32",
        axis="D_caching_memory_scheduling",
        description="Latency-protecting prefill chunking: chunked_prefill_size=4096, max_prefill_tokens=8192, max_running_requests=32",
        chunked_prefill_size=4096,
        max_prefill_tokens=8192,
        max_running_requests=32,
        torch_compile_max_bs=32,
        groups=[
            SweepPointGroup("4096:1024", "0.0,0.95", "1,32,64,128", req_mult=1, max_reqs=128),
            SweepPointGroup("65536:1024", "0.0,0.95", "1,16,32", req_mult=1, max_reqs=32),
        ],
    ),
]


SPEC_CODE_PROMPTS = [
    (
        "Refactor the following Python module so every method adds type annotations, docstrings, and logging while "
        "preserving exact function names, field names, and return dictionaries:\n\n"
        "```python\n"
        "class OrderRouter:\n"
        "    def __init__(self, venues, max_slippage_bps=12, timeout_ms=250, enable_hedging=True):\n"
        "        self.venues = venues\n"
        "        self.max_slippage_bps = max_slippage_bps\n"
        "        self.timeout_ms = timeout_ms\n"
        "        self.enable_hedging = enable_hedging\n"
        "        self.metrics = {'routed': 0, 'filled': 0, 'rejected': 0, 'hedged': 0}\n\n"
        "    def route_order(self, order_id, symbol, side, quantity, limit_price):\n"
        "        best_venue = None\n"
        "        best_px = None\n"
        "        for venue in self.venues:\n"
        "            quote = venue.get_quote(symbol, side, quantity)\n"
        "            if quote is None or not quote.get('available', False):\n"
        "                continue\n"
        "            if best_px is None or (side == 'BUY' and quote['price'] < best_px) or (side == 'SELL' and quote['price'] > best_px):\n"
        "                best_px = quote['price']\n"
        "                best_venue = venue\n"
        "        if best_venue is None:\n"
        "            self.metrics['rejected'] += 1\n"
        "            return {'order_id': order_id, 'status': 'REJECTED', 'venue': None, 'fill_price': None}\n"
        "        self.metrics['routed'] += 1\n"
        "        fill = best_venue.execute(order_id, symbol, side, quantity, limit_price, self.timeout_ms)\n"
        "        self.metrics['filled'] += int(fill['status'] == 'FILLED')\n"
        "        return {'order_id': order_id, 'status': fill['status'], 'venue': best_venue.name, 'fill_price': fill['price']}\n"
        "```\n"
        "Output the complete refactored Python module now:\n```python\n"
    ),
    (
        "Convert the following Kubernetes deployment JSON manifest into an updated JSON manifest where `replicas` is changed "
        "from 2 to 4 and `LOG_LEVEL` is changed from `INFO` to `DEBUG`, keeping every other key and value verbatim:\n\n"
        '{\n  "apiVersion": "apps/v1",\n  "kind": "Deployment",\n  "metadata": {\n    "name": "flint-router-gateway",\n'
        '    "namespace": "serving-prod",\n    "labels": {"app": "flint-router-gateway", "tier": "edge", "owner": "ml-infra"}\n  },\n'
        '  "spec": {\n    "replicas": 2,\n    "selector": {"matchLabels": {"app": "flint-router-gateway"}},\n'
        '    "template": {\n      "metadata": {"labels": {"app": "flint-router-gateway", "tier": "edge"}},\n'
        '      "spec": {\n        "containers": [{\n          "name": "router",\n          "image": "registry.internal/serving/flint-router:v2.4.1",\n'
        '          "ports": [{"containerPort": 8000, "protocol": "TCP"}, {"containerPort": 9090, "protocol": "TCP"}],\n'
        '          "env": [\n            {"name": "LOG_LEVEL", "value": "INFO"},\n            {"name": "MAX_INFLIGHT", "value": "256"},\n'
        '            {"name": "CACHE_AFFINITY_WINDOW", "value": "64"},\n            {"name": "UPSTREAM_TIMEOUT_MS", "value": "30000"}\n          ]\n'
        "        }]\n      }\n    }\n  }\n}\n\nUpdated JSON:\n"
    ),
]


def free_dshm_gb() -> float:
    st = os.statvfs("/dev/shm")
    return (st.f_bavail * st.f_frsize) / 1e9


def prune_inductor_caches_if_needed(min_free_gb: float = 120.0) -> None:
    if free_dshm_gb() >= min_free_gb:
        return
    cands = sorted(Path("/dev/shm").glob("ind_*"), key=lambda p: p.stat().st_mtime)
    for p in cands:
        shutil.rmtree(p, ignore_errors=True)
        if free_dshm_gb() >= min_free_gb + 40.0:
            break


def is_config_complete(cfg: MatrixConfig) -> bool:
    out_file = RUNS_DIR / f"{cfg.config_id}.json"
    if not out_file.exists():
        return False
    try:
        data = json.loads(out_file.read_text())
        return data.get("status") in ("completed", "unsupported")
    except Exception:
        return False


def try_claim_config(cfg: MatrixConfig, worker_id: str) -> bool:
    CLAIMS_DIR.mkdir(parents=True, exist_ok=True)
    claim_path = CLAIMS_DIR / f"{cfg.config_id}.claim"
    if is_config_complete(cfg):
        return False
    if claim_path.exists():
        try:
            info = json.loads(claim_path.read_text())
            age = time.time() - float(info.get("ts", 0))
            owner = info.get("worker_id", "")
            # Allow reclaiming own interrupted job or stale claim > 45 min
            if owner != worker_id and age < 2700:
                return False
        except Exception:
            pass
    payload = json.dumps({"config_id": cfg.config_id, "worker_id": worker_id, "host": os.uname().nodename, "ts": time.time()})
    claim_path.write_text(payload)
    time.sleep(0.5)
    try:
        verify = json.loads(claim_path.read_text())
        return verify.get("worker_id") == worker_id
    except Exception:
        return True


async def run_realistic_spec_benchmark(base_url: str, model_name: str, concurrencies: list[int] = [1, 4, 16, 64]) -> list[dict[str, Any]]:
    """Benchmarks structured code-refactoring + JSON-editing prompts with high N-gram overlap."""
    results: list[dict[str, Any]] = []
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for conc in concurrencies:
            async with session.post(f"{base_url}/flush_cache") as rf:
                await rf.read()
            n_reqs = 4 if conc == 1 else min(max(conc * 2, 8), 64)
            sem = asyncio.Semaphore(conc)
            ttfts: list[float] = []
            tpots: list[float] = []
            e2es: list[float] = []
            out_toks: list[int] = []
            in_toks: list[int] = []
            errs: list[str] = []

            async def one_call(idx: int) -> None:
                prompt_str = SPEC_CODE_PROMPTS[idx % len(SPEC_CODE_PROMPTS)] + f"# request_nonce={idx}\n"
                body = {
                    "model": model_name,
                    "prompt": prompt_str,
                    "max_tokens": 384,
                    "min_tokens": 384,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                async with sem:
                    t0 = time.perf_counter()
                    t_first = None
                    u_in = 0
                    u_out = 0
                    chunks = 0
                    try:
                        async with session.post(f"{base_url}/v1/completions", json=body, headers={"X-Policy": "least_inflight"}) as resp:
                            if resp.status != 200:
                                errs.append(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                                return
                            buf = b""
                            async for raw in resp.content.iter_any():
                                now = time.perf_counter()
                                buf += raw
                                while b"\n" in buf:
                                    line_b, buf = buf.split(b"\n", 1)
                                    line = line_b.strip()
                                    if not line.startswith(b"data:"):
                                        continue
                                    pb = line[5:].strip()
                                    if pb == b"[DONE]":
                                        continue
                                    try:
                                        obj = json.loads(pb)
                                    except Exception:
                                        continue
                                    ch = obj.get("choices") or []
                                    if ch and ch[0].get("text"):
                                        if t_first is None:
                                            t_first = now
                                        chunks += 1
                                    u = obj.get("usage")
                                    if u:
                                        u_in = int(u.get("prompt_tokens") or u_in)
                                        u_out = int(u.get("completion_tokens") or u_out)
                        t1 = time.perf_counter()
                        if t_first is None:
                            t_first = t1
                        n_out = u_out if u_out > 0 else max(chunks, 384)
                        ttfts.append((t_first - t0) * 1000.0)
                        tpots.append(((t1 - t_first) * 1000.0) / max(n_out - 1, 1))
                        e2es.append(t1 - t0)
                        out_toks.append(n_out)
                        in_toks.append(u_in)
                    except Exception as e:
                        errs.append(repr(e))

            tw0 = time.perf_counter()
            await asyncio.gather(*(one_call(i) for i in range(n_reqs)))
            wall = max(time.perf_counter() - tw0, 1e-6)
            spec_stats: list[dict] = []
            try:
                async with session.get(f"{base_url}/get_server_info") as rs:
                    if rs.status == 200:
                        sinfo = await rs.json()
                        for wi in sinfo.get("workers_info") or []:
                            for st in wi.get("internal_states") or []:
                                keep = {k: v for k, v in st.items() if "spec" in k.lower() or "accept" in k.lower() or "draft" in k.lower() or "avg_spec" in k.lower()}
                                if keep:
                                    spec_stats.append(keep)
            except Exception:
                pass
            tot_out = sum(out_toks)
            tpot_m = float(np.mean(tpots)) if tpots else 0.0
            results.append(
                {
                    "workload": "structured_code_json_edit_384osl",
                    "concurrency": conc,
                    "num_requests": n_reqs,
                    "completed_requests": len(out_toks),
                    "failed_requests": len(errs),
                    "wall_sec": round(wall, 3),
                    "host_output_tok_s": round(tot_out / wall, 2),
                    "per_chip_output_tok_s": round((tot_out / wall) / 4.0, 2),
                    "single_stream_tok_s": round(1000.0 / tpot_m, 2) if tpot_m > 0 else 0.0,
                    "ttft_ms_p50": round(float(np.percentile(ttfts, 50)), 2) if ttfts else 0.0,
                    "tpot_ms_mean": round(tpot_m, 3),
                    "tpot_ms_p50": round(float(np.percentile(tpots, 50)), 3) if tpots else 0.0,
                    "e2e_s_p50": round(float(np.percentile(e2es, 50)), 3) if e2es else 0.0,
                    "spec_internal_states": spec_stats[:4],
                }
            )
            print(
                f"  [spec-suite] c={conc} | host_out={results[-1]['host_output_tok_s']} tok/s | "
                f"1-stream={results[-1]['single_stream_tok_s']} tok/s (TPOT p50={results[-1]['tpot_ms_p50']} ms) | spec={spec_stats[:1]}",
                flush=True,
            )
    return results


def _try_adopt_running_servers(cfg: MatrixConfig) -> tuple[bool, dict[str, Any]] | None:
    import urllib.request

    log0 = Path(f"/workspace/logs/{cfg.config_id}_rep0_port30000.log")
    if not log0.exists() or (time.time() - log0.stat().st_mtime) > 1200:
        return None
    try:
        ps_out = subprocess.check_output(["ps", "-eo", "args"], text=True)
    except Exception:
        return None
    live_ports = [r for r in range(cfg.num_replicas) if f"sglang.launch_server" in ps_out and f"--port {30000 + r}" in ps_out]
    if len(live_ports) != cfg.num_replicas:
        return None
    print(f"[{cfg.config_id}] Adopting {cfg.num_replicas} in-flight SGLang servers and waiting for /health...", flush=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        ok_cnt = 0
        for r in range(cfg.num_replicas):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{30000 + r}/health", timeout=2.5) as resp:
                    if resp.status == 200:
                        ok_cnt += 1
            except Exception:
                pass
        if ok_cnt == cfg.num_replicas:
            import launch_host_replicas as lhr

            rep_info = []
            for r in range(cfg.num_replicas):
                lp = Path(f"/workspace/logs/{cfg.config_id}_rep{r}_port{30000 + r}.log")
                sinfo = lhr.http_get_json(f"http://127.0.0.1:{30000 + r}/get_server_info") or {}
                telem = lhr.parse_log_telemetry(lp)
                rep_info.append(
                    {
                        "replica": r,
                        "port": 30000 + r,
                        "log_path": str(lp),
                        "max_total_num_tokens": sinfo.get("max_total_num_tokens", telem.get("max_total_num_tokens")),
                        "internal_states": sinfo.get("internal_states"),
                        "telemetry": telem,
                    }
                )
            workers = ",".join(f"http://127.0.0.1:{30000 + r}" for r in range(cfg.num_replicas))
            subprocess.run(
                [
                    sys.executable,
                    "/workspace/gateway_flint_v7x.py",
                    "--port",
                    "8000",
                    "--workers",
                    workers,
                    "--daemon",
                    "--log-file",
                    f"/workspace/logs/gw_{cfg.config_id}.log",
                ],
                check=True,
            )
            time.sleep(1.5)
            return True, {
                "host": os.uname().nodename,
                "status": "ready",
                "adopted": True,
                "adopt_wait_sec": round(time.time() - t0, 1),
                "num_replicas": cfg.num_replicas,
                "tp_size": cfg.tp_size,
                "ep_size": cfg.ep_size,
                "replicas": rep_info,
            }
        # Verify none exited
        ps_now = subprocess.check_output(["ps", "-eo", "args"], text=True)
        if sum(1 for r in range(cfg.num_replicas) if f"--port {30000 + r}" in ps_now) < cfg.num_replicas:
            break
        time.sleep(3.0)
    return None


def start_servers_for_config(cfg: MatrixConfig, disable_compile_override: bool = False) -> tuple[bool, dict[str, Any]]:
    if not disable_compile_override:
        adopted = _try_adopt_running_servers(cfg)
        if adopted is not None:
            return adopted
    eff_timeout = max(cfg.wait_timeout, 1800)
    cmd = [
        sys.executable,
        "/workspace/launch_host_replicas.py",
        "--mode",
        "start",
        "--tag",
        cfg.config_id,
        "--model-path",
        cfg.model_path,
        "--tp-size",
        str(cfg.tp_size),
        "--dp-size",
        str(cfg.dp_size),
        "--ep-size",
        str(cfg.ep_size),
        "--num-replicas",
        str(cfg.num_replicas),
        "--kv-cache-dtype",
        cfg.kv_cache_dtype,
        "--moe-runner-backend",
        cfg.moe_runner_backend,
        "--context-length",
        str(cfg.context_length),
        "--mem-fraction-static",
        str(cfg.mem_fraction_static),
        "--chunked-prefill-size",
        str(cfg.chunked_prefill_size),
        "--max-prefill-tokens",
        str(cfg.max_prefill_tokens),
        "--max-running-requests",
        str(cfg.max_running_requests),
        "--torch-compile-max-bs",
        str(cfg.torch_compile_max_bs),
        "--swa-full-tokens-ratio",
        str(cfg.swa_full_tokens_ratio),
        "--wait-timeout",
        str(eff_timeout),
    ]
    if cfg.disable_torch_compile or disable_compile_override:
        cmd.append("--disable-torch-compile")
    if cfg.extra_args.strip():
        cmd.append(f"--extra-args={cfg.extra_args.strip()}")

    print(f"\n[{cfg.config_id}] Launching servers: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, text=True, capture_output=True)
    stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    info: dict[str, Any] = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:] if proc.stderr else ""}
    if stdout_lines:
        try:
            info.update(json.loads(stdout_lines[-1]))
        except Exception:
            info["raw_stdout_tail"] = "\n".join(stdout_lines[-20:])
    if proc.returncode == 0 and info.get("status") == "ready":
        workers = ",".join(f"http://127.0.0.1:{30000 + r}" for r in range(cfg.num_replicas))
        subprocess.run(
            [
                sys.executable,
                "/workspace/gateway_flint_v7x.py",
                "--port",
                "8000",
                "--workers",
                workers,
                "--daemon",
                "--log-file",
                f"/workspace/logs/gw_{cfg.config_id}.log",
            ],
            check=True,
        )
        time.sleep(1.5)
        return True, info
    return False, info


def postprocess_single_host_bundle(bundle: dict[str, Any], cfg: MatrixConfig, launch_info: dict[str, Any], spec_suite: list[dict[str, Any]]) -> dict[str, Any]:
    bundle["hardware"] = f"Single-Host TPU v7x ({os.uname().nodename}, 4 dual-core chips = 8 TensorCores, 768 GB HBM)"
    bundle["axis"] = cfg.axis
    bundle["description"] = cfg.description
    bundle["server_launch"] = launch_info
    bundle["realistic_spec_suite"] = spec_suite
    bundle["config"].update(
        {
            "config_id": cfg.config_id,
            "tp_size": cfg.tp_size,
            "dp_size": cfg.dp_size,
            "ep_size": cfg.ep_size,
            "num_replicas": cfg.num_replicas,
            "moe_runner_backend": cfg.moe_runner_backend,
            "swa_full_tokens_ratio": cfg.swa_full_tokens_ratio,
            "chunked_prefill_size": cfg.chunked_prefill_size,
            "max_prefill_tokens": cfg.max_prefill_tokens,
            "max_running_requests": cfg.max_running_requests,
            "extra_args": cfg.extra_args,
        }
    )
    for pt in bundle.get("points", []):
        host_tps = float(pt.get("slice_output_tok_s", 0.0))
        pt["host_output_tok_s"] = round(host_tps, 2)
        pt["per_chip_output_tok_s"] = round(host_tps / 4.0, 2)
        pt["per_tensorcore_output_tok_s"] = round(host_tps / 8.0, 2)
        pt["host_total_tok_s"] = round(float(pt.get("slice_total_tok_s", 0.0)), 2)
    return bundle


def run_one_config(cfg: MatrixConfig, worker_id: str) -> None:
    prune_inductor_caches_if_needed()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    local_json = LOCAL_RESULTS / f"{cfg.config_id}.json"
    gcs_json = RUNS_DIR / f"{cfg.config_id}.json"
    if gcs_json.exists() and not local_json.exists():
        shutil.copyfile(gcs_json, local_json)

    ok, launch_info = start_servers_for_config(cfg, disable_compile_override=False)
    used_disable_compile = cfg.disable_torch_compile
    if not ok and cfg.fallback_disable_compile and not cfg.disable_torch_compile:
        print(f"[{cfg.config_id}] Retrying with --disable-torch-compile after initial failure...", flush=True)
        ok, launch_info_retry = start_servers_for_config(cfg, disable_compile_override=True)
        launch_info = {"initial_attempt": launch_info, "fallback_no_compile": launch_info_retry}
        used_disable_compile = True

    if not ok:
        print(f"[{cfg.config_id}] Server launch unsupported/failed; recording diagnostic JSON and continuing.", flush=True)
        fail_doc = {
            "run_name": cfg.config_id,
            "status": "unsupported",
            "worker_id": worker_id,
            "host": os.uname().nodename,
            "axis": cfg.axis,
            "description": cfg.description,
            "launch_diagnostics": launch_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        payload = json.dumps(fail_doc, indent=2) + "\n"
        local_json.write_text(payload)
        gcs_json.write_text(payload)
        subprocess.run([sys.executable, "/workspace/launch_host_replicas.py", "--mode", "stop"], check=False)
        return

    print(f"[{cfg.config_id}] READY in {launch_info.get('startup_sec')}s on {os.uname().nodename}!", flush=True)
    spec_suite: list[dict[str, Any]] = []
    if cfg.run_realistic_spec_suite:
        print(f"[{cfg.config_id}] Running realistic code/JSON N-gram workload suite...", flush=True)
        spec_suite = asyncio.run(run_realistic_spec_benchmark("http://127.0.0.1:8000", cfg.model_path))

    for g_idx, grp in enumerate(cfg.groups):
        bench_cmd = [
            sys.executable,
            "/workspace/bench_flint_v7x.py",
            "--run-name",
            cfg.config_id,
            "--base-url",
            "http://127.0.0.1:8000",
            "--model-name",
            cfg.model_path,
            "--weights-dtype",
            cfg.weights_dtype,
            "--kv-cache-dtype",
            cfg.kv_cache_dtype,
            "--tp-size",
            str(cfg.tp_size),
            "--ep-size",
            str(cfg.ep_size),
            "--num-replicas",
            str(cfg.num_replicas),
            "--notes",
            cfg.description + (" [eager]" if used_disable_compile else ""),
            "--shapes",
            grp.shapes,
            "--hit-rates",
            grp.hit_rates,
            "--concurrencies",
            grp.concurrencies,
            "--req-mult",
            str(grp.req_mult),
            "--max-reqs",
            str(grp.max_reqs),
            "--max-total-seq-tokens",
            str(grp.max_total_seq_tokens),
            "--resume",
            "--output",
            str(local_json),
            "--gcs-output",
            str(gcs_json),
        ]
        if cfg.verify_quality and g_idx == 0:
            bench_cmd.append("--verify-quality")
        subprocess.run(bench_cmd, check=False)

    if local_json.exists():
        bundle = json.loads(local_json.read_text())
    else:
        bundle = {"run_name": cfg.config_id, "config": {}, "points": []}
    bundle["status"] = "completed"
    bundle["worker_id"] = worker_id
    bundle["used_disable_compile"] = used_disable_compile
    bundle = postprocess_single_host_bundle(bundle, cfg, launch_info, spec_suite)
    payload = json.dumps(bundle, indent=2) + "\n"
    local_json.write_text(payload)
    gcs_json.write_text(payload)
    subprocess.run([sys.executable, "/workspace/launch_host_replicas.py", "--mode", "stop"], check=False)
    print(f"[{cfg.config_id}] COMPLETED ({len(bundle.get('points', []))} points saved to {gcs_json})", flush=True)


def rebuild_master_summary() -> None:
    summary = {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": {}}
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                pts = d.get("points", [])
                peak_4k = max((x["host_output_tok_s"] for x in pts if x.get("isl") == 4096 and "host_output_tok_s" in x), default=None)
                peak_1k = max((x["host_output_tok_s"] for x in pts if x.get("isl") == 1024 and "host_output_tok_s" in x), default=None)
                peak_64k = max((x["host_output_tok_s"] for x in pts if x.get("isl") == 65536 and "host_output_tok_s" in x), default=None)
                peak_128k = max((x["host_output_tok_s"] for x in pts if x.get("isl") == 131072 and "host_output_tok_s" in x), default=None)
                c1_1k = next((x["single_stream_tok_s"] for x in pts if x.get("isl") == 1024 and x.get("concurrency") == 1), None)
                summary["runs"][p.stem] = {
                    "status": d.get("status", "in_progress"),
                    "host": d.get("server_launch", {}).get("host") or d.get("host"),
                    "axis": d.get("axis"),
                    "num_points": len(pts),
                    "peak_host_tok_s_1k": peak_1k,
                    "peak_host_tok_s_4k": peak_4k,
                    "peak_host_tok_s_64k": peak_64k,
                    "peak_host_tok_s_128k": peak_128k,
                    "single_stream_tok_s_1k": c1_1k,
                    "spec_suite_c1_tok_s": next((s["single_stream_tok_s"] for s in d.get("realistic_spec_suite", []) if s["concurrency"] == 1), None),
                }
            except Exception:
                pass
    (GCS_ROOT / "sglang_matrix_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker-id", default="worker-0")
    ap.add_argument("--reverse", action="store_true", help="Traverse MATRIX in reverse order for lock-free 2-host parallelism")
    ap.add_argument("--only", default="", help="Optional comma-separated config_ids")
    args = ap.parse_args()

    only_set = {x.strip() for x in args.only.split(",") if x.strip()}
    order = list(reversed(MATRIX)) if args.reverse else list(MATRIX)
    if only_set:
        order = [c for c in order if c.config_id in only_set]

    print(f"[worker={args.worker_id} host={os.uname().nodename}] Starting SGLang optimization queue ({len(order)} configs, reverse={args.reverse})", flush=True)
    for cfg in order:
        if is_config_complete(cfg):
            print(f"[{cfg.config_id}] Already complete on GCS, skipping.", flush=True)
            continue
        if not try_claim_config(cfg, args.worker_id):
            print(f"[{cfg.config_id}] Claimed by peer worker, skipping.", flush=True)
            continue
        try:
            run_one_config(cfg, args.worker_id)
        except Exception as e:
            print(f"[{cfg.config_id}] Unhandled error: {e!r}", flush=True)
        rebuild_master_summary()

    rebuild_master_summary()
    print(f"[worker={args.worker_id}] ALL QUEUED SGLANG CONFIGS FINISHED!", flush=True)


if __name__ == "__main__":
    main()
