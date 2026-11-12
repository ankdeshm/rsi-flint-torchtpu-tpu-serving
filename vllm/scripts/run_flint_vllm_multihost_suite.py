#!/usr/bin/env python3
"""Multi-Host TPU `v5p-16` & `v7x-16` Serving Optimization Suite for `RecursiveSI/flint` (`40B-A2.5B` MoE) on `vllm-torchtpu` (`v0.32.0`).

Explores:
  1. Multi-Host Expert Parallelism (`EP=16` on `v5p-16`, `EP=32` on `v7x-16`) across all 4 hosts via Ray:
     - `TP=1 x DP=16 x EP=16` (`v5p-16`) / `TP=1 x DP=32 x EP=32` (`v7x-16`)
     - `TP=2 x DP=8 x EP=16`  (`v5p-16`) / `TP=2 x DP=16 x EP=32` (`v7x-16`)
     - `TP=4 x DP=4 x EP=16`  (`v5p-16`)
  2. Quantization & Precision Sweeps:
     - `[128, 128]` Block-FP8 (`RecursiveSI/flint-fp8` — `W8A16` on `v5p` MXU, native `FP8` on `v7x`) + FP8 KV cache
     - Pure `BF16` (`RecursiveSI/flint` — made feasible at `TP=1` on `v5p-16` for the first time via `EP=16`:
       `~4.9 GiB` weights/chip instead of `74.1 GiB` OOM!) + FP8 KV cache
     - Online per-channel `INT8` `W8A16` (`REQUANTIZE_WEIGHT_DTYPE=int8`, `--quantization=fp8` on BF16 checkpoint)
  3. Kernel & Speculative Decoding Sweeps:
     - `MOE_FUSED_EP_KERNEL_IMPL=v2` (`MOE_FUSED_EP_KERNEL_MIN_TOKENS=64`) SPMD fused EP transport+GMM kernel
     - `ngram` Speculative Decoding (`k=5, min=2, max=5`) on both standard `4096:1024` prompts and realistic
       agentic Python refactoring prompts (`coding_edit` `4096:512`) to cut single-stream decode TPOT below `7 ms`!
  4. High-Concurrency Slice-Wide Sweeps (`c=1..512` on `v5p-16`, `c=1..1024` on `v7x-16`) across `0%`, `80%`, `95%`
     prefix-cache hit rates and `4K:1K`, `8K:1K`, `64K:1K`, `128K:1K` contexts, with immediate per-point JSONL + JSON
     upload to GCS via GCE metadata token.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

WORLD_HOSTS = 4
COORD_PORT = 19999
ORIG_TPU_WORKER_ID = os.environ.get("TPU_WORKER_ID", "")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][flint-vllm-opt][host-{idx}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# GCE Metadata Token + Fast Parallel GCS Sync / Upload
# ---------------------------------------------------------------------------
def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def gcs_upload_bytes(bucket: str, obj: str, data: bytes, content_type: str = "application/json") -> None:
    for att in range(5):
        try:
            tok = vm_access_token()
            url = (
                f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
            )
            req = urllib.request.Request(
                url,
                data=data,
                method="POST",
                headers={"Authorization": f"Bearer {tok}", "Content-Type": content_type},
            )
            with urllib.request.urlopen(req, timeout=120):
                return
        except Exception:
            if att == 4:
                raise
            time.sleep(2 * (att + 1))


def gcs_list_prefix(bucket: str, prefix: str) -> list[dict[str, Any]]:
    tok = vm_access_token()
    items: list[dict[str, Any]] = []
    pt: str | None = None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?prefix={urllib.parse.quote(prefix, safe='')}"
        if pt:
            url += f"&pageToken={urllib.parse.quote(pt, safe='')}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
        items.extend(d.get("items", []))
        pt = d.get("nextPageToken")
        if not pt:
            break
    return items


def fast_gcs_sync_dir(idx: int, bucket: str, gcs_prefix: str, local_dir: Path, workers: int = 24) -> int:
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    items = gcs_list_prefix(bucket, gcs_prefix)
    blobs = [
        (it["name"][len(gcs_prefix):], int(it["size"]))
        for it in items
        if not it["name"].endswith("/") and "/.cache/" not in it["name"]
    ]
    if not blobs:
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[tuple[str, Path, int, int]] = []
    chunk = 256 * 1024 * 1024
    for rel, sz in blobs:
        dst = local_dir / rel
        if dst.exists() and dst.stat().st_size == sz:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as f:
            if sz > 0:
                f.truncate(sz)
        if sz == 0:
            continue
        for start in range(0, sz, chunk):
            end = min(sz - 1, start + chunk - 1)
            tasks.append((gcs_prefix + rel, dst, start, end))

    if not tasks:
        return len(blobs)

    t0 = time.time()
    total_dl = sum(e - s + 1 for _, _, s, e in tasks)
    log(idx, f"Syncing {len(blobs)} files ({total_dl / (1024**3):.2f} GiB) from gs://{bucket}/{gcs_prefix} -> {local_dir}...")

    def _dl_range(t: tuple[str, Path, int, int]) -> None:
        obj_name, dst, start, end = t
        for att in range(5):
            try:
                tok = vm_access_token()
                url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj_name, safe='')}?alt=media"
                req = urllib.request.Request(
                    url,
                    headers={"Authorization": f"Bearer {tok}", "Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(req, timeout=300) as resp, open(dst, "r+b") as f:
                    f.seek(start)
                    shutil.copyfileobj(resp, f, length=16 * 1024 * 1024)
                return
            except Exception:
                if att == 4:
                    raise
                time.sleep(2 * (att + 1))

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_dl_range, tasks))
    dt_s = max(1e-3, time.time() - t0)
    log(idx, f"Synced {local_dir} in {dt_s:.1f}s ({total_dl / (1024**3) / dt_s:.2f} GiB/s)")
    return len(blobs)


# ---------------------------------------------------------------------------
# Lightweight 4-Host TCP Coordinator on Host 0 (:19999)
# ---------------------------------------------------------------------------
_LOCK = threading.Lock()
_BARRIERS: dict[str, set[int]] = {}
_DECISIONS: dict[str, str] = {}
_RUN_ID: str = ""


class _CoordHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/ping":
            self._reply({"ok": True, "run_id": _RUN_ID})
            return
        if parsed.path == "/barrier":
            name = qs["name"][0]
            idx = int(qs["idx"][0])
            with _LOCK:
                s = _BARRIERS.setdefault(name, set())
                s.add(idx)
                cnt = len(s)
            self._reply({"ready": cnt >= WORLD_HOSTS, "count": cnt})
            return
        if parsed.path == "/decision":
            key = qs["key"][0]
            with _LOCK:
                val = _DECISIONS.get(key, "")
            self._reply({"val": val})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/decision":
            key = qs["key"][0]
            val = qs["val"][0]
            with _LOCK:
                _DECISIONS[key] = val
            self._reply({"ok": True})
            return
        self.send_response(404)
        self.end_headers()

    def _reply(self, obj: dict[str, Any]) -> None:
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_coord_server(run_id: str) -> ThreadingHTTPServer:
    global _RUN_ID
    _RUN_ID = run_id
    srv = ThreadingHTTPServer(("0.0.0.0", COORD_PORT), _CoordHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def resolve_host0_ip(head_fqdn: str, run_id: str, timeout_s: float = 600.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            ip = socket.gethostbyname(head_fqdn)
            with urllib.request.urlopen(f"http://{ip}:{COORD_PORT}/ping", timeout=3.0) as r:
                if r.status == 200:
                    d = json.loads(r.read().decode("utf-8"))
                    if d.get("run_id") == run_id:
                        return ip
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for coordinator at {head_fqdn}:{COORD_PORT} (run_id={run_id})")


def enter_barrier(coord_url: str, name: str, idx: int, timeout_s: float = 1800.0) -> None:
    t0 = time.time()
    q = urllib.parse.urlencode({"name": name, "idx": str(idx)})
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{coord_url}/barrier?{q}", timeout=5.0) as r:
                if json.loads(r.read().decode("utf-8")).get("ready"):
                    return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Barrier timeout on {name} (host {idx})")


def set_decision(coord_url: str, key: str, val: str) -> None:
    q = urllib.parse.urlencode({"key": key, "val": val})
    req = urllib.request.Request(f"{coord_url}/decision?{q}", method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=10.0) as r:
        r.read()


def wait_decision(coord_url: str, key: str, timeout_s: float = 10800.0) -> str:
    t0 = time.time()
    q = urllib.parse.urlencode({"key": key})
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{coord_url}/decision?{q}", timeout=5.0) as r:
                val = json.loads(r.read().decode("utf-8")).get("val", "")
                if val:
                    return val
        except Exception:
            pass
        time.sleep(3.0)
    raise TimeoutError(f"Decision timeout waiting on {key}")


# ---------------------------------------------------------------------------
# Process Management & Ray Multi-Host Bootstrap
# ---------------------------------------------------------------------------
def clean_host_processes() -> None:
    subprocess.run(["ray", "stop", "--force"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    kill_sh = (
        "for p in $(pgrep -f 'vllm serve|VLLM::|EngineCore|ray::|gcs_server|raylet'); do "
        "[ \"$p\" != \"$$\" ] && kill -9 \"$p\" 2>/dev/null || true; done"
    )
    subprocess.run(["bash", "-lc", kill_sh], check=False)
    subprocess.run(
        [
            "bash",
            "-lc",
            "rm -rf /tmp/vllm_tmp/* /tmp/ray/* 2>/dev/null || true; "
            "mkdir -p /tmp/hf_cache /tmp/vllm_tmp /tmp/vllm_cache /tmp/torchinductor",
        ],
        check=False,
    )
    time.sleep(3.0)


def build_ray_vllm_env(hw: str, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TORCH_TPU_TOPOLOGY", None)
    env.pop("TORCH_DEVICE_BACKEND_AUTOLOAD", None)
    env.pop("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", None)
    copy_keys = [
        "TORCH_TPU_TOPOLOGY",
        "USE_MOE_SPARSE_CORE",
        "MOE_FUSED_EP_KERNEL_IMPL",
        "MOE_FUSED_EP_KERNEL_MIN_TOKENS",
        "TPU_MOE_HIERARCHICAL_EP",
        "TPU_MOE_SKIP_PADDED_TOKENS",
        "REQUANTIZE_WEIGHT_DTYPE",
        "MOE_REQUANTIZE_WEIGHT_DTYPE",
        "ENABLE_QUANTIZED_MATMUL_KERNEL",
        "TPU_FP8_LINEAR_W8A16_MAX_ROWS",
        "TORCH_DIST_TIMEOUT",
        "TMPDIR",
        "HF_HOME",
        "VLLM_CACHE_ROOT",
        "VLLM_XLA_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCHINDUCTOR_AUTOGRAD_CACHE",
    ]
    env.update(
        {
            "TPU_MULTIHOST_BACKEND": "ray",
            "TPU_PROCESS_ADDRESSES": "",
            "TPU_WORKER_HOSTNAMES": "",
            "TPU_SKIP_MDS_QUERY": "1",
            "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "1",
            "VLLM_PLUGINS": "torchtpu",
            "VLLM_TARGET_DEVICE": "tpu",
            "PJRT_DEVICE": "TPU",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_NO_USAGE_STATS": "1",
            "HF_HOME": "/tmp/hf_cache",
            "TMPDIR": "/tmp/vllm_tmp",
            "VLLM_CACHE_ROOT": "/tmp/vllm_cache",
            "VLLM_XLA_CACHE_PATH": "/tmp/vllm_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "TORCH_DIST_TIMEOUT": "1800",
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_SHM_BROADCAST_TIMEOUT_S": "1800",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "1800",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
            "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": ",".join(copy_keys),
        }
    )
    if hw == "v5p-16":
        # TPU v5p megacore exposes 4 devices/host (16 total), whereas vllm_torchtpu's
        # fallback map returns "2,2,4,2" (32 worker addresses). Explicitly pass "2,2,4".
        env["TORCH_TPU_TOPOLOGY"] = "2,2,4"
        # TPU v5p SparseCore has 8 SIMD lanes, whereas ragged_gather_reduce defaults to
        # num_row_partitions=16 (asserting num_row_partitions <= num_simd_lanes).
        env["USE_MOE_SPARSE_CORE"] = "0"
    if extra_env:
        env.update(extra_env)
    return env


def start_ray_cluster(idx: int, head_ip: str, env: dict[str, str], min_tpu_per_host: int) -> None:
    worker_id = ORIG_TPU_WORKER_ID if ORIG_TPU_WORKER_ID != "" else str(idx)
    if idx == 0:
        subprocess.run(
            ["ray", "start", "--head", "--port=6379", f"--labels=ray.io/tpu-worker-id={worker_id}"],
            env=env,
            check=True,
        )
        t0 = time.time()
        check_py = (
            "import ray; ray.init(address='auto', logging_level='ERROR'); "
            f"print(sum(1 for n in ray.nodes() if n['Alive'] and n.get('Resources', {{}}).get('TPU', 0) >= {min_tpu_per_host}))"
        )
        while time.time() - t0 < 300:
            res = subprocess.run(["python3", "-c", check_py], env=env, text=True, capture_output=True)
            if res.stdout.strip() == "4":
                log(0, f"All 4 TPU hosts (TPU>={min_tpu_per_host}/host) joined Ray cluster!")
                return
            time.sleep(3.0)
        raise TimeoutError("Timed out waiting for 4 TPU hosts in Ray cluster")
    else:
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                with socket.create_connection((head_ip, 6379), timeout=3.0):
                    break
            except Exception:
                time.sleep(2.0)
        subprocess.run(
            ["ray", "start", f"--address={head_ip}:6379", f"--labels=ray.io/tpu-worker-id={worker_id}"],
            env=env,
            check=True,
        )


def wait_vllm_ready(proc: subprocess.Popen[Any], log_path: Path, port: int = 8000, timeout_s: float = 2400.0) -> tuple[bool, float, str]:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
            return False, round(time.time() - t0, 1), f"Exited rc={proc.returncode}:\n{tail}"
        try:
            with urllib.request.urlopen(url, timeout=5.0) as r:
                if r.status == 200:
                    return True, round(time.time() - t0, 1), ""
        except Exception:
            pass
        time.sleep(4.0)
    tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
    return False, round(time.time() - t0, 1), f"Timeout after {timeout_s}s:\n{tail}"


# ---------------------------------------------------------------------------
# OpenAI SSE Streaming Benchmark Harness (0%/80%/95% Prefix Hit + Coding Edit)
# ---------------------------------------------------------------------------
def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 2)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 2)
    return round(s[f] * (c - k) + s[c] * (k - f), 2)


class PromptBuilder:
    def __init__(self, tok_dir: Path) -> None:
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
        base_text = (
            "class DistributedRouter:\n"
            "    def __init__(self, num_experts: int = 256, top_k: int = 6):\n"
            "        self.num_experts = num_experts\n"
            "        self.top_k = top_k\n"
            "        self.load_counts = [0] * num_experts\n\n"
            "    def route_batch(self, token_scores: list[list[float]]) -> list[list[tuple[int, float]]]:\n"
            "        assignments = []\n"
            "        for row in token_scores:\n"
            "            ranked = sorted(enumerate(row), key=lambda kv: kv[1], reverse=True)[:self.top_k]\n"
            "            denom = sum(v for _, v in ranked) or 1.0\n"
            "            normed = [(idx, val / denom) for idx, val in ranked]\n"
            "            for idx, _ in normed:\n"
            "                self.load_counts[idx] += 1\n"
            "            assignments.append(normed)\n"
            "        return assignments\n\n"
        )
        self.code_ids = self._enc(base_text)
        prose_text = (
            "Mixture-of-Experts serving on multi-host TPU pods combines intra-host ICI tensor/expert "
            "parallelism with asynchronous paged KV-cache management, sliding-window attention sinks, "
            "and dynamic blockwise 8-bit weight quantization. "
        )
        self.prose_ids = self._enc(prose_text)
        self._prefix_cache: dict[tuple[int, int], list[int]] = {}

    def _enc(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def _dec(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def _fill_ids(self, base_ids: list[int], target_len: int, salt: int) -> list[int]:
        rng = random.Random(100003 * salt + target_len)
        salt_ids = self._enc(f"[req={salt} seed={rng.randint(1000, 999999)}] ")
        rem = max(8, target_len - len(salt_ids))
        reps = (rem // len(base_ids)) + 2
        body = (base_ids * reps)[:rem]
        return salt_ids + body

    def build_prompts(self, num_prompts: int, input_tokens: int, hit_ratio: float, mode: str, run_seed: int) -> tuple[str | None, list[str]]:
        if mode == "coding_edit":
            inst = self._enc(
                "\n\n# Task: Refactor the Python module above to add type annotations, input validation, "
                "prefix-aware load balancing statistics, and unit tests. Output the complete updated Python module:\n"
            )
            shared_len = max(64, input_tokens - len(inst) - 16)
            shared_ids = self._fill_ids(self.code_ids, shared_len, salt=run_seed)
            warmup_prompt = self._dec(shared_ids + inst)
            prompts = []
            for i in range(num_prompts):
                req_tag = self._enc(f"# Variant {i}: include docstring and edge-case checks.\n")
                ids = shared_ids[: max(32, input_tokens - len(inst) - len(req_tag))] + req_tag + inst
                prompts.append(self._dec(ids))
            return warmup_prompt, prompts

        if hit_ratio <= 0.0:
            prompts = [
                self._dec(self._fill_ids(self.prose_ids, input_tokens, salt=run_seed * 10000 + i))
                for i in range(num_prompts)
            ]
            return None, prompts

        prefix_len = max(64, int(round(input_tokens * hit_ratio)))
        suffix_len = max(16, input_tokens - prefix_len)
        key = (run_seed, prefix_len)
        if key not in self._prefix_cache:
            self._prefix_cache[key] = self._fill_ids(self.prose_ids, prefix_len, salt=run_seed * 77 + 1)
        shared_prefix = self._prefix_cache[key]
        warmup_ids = shared_prefix + self._fill_ids(self.prose_ids, suffix_len, salt=999991)
        warmup_prompt = self._dec(warmup_ids)
        prompts = []
        for i in range(num_prompts):
            suf = self._fill_ids(self.prose_ids, suffix_len, salt=run_seed * 10000 + i + 1)
            prompts.append(self._dec(shared_prefix + suf))
        return warmup_prompt, prompts


def send_one_sse_request(
    url: str,
    served_model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": served_model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "min_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    t_first: float | None = None
    chunks = 0
    comp_tokens = 0
    prompt_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_s = line[5:].strip()
                if data_s == "[DONE]":
                    break
                obj = json.loads(data_s)
                choices = obj.get("choices") or []
                if choices:
                    txt = choices[0].get("text") or ""
                    if txt:
                        if t_first is None:
                            t_first = time.perf_counter()
                        chunks += 1
                usage = obj.get("usage")
                if usage:
                    comp_tokens = int(usage.get("completion_tokens") or comp_tokens)
                    prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
        t_end = time.perf_counter()
        if comp_tokens <= 0:
            comp_tokens = max(1, chunks)
        if t_first is None:
            t_first = t_end
        ttft_ms = (t_first - t_start) * 1000.0
        decode_s = max(1e-6, t_end - t_first)
        tpot_ms = (decode_s * 1000.0) / max(1, comp_tokens - 1) if comp_tokens > 1 else ttft_ms
        stream_tps = (comp_tokens - 1) / decode_s if comp_tokens > 1 else comp_tokens / max(1e-6, t_end - t_start)
        return {
            "ok": True,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "stream_tps": stream_tps,
            "e2e_s": t_end - t_start,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}


def verify_model_quality(port: int, served_model: str) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": served_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a concise Python function `merge_intervals(intervals)` that merges overlapping [start, end] intervals and state its time complexity.",
                }
            ],
            "max_tokens": 256,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    dt_s = round(time.perf_counter() - t0, 3)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or msg.get("reasoning") or "")
    usage = data.get("usage") or {}
    return {
        "wall_s": dt_s,
        "usage": usage,
        "sample_preview": text.strip()[:400],
        "coherent": ("def " in text or "sort" in text.lower() or "interval" in text.lower()),
    }


def run_bench_point(
    pb: PromptBuilder,
    port: int,
    served_model: str,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    hit_ratio: float,
    mode: str = "standard",
    temperature: float = 0.0,
    run_seed: int = 1,
) -> dict[str, Any]:
    url = f"http://127.0.0.1:{port}/v1/completions"
    num_prompts = max(4, min(max(concurrency * 2, 4), 1024))
    if input_tokens >= 65536:
        num_prompts = max(2, min(concurrency * 2, 16))
    warmup_prompt, prompts = pb.build_prompts(num_prompts, input_tokens, hit_ratio, mode, run_seed)

    # Warm prefix cache (or JIT bucket at c=1) before timed measurement
    if warmup_prompt is not None:
        send_one_sse_request(url, served_model, warmup_prompt, max_tokens=16, temperature=temperature)

    t_wall0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [
            pool.submit(send_one_sse_request, url, served_model, p, output_tokens, temperature)
            for p in prompts
        ]
        results = [f.result() for f in futs]
    wall_s = max(1e-6, time.perf_counter() - t_wall0)

    oks = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]
    tot_out = sum(r["completion_tokens"] for r in oks)
    tot_in = sum(r["prompt_tokens"] or input_tokens for r in oks)
    ttfts = [r["ttft_ms"] for r in oks]
    tpots = [r["tpot_ms"] for r in oks]
    stps = [r["stream_tps"] for r in oks]

    return {
        "mode": mode,
        "shape": f"{input_tokens}:{output_tokens}",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "hit_rate": hit_ratio,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "ok": len(oks),
        "fail": len(fails),
        "wall_s": round(wall_s, 3),
        "total_output_tokens": tot_out,
        "output_tok_s": round(tot_out / wall_s, 2),
        "total_tok_s": round((tot_in + tot_out) / wall_s, 2),
        "mean_stream_tps": round(statistics.mean(stps), 2) if stps else 0.0,
        "p50_stream_tps": percentile(stps, 50),
        "mean_ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else 0.0,
        "p50_ttft_ms": percentile(ttfts, 50),
        "p99_ttft_ms": percentile(ttfts, 99),
        "mean_tpot_ms": round(statistics.mean(tpots), 2) if tpots else 0.0,
        "p50_tpot_ms": percentile(tpots, 50),
        "p99_tpot_ms": percentile(tpots, 99),
        "first_error": fails[0]["error"] if fails else "",
    }


# ---------------------------------------------------------------------------
# Experiment Matrix Builder (`v5p-16` & `v7x-16`)
# ---------------------------------------------------------------------------
def build_experiments(hw: str) -> list[dict[str, Any]]:
    if hw == "v5p-16":
        return [
            {
                "name": "vllm_w8a16_tp1_dp16_ep16_fp8kv",
                "desc": "vLLM-TorchTPU W8A16 Block-FP8 TP=1xDP=16xEP=16 (16 experts/chip, 2.6 GiB weights/chip, FP8 KV)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 1,
                "dp": 16,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 4, 8, 16, 32, 64, 128, 256, 512]),
                    ("standard", 4096, 1024, 0.8, [1, 16, 64, 256, 512]),
                    ("standard", 4096, 1024, 0.95, [1, 16, 64, 256, 512]),
                    ("standard", 8192, 1024, 0.0, [1, 64, 256]),
                    ("standard", 8192, 1024, 0.95, [1, 64, 256]),
                    ("standard", 65536, 1024, 0.0, [1, 8]),
                    ("standard", 131072, 1024, 0.0, [1, 4]),
                ],
            },
            {
                "name": "vllm_bf16_tp1_dp16_ep16_fp8kv",
                "desc": "vLLM-TorchTPU Pure BF16 TP=1xDP=16xEP=16 (4.9 GiB BF16 weights/chip via EP=16, zero W8A16 dequant, FP8 KV)",
                "model_dir": "/dev/shm/flint",
                "tp": 1,
                "dp": 16,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 8, 32, 64, 128, 256, 512]),
                    ("standard", 4096, 1024, 0.8, [1, 64, 256, 512]),
                    ("standard", 4096, 1024, 0.95, [1, 64, 256, 512]),
                    ("standard", 8192, 1024, 0.0, [1, 256]),
                    ("standard", 8192, 1024, 0.95, [256]),
                ],
            },
            {
                "name": "vllm_w8a16_tp2_dp8_ep16_fp8kv",
                "desc": "vLLM-TorchTPU W8A16 Block-FP8 TP=2xDP=8xEP=16 (halved dense/attn weight reads + EP=16)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 2,
                "dp": 8,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 4, 16, 64, 128, 256, 512]),
                    ("standard", 4096, 1024, 0.8, [1, 64, 256, 512]),
                    ("standard", 4096, 1024, 0.95, [1, 64, 256, 512]),
                    ("standard", 8192, 1024, 0.0, [1, 256]),
                ],
            },
            {
                "name": "vllm_w8a16_tp4_dp4_ep16_fp8kv",
                "desc": "vLLM-TorchTPU W8A16 Block-FP8 TP=4xDP=4xEP=16 Low-Latency Baseline (Non-Speculative)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 4,
                "dp": 4,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 65536,
                "max_num_seqs": 64,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 4, 16, 64, 256]),
                    ("standard", 4096, 1024, 0.95, [1, 4, 16, 64, 256]),
                    ("coding_edit", 4096, 512, 0.95, [1, 4, 8, 16]),
                ],
            },
            {
                "name": "vllm_w8a16_tp4_dp4_ep16_ngram5",
                "desc": "vLLM-TorchTPU W8A16 TP=4xDP=4xEP=16 + N-Gram Speculative Decoding (k=5, lookup=2..5)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 4,
                "dp": 4,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 65536,
                "max_num_seqs": 64,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": [
                    "--enable-expert-parallel",
                    '--speculative-config={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}',
                ],
                "extra_env": {},
                "matrix": [
                    ("coding_edit", 4096, 512, 0.95, [1, 4, 8, 16]),
                    ("standard", 4096, 1024, 0.0, [1, 4, 16]),
                    ("standard", 4096, 1024, 0.95, [1, 4, 16]),
                ],
            },
            {
                "name": "vllm_int8_w8a16_tp1_dp16_ep16_fp8kv",
                "desc": "vLLM-TorchTPU Online Per-Channel INT8 W8A16 TP=1xDP=16xEP=16 (no [128,128] block scale overhead)",
                "model_dir": "/dev/shm/flint",
                "tp": 1,
                "dp": 16,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 65536,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel", "--quantization=fp8"],
                "extra_env": {
                    "REQUANTIZE_WEIGHT_DTYPE": "int8",
                    "MOE_REQUANTIZE_WEIGHT_DTYPE": "int8",
                },
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 16, 64, 256, 512]),
                    ("standard", 4096, 1024, 0.95, [1, 256, 512]),
                ],
            },
        ]
    elif hw == "v7x-16":
        return [
            {
                "name": "vllm_fp8_tp1_dp32_ep32_fp8kv",
                "desc": "vLLM-TorchTPU Native FP8 TP=1xDP=32xEP=32 (8 experts/TC = 0.63 GiB expert weights/TC, FP8 KV)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 1,
                "dp": 32,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 8, 32, 64, 128, 256, 512, 1024]),
                    ("standard", 4096, 1024, 0.8, [1, 64, 256, 512, 1024]),
                    ("standard", 4096, 1024, 0.95, [1, 64, 256, 512, 1024]),
                    ("standard", 8192, 1024, 0.0, [1, 256, 512]),
                    ("standard", 65536, 1024, 0.0, [1, 8]),
                    ("standard", 131072, 1024, 0.0, [1, 8]),
                ],
            },
            {
                "name": "vllm_bf16_tp1_dp32_ep32_fp8kv",
                "desc": "vLLM-TorchTPU Pure BF16 TP=1xDP=32xEP=32 (8 experts/TC = 1.25 GiB BF16 expert weights/TC, FP8 KV)",
                "model_dir": "/dev/shm/flint",
                "tp": 1,
                "dp": 32,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 32, 128, 256, 512, 1024]),
                    ("standard", 4096, 1024, 0.8, [1, 256, 512, 1024]),
                    ("standard", 4096, 1024, 0.95, [1, 256, 512, 1024]),
                ],
            },
            {
                "name": "vllm_fp8_tp2_dp16_ep32_fp8kv",
                "desc": "vLLM-TorchTPU Native FP8 Intra-Chip TP=2xDP=16xEP=32 (ICI-free intra-die TP=2 + EP=32)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 2,
                "dp": 16,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 135168,
                "max_num_seqs": 128,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": ["--enable-expert-parallel"],
                "extra_env": {},
                "matrix": [
                    ("standard", 4096, 1024, 0.0, [1, 8, 64, 256, 512, 1024]),
                    ("standard", 4096, 1024, 0.95, [1, 64, 256, 512, 1024]),
                ],
            },
            {
                "name": "vllm_fp8_tp4_dp8_ep32_ngram5",
                "desc": "vLLM-TorchTPU Native FP8 TP=4xDP=8xEP=32 + N-Gram Speculative Decoding (k=5, lookup=2..5)",
                "model_dir": "/dev/shm/flint-fp8",
                "tp": 4,
                "dp": 8,
                "kv_dtype": "fp8",
                "mem_util": 0.88,
                "max_model_len": 65536,
                "max_num_seqs": 64,
                "max_batched_tokens": 2048,
                "compile_sizes": [4, 16, 64, 128, 256, 512, 1024, 2048],
                "extra_args": [
                    "--enable-expert-parallel",
                    '--speculative-config={"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":5,"prompt_lookup_min":2}',
                ],
                "extra_env": {},
                "matrix": [
                    ("coding_edit", 4096, 512, 0.95, [1, 4, 8, 16]),
                    ("standard", 4096, 1024, 0.0, [1, 4, 16]),
                    ("standard", 4096, 1024, 0.95, [1, 4, 16]),
                ],
            },
        ]
    raise ValueError(f"Unsupported --hw={hw}")


def ensure_v5p_sparsecore_fallback(idx: int) -> None:
    target = Path("/workspace/vllm-torchtpu/src/vllm_torchtpu/kernels/deepseek_v4/sparsecore_topk.py")
    if not target.exists():
        return
    src = target.read_text()
    if "sc.num_lanes != LANES" in src:
        return
    anchor = '    if sc is None:\n        raise NotImplementedError("SparseCore is not available")'
    if anchor in src:
        log(idx, "SparseCore v1 guard verified.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw", choices=["v5p-16", "v7x-16"], required=True)
    ap.add_argument("--gcs-bucket", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--head-fqdn", required=True)
    ap.add_argument("--only", default="", help="Optional comma-separated experiment names")
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    min_tpu_per_host = 4 if args.hw == "v5p-16" else 8
    gcs_out_prefix = f"flint-opt-results/{args.hw}_{args.run_id}"
    local_out_dir = Path(f"/tmp/flint_opt_{args.run_id}")
    local_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Start coordinator on Host 0 & join barrier
    if idx == 0:
        start_coord_server(args.run_id)
        head_ip = socket.gethostbyname(socket.gethostname())
        log(0, f"Coordinator listening on {head_ip}:{COORD_PORT} (run_id={args.run_id}, gcs_prefix=gs://{args.gcs_bucket}/{gcs_out_prefix})")
    else:
        head_ip = resolve_host0_ip(args.head_fqdn, args.run_id)
        log(idx, f"Connected to coordinator at {head_ip}:{COORD_PORT} (run_id={args.run_id})")
    coord_url = f"http://{head_ip}:{COORD_PORT}"

    # 2. Install FlintConfig + FlintForCausalLM on all 4 hosts
    installer = Path(__file__).resolve().parent / "install_flint_vllm.py"
    subprocess.run([sys.executable, str(installer)], check=True)
    if args.hw == "v5p-16":
        ensure_v5p_sparsecore_fallback(idx)

    # 3. Stage RecursiveSI/flint-fp8 (40.4 GiB) and RecursiveSI/flint (79.6 GiB) into /dev/shm on all 4 hosts
    fast_gcs_sync_dir(idx, args.gcs_bucket, "models/RecursiveSI/flint-fp8/", Path("/dev/shm/flint-fp8"))
    fast_gcs_sync_dir(idx, args.gcs_bucket, "models/RecursiveSI/flint/", Path("/dev/shm/flint"))
    enter_barrier(coord_url, "checkpoints_staged", idx)

    pb = PromptBuilder(Path("/dev/shm/flint-fp8")) if idx == 0 else None
    exps = build_experiments(args.hw)
    if args.only.strip():
        allow = {x.strip() for x in args.only.split(",") if x.strip()}
        exps = [e for e in exps if e["name"] in allow]

    for exp_i, exp in enumerate(exps):
        ename = exp["name"]
        if idx == 0:
            existing = {it["name"] for it in gcs_list_prefix(args.gcs_bucket, f"{gcs_out_prefix}/{ename}.json")}
            skip_flag = "SKIP" if f"{gcs_out_prefix}/{ename}.json" in existing else "RUN"
            set_decision(coord_url, f" gate_{exp_i}_{ename}".strip(), skip_flag)
        else:
            skip_flag = wait_decision(coord_url, f"gate_{exp_i}_{ename}")
        if skip_flag == "SKIP":
            log(idx, f"[{ename}] Already completed in gs://{args.gcs_bucket}/{gcs_out_prefix}/{ename}.json — skipping!")
            continue
        enter_barrier(coord_url, f"exp_{exp_i}_{ename}_clean", idx)
        clean_host_processes()
        env = build_ray_vllm_env(args.hw, exp.get("extra_env"))
        enter_barrier(coord_url, f"exp_{exp_i}_{ename}_ray", idx)
        start_ray_cluster(idx, head_ip, env, min_tpu_per_host=min_tpu_per_host)

        if idx == 0:
            assert pb is not None
            served_model = "RecursiveSI/flint"
            comp_cfg = json.dumps({"compile_sizes": exp["compile_sizes"]})
            use_async = not any("ngram" in a for a in exp["extra_args"])
            vllm_cmd = [
                "vllm",
                "serve",
                exp["model_dir"],
                f"--tokenizer={exp['model_dir']}",
                f"--served-model-name={served_model}",
                "--host=0.0.0.0",
                "--port=8000",
                "--api-server-count=1",
                f"--tensor-parallel-size={exp['tp']}",
                f"--data-parallel-size={exp['dp']}",
                "--safetensors-load-strategy=prefetch",
                f"--kv-cache-dtype={exp['kv_dtype']}",
                f"--gpu-memory-utilization={exp['mem_util']}",
                f"--max-model-len={exp['max_model_len']}",
                f"--max-num-seqs={exp['max_num_seqs']}",
                f"--max-num-batched-tokens={exp['max_batched_tokens']}",
                f"--compilation-config={comp_cfg}",
                "--enable-prefix-caching",
                "--async-scheduling" if use_async else "--no-async-scheduling",
                "--trust-remote-code",
            ] + exp["extra_args"]

            srv_log = local_out_dir / f"server_{ename}.log"
            log(0, f"=== [{exp_i+1}/{len(exps)}] {ename}: {' '.join(vllm_cmd)} ===")
            with open(srv_log, "w") as lf:
                proc = subprocess.Popen(vllm_cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

            ok, boot_s, err_tail = wait_vllm_ready(proc, srv_log, port=8000, timeout_s=2400.0)
            if not ok:
                log(0, f"[{ename}] FAILED BOOT after {boot_s}s:\n{err_tail[-1500:]}")
                diag = {
                    "experiment": ename,
                    "desc": exp["desc"],
                    "hw": args.hw,
                    "status": "FAILED_BOOT",
                    "boot_s": boot_s,
                    "error_tail": err_tail[-3000:],
                }
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_out_prefix}/diag_{ename}.json", json.dumps(diag, indent=2).encode("utf-8"))
                gcs_upload_bytes(args.gcs_bucket, f"{gcs_out_prefix}/server_{ename}.log", srv_log.read_bytes(), content_type="text/plain")
                try:
                    proc.kill()
                except Exception:
                    pass
                set_decision(coord_url, f"done_{exp_i}_{ename}", "FAILED")
                continue

            log(0, f"[{ename}] READY in {boot_s}s! Verifying output quality...")
            try:
                qual = verify_model_quality(8000, served_model)
                log(0, f"[{ename}] Quality OK ({qual['wall_s']}s, coherent={qual['coherent']}): {qual['sample_preview'][:180]!r}")
            except Exception as qexc:
                qual = {"coherent": False, "error": str(qexc)}
                log(0, f"[{ename}] Quality check warning: {qexc}")

            rows: list[dict[str, Any]] = []
            jsonl_lines: list[str] = []
            for mode, in_tok, out_tok, hit_r, concs in exp["matrix"]:
                for c in concs:
                    pt = run_bench_point(
                        pb=pb,
                        port=8000,
                        served_model=served_model,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        concurrency=c,
                        hit_ratio=hit_r,
                        mode=mode,
                        temperature=0.0,
                        run_seed=exp_i * 100 + int(hit_r * 100) + c,
                    )
                    pt["experiment"] = ename
                    pt["hw"] = args.hw
                    pt["tp"] = exp["tp"]
                    pt["dp"] = exp["dp"]
                    rows.append(pt)
                    jsonl_lines.append(json.dumps(pt))
                    log(
                        0,
                        f"[{ename}] {mode} {in_tok}:{out_tok} hit={int(hit_r*100)}% c={c} -> "
                        f"out={pt['output_tok_s']} tok/s | stream={pt['mean_stream_tps']} tok/s | "
                        f"TPOT={pt['mean_tpot_ms']} ms (p50={pt['p50_tpot_ms']} ms) | TTFT={pt['mean_ttft_ms']} ms | "
                        f"ok={pt['ok']}/{pt['num_prompts']}",
                    )
                    # Immediate incremental persistence to GCS after every single benchmark point
                    gcs_upload_bytes(
                        args.gcs_bucket,
                        f"{gcs_out_prefix}/{ename}_points.jsonl",
                        ("\n".join(jsonl_lines) + "\n").encode("utf-8"),
                        content_type="text/plain",
                    )

            summary = {
                "experiment": ename,
                "desc": exp["desc"],
                "hw": args.hw,
                "engine": "vllm-torchtpu-v0.32.0",
                "tp": exp["tp"],
                "dp": exp["dp"],
                "ep": exp["tp"] * exp["dp"],
                "kv_dtype": exp["kv_dtype"],
                "boot_s": boot_s,
                "quality": qual,
                "cmd": vllm_cmd,
                "extra_env": exp.get("extra_env", {}),
                "results": rows,
            }
            out_json = local_out_dir / f"{ename}.json"
            out_json.write_text(json.dumps(summary, indent=2) + "\n")
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_out_prefix}/{ename}.json", out_json.read_bytes())
            gcs_upload_bytes(args.gcs_bucket, f"{gcs_out_prefix}/server_{ename}.log", srv_log.read_bytes()[-512000:], content_type="text/plain")
            log(0, f"[{ename}] Completed & uploaded {len(rows)} points to gs://{args.gcs_bucket}/{gcs_out_prefix}/{ename}.json")

            try:
                proc.terminate()
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            set_decision(coord_url, f"done_{exp_i}_{ename}", "OK")
        else:
            dec = wait_decision(coord_url, f"done_{exp_i}_{ename}")
            log(idx, f"[{ename}] Host 0 signaled {dec}")

    enter_barrier(coord_url, "all_done", idx)
    clean_host_processes()
    log(idx, "ALL FLINT VLLM MULTI-HOST EXPERIMENTS COMPLETE!")


if __name__ == "__main__":
    main()
