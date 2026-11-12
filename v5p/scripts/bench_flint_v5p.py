#!/usr/bin/env python3
"""Comprehensive Flint (`RecursiveSI/flint`, 40B-A2.5B MoE) Serving Benchmark Suite on TPU v5p-16.

Supports two execution modes on GKE Cluster 1's `tpu-v5p-16-pool` (`4 hosts x 4 v5p chips = 16 chips`):
  1. `--mode smoke` (Step 1):
     - Stages `RecursiveSI/flint` BF16 (~79.6 GB) to `/dev/shm/flint`.
     - Converts BF16 -> Block-FP8 `[128, 128]` (~40.4 GB) into `/dev/shm/flint-fp8` and uploads to GCS.
     - Smoke-tests coherent output & reasoning (`"Write a Python function that merges overlapping intervals."`)
       and single-stream + batched probes across:
         * `BF16 TP=4` (1 replica/host = 4 chips)
         * `BF16 TP=2` (2 replicas/host = 4 chips)
         * `FP8  TP=1` (4 replicas/host = 4 chips)
         * `FP8  TP=2` (2 replicas/host = 4 chips)
         * `FP8  TP=4` (1 replica/host = 4 chips)
  2. `--mode full_sweep` (Step 2):
     - Phase A (4-Host Simultaneous Parallel Matrix): Each of the 4 `ct5p-hightpu-4t` hosts runs an assigned
       partition of the `(Precision, TP/DP, Scheduler/Kernel Knobs, Context Length, Prefix-Cache Hit Rate, Concurrency)`
       matrix with immediate per-point JSONL persistence to GCS.
     - Phase B (True 16-Chip `v5p-16` 4-Host Slice-Wide Multi-Replica Serving): Coordinates all 4 hosts via GCS
       rendezvous and runs true 16-chip load-balanced serving across `DP=4 (BF16 TP=4)`, `DP=8 (FP8 TP=2)`,
       and `DP=16 (FP8 TP=1)` at `0%`, `80%`, and `95%` prefix-cache hit rates!
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

METADATA_HOOK_C = r"""
#define _GNU_SOURCE
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static int (*real_connect)(int, const struct sockaddr*, socklen_t) = NULL;

int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
    if (!real_connect) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    if (addr && addr->sa_family == AF_INET) {
        struct sockaddr_in copy_addr;
        memcpy(&copy_addr, addr, sizeof(struct sockaddr_in));
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(copy_addr.sin_addr), ip, INET_ADDRSTRLEN);
        if (strcmp(ip, "169.254.169.254") == 0) {
            const char *port_s = getenv("MOCK_METADATA_PORT");
            int port = port_s ? atoi(port_s) : 80;
            copy_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
            copy_addr.sin_port = htons(port);
            return real_connect(sockfd, (const struct sockaddr *)&copy_addr, sizeof(struct sockaddr_in));
        }
    }
    return real_connect(sockfd, addr, addrlen);
}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(idx: int, msg: str) -> None:
    print(f"[{utc_now()}][v5p-pod-{idx}] {msg}", flush=True)


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return round(s[0], 2)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return round(s[int(k)], 2)
    return round(s[f] * (c - k) + s[c] * (k - f), 2)


def vm_access_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]


def vm_meta(rel: str, default: str = "") -> str:
    try:
        req = urllib.request.Request(
            f"http://169.254.169.254/computeMetadata/v1/{rel.lstrip('/')}",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode("utf-8").strip()
    except Exception:
        return default


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


def gcs_download_bytes(bucket: str, obj: str) -> bytes | None:
    tok = vm_access_token()
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


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
        (it["name"][len(gcs_prefix) :], int(it["size"]))
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
    log(idx, f"Streaming {len(blobs)} files ({total_dl / (1024**3):.2f} GiB) from gs://{bucket}/{gcs_prefix} -> {local_dir}...")

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


def fast_gcs_upload_dir(idx: int, bucket: str, local_dir: Path, gcs_prefix: str, workers: int = 16) -> int:
    if not local_dir.exists():
        return 0
    if not gcs_prefix.endswith("/"):
        gcs_prefix += "/"
    files = [p for p in local_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]
    if not files:
        return 0
    t0 = time.time()
    total_sz = sum(p.stat().st_size for p in files)

    def _up_one(p: Path) -> None:
        rel = p.relative_to(local_dir).as_posix()
        gcs_upload_bytes(bucket, gcs_prefix + rel, p.read_bytes(), content_type="application/octet-stream")

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_up_one, files))
    dt_s = max(1e-3, time.time() - t0)
    log(idx, f"Uploaded {len(files)} XLA cache files ({total_sz / (1024**3):.2f} GiB) -> gs://{bucket}/{gcs_prefix} in {dt_s:.1f}s")
    return len(files)


def bootstrap_pod(idx: int, bucket: str, need_bf16: bool = True, need_fp8: bool = True) -> tuple[Path, Path]:
    Path("/workspace").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/results").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/sglang_xla_cache/xla").mkdir(parents=True, exist_ok=True)

    # 1. Install flint_sglang wheel
    whl_dst = Path("/tmp/flint_sglang-0.1.4-py3-none-any.whl")
    if not whl_dst.exists():
        whl_bytes = (
            gcs_download_bytes(bucket, "flint-serving-scripts/v5p/flint_sglang-0.1.4-py3-none-any.whl")
            or gcs_download_bytes(bucket, "models/RecursiveSI/flint/sglang/flint_sglang-0.1.4-py3-none-any.whl")
        )
        assert whl_bytes is not None, "Missing flint_sglang wheel in GCS!"
        whl_dst.write_bytes(whl_bytes)
    subprocess.run(["pip", "install", "--no-deps", "--quiet", "--force-reinstall", str(whl_dst)], check=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "patch_flint_sglang_tpu.py")], check=True)
    log(idx, "Installed & patched flint_sglang-0.1.4 + 256-expert Pallas router_topk.")

    # 2. Compile metadata_hook.so & start v5p_mock_metadata.py on 127.0.0.1:18080 (and :80)
    hook_c = Path("/workspace/metadata_hook.c")
    hook_so = Path("/workspace/metadata_hook.so")
    hook_c.write_text(METADATA_HOOK_C)
    subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(hook_so), str(hook_c), "-ldl"], check=True)

    meta_running = False
    try:
        with socket.create_connection(("127.0.0.1", 18080), timeout=0.3):
            meta_running = True
    except OSError:
        pass
    if not meta_running:
        env_clean = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
        mf = open("/workspace/v5p_mock_metadata.log", "w")
        subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "v5p_mock_metadata.py"), "--port", "18080"],
            env=env_clean,
            stdout=mf,
            stderr=subprocess.STDOUT,
        )
        mf80 = open("/workspace/v5p_mock_metadata_80.log", "w")
        subprocess.Popen(
            [sys.executable, str(SCRIPT_DIR / "v5p_mock_metadata.py"), "--port", "80"],
            env=env_clean,
            stdout=mf80,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["bash", "-lc", "grep -q 'metadata.google.internal' /etc/hosts || echo '127.0.0.1 metadata.google.internal' >> /etc/hosts"],
            check=False,
        )
        for _ in range(25):
            try:
                with socket.create_connection(("127.0.0.1", 18080), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        log(idx, "v5p_mock_metadata.py active on 127.0.0.1:18080 and 127.0.0.1:80.")

    bf16_dir = Path("/dev/shm/flint")
    fp8_dir = Path("/dev/shm/flint-fp8")

    # Check if FP8 checkpoint already exists in GCS
    gcs_fp8_cfg = gcs_download_bytes(bucket, "models/RecursiveSI/flint-fp8/config.json") if need_fp8 else None
    if need_fp8 and gcs_fp8_cfg is not None:
        fast_gcs_sync_dir(idx, bucket, "models/RecursiveSI/flint-fp8/", fp8_dir)
        (fp8_dir / ".ready_fp8_128").write_text("ok\n")

    if need_bf16 or (need_fp8 and not (fp8_dir / ".ready_fp8_128").exists()):
        fast_gcs_sync_dir(idx, bucket, "models/RecursiveSI/flint/", bf16_dir)

    if need_fp8 and not (fp8_dir / ".ready_fp8_128").exists():
        conv_env = os.environ.copy()
        conv_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        conv_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "quantize_flint_fp8.py"),
            "--src",
            str(bf16_dir),
            "--dst",
            str(fp8_dir),
            "--workers",
            "15",
        ]
        if idx == 0 and gcs_fp8_cfg is None:
            conv_cmd.extend(["--gcs-bucket", bucket, "--gcs-prefix", "models/RecursiveSI/flint-fp8"])
        subprocess.run(conv_cmd, env=conv_env, check=True)

    return bf16_dir, fp8_dir


def set_mock_meta_single_host() -> None:
    """Present the pod as a standalone 1-host 4-chip v5p-8 VM so physical chips 0..3 are all valid."""
    st = {
        "chips_per_host_bounds": "2,2,1",
        "host_bounds": "1,1,1",
        "topology": "2x2x1",
        "accelerator_type": "v5p-8",
        "metrics_ports": "8431,8432,8433,8434",
    }
    Path("/tmp/v5p_mock_meta_state.json").write_text(json.dumps(st, indent=2))


def kill_sglang_and_gateways() -> None:
    subprocess.run(
        [
            "bash",
            "-lc",
            "for p in $(pgrep -f 'sglang.launch_server|sglang::|flint_v5p_gateway.py|sglang.srt'); do "
            '[ "$p" != "$$" ] && kill -9 $p 2>/dev/null || true; done; '
            "rm -f /tmp/libtpu_lockfile /dev/shm/tmp_r*/libtpu_lockfile; sleep 3",
        ],
        check=False,
    )


def build_replica_env(
    tp_size: int,
    replica_idx: int,
    replicas_per_host: int = 1,
    use_sglang_dp_controller: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    for k in (
        "JOB_COMPLETION_INDEX",
        "JOB_COMPLETION_MODE",
        "TPU_PROCESS_PORT",
        "TPU_PROCESS_ADDRESSES",
        "TPU_CHIPS_PER_PROCESS_BOUNDS",
        "TPU_PROCESS_BOUNDS",
        "TPU_WORKER_ID",
        "TPU_VISIBLE_CHIPS",
        "TPU_VISIBLE_DEVICES",
        "TPU_RUNTIME_METRICS_PORTS",
        "TPU_SKIP_MDS_QUERY",
        "VBAR_CONTROL_SERVICE_URL",
        "TPU_WORKER_HOSTNAMES",
        "TPU_TOPOLOGY_WRAP",
        "TPU_TOPOLOGY_ALT",
        "TPU_TOPOLOGY",
        "TOPOLOGY",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS",
        "HOST_BOUNDS",
        "TPU_ACCELERATOR_TYPE",
        "TORCH_TPU_SLICEBUILDER_ADDRESSES",
        "TORCH_TPU_TOPOLOGY",
        "TORCH_TPU_DEVICES_PER_HOST",
        "SGLANG_TPU_PD_COLOCATED",
    ):
        env.pop(k, None)

    port_seed = 29600 + replica_idx * 32
    sb = ",".join(f"localhost:{port_seed + i}" for i in range(tp_size))
    metrics_ports = ",".join(str(8440 + replica_idx * 8 + k) for k in range(tp_size))

    tmp_dir = Path(f"/dev/shm/tmp_r{replica_idx}_tp{tp_size}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    inductor_dir = Path(f"/dev/shm/inductor_r{replica_idx}_tp{tp_size}")
    inductor_dir.mkdir(parents=True, exist_ok=True)
    xla_cache = Path(f"/dev/shm/sglang_xla_cache/xla_tp{tp_size}_r{replica_idx}")
    xla_cache.mkdir(parents=True, exist_ok=True)

    env.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port_seed - 1),
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": sb,
            "TPU_SKIP_MDS_QUERY": "1",
            "TPU_ACCELERATOR_TYPE": "v5p-8",
            "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
            "TPU_HOST_BOUNDS": "1,1,1",
            "TPU_TOPOLOGY": "2x2x1",
            "TPU_WORKER_ID": "0",
            "TPU_RUNTIME_METRICS_PORTS": metrics_ports,
            "GCE_METADATA_HOST": "127.0.0.1:18080",
            "GCE_METADATA_IP": "127.0.0.1",
            "MOCK_METADATA_PORT": "18080",
            "GOOGLE_CLOUD_PROJECT": "mock-tpu-project",
            "GCLOUD_PROJECT": "mock-tpu-project",
            "SGLANG_EXTERNAL_MODEL_PACKAGE": "flint_sglang",
            "FLINT_TPU_ROUTER_MODE": "fused",
            "FLASHINFER_USE_CUDA_NORM": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_TPU_TIER2_COMPILATION_CACHE": "1",
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": str(xla_cache),
            "SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR": str(inductor_dir),
            "TORCHINDUCTOR_CACHE_DIR": str(inductor_dir),
            "TORCHINDUCTOR_AUTOGRAD_CACHE": "0",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "TMPDIR": str(tmp_dir),
            "TPU_LOCKFILE_DIR": str(tmp_dir),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if replicas_per_host > 1 and not use_sglang_dp_controller:
        # Required on TorchTPU so monkey_patch_torch_tpu_env offsets LOCAL_RANK by --base-gpu-id!
        env["SGLANG_TPU_PD_COLOCATED"] = "1"
    if tp_size == 1:
        env["TORCH_TPU_TOPOLOGY"] = "1,1,1"
        env["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_PROCESS_BOUNDS"] = "1,1,1"
        env["TPU_PROCESS_ADDRESSES"] = "local"
    elif tp_size == 4:
        env["TORCH_TPU_TOPOLOGY"] = "2,2,1"
    if Path("/var/run/tpu-plugin/vbar.sock").is_socket():
        env["VBAR_CONTROL_SERVICE_URL"] = "unix:///var/run/tpu-plugin/vbar.sock"
    if extra_env:
        env.update(extra_env)
    return env


def check_http_ok(url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def parse_server_memory_stats(log_paths: list[Path], tp_size: int, precision: str) -> dict[str, Any]:
    """Extract Full & SWA KV pool token capacities and exact HBM usage from SGLang TPU logs."""
    max_total_tokens = None
    swa_max_total_tokens = None
    full_kv_gb = None
    swa_kv_gb = None
    avail_after_weights_gb = None
    weight_gib_per_chip = (79.57 / (1.073741824 * tp_size)) if precision == "bf16" else (40.44 / (1.073741824 * tp_size))
    compiled_peak_mb = 0

    for lp in log_paths:
        if not lp.exists():
            continue
        txt = lp.read_text(errors="ignore")
        for m in re.finditer(r"Load weight end\..*?avail mem=([0-9.]+)\s*GB,\s*mem usage=([0-9.]+)\s*GB", txt):
            avail_after_weights_gb = float(m.group(1))
            weight_gib_per_chip = float(m.group(2))
        for m in re.finditer(
            r"Full KV Cache is allocated\..*?#tokens:\s*(\d+),\s*K size:\s*([0-9.]+)\s*GB,\s*V size:\s*([0-9.]+)\s*GB",
            txt,
        ):
            max_total_tokens = int(m.group(1))
            full_kv_gb = float(m.group(2)) + float(m.group(3))
        for m in re.finditer(
            r"SWA KV Cache is allocated\..*?#tokens:\s*(\d+),\s*K size:\s*([0-9.]+)\s*GB,\s*V size:\s*([0-9.]+)\s*GB",
            txt,
        ):
            swa_max_total_tokens = int(m.group(1))
            swa_kv_gb = float(m.group(2)) + float(m.group(3))
        if max_total_tokens is None:
            for m in re.finditer(r"max_total_num_tokens=(\d+)", txt):
                max_total_tokens = int(m.group(1))
            for m in re.finditer(r"#tokens:\s*(\d+)", txt):
                if max_total_tokens is None:
                    max_total_tokens = int(m.group(1))
        for m in re.finditer(r"HBM:\s*\+\d+\s*MB\s*\((\d+)/(\d+)\s*MB\)", txt):
            compiled_peak_mb = max(compiled_peak_mb, int(m.group(1)))

    if full_kv_gb is not None or swa_kv_gb is not None:
        kv_gib_per_chip = round((full_kv_gb or 0.0) + (swa_kv_gb or 0.0), 2)
    else:
        kv_gib_per_chip = None

    # Query libtpu runtime metrics port if HTTP Prometheus is exposed
    libtpu_hbm_gib = None
    for mp in (8440, 8431, 8432, 8448):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{mp}/metrics", timeout=1.5) as r:
                mtxt = r.read().decode("utf-8", errors="ignore")
            vals = [float(x) for x in re.findall(r"memory_usage_bytes[^\n]*?\s+([0-9.eE+-]+)", mtxt)]
            if vals:
                libtpu_hbm_gib = max(vals) / (1024**3)
                break
        except Exception:
            pass

    if compiled_peak_mb > 0:
        peak_chip_gib = round(compiled_peak_mb / 1024.0, 2)
    elif libtpu_hbm_gib is not None and libtpu_hbm_gib > 1.0:
        peak_chip_gib = round(libtpu_hbm_gib, 2)
    elif kv_gib_per_chip is not None:
        peak_chip_gib = round(weight_gib_per_chip + kv_gib_per_chip, 2)
    elif avail_after_weights_gb is not None:
        peak_chip_gib = round(95.73 - avail_after_weights_gb, 2)
    else:
        peak_chip_gib = None

    return {
        "weight_gib_per_chip": round(weight_gib_per_chip, 2),
        "kv_gib_per_chip": kv_gib_per_chip,
        "peak_hbm_gib_per_chip": peak_chip_gib,
        "max_total_tokens_per_replica": max_total_tokens,
        "swa_max_total_tokens_per_replica": swa_max_total_tokens,
    }


class HostClusterHandle:
    def __init__(
        self,
        *,
        config_tag: str,
        precision: str,
        tp_size: int,
        replicas_per_host: int,
        backend_urls: list[str],
        entry_url: str,
        procs: list[subprocess.Popen],
        log_paths: list[Path],
        mem_fraction_static: float,
        boot_s: float,
    ):
        self.config_tag = config_tag
        self.precision = precision
        self.tp_size = tp_size
        self.replicas_per_host = replicas_per_host
        self.backend_urls = backend_urls
        self.entry_url = entry_url
        self.procs = procs
        self.log_paths = log_paths
        self.mem_fraction_static = mem_fraction_static
        self.boot_s = boot_s


def _launch_host_servers_impl(
    idx: int,
    *,
    config_tag: str,
    model_path: Path,
    precision: str,
    tp_size: int,
    replicas_per_host: int,
    use_sglang_dp_controller: bool,
    attention_backend: str,
    kv_cache_dtype: str,
    mem_fraction_static: float,
    max_running_requests: int,
    chunked_prefill_size: int,
    context_length: int,
    enable_torch_compile: bool,
    extra_args: list[str] | None,
    extra_env: dict[str, str] | None,
    ready_timeout_s: int,
) -> HostClusterHandle:
    kill_sglang_and_gateways()
    set_mock_meta_single_host()

    t0 = time.time()
    procs: list[subprocess.Popen] = []
    log_paths: list[Path] = []
    backend_urls: list[str] = []

    num_processes = 1 if use_sglang_dp_controller else replicas_per_host
    mode_desc = f"sglang_dp={replicas_per_host}" if use_sglang_dp_controller else f"multi_proc={replicas_per_host}"
    log(
        idx,
        f"Launching {config_tag} ({mode_desc}): precision={precision}, tp={tp_size}, "
        f"attn={attention_backend}, kv={kv_cache_dtype}, mem_frac={mem_fraction_static}, "
        f"mrr={max_running_requests}, chunk={chunked_prefill_size}, ctx={context_length}",
    )

    for r in range(num_processes):
        port = 8000 + r
        base_gpu_id = 0 if use_sglang_dp_controller else (r * tp_size)
        lp = Path(f"/workspace/sglang_{config_tag}_{'dp' if use_sglang_dp_controller else 'r'}{r}.log")
        log_paths.append(lp)
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--device",
            "tpu",
            "--model-path",
            str(model_path),
            "--served-model-name",
            "RecursiveSI/flint",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--tp-size",
            str(tp_size),
            "--base-gpu-id",
            str(base_gpu_id),
            "--attention-backend",
            attention_backend,
            "--kv-cache-dtype",
            kv_cache_dtype,
            "--page-size",
            "64",
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--max-running-requests",
            str(max_running_requests),
            "--chunked-prefill-size",
            str(chunked_prefill_size),
            "--context-length",
            str(context_length),
            "--moe-runner-backend",
            "auto",
            "--grammar-backend",
            "flint",
            "--reasoning-parser",
            "flint",
            "--tool-call-parser",
            "flint",
            "--enable-fp32-lm-head",
            "--enable-metrics",
        ]
        if use_sglang_dp_controller and replicas_per_host > 1:
            cmd.extend(["--dp-size", str(replicas_per_host)])
        if enable_torch_compile:
            cmd.append("--enable-torch-compile")
        if extra_args:
            cmd.extend(extra_args)

        env = build_replica_env(
            tp_size,
            r,
            replicas_per_host=replicas_per_host,
            use_sglang_dp_controller=use_sglang_dp_controller,
            extra_env=extra_env,
        )
        if use_sglang_dp_controller and replicas_per_host > 1:
            env.pop("TORCH_TPU_SLICEBUILDER_ADDRESSES", None)
        lf = open(lp, "w")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append(p)
        backend_urls.append(f"http://127.0.0.1:{port}")
        if num_processes > 1 and r < num_processes - 1:
            time.sleep(8)

    deadline = time.time() + ready_timeout_s
    ready = [False] * num_processes
    while time.time() < deadline:
        for r in range(num_processes):
            if not ready[r]:
                if procs[r].poll() is not None:
                    tail = log_paths[r].read_text(errors="ignore")[-4000:]
                    raise RuntimeError(
                        f"SGLang replica {r} ({config_tag}, {mode_desc}) exited early with code {procs[r].returncode}:\n{tail}"
                    )
                if check_http_ok(f"{backend_urls[r]}/health"):
                    ready[r] = True
                    log(idx, f"Replica {r}/{num_processes} ({config_tag}) READY in {time.time() - t0:.1f}s")
        if all(ready):
            break
        time.sleep(3)

    if not all(ready):
        tails = "\n---\n".join(lp.read_text(errors="ignore")[-2500:] for lp in log_paths)
        raise TimeoutError(f"Timed out waiting for {config_tag} replicas {ready}:\n{tails}")

    gw_log = Path(f"/workspace/gateway_{config_tag}.log")
    gw_f = open(gw_log, "w")
    gw_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    gw_proc = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT_DIR / "flint_v5p_gateway.py"),
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--backends",
            *backend_urls,
        ],
        env=gw_env,
        stdout=gw_f,
        stderr=gw_f,
    )
    procs.append(gw_proc)
    for _ in range(30):
        if check_http_ok("http://127.0.0.1:8080/health"):
            break
        time.sleep(0.5)

    boot_s = round(time.time() - t0, 2)
    log(idx, f"Host cluster {config_tag} ({mode_desc}) fully READY on :8080 in {boot_s}s")
    return HostClusterHandle(
        config_tag=config_tag,
        precision=precision,
        tp_size=tp_size,
        replicas_per_host=replicas_per_host,
        backend_urls=backend_urls,
        entry_url="http://127.0.0.1:8080",
        procs=procs,
        log_paths=log_paths,
        mem_fraction_static=mem_fraction_static,
        boot_s=boot_s,
    )


def launch_host_servers(
    idx: int,
    *,
    config_tag: str,
    model_path: Path,
    precision: str,
    tp_size: int,
    replicas_per_host: int | None = None,
    attention_backend: str = "pallas_tpu_rpa_v3",
    kv_cache_dtype: str = "fp8_e4m3",
    mem_fraction_static: float = 0.84,
    max_running_requests: int = 64,
    chunked_prefill_size: int = 8192,
    context_length: int = 135168,
    enable_torch_compile: bool = True,
    extra_args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
    ready_timeout_s: int = 2400,
) -> HostClusterHandle:
    if replicas_per_host is None:
        replicas_per_host = 4 // tp_size
    try:
        return _launch_host_servers_impl(
            idx,
            config_tag=config_tag,
            model_path=model_path,
            precision=precision,
            tp_size=tp_size,
            replicas_per_host=replicas_per_host,
            use_sglang_dp_controller=False,
            attention_backend=attention_backend,
            kv_cache_dtype=kv_cache_dtype,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            chunked_prefill_size=chunked_prefill_size,
            context_length=context_length,
            enable_torch_compile=enable_torch_compile,
            extra_args=extra_args,
            extra_env=extra_env,
            ready_timeout_s=ready_timeout_s,
        )
    except Exception as first_err:
        if replicas_per_host <= 1:
            raise
        log(idx, f"WARNING: Multi-process launch for {config_tag} failed ({first_err}); retrying with --dp-size {replicas_per_host}...")
        return _launch_host_servers_impl(
            idx,
            config_tag=config_tag,
            model_path=model_path,
            precision=precision,
            tp_size=tp_size,
            replicas_per_host=replicas_per_host,
            use_sglang_dp_controller=True,
            attention_backend=attention_backend,
            kv_cache_dtype=kv_cache_dtype,
            mem_fraction_static=mem_fraction_static,
            max_running_requests=max_running_requests,
            chunked_prefill_size=chunked_prefill_size,
            context_length=context_length,
            enable_torch_compile=enable_torch_compile,
            extra_args=extra_args,
            extra_env=extra_env,
            ready_timeout_s=ready_timeout_s,
        )


def run_coherent_smoke_test(idx: int, entry_url: str, config_tag: str) -> dict[str, Any]:
    prompt_msg = "Write a Python function that merges overlapping intervals."
    payload = {
        "model": "RecursiveSI/flint",
        "messages": [{"role": "user", "content": prompt_msg}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        f"{entry_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt_s = time.perf_counter() - t0
    msg = body["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    combined = f"{reasoning}\n{content}".strip()
    has_code = ("def " in combined) and ("return" in combined)
    log(
        idx,
        f"[SMOKE {config_tag}] latency={dt_s:.2f}s has_code={has_code} "
        f"reasoning_chars={len(reasoning)} content_preview={content[:180]!r}",
    )
    if not has_code:
        raise RuntimeError(f"Smoke test for {config_tag} did not contain valid Python function: {body}")
    return {
        "config_tag": config_tag,
        "latency_s": round(dt_s, 3),
        "has_valid_python_fn": has_code,
        "reasoning_preview": reasoning[:400],
        "content": content,
        "usage": body.get("usage", {}),
    }


def _scrape_sglang_cache_counters(backend_urls: list[str]) -> tuple[float, float]:
    hits = 0.0
    total = 0.0
    for u in backend_urls:
        try:
            with urllib.request.urlopen(f"{u.rstrip('/')}/metrics", timeout=5.0) as r:
                txt = r.read().decode("utf-8", errors="ignore")
            for line in txt.splitlines():
                if line.startswith("#"):
                    continue
                if "cached_tokens_total" in line:
                    hits += float(line.rsplit(" ", 1)[-1])
                elif "prompt_tokens_total" in line or "input_tokens_total" in line:
                    total += float(line.rsplit(" ", 1)[-1])
        except Exception:
            pass
    return hits, total


def _flush_all_caches(entry_url: str, backend_urls: list[str]) -> None:
    for u in [entry_url, *backend_urls]:
        try:
            req = urllib.request.Request(f"{u.rstrip('/')}/flush_cache", data=b"{}", method="POST")
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            pass
    time.sleep(0.5)


def build_matched_prompts(
    isl: int, conc: int, n_reqs: int, hit_rate: float
) -> tuple[list[tuple[list[int], int | None]], list[list[int]], int]:
    """Deterministic prompt generator shared 1:1 between v5p-16 and v7x-16."""
    if hit_rate <= 0.0:
        prompts: list[tuple[list[int], int | None]] = []
        for i in range(n_reqs):
            rng = random.Random(100000 + isl * 1000 + conc * 10 + i)
            prompts.append(([rng.randint(1000, 32999) for _ in range(isl)], None))
        return prompts, [], 0

    shared_len = (int(isl * hit_rate) // 64) * 64
    unique_len = max(1, isl - shared_len)
    shared_len = isl - unique_len
    num_trunks = max(1, min(8, conc // 4))
    trunks: list[list[int]] = []
    for t in range(num_trunks):
        rng_t = random.Random(50000 + isl * 100 + t)
        trunks.append([rng_t.randint(1000, 32999) for _ in range(shared_len)])

    prompts = []
    for i in range(n_reqs):
        t_id = i % num_trunks
        rng_u = random.Random(900000 + isl * 1000 + conc * 10 + i)
        suffix = [rng_u.randint(1000, 32999) for _ in range(unique_len)]
        prompts.append((trunks[t_id] + suffix, t_id))
    return prompts, trunks, shared_len


def _stream_one_completion(
    entry_url: str,
    prompt_ids: list[int],
    osl: int,
    trunk_id: int | None = None,
    num_trunks: int = 0,
) -> dict[str, Any]:
    payload = {
        "model": "RecursiveSI/flint",
        "prompt": prompt_ids,
        "max_tokens": osl,
        "min_tokens": osl,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    if trunk_id is not None:
        headers["X-Prefix-Trunk"] = str(trunk_id)
        headers["X-Num-Trunks"] = str(num_trunks)

    req = urllib.request.Request(
        f"{entry_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    t0 = time.perf_counter()
    ttft: float | None = None
    t_prev: float | None = None
    t_last = t0
    itls_ms: list[float] = []
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=2400) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue
                data_s = line[6:]
                if data_s == "[DONE]":
                    break
                obj = json.loads(data_s)
                choices = obj.get("choices")
                if choices and choices[0].get("text") is not None and choices[0].get("text") != "":
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    elif t_prev is not None:
                        itls_ms.append((now - t_prev) * 1000.0)
                    t_prev = now
                    t_last = now
                    chunks += 1
        if ttft is None:
            return {"ok": False, "error": "no_tokens_streamed"}
        e2e = t_last - t0
        dec_s = max(1e-6, e2e - ttft)
        return {
            "ok": True,
            "ttft_ms": ttft * 1000.0,
            "e2e_s": e2e,
            "tpot_ms": dec_s * 1000.0 / max(1, osl - 1),
            "stream_tps": max(1, osl - 1) / dec_s,
            "itls_ms": itls_ms,
            "chunks": chunks,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def run_benchmark_point(
    idx: int,
    *,
    entry_url: str,
    backend_urls: list[str],
    meta: dict[str, Any],
    isl: int,
    osl: int,
    conc: int,
    hit_rate: float,
    num_hosts_active: int,
    local_jsonl: Path,
    gcs_bucket: str,
    gcs_jsonl_obj: str,
) -> dict[str, Any]:
    if conc == 1:
        n_reqs = 2 if isl >= 65536 else 3
    elif isl >= 131072:
        n_reqs = max(conc, 2) if hit_rate == 0.0 else max(conc, 4)
    elif isl >= 65536:
        n_reqs = max(conc, 4) if hit_rate == 0.0 else max(conc * 2, 8)
    else:
        n_reqs = max(conc * 2, 8)

    _flush_all_caches(entry_url, backend_urls)
    prompts, trunks, shared_len = build_matched_prompts(isl, conc, n_reqs, hit_rate)
    num_trunks = len(trunks)

    # Pre-warm prefix trunks on the exact target replicas for 80% / 95% multi-turn cache hits.
    # IMPORTANT (Hybrid 35x SWA-513 + 14x Full Attention): pre-warm ONLY trunk_tokens (osl=1)
    # without a >513-token pad_tail, otherwise SWARadixCache slides/evicts the SWA KV window
    # past shared_len and triggers prefill_swa_miss_tokens_total recomputation on the initial batch!
    if hit_rate > 0.0 and trunks:
        prewarm_jobs: list[tuple[str, list[int]]] = []
        n_b = len(backend_urls)
        for t_id, trunk_tokens in enumerate(trunks):
            if num_trunks < n_b:
                targets = [backend_urls[b_i] for b_i in range(n_b) if (b_i % num_trunks) == (t_id % num_trunks)]
            else:
                targets = [backend_urls[t_id % n_b]]
            for u in targets:
                prewarm_jobs.append((u, trunk_tokens))

        with cf.ThreadPoolExecutor(max_workers=min(16, len(prewarm_jobs))) as ex:
            list(ex.map(lambda jb: _stream_one_completion(jb[0], jb[1], 1), prewarm_jobs))

    hits_before, tot_before = _scrape_sglang_cache_counters(backend_urls)
    t_start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=conc) as ex:
        results = list(
            ex.map(
                lambda item: _stream_one_completion(entry_url, item[0], osl, trunk_id=item[1], num_trunks=num_trunks),
                prompts,
            )
        )
    wall_s = time.perf_counter() - t_start
    hits_after, tot_after = _scrape_sglang_cache_counters(backend_urls)

    oks = [r for r in results if r.get("ok")]
    errs = [r.get("error") for r in results if not r.get("ok")]
    completed = len(oks)
    out_toks = completed * osl
    in_toks = completed * isl

    measured_out_tps = out_toks / max(1e-6, wall_s)
    measured_tot_tps = (in_toks + out_toks) / max(1e-6, wall_s)
    per_host_out_tps = measured_out_tps / num_hosts_active
    slice_16chip_out_tps = per_host_out_tps * 4.0
    per_host_tot_tps = measured_tot_tps / num_hosts_active
    slice_16chip_tot_tps = per_host_tot_tps * 4.0

    ttfts = [r["ttft_ms"] for r in oks]
    tpots = [r["tpot_ms"] for r in oks]
    e2es = [r["e2e_s"] for r in oks]
    stps = [r["stream_tps"] for r in oks]
    all_itls = [x for r in oks for x in r.get("itls_ms", [])]

    obs_hit_pct = None
    if tot_after > tot_before:
        obs_hit_pct = round(100.0 * max(0.0, hits_after - hits_before) / (tot_after - tot_before), 2)

    rec = {
        "timestamp_utc": utc_now(),
        **meta,
        "num_hosts_measured": num_hosts_active,
        "chips_measured": num_hosts_active * 4,
        "isl": isl,
        "osl": osl,
        "prefix_hit_target_pct": int(round(hit_rate * 100)),
        "shared_prefix_tokens": shared_len,
        "observed_cache_hit_pct": obs_hit_pct,
        "concurrency": conc,
        "concurrency_slice_equiv": conc * (4 // num_hosts_active),
        "num_prompts": n_reqs,
        "completed": completed,
        "failed": len(results) - completed,
        "first_error": errs[0] if errs else None,
        "wall_s": round(wall_s, 3),
        "measured_output_tok_s": round(measured_out_tps, 2),
        "measured_total_tok_s": round(measured_tot_tps, 2),
        "per_host_output_tok_s": round(per_host_out_tps, 2),
        "per_host_total_tok_s": round(per_host_tot_tps, 2),
        "slice_16chip_output_tok_s": round(slice_16chip_out_tps, 2),
        "slice_16chip_total_tok_s": round(slice_16chip_tot_tps, 2),
        "mean_ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else None,
        "median_ttft_ms": round(statistics.median(ttfts), 2) if ttfts else None,
        "p99_ttft_ms": percentile(ttfts, 99.0),
        "mean_tpot_ms": round(statistics.mean(tpots), 2) if tpots else None,
        "median_tpot_ms": round(statistics.median(tpots), 2) if tpots else None,
        "p99_tpot_ms": percentile(tpots, 99.0),
        "mean_itl_ms": round(statistics.mean(all_itls), 2) if all_itls else None,
        "median_itl_ms": round(statistics.median(all_itls), 2) if all_itls else None,
        "p99_itl_ms": percentile(all_itls, 99.0),
        "mean_e2e_s": round(statistics.mean(e2es), 3) if e2es else None,
        "median_e2e_s": round(statistics.median(e2es), 3) if e2es else None,
        "p99_e2e_s": round(percentile(e2es, 99.0) or 0.0, 3) if e2es else None,
        "median_stream_tok_s": round(statistics.median(stps), 2) if stps else None,
    }

    local_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(local_jsonl, "a") as f:
        f.write(json.dumps(rec) + "\n")
    gcs_upload_bytes(gcs_bucket, gcs_jsonl_obj, local_jsonl.read_bytes(), content_type="application/jsonl")

    log(
        idx,
        f"[BENCH {meta['config_tag']}] {isl}:{osl} hit={int(hit_rate*100)}% c={conc} "
        f"ok={completed}/{n_reqs} host_out={rec['per_host_output_tok_s']:.1f} tok/s "
        f"slice16_out={rec['slice_16chip_output_tok_s']:.1f} tok/s "
        f"med_ttft={rec['median_ttft_ms']}ms med_tpot={rec['median_tpot_ms']}ms",
    )
    return rec


def run_smoke_mode(idx: int, bucket: str, run_id: str) -> None:
    """Step 1: Verify BF16 & FP8 across TP=4, TP=2, TP=1 across the 4 hosts in parallel!
    Pod 0: Converts FP8 + uploads to GCS, then tests FP8 TP=1 (4 replicas/host) & FP8 TP=4
    Pod 1: Tests BF16 TP=4 (1 replica/host) + coherent smoke + 4K/8K probe
    Pod 2: Tests BF16 TP=2 (2 replicas/host) + coherent smoke + 4K/8K probe
    Pod 3: Waits for FP8 in GCS (or converts locally), tests FP8 TP=2 (2 replicas/host)
    """
    need_bf16 = idx in (1, 2)
    need_fp8 = idx in (0, 3)
    bf16_dir, fp8_dir = bootstrap_pod(idx, bucket, need_bf16=need_bf16, need_fp8=need_fp8)

    local_jsonl = Path(f"/dev/shm/results/{run_id}_smoke_pod{idx}.jsonl")
    gcs_jsonl = f"flint-serving-results/raw/{run_id}_smoke_pod{idx}.jsonl"
    smoke_summary: dict[str, Any] = {"pod_idx": idx, "hostname": socket.gethostname(), "tests": []}

    configs_by_pod = {
        0: [
            ("fp8_tp1_dp4", fp8_dir, "fp8", 1, 4, 0.82),
            ("fp8_tp4_dp1", fp8_dir, "fp8", 4, 1, 0.85),
        ],
        1: [
            ("bf16_tp4_dp1", bf16_dir, "bf16", 4, 1, 0.85),
        ],
        2: [
            ("bf16_tp2_dp2", bf16_dir, "bf16", 2, 2, 0.83),
        ],
        3: [
            ("fp8_tp2_dp2", fp8_dir, "fp8", 2, 2, 0.84),
        ],
    }

    for tag, mdir, prec, tp, rph, mem_frac in configs_by_pod.get(idx, []):
        handle = launch_host_servers(
            idx,
            config_tag=tag,
            model_path=mdir,
            precision=prec,
            tp_size=tp,
            replicas_per_host=rph,
            mem_fraction_static=mem_frac,
            max_running_requests=64,
            chunked_prefill_size=8192,
            context_length=135168,
        )
        mem_stats = parse_server_memory_stats(handle.log_paths, tp, prec)
        smoke_res = run_coherent_smoke_test(idx, handle.entry_url, tag)
        meta = {
            "phase": "smoke",
            "config_tag": tag,
            "precision": prec,
            "tp_size": tp,
            "replicas_per_host": rph,
            "slice_replicas_dp": rph * 4,
            "attention_backend": "pallas_tpu_rpa_v3",
            "kv_cache_dtype": "fp8_e4m3",
            "router_mode": "fused",
            "mem_fraction_static": mem_frac,
            "max_running_requests": 64,
            "chunked_prefill_size": 8192,
            "boot_s": handle.boot_s,
            **mem_stats,
        }
        # Quick single-stream c=1 and c=8 sanity probes at 4K:1K (0% and 80% cache hit)
        b_c1 = run_benchmark_point(
            idx,
            entry_url=handle.entry_url,
            backend_urls=handle.backend_urls,
            meta=meta,
            isl=4096,
            osl=1024,
            conc=1,
            hit_rate=0.0,
            num_hosts_active=1,
            local_jsonl=local_jsonl,
            gcs_bucket=bucket,
            gcs_jsonl_obj=gcs_jsonl,
        )
        b_c8 = run_benchmark_point(
            idx,
            entry_url=handle.entry_url,
            backend_urls=handle.backend_urls,
            meta=meta,
            isl=8192,
            osl=1024,
            conc=8,
            hit_rate=0.80,
            num_hosts_active=1,
            local_jsonl=local_jsonl,
            gcs_bucket=bucket,
            gcs_jsonl_obj=gcs_jsonl,
        )
        smoke_summary["tests"].append(
            {
                "config_tag": tag,
                "boot_s": handle.boot_s,
                "mem_stats": mem_stats,
                "smoke": smoke_res,
                "probe_4k_c1": b_c1,
                "probe_8k_80pct_c8": b_c8,
            }
        )
        kill_sglang_and_gateways()

    gcs_upload_bytes(
        bucket,
        f"flint-serving-results/raw/{run_id}_smoke_summary_pod{idx}.json",
        json.dumps(smoke_summary, indent=2).encode("utf-8"),
    )
    log(idx, f"Smoke test pod {idx} COMPLETE!")


def run_full_sweep_mode(idx: int, bucket: str, run_id: str) -> None:
    """Step 2: Full Parallelized Matrix (Phase A across 4 hosts + Phase B 16-chip Slice-Wide)."""
    need_bf16 = True
    need_fp8 = True
    bf16_dir, fp8_dir = bootstrap_pod(idx, bucket, need_bf16=need_bf16, need_fp8=need_fp8)

    local_jsonl = Path(f"/dev/shm/results/{run_id}_sweep_pod{idx}.jsonl")
    gcs_jsonl = f"flint-serving-results/raw/{run_id}_sweep_pod{idx}.jsonl"

    # Register pod IP for Phase B 4-host slice coordination
    host_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    node_reg = {"idx": idx, "ip": host_ip, "hostname": socket.gethostname(), "ts": utc_now()}
    gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{idx}.json", json.dumps(node_reg).encode())

    def _sweep_server_matrix(
        *,
        tag: str,
        mdir: Path,
        prec: str,
        tp: int,
        rph: int,
        mem_frac: float,
        attn: str = "pallas_tpu_rpa_v3",
        kv_dtype: str = "fp8_e4m3",
        router_mode: str = "fused",
        mrr: int = 64,
        chunk: int = 8192,
        ctx_len: int = 135168,
        matrix: list[tuple[int, int, float, list[int]]],
    ) -> None:
        handle = launch_host_servers(
            idx,
            config_tag=tag,
            model_path=mdir,
            precision=prec,
            tp_size=tp,
            replicas_per_host=rph,
            attention_backend=attn,
            kv_cache_dtype=kv_dtype,
            mem_fraction_static=mem_frac,
            max_running_requests=mrr,
            chunked_prefill_size=chunk,
            context_length=ctx_len,
            extra_env={"FLINT_TPU_ROUTER_MODE": router_mode},
        )
        mem_stats = parse_server_memory_stats(handle.log_paths, tp, prec)
        meta = {
            "phase": "single_host_matrix",
            "config_tag": tag,
            "precision": prec,
            "tp_size": tp,
            "replicas_per_host": rph,
            "slice_replicas_dp": rph * 4,
            "attention_backend": attn,
            "kv_cache_dtype": kv_dtype,
            "router_mode": router_mode,
            "mem_fraction_static": mem_frac,
            "max_running_requests": mrr,
            "chunked_prefill_size": chunk,
            "boot_s": handle.boot_s,
            **mem_stats,
        }
        for isl, osl, hit, concs in matrix:
            for c in concs:
                run_benchmark_point(
                    idx,
                    entry_url=handle.entry_url,
                    backend_urls=handle.backend_urls,
                    meta=meta,
                    isl=isl,
                    osl=osl,
                    conc=c,
                    hit_rate=hit,
                    num_hosts_active=1,
                    local_jsonl=local_jsonl,
                    gcs_bucket=bucket,
                    gcs_jsonl_obj=gcs_jsonl,
                )
        kill_sglang_and_gateways()

    # Standard comprehensive workload matrix per primary serving topology:
    # Covers 4K:1K ref, Short (8K:1K, 16K:1K), Long (64K:1K, 128K:1K) x Prefix Hit (0%, 80%, 95%) x Concurrencies
    core_matrix_high_kv = [
        (4096, 1024, 0.0, [1, 2, 4, 8, 16, 32, 64, 128]),
        (4096, 1024, 0.80, [1, 8, 32, 64, 128]),
        (4096, 1024, 0.95, [1, 8, 32, 64, 128]),
        (8192, 1024, 0.0, [1, 2, 4, 8, 16, 32, 64, 128]),
        (8192, 1024, 0.80, [1, 4, 8, 16, 32, 64, 128]),
        (8192, 1024, 0.95, [1, 4, 8, 16, 32, 64, 128]),
        (16384, 1024, 0.0, [1, 2, 4, 8, 16, 32, 64]),
        (16384, 1024, 0.80, [1, 4, 8, 16, 32, 64, 128]),
        (16384, 1024, 0.95, [1, 4, 8, 16, 32, 64, 128]),
        (65536, 1024, 0.0, [1, 2, 4, 8, 16]),
        (65536, 1024, 0.80, [1, 4, 8, 16, 32]),
        (65536, 1024, 0.95, [1, 4, 8, 16, 32]),
        (131072, 1024, 0.0, [1, 2, 4, 8]),
        (131072, 1024, 0.80, [1, 2, 4, 8, 16]),
        (131072, 1024, 0.95, [1, 2, 4, 8, 16]),
    ]

    if idx == 0:
        # Pod 0: BF16 TP=4 full matrix + BF16 KV-cache & pallas_tpu_rpa_batched kernel ablations
        _sweep_server_matrix(
            tag="bf16_tp4_fp8kv_c8k",
            mdir=bf16_dir,
            prec="bf16",
            tp=4,
            rph=1,
            mem_frac=0.85,
            kv_dtype="fp8_e4m3",
            mrr=64,
            chunk=8192,
            matrix=core_matrix_high_kv,
        )
        # Ablation 1: BF16 KV cache vs FP8 KV cache on BF16 TP=4
        _sweep_server_matrix(
            tag="bf16_tp4_bf16kv_c8k",
            mdir=bf16_dir,
            prec="bf16",
            tp=4,
            rph=1,
            mem_frac=0.85,
            kv_dtype="bfloat16",
            mrr=64,
            chunk=8192,
            matrix=[
                (8192, 1024, 0.0, [1, 8, 32, 64]),
                (8192, 1024, 0.80, [8, 32, 64]),
                (65536, 1024, 0.0, [1, 4, 8]),
                (131072, 1024, 0.80, [1, 4, 8]),
            ],
        )
        # Ablation 2: pallas_tpu_rpa_batched attention backend
        try:
            _sweep_server_matrix(
                tag="bf16_tp4_fp8kv_rpa_batched",
                mdir=bf16_dir,
                prec="bf16",
                tp=4,
                rph=1,
                mem_frac=0.85,
                attn="pallas_tpu_rpa_batched",
                kv_dtype="fp8_e4m3",
                mrr=64,
                chunk=8192,
                matrix=[
                    (8192, 1024, 0.0, [1, 8, 32, 64]),
                    (65536, 1024, 0.80, [4, 16]),
                ],
            )
        except Exception as attn_err:
            log(idx, f"NOTE: pallas_tpu_rpa_batched ablation skipped/unsupported: {attn_err}")
            kill_sglang_and_gateways()
    elif idx == 1:
        # Pod 1: BF16 TP=2 (2 replicas/host = DP=8 slice) full matrix + FP8 TP=4 chunked-prefill ablations
        _sweep_server_matrix(
            tag="bf16_tp2_dp2_fp8kv_c8k",
            mdir=bf16_dir,
            prec="bf16",
            tp=2,
            rph=2,
            mem_frac=0.83,
            kv_dtype="fp8_e4m3",
            mrr=64,
            chunk=8192,
            matrix=core_matrix_high_kv,
        )
        for chunk_sz in (4096, 16384):
            _sweep_server_matrix(
                tag=f"fp8_tp4_fp8kv_c{chunk_sz // 1024}k",
                mdir=fp8_dir,
                prec="fp8",
                tp=4,
                rph=1,
                mem_frac=0.85,
                kv_dtype="fp8_e4m3",
                mrr=64,
                chunk=chunk_sz,
                matrix=[
                    (8192, 1024, 0.0, [1, 8, 32, 64]),
                    (16384, 1024, 0.80, [8, 32, 64]),
                    (65536, 1024, 0.0, [1, 4, 8]),
                    (131072, 1024, 0.80, [1, 4, 8]),
                ],
            )
    elif idx == 2:
        # Pod 2: FP8 TP=1 (4 replicas/host = DP=16 slice!) full matrix + max-running-requests & stock router ablations
        _sweep_server_matrix(
            tag="fp8_tp1_dp4_fp8kv_c8k",
            mdir=fp8_dir,
            prec="fp8",
            tp=1,
            rph=4,
            mem_frac=0.82,
            kv_dtype="fp8_e4m3",
            mrr=64,
            chunk=8192,
            matrix=core_matrix_high_kv,
        )
        for mrr_val in (16, 32):
            _sweep_server_matrix(
                tag=f"fp8_tp1_dp4_mrr{mrr_val}",
                mdir=fp8_dir,
                prec="fp8",
                tp=1,
                rph=4,
                mem_frac=0.82,
                kv_dtype="fp8_e4m3",
                mrr=mrr_val,
                chunk=8192,
                matrix=[
                    (8192, 1024, 0.0, [16, 32, 64, 128]),
                    (8192, 1024, 0.80, [32, 64, 128]),
                ],
            )
        # Ablation 3: Stock Python sigmoid_routing vs Fused Pallas 256-expert router_topk
        _sweep_server_matrix(
            tag="fp8_tp4_fp8kv_stock_router",
            mdir=fp8_dir,
            prec="fp8",
            tp=4,
            rph=1,
            mem_frac=0.85,
            kv_dtype="fp8_e4m3",
            router_mode="stock",
            mrr=64,
            chunk=8192,
            matrix=[
                (4096, 1024, 0.0, [1, 8, 32, 64]),
                (8192, 1024, 0.0, [1, 8, 32, 64]),
            ],
        )
    elif idx == 3:
        # Pod 3: FP8 TP=2 (2 replicas/host = DP=8 slice) full matrix + FP8 TP=4 full matrix
        _sweep_server_matrix(
            tag="fp8_tp2_dp2_fp8kv_c8k",
            mdir=fp8_dir,
            prec="fp8",
            tp=2,
            rph=2,
            mem_frac=0.84,
            kv_dtype="fp8_e4m3",
            mrr=64,
            chunk=8192,
            matrix=core_matrix_high_kv,
        )
        _sweep_server_matrix(
            tag="fp8_tp4_fp8kv_c8k",
            mdir=fp8_dir,
            prec="fp8",
            tp=4,
            rph=1,
            mem_frac=0.85,
            kv_dtype="fp8_e4m3",
            mrr=64,
            chunk=8192,
            matrix=core_matrix_high_kv,
        )

    # =========================================================================
    # PHASE B: True 4-Host 16-Chip (`v5p-16`) Slice-Wide Multi-Replica Serving!
    # =========================================================================
    try:
        fast_gcs_upload_dir(idx, bucket, Path("/dev/shm/sglang_xla_cache"), f"flint-serving-results/xla_cache/{run_id}/")
    except Exception as cache_up_err:
        log(idx, f"NOTE: XLA cache upload warning (non-fatal): {cache_up_err}")

    gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/phaseA_done_{idx}.json", b'{"ok":true}')
    log(idx, "Finished Phase A single-host matrix; waiting at barrier for all 4 hosts to enter Phase B 16-chip slice runs...")
    t_bar = time.time()
    while time.time() - t_bar < 7200:
        done_cnt = sum(
            1 for i in range(4) if gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/phaseA_done_{i}.json") is not None
        )
        if done_cnt == 4:
            break
        time.sleep(10)

    try:
        fast_gcs_sync_dir(idx, bucket, f"flint-serving-results/xla_cache/{run_id}/", Path("/dev/shm/sglang_xla_cache"), workers=16)
    except Exception as cache_dl_err:
        log(idx, f"NOTE: XLA cache download warning (non-fatal): {cache_dl_err}")

    nodes = {
        i: json.loads(gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{i}.json").decode())
        for i in range(4)
    }

    slice_configs = [
        ("slice16_bf16_tp2_dp8", bf16_dir, "bf16", 2, 2, 0.83, 64),
        ("slice16_bf16_tp4_dp4", bf16_dir, "bf16", 4, 1, 0.85, 64),
        ("slice16_fp8_tp1_dp16", fp8_dir, "fp8", 1, 4, 0.82, 64),
        ("slice16_fp8_tp2_dp8", fp8_dir, "fp8", 2, 2, 0.84, 64),
        ("slice16_fp8_tp4_dp4", fp8_dir, "fp8", 4, 1, 0.85, 64),
    ]

    for s_idx, (stag, smdir, sprec, stp, srph, smem, smrr) in enumerate(slice_configs):
        handle = launch_host_servers(
            idx,
            config_tag=stag,
            model_path=smdir,
            precision=sprec,
            tp_size=stp,
            replicas_per_host=srph,
            mem_fraction_static=smem,
            max_running_requests=smrr,
            chunked_prefill_size=8192,
            context_length=135168,
        )
        local_ports = [int(u.rsplit(":", 1)[-1]) for u in handle.backend_urls]
        ready_payload = json.dumps({"ok": True, "ip": host_ip, "ports": local_ports}).encode("utf-8")
        gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{idx}.json", ready_payload)
        ready_infos: dict[int, dict[str, Any]] = {}
        while len(ready_infos) < 4:
            for i in range(4):
                if i not in ready_infos:
                    b_raw = gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{i}.json")
                    if b_raw is not None:
                        ready_infos[i] = json.loads(b_raw.decode("utf-8"))
            if len(ready_infos) < 4:
                time.sleep(3)

        if idx == 0:
            all_16chip_backends = [
                f"http://{ready_infos[h].get('ip', nodes[h]['ip'])}:{p}"
                for h in range(4)
                for p in ready_infos[h].get("ports", [8000 + r for r in range(srph)])
            ]
            gw16_log = open(f"/workspace/gw16_{stag}.log", "w")
            gw_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            gw16 = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "flint_v5p_gateway.py"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8090",
                    "--backends",
                    *all_16chip_backends,
                ],
                env=gw_env,
                stdout=gw16_log,
                stderr=gw16_log,
            )
            for _ in range(30):
                if check_http_ok("http://127.0.0.1:8090/health"):
                    break
                time.sleep(1.0)

            mem_stats = parse_server_memory_stats(handle.log_paths, stp, sprec)
            s_meta = {
                "phase": "slice_16chip_empirical",
                "config_tag": stag,
                "precision": sprec,
                "tp_size": stp,
                "replicas_per_host": srph,
                "slice_replicas_dp": srph * 4,
                "attention_backend": "pallas_tpu_rpa_v3",
                "kv_cache_dtype": "fp8_e4m3",
                "mem_fraction_static": smem,
                "max_running_requests": smrr,
                "chunked_prefill_size": 8192,
                "boot_s": handle.boot_s,
                **mem_stats,
            }
            slice_matrix = [
                (4096, 1024, 0.0, [8, 32, 64, 128, 256]),
                (8192, 1024, 0.0, [4, 16, 64, 128, 256]),
                (8192, 1024, 0.80, [16, 64, 128, 256]),
                (8192, 1024, 0.95, [16, 64, 128, 256]),
                (16384, 1024, 0.0, [4, 16, 64, 128]),
                (16384, 1024, 0.80, [16, 64, 128, 256]),
                (16384, 1024, 0.95, [16, 64, 128, 256]),
                (65536, 1024, 0.0, [4, 16, 32, 64]),
                (65536, 1024, 0.80, [16, 32, 64]),
                (65536, 1024, 0.95, [16, 32, 64]),
                (131072, 1024, 0.0, [4, 8, 16, 32]),
                (131072, 1024, 0.80, [4, 16, 32]),
                (131072, 1024, 0.95, [4, 16, 32]),
            ]
            for isl, osl, hit, concs in slice_matrix:
                for c in concs:
                    if c > smrr * srph * 4:
                        continue
                    run_benchmark_point(
                        idx,
                        entry_url="http://127.0.0.1:8090",
                        backend_urls=all_16chip_backends,
                        meta=s_meta,
                        isl=isl,
                        osl=osl,
                        conc=c,
                        hit_rate=hit,
                        num_hosts_active=4,
                        local_jsonl=local_jsonl,
                        gcs_bucket=bucket,
                        gcs_jsonl_obj=gcs_jsonl,
                    )
            gw16.terminate()
            gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json", b'{"ok":true}')
        else:
            while gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json") is None:
                time.sleep(5)

        kill_sglang_and_gateways()

    log(idx, "ALL Phase A + Phase B benchmarks COMPLETE!")


def run_slice16_high_conc_mode(idx: int, bucket: str, run_id: str) -> None:
    """Dedicated 4-Host 16-Chip (`v5p-16`) High-Concurrency Empirical Sweep (`swa_full_tokens_ratio=0.10`)."""
    bf16_dir, fp8_dir = bootstrap_pod(idx, bucket, need_bf16=True, need_fp8=True)
    local_jsonl = Path(f"/dev/shm/results/{run_id}_slice16_pod{idx}.jsonl")
    gcs_jsonl = f"flint-serving-results/raw/{run_id}_slice16_pod{idx}.jsonl"

    host_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    node_reg = {"idx": idx, "ip": host_ip, "ts": utc_now()}
    gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{idx}.json", json.dumps(node_reg).encode())

    t_bar = time.time()
    while time.time() - t_bar < 3600:
        if sum(1 for i in range(4) if gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{i}.json") is not None) == 4:
            break
        time.sleep(3)

    nodes = {
        i: json.loads(gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{i}.json").decode())
        for i in range(4)
    }

    slice_configs = [
        (
            "slice16_bf16_tp2_dp8_mrr96_swa010",
            bf16_dir,
            "bf16",
            2,
            2,
            0.85,
            96,
            ["--swa-full-tokens-ratio", "0.10", "--torch-compile-max-bs", "96"],
        ),
        (
            "slice16_fp8_tp1_dp16_mrr64_swa010",
            fp8_dir,
            "fp8",
            1,
            4,
            0.84,
            64,
            ["--swa-full-tokens-ratio", "0.10", "--torch-compile-max-bs", "64"],
        ),
    ]

    for stag, smdir, sprec, stp, srph, smem, smrr, sextra in slice_configs:
        handle = launch_host_servers(
            idx,
            config_tag=stag,
            model_path=smdir,
            precision=sprec,
            tp_size=stp,
            replicas_per_host=srph,
            mem_fraction_static=smem,
            max_running_requests=smrr,
            chunked_prefill_size=8192,
            context_length=135168,
            extra_args=sextra,
        )
        local_ports = [int(u.rsplit(":", 1)[-1]) for u in handle.backend_urls]
        ready_payload = json.dumps({"ok": True, "ip": host_ip, "ports": local_ports}).encode("utf-8")
        gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{idx}.json", ready_payload)
        ready_infos: dict[int, dict[str, Any]] = {}
        while len(ready_infos) < 4:
            for i in range(4):
                if i not in ready_infos:
                    b_raw = gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{i}.json")
                    if b_raw is not None:
                        ready_infos[i] = json.loads(b_raw.decode("utf-8"))
            if len(ready_infos) < 4:
                time.sleep(3)

        if idx == 0:
            all_16chip_backends = [
                f"http://{ready_infos[h].get('ip', nodes[h]['ip'])}:{p}"
                for h in range(4)
                for p in ready_infos[h].get("ports", [8000 + r for r in range(srph)])
            ]
            gw16_log = open(f"/workspace/gw16_{stag}.log", "w")
            gw_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            gw16 = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "flint_v5p_gateway.py"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8090",
                    "--backends",
                    *all_16chip_backends,
                ],
                env=gw_env,
                stdout=gw16_log,
                stderr=gw16_log,
            )
            for _ in range(30):
                if check_http_ok("http://127.0.0.1:8090/health"):
                    break
                time.sleep(1.0)

            run_coherent_smoke_test(idx, "http://127.0.0.1:8090", stag)
            mem_stats = parse_server_memory_stats(handle.log_paths, stp, sprec)
            s_meta = {
                "phase": "slice_16chip_empirical",
                "config_tag": stag,
                "precision": sprec,
                "tp_size": stp,
                "replicas_per_host": srph,
                "slice_replicas_dp": srph * 4,
                "attention_backend": "pallas_tpu_rpa_v3",
                "kv_cache_dtype": "fp8_e4m3",
                "swa_full_tokens_ratio": 0.10,
                "mem_fraction_static": smem,
                "max_running_requests": smrr,
                "chunked_prefill_size": 8192,
                "boot_s": handle.boot_s,
                **mem_stats,
            }
            slice_matrix = [
                (4096, 1024, 0.0, [1, 8, 32, 64, 128, 256, 512, 768]),
                (4096, 1024, 0.80, [32, 128, 256, 512, 768]),
                (4096, 1024, 0.95, [32, 128, 256, 512, 768]),
                (8192, 1024, 0.0, [32, 128, 256, 512]),
                (8192, 1024, 0.95, [32, 128, 256, 512]),
                (65536, 1024, 0.95, [16, 64]),
                (131072, 1024, 0.95, [8, 32]),
            ]
            for isl, osl, hit, concs in slice_matrix:
                for c in concs:
                    if c > smrr * srph * 4:
                        continue
                    run_benchmark_point(
                        idx,
                        entry_url="http://127.0.0.1:8090",
                        backend_urls=all_16chip_backends,
                        meta=s_meta,
                        isl=isl,
                        osl=osl,
                        conc=c,
                        hit_rate=hit,
                        num_hosts_active=4,
                        local_jsonl=local_jsonl,
                        gcs_bucket=bucket,
                        gcs_jsonl_obj=gcs_jsonl,
                    )
            gw16.terminate()
            gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json", b'{"ok":true}')
        else:
            while gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json") is None:
                time.sleep(5)

        kill_sglang_and_gateways()

    log(idx, "ALL slice16_high_conc benchmarks COMPLETE!")


def run_slice16_longctx_swa_mode(idx: int, bucket: str, run_id: str) -> None:
    """Dedicated 4-Host 16-Chip (`v5p-16`) 64K/128K Long-Context, Prefix-Cache (0%/50%/95%), and SWA Ratio Sweep."""
    bf16_dir, fp8_dir = bootstrap_pod(idx, bucket, need_bf16=True, need_fp8=True)
    local_jsonl = Path(f"/dev/shm/results/{run_id}_longctx_swa_pod{idx}.jsonl")
    gcs_jsonl = f"flint-serving-results/raw/{run_id}_longctx_swa_pod{idx}.jsonl"

    host_ip = os.environ.get("POD_IP") or socket.gethostbyname(socket.gethostname())
    node_reg = {"idx": idx, "ip": host_ip, "ts": utc_now()}
    gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{idx}.json", json.dumps(node_reg).encode())

    t_bar = time.time()
    while time.time() - t_bar < 3600:
        if sum(1 for i in range(4) if gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{i}.json") is not None) == 4:
            break
        time.sleep(3)

    nodes = {
        i: json.loads(gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/node_{i}.json").decode())
        for i in range(4)
    }

    full_longctx_matrix = [
        (65536, 1024, 0.0, [1, 8, 16, 32, 64]),
        (65536, 1024, 0.50, [1, 8, 16, 32, 64, 128]),
        (65536, 1024, 0.95, [1, 8, 16, 32, 64, 128]),
        (131072, 1024, 0.0, [1, 4, 8, 16, 32]),
        (131072, 1024, 0.50, [1, 4, 8, 16, 32, 64]),
        (131072, 1024, 0.95, [1, 4, 8, 16, 32, 64]),
    ]
    swa_ablation_matrix = [
        (65536, 1024, 0.0, [1, 16, 64]),
        (65536, 1024, 0.50, [1, 16, 64, 128]),
        (65536, 1024, 0.95, [1, 16, 64, 128]),
        (131072, 1024, 0.0, [1, 8, 32]),
        (131072, 1024, 0.50, [1, 8, 32, 64]),
        (131072, 1024, 0.95, [1, 8, 32, 64]),
    ]

    slice_configs = [
        (
            "slice16_bf16_tp2_dp8_longctx_swa010",
            bf16_dir,
            "bf16",
            2,
            2,
            0.85,
            96,
            0.10,
            full_longctx_matrix,
        ),
        (
            "slice16_fp8_tp2_dp8_longctx_swa005",
            fp8_dir,
            "fp8",
            2,
            2,
            0.85,
            96,
            0.05,
            full_longctx_matrix,
        ),
        (
            "slice16_bf16_tp2_dp8_longctx_swa030",
            bf16_dir,
            "bf16",
            2,
            2,
            0.85,
            96,
            0.30,
            swa_ablation_matrix,
        ),
        (
            "slice16_bf16_tp2_dp8_longctx_swa080",
            bf16_dir,
            "bf16",
            2,
            2,
            0.85,
            96,
            0.80,
            swa_ablation_matrix,
        ),
    ]

    for stag, smdir, sprec, stp, srph, smem, smrr, sswa, smatrix in slice_configs:
        handle = launch_host_servers(
            idx,
            config_tag=stag,
            model_path=smdir,
            precision=sprec,
            tp_size=stp,
            replicas_per_host=srph,
            mem_fraction_static=smem,
            max_running_requests=smrr,
            chunked_prefill_size=8192,
            context_length=135168,
            extra_args=["--swa-full-tokens-ratio", f"{sswa:.2f}", "--torch-compile-max-bs", str(smrr)],
        )
        local_ports = [int(u.rsplit(":", 1)[-1]) for u in handle.backend_urls]
        ready_payload = json.dumps({"ok": True, "ip": host_ip, "ports": local_ports}).encode("utf-8")
        gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{idx}.json", ready_payload)
        ready_infos: dict[int, dict[str, Any]] = {}
        while len(ready_infos) < 4:
            for i in range(4):
                if i not in ready_infos:
                    b_raw = gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_ready_{i}.json")
                    if b_raw is not None:
                        ready_infos[i] = json.loads(b_raw.decode("utf-8"))
            if len(ready_infos) < 4:
                time.sleep(3)

        if idx == 0:
            all_16chip_backends = [
                f"http://{ready_infos[h].get('ip', nodes[h]['ip'])}:{p}"
                for h in range(4)
                for p in ready_infos[h].get("ports", [8000 + r for r in range(srph)])
            ]
            gw16_log = open(f"/workspace/gw16_{stag}.log", "w")
            gw_env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            gw16 = subprocess.Popen(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "flint_v5p_gateway.py"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8090",
                    "--backends",
                    *all_16chip_backends,
                ],
                env=gw_env,
                stdout=gw16_log,
                stderr=gw16_log,
            )
            for _ in range(30):
                if check_http_ok("http://127.0.0.1:8090/health"):
                    break
                time.sleep(1.0)

            run_coherent_smoke_test(idx, "http://127.0.0.1:8090", stag)
            mem_stats = parse_server_memory_stats(handle.log_paths, stp, sprec)
            s_meta = {
                "phase": "slice_16chip_longctx_swa",
                "config_tag": stag,
                "precision": sprec,
                "tp_size": stp,
                "replicas_per_host": srph,
                "slice_replicas_dp": srph * 4,
                "attention_backend": "pallas_tpu_rpa_v3",
                "kv_cache_dtype": "fp8_e4m3",
                "swa_full_tokens_ratio": sswa,
                "mem_fraction_static": smem,
                "max_running_requests": smrr,
                "chunked_prefill_size": 8192,
                "boot_s": handle.boot_s,
                **mem_stats,
            }
            for isl, osl, hit, concs in smatrix:
                for c in concs:
                    if c > smrr * srph * 4:
                        continue
                    run_benchmark_point(
                        idx,
                        entry_url="http://127.0.0.1:8090",
                        backend_urls=all_16chip_backends,
                        meta=s_meta,
                        isl=isl,
                        osl=osl,
                        conc=c,
                        hit_rate=hit,
                        num_hosts_active=4,
                        local_jsonl=local_jsonl,
                        gcs_bucket=bucket,
                        gcs_jsonl_obj=gcs_jsonl,
                    )
            gw16.terminate()
            gcs_upload_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json", b'{"ok":true}')
        else:
            while gcs_download_bytes(bucket, f"flint-serving-results/coord/{run_id}/{stag}_done.json") is None:
                time.sleep(5)

        kill_sglang_and_gateways()

    log(idx, "ALL slice16_longctx_swa benchmarks COMPLETE!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=["smoke", "full_sweep", "slice16_high_conc", "slice16_longctx_swa"],
        required=True,
    )
    ap.add_argument("--gcs-bucket", default="YOUR_GCS_BUCKET")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    idx = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    if args.mode == "smoke":
        run_smoke_mode(idx, args.gcs_bucket, args.run_id)
    elif args.mode == "slice16_high_conc":
        run_slice16_high_conc_mode(idx, args.gcs_bucket, args.run_id)
    elif args.mode == "slice16_longctx_swa":
        run_slice16_longctx_swa_mode(idx, args.gcs_bucket, args.run_id)
    else:
        run_full_sweep_mode(idx, args.gcs_bucket, args.run_id)


if __name__ == "__main__":
    main()

