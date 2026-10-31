#!/usr/bin/env python3
"""Host-local multi-replica launcher & health verifier for RecursiveSI/flint on TPU v7x.

Supports every intra-host/sub-host TPU v7x parallelism layout across 8 TensorCores (4 chips):
  - TP=1 -> up to 8 independent replicas per host (1 TensorCore / 95 GiB HBM each; requires FP8 weights)
  - TP=2 -> up to 4 independent replicas per host (1 physical v7x chip / 190 GiB HBM each; zero inter-chip ICI)
  - TP=4 -> up to 2 independent replicas per host (2 physical v7x chips / 380 GiB HBM each)
  - TP=8 -> 1 replica per host (4 physical v7x chips / 760 GiB HBM; optional EP=8)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


UNSET_MULTIHOST_ENVS = [
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_WORKER_ID",
    "TPU_PROCESS_PORT",
    "CLOUD_TPU_TASK_ID",
    "TPU_CHIPS_PER_HOST_BOUNDS",
    "TPU_HOST_BOUNDS",
    "TPU_STDERR_LOG_LEVEL",
]


def kill_existing_sglang() -> None:
    my_pid = os.getpid()
    ppid = os.getppid()
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
    except Exception:
        return
    victims: list[int] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        cmd = parts[1]
        if pid in (1, my_pid, ppid):
            continue
        if any(
            pat in cmd
            for pat in (
                "sglang.launch_server",
                "sglang::scheduler",
                "sglang::detokenizer",
                "sglang.srt",
            )
        ):
            victims.append(pid)
    if victims:
        for pid in victims:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(2.0)


def ensure_mock_metadata() -> None:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:80/computeMetadata/v1/instance/attributes/TPU_WORKER_ID",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return
    except Exception:
        pass
    if Path("/workspace/mock_metadata.py").exists():
        with open("/workspace/mock_metadata.log", "ab") as logf:
            subprocess.Popen(
                [sys.executable, "/workspace/mock_metadata.py"],
                stdout=logf,
                stderr=logf,
                start_new_session=True,
            )
        time.sleep(1.0)


def supported_server_arg_fields() -> set[str]:
    try:
        from sglang.srt.server_args import ServerArgs

        return {f.name for f in dataclasses.fields(ServerArgs)}
    except Exception:
        return set()


def parse_log_telemetry(log_path: Path) -> dict:
    info: dict = {}
    if not log_path.exists():
        return info
    text = log_path.read_text(errors="replace")
    # Look for KV cache token counts, weight memory, available HBM
    for line in text.splitlines():
        if "Flint weights loaded:" in line:
            info["weights_loaded_line"] = line.strip()
        if "Memory usage" in line or "available_gpu_mem" in line or "KV Cache" in line or "max_total_num_tokens" in line:
            info.setdefault("mem_lines", []).append(line.strip())
        if "swa" in line.lower() and ("token" in line.lower() or "pool" in line.lower() or "size" in line.lower()):
            info.setdefault("swa_lines", []).append(line.strip())
        m_tok = re.search(r"max_total_num_tokens[=:\s]+(\d+)", line)
        if m_tok:
            info["max_total_num_tokens"] = int(m_tok.group(1))
        m_full = re.search(r"full[_\s]available[_\s](?:size|tokens)[=:\s]+(\d+)", line, re.IGNORECASE)
        if m_full:
            info["full_kv_tokens"] = int(m_full.group(1))
        m_swa = re.search(r"swa[_\s]available[_\s](?:size|tokens)[=:\s]+(\d+)", line, re.IGNORECASE)
        if m_swa:
            info["swa_kv_tokens"] = int(m_swa.group(1))
    return info


def http_get_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode())
    except Exception:
        return None
    return None


def http_health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["start", "stop", "status"], default="start")
    ap.add_argument("--tag", default="pass")
    ap.add_argument("--model-path", default="/dev/shm/flint")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--num-replicas", type=int, default=0, help="0 = fill all 8 TensorCores (8 // tp_size)")
    ap.add_argument("--base-port", type=int, default=30000)
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3", choices=["fp8_e4m3", "bfloat16"])
    ap.add_argument("--attention-backend", default="pallas_tpu_rpa_v3")
    ap.add_argument("--moe-runner-backend", default="auto")
    ap.add_argument("--context-length", type=int, default=135168)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--max-prefill-tokens", type=int, default=16384)
    ap.add_argument("--max-running-requests", type=int, default=128)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--swa-full-tokens-ratio", type=float, default=-1.0, help="<0 uses SGLang default")
    ap.add_argument("--disable-torch-compile", action="store_true")
    ap.add_argument("--extra-args", default="", help="Space-separated extra CLI args for sglang.launch_server")
    ap.add_argument("--wait-timeout", type=int, default=600)
    args = ap.parse_args()

    if args.mode == "stop":
        kill_existing_sglang()
        print(json.dumps({"host": os.uname().nodename, "status": "stopped"}))
        return

    num_replicas = args.num_replicas if args.num_replicas > 0 else (8 // args.tp_size)
    assert num_replicas * args.tp_size <= 8, f"num_replicas={num_replicas} * tp_size={args.tp_size} > 8 TensorCores!"

    if args.mode == "status":
        statuses = {args.base_port + r: http_health(args.base_port + r) for r in range(num_replicas)}
        print(json.dumps({"host": os.uname().nodename, "ports": statuses}))
        return

    kill_existing_sglang()
    ensure_mock_metadata()

    fields = supported_server_arg_fields()
    log_dir = Path("/workspace/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, int, subprocess.Popen, Path]] = []
    model_short = Path(args.model_path).name
    cache_stem = (
        f"{model_short}_tp{args.tp_size}_ep{args.ep_size}_{args.kv_cache_dtype}_"
        f"{args.attention_backend}_ps{args.page_size}_cp{args.chunked_prefill_size}"
    )

    for r in range(num_replicas):
        base_gpu_id = r * args.tp_size
        port = args.base_port + r
        nccl_port = 27500 + r * 20
        master_port = 29500 + r * 20
        slice_addrs = ",".join(f"127.0.0.1:{28500 + r * 20 + k}" for k in range(args.tp_size))
        metrics_ports = ",".join(str(8440 + r * 8 + k) for k in range(args.tp_size))

        tmp_dir = Path(f"/dev/shm/tmp_rep{r}")
        ind_dir = Path(f"/dev/shm/ind_{cache_stem}_r{r}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ind_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        for k in UNSET_MULTIHOST_ENVS:
            env.pop(k, None)

        if Path("/workspace/metadata_hook.so").exists():
            env["LD_PRELOAD"] = "/workspace/metadata_hook.so"
        env["GCE_METADATA_HOST"] = "127.0.0.1:80"
        env["GCE_METADATA_IP"] = "127.0.0.1"
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
        env["FLASHINFER_USE_CUDA_NORM"] = "1"
        env["SGLANG_TPU_PD_COLOCATED"] = "1"
        env["SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR"] = str(ind_dir)
        env["TMPDIR"] = str(tmp_dir)
        env["MASTER_ADDR"] = "127.0.0.1"
        env["MASTER_PORT"] = str(master_port)
        env["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = slice_addrs
        env["TPU_RUNTIME_METRICS_PORTS"] = metrics_ports

        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--device",
            "tpu",
            "--tp-size",
            str(args.tp_size),
            "--base-gpu-id",
            str(base_gpu_id),
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--nccl-port",
            str(nccl_port),
            "--trust-remote-code",
            "--grammar-backend",
            "flint",
            "--reasoning-parser",
            "flint",
            "--tool-call-parser",
            "flint",
            "--moe-runner-backend",
            args.moe_runner_backend,
            "--attention-backend",
            args.attention_backend,
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--enable-fp32-lm-head",
            "--context-length",
            str(args.context_length),
            "--mem-fraction-static",
            str(args.mem_fraction_static),
            "--chunked-prefill-size",
            str(args.chunked_prefill_size),
            "--max-prefill-tokens",
            str(args.max_prefill_tokens),
            "--max-running-requests",
            str(args.max_running_requests),
            "--page-size",
            str(args.page_size),
        ]
        if args.ep_size > 1:
            cmd.extend(["--ep-size", str(args.ep_size)])
        if args.swa_full_tokens_ratio >= 0:
            cmd.extend(["--swa-full-tokens-ratio", str(args.swa_full_tokens_ratio)])
        if not args.disable_torch_compile:
            cmd.append("--enable-torch-compile")
        if "disaggregation_bootstrap_port" in fields:
            cmd.extend(["--disaggregation-bootstrap-port", str(18900 + r)])
        if "engine_info_bootstrap_port" in fields:
            cmd.extend(["--engine-info-bootstrap-port", str(16700 + r)])
        if args.extra_args.strip():
            cmd.extend(args.extra_args.strip().split())

        log_path = log_dir / f"{args.tag}_rep{r}_port{port}.log"
        logf = open(log_path, "wb")
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((r, port, proc, log_path))

    t0 = time.time()
    ready = set()
    while time.time() - t0 < args.wait_timeout:
        for r, port, proc, log_path in procs:
            if r in ready:
                continue
            if http_health(port):
                ready.add(r)
                continue
            rc = proc.poll()
            if rc is not None:
                tail = ""
                if log_path.exists():
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
                print(
                    json.dumps(
                        {
                            "host": os.uname().nodename,
                            "status": "error",
                            "failed_replica": r,
                            "port": port,
                            "returncode": rc,
                            "log_path": str(log_path),
                            "log_tail": tail,
                        }
                    ),
                    flush=True,
                )
                sys.exit(1)
        if len(ready) == num_replicas:
            break
        time.sleep(3.0)

    if len(ready) != num_replicas:
        tails = {
            r: "\n".join(lp.read_text(errors="replace").splitlines()[-35:]) if lp.exists() else ""
            for r, _, _, lp in procs
            if r not in ready
        }
        print(
            json.dumps(
                {
                    "host": os.uname().nodename,
                    "status": "timeout",
                    "ready": sorted(ready),
                    "expected": num_replicas,
                    "tails": tails,
                }
            ),
            flush=True,
        )
        sys.exit(2)

    elapsed = round(time.time() - t0, 1)
    rep_info = []
    for r, port, _, log_path in procs:
        srv_info = http_get_json(f"http://127.0.0.1:{port}/get_server_info") or {}
        telem = parse_log_telemetry(log_path)
        rep_info.append(
            {
                "replica": r,
                "port": port,
                "log_path": str(log_path),
                "max_total_num_tokens": srv_info.get("max_total_num_tokens", telem.get("max_total_num_tokens")),
                "internal_states": srv_info.get("internal_states"),
                "telemetry": telem,
            }
        )

    print(
        json.dumps(
            {
                "host": os.uname().nodename,
                "status": "ready",
                "startup_sec": elapsed,
                "num_replicas": num_replicas,
                "tp_size": args.tp_size,
                "ep_size": args.ep_size,
                "replicas": rep_info,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
