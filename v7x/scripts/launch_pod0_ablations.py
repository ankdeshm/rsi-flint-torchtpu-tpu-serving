#!/usr/bin/env python3
"""Launches 4 simultaneous 1-chip (TP=2) ablation servers across Pod 0's 4 isolated VFIO chips.

  - Chip 0 (base_gpu_id=0, :30000): Block-FP8 [128,128] weights + FP8 KV + Fused Router (warms FP8 TP=2 Inductor cache!)
  - Chip 1 (base_gpu_id=2, :30001): BF16 weights + BF16 KV Cache + Fused Router (Pass 5 KV dtype ablation)
  - Chip 2 (base_gpu_id=4, :30002): BF16 weights + FP8 KV + Stock Unfused Router (Pass 6a Pallas router_topk ablation)
  - Chip 3 (base_gpu_id=6, :30003): BF16 weights + FP8 KV + Fused Router + --swa-full-tokens-ratio 0.10 (Pass 6b SWA pool ratio ablation)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from launch_host_replicas import (
    UNSET_MULTIHOST_ENVS,
    ensure_mock_metadata,
    http_get_json,
    http_health,
    kill_existing_sglang,
    parse_log_telemetry,
    supported_server_arg_fields,
)


CONFIGS = [
    {
        "r": 0,
        "name": "ablation_fp8_tp2_1chip",
        "model_path": "/dev/shm/flint-fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "router_mode": "fused",
        "swa_ratio": -1.0,
        "ind_dir": "/dev/shm/ind_flint-fp8_tp2_dp1_ep1_fp8_e4m3_pallas_tpu_rpa_v3_rmfused_ps64_cp8192_bs64_r0",
        "warm_from": "",
    },
    {
        "r": 1,
        "name": "ablation_bf16kv_tp2_1chip",
        "model_path": "/dev/shm/flint",
        "kv_cache_dtype": "bfloat16",
        "router_mode": "fused",
        "swa_ratio": -1.0,
        "ind_dir": "/dev/shm/ind_ablation_bf16kv_r1",
        "warm_from": "",
    },
    {
        "r": 2,
        "name": "ablation_stock_router_tp2_1chip",
        "model_path": "/dev/shm/flint",
        "kv_cache_dtype": "fp8_e4m3",
        "router_mode": "stock",
        "swa_ratio": -1.0,
        "ind_dir": "/dev/shm/ind_ablation_stockrouter_r2",
        "warm_from": "",
    },
    {
        "r": 3,
        "name": "ablation_swa010_tp2_1chip",
        "model_path": "/dev/shm/flint",
        "kv_cache_dtype": "fp8_e4m3",
        "router_mode": "fused",
        "swa_ratio": 0.10,
        "ind_dir": "/dev/shm/ind_ablation_swa010_r3",
        "warm_from": "/dev/shm/ind_flint_tp2_dp1_ep1_fp8_e4m3_pallas_tpu_rpa_v3_rmfused_ps64_cp8192_bs64_r0",
    },
]


def main() -> None:
    kill_existing_sglang()
    ensure_mock_metadata()
    subprocess.run([sys.executable, "/workspace/patch_flint_sglang_tpu.py"], check=True)
    fields = supported_server_arg_fields()
    log_dir = Path("/workspace/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for cfg in CONFIGS:
        r = cfg["r"]
        base_gpu_id = r * 2
        port = 30000 + r
        nccl_port = 27500 + r * 20
        master_port = 29500 + r * 20
        slice_addrs = f"127.0.0.1:{28500 + r * 20},127.0.0.1:{28500 + r * 20 + 1}"
        metrics_ports = f"{8440 + r * 8},{8440 + r * 8 + 1}"

        tmp_dir = Path(f"/dev/shm/tmp_rep{r}")
        ind_dir = Path(cfg["ind_dir"])
        tmp_dir.mkdir(parents=True, exist_ok=True)
        warm_src = Path(cfg["warm_from"]) if cfg["warm_from"] else None
        if warm_src and warm_src.exists() and any(warm_src.iterdir()) and (not ind_dir.exists() or not any(ind_dir.iterdir())):
            shutil.copytree(warm_src, ind_dir, dirs_exist_ok=True)
        else:
            ind_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        for k in UNSET_MULTIHOST_ENVS:
            env.pop(k, None)
        if Path("/workspace/metadata_hook.so").exists():
            env["LD_PRELOAD"] = "/workspace/metadata_hook.so"
        env["GCE_METADATA_HOST"] = "127.0.0.1:80"
        env["GCE_METADATA_IP"] = "127.0.0.1"
        env["TPU_SKIP_MDS_QUERY"] = "1"
        env["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "flint_sglang"
        env["FLINT_TPU_ROUTER_MODE"] = cfg["router_mode"]
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
            cfg["model_path"],
            "--device",
            "tpu",
            "--tp-size",
            "2",
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
            "auto",
            "--attention-backend",
            "pallas_tpu_rpa_v3",
            "--kv-cache-dtype",
            cfg["kv_cache_dtype"],
            "--enable-fp32-lm-head",
            "--context-length",
            "135168",
            "--mem-fraction-static",
            "0.85",
            "--chunked-prefill-size",
            "8192",
            "--max-prefill-tokens",
            "16384",
            "--max-running-requests",
            "64",
            "--page-size",
            "64",
            "--enable-torch-compile",
        ]
        if "cuda_graph_max_bs" in fields:
            cmd.extend(["--cuda-graph-max-bs", "64"])
        if "torch_compile_max_bs" in fields:
            cmd.extend(["--torch-compile-max-bs", "64"])
        if cfg["swa_ratio"] >= 0:
            cmd.extend(["--swa-full-tokens-ratio", str(cfg["swa_ratio"])])
        if "disaggregation_bootstrap_port" in fields:
            cmd.extend(["--disaggregation-bootstrap-port", str(18900 + r)])
        if "engine_info_bootstrap_port" in fields:
            cmd.extend(["--engine-info-bootstrap_port".replace("_port", "-port"), str(16700 + r)])

        log_path = log_dir / f"{cfg['name']}_port{port}.log"
        logf = open(log_path, "wb")
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((cfg, port, proc, log_path))

    t0 = time.time()
    ready = {}
    while time.time() - t0 < 2100:
        for cfg, port, proc, log_path in procs:
            name = cfg["name"]
            if name in ready:
                continue
            if http_health(port):
                ready[name] = round(time.time() - t0, 1)
                print(f"[pod0-ablations] READY {name} on :{port} in {ready[name]}s", flush=True)
                continue
            rc = proc.poll()
            if rc is not None:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-50:]) if log_path.exists() else ""
                print(json.dumps({"status": "error", "failed": name, "port": port, "rc": rc, "tail": tail}), flush=True)
                sys.exit(1)
        if len(ready) == len(procs):
            break
        time.sleep(3.0)

    if len(ready) != len(procs):
        print(json.dumps({"status": "timeout", "ready": ready}), flush=True)
        sys.exit(2)

    summary = {"status": "ready", "elapsed_s": round(time.time() - t0, 1), "servers": []}
    for cfg, port, _, log_path in procs:
        srv_info = http_get_json(f"http://127.0.0.1:{port}/get_server_info") or {}
        telem = parse_log_telemetry(log_path)
        mem = {}
        istates = srv_info.get("internal_states") or []
        if istates and isinstance(istates, list) and isinstance(istates[0], dict):
            mem = istates[0].get("memory_usage") or {}
        summary["servers"].append(
            {
                "name": cfg["name"],
                "port": port,
                "ready_s": ready[cfg["name"]],
                "memory_usage": mem,
                "telemetry": telem,
            }
        )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
