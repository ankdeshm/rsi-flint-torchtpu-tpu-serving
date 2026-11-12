#!/usr/bin/env python3
"""Launches 3 simultaneous ablation servers across Pod 2's 8 TensorCores (4 physical v7x chips):
  - Chips 0+1 (base_gpu_id=0, TP=4, :30000): Block-FP8 [128,128] weights + FP8 KV + Fused Router (FP8 TP=4!)
  - Chip 2    (base_gpu_id=4, TP=2, :30001): BF16 weights + FP8 KV + pallas_tpu_rpa_batched (Attention backend ablation)
  - Chip 3    (base_gpu_id=6, TP=2, :30002): BF16 weights + FP8 KV + chunked_prefill=16384 + page_size=128 (Prefill/page ablation)
"""

from __future__ import annotations

import json
import os
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
        "slot": 0,
        "name": "ablation_fp8_tp4_1rep",
        "model_path": "/dev/shm/flint-fp8",
        "tp_size": 4,
        "base_gpu_id": 0,
        "port": 30000,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_v3",
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 16384,
        "page_size": 64,
        "ind_dir": "/dev/shm/ind_flint-fp8_tp4_dp1_ep1_fp8_e4m3_pallas_tpu_rpa_v3_rmfused_ps64_cp8192_bs64_r0",
    },
    {
        "slot": 1,
        "name": "ablation_rpabatched_tp2_1chip",
        "model_path": "/dev/shm/flint",
        "tp_size": 2,
        "base_gpu_id": 4,
        "port": 30001,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_batched",
        "chunked_prefill_size": 8192,
        "max_prefill_tokens": 16384,
        "page_size": 64,
        "ind_dir": "/dev/shm/ind_ablation_rpabatched_r1",
    },
    {
        "slot": 2,
        "name": "ablation_cp16k_ps128_tp2_1chip",
        "model_path": "/dev/shm/flint",
        "tp_size": 2,
        "base_gpu_id": 6,
        "port": 30002,
        "kv_cache_dtype": "fp8_e4m3",
        "attention_backend": "pallas_tpu_rpa_v3",
        "chunked_prefill_size": 16384,
        "max_prefill_tokens": 32768,
        "page_size": 128,
        "ind_dir": "/dev/shm/ind_ablation_cp16k_ps128_r2",
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
        s = cfg["slot"]
        tp = cfg["tp_size"]
        port = cfg["port"]
        nccl_port = 27500 + s * 20
        master_port = 29500 + s * 20
        slice_addrs = ",".join(f"127.0.0.1:{28500 + s * 20 + k}" for k in range(tp))
        metrics_ports = ",".join(str(8440 + s * 8 + k) for k in range(tp))

        tmp_dir = Path(f"/dev/shm/tmp_pod2_s{s}")
        ind_dir = Path(cfg["ind_dir"])
        tmp_dir.mkdir(parents=True, exist_ok=True)
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
        env["FLINT_TPU_ROUTER_MODE"] = "fused"
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
            str(tp),
            "--base-gpu-id",
            str(cfg["base_gpu_id"]),
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
            cfg["attention_backend"],
            "--kv-cache-dtype",
            cfg["kv_cache_dtype"],
            "--enable-fp32-lm-head",
            "--context-length",
            "135168",
            "--mem-fraction-static",
            "0.85",
            "--chunked-prefill-size",
            str(cfg["chunked_prefill_size"]),
            "--max-prefill-tokens",
            str(cfg["max_prefill_tokens"]),
            "--max-running-requests",
            "64",
            "--page-size",
            str(cfg["page_size"]),
            "--enable-torch-compile",
        ]
        if "cuda_graph_max_bs" in fields:
            cmd.extend(["--cuda-graph-max-bs", "64"])
        if "torch_compile_max_bs" in fields:
            cmd.extend(["--torch-compile-max-bs", "64"])
        if "disaggregation_bootstrap_port" in fields:
            cmd.extend(["--disaggregation-bootstrap-port", str(18900 + s)])
        if "engine_info_bootstrap_port" in fields:
            cmd.extend(["--engine-info-bootstrap-port", str(16700 + s)])

        log_path = log_dir / f"{cfg['name']}_port{port}.log"
        logf = open(log_path, "wb")
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((cfg, port, proc, log_path))

    t0 = time.time()
    ready = {}
    failed = {}
    while time.time() - t0 < 2100:
        for cfg, port, proc, log_path in procs:
            name = cfg["name"]
            if name in ready or name in failed:
                continue
            if http_health(port):
                ready[name] = round(time.time() - t0, 1)
                print(f"[pod2-ablations] READY {name} on :{port} in {ready[name]}s", flush=True)
                continue
            rc = proc.poll()
            if rc is not None:
                tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-45:]) if log_path.exists() else ""
                failed[name] = {"port": port, "rc": rc, "tail": tail}
                print(f"[pod2-ablations] FAILED {name} on :{port} rc={rc} (continuing others)", flush=True)
        if len(ready) + len(failed) == len(procs):
            break
        time.sleep(3.0)

    summary = {"status": "done", "elapsed_s": round(time.time() - t0, 1), "ready": ready, "failed": failed, "servers": []}
    for cfg, port, _, log_path in procs:
        name = cfg["name"]
        if name in ready:
            srv_info = http_get_json(f"http://127.0.0.1:{port}/get_server_info") or {}
            telem = parse_log_telemetry(log_path)
            mem = {}
            istates = srv_info.get("internal_states") or []
            if istates and isinstance(istates, list) and isinstance(istates[0], dict):
                mem = istates[0].get("memory_usage") or {}
            summary["servers"].append({"name": name, "port": port, "ready_s": ready[name], "memory_usage": mem, "telemetry": telem})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
