#!/usr/bin/env python3
"""Master 4-Host TPU v7x-16 Slice Orchestrator for RecursiveSI/flint Serving Benchmarks.

Dynamically discovers the 4 live flint-v7x-16-spot-runner pods + internal IPs on Cluster 1,
pushes scripts, launches isolated sub-host/intra-host SGLang replicas across all 4 hosts in parallel,
starts the cache-aware prefix-affinity gateway on Pod 0 (:8000), and executes bench_flint_v7x.py
with immediate local + GCS persistence.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


CTX = "--context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1"
LOCAL_DIR = Path("/Users/deshmukhankita/ankita-dev/Atlas Cloud POC/flint_v7x_serving")
RESULTS_DIR = LOCAL_DIR / "results"
SCRIPTS = [
    "mock_metadata.py",
    "metadata_hook.c",
    "convert_flint_fp8.py",
    "launch_host_replicas.py",
    "gateway_flint_v7x.py",
    "bench_flint_v7x.py",
    "patch_flint_sglang_tpu.py",
    "patch_flint_router_topk.py",
]


def k_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CLOUDSDK_METRICS_ENVIRONMENT"] = "datacloud.antigravity"
    return env


def run_kubectl(args: list[str], check: bool = True, timeout: int = 900) -> subprocess.CompletedProcess:
    cmd = ["kubectl", CTX, *args]
    return subprocess.run(cmd, env=k_env(), text=True, capture_output=True, check=check, timeout=timeout)


def discover_ready_pods(wait_timeout: int = 7200) -> list[dict[str, str]]:
    """Wait until all 4 indexed runner pods are Running and have READY_FOR_BENCHMARKS."""
    t0 = time.time()
    while time.time() - t0 < wait_timeout:
        res = run_kubectl(["get", "pods", "-l", "app=flint-v7x-16-spot-runner", "-o", "json"], check=False, timeout=30)
        if res.returncode == 0 and res.stdout.strip():
            data = json.loads(res.stdout)
            by_idx: dict[int, dict[str, str]] = {}
            for item in data.get("items", []):
                phase = item.get("status", {}).get("phase")
                if phase != "Running":
                    continue
                labels = item.get("metadata", {}).get("labels", {})
                idx_s = labels.get("batch.kubernetes.io/job-completion-index")
                if idx_s is None:
                    continue
                cs = item.get("status", {}).get("containerStatuses", [])
                if not cs or not all(c.get("ready") for c in cs):
                    continue
                pod_name = item["metadata"]["name"]
                pod_ip = item.get("status", {}).get("podIP", "")
                node_name = item.get("spec", {}).get("nodeName", "")
                if pod_ip:
                    by_idx[int(idx_s)] = {"index": int(idx_s), "pod": pod_name, "ip": pod_ip, "node": node_name}
            if len(by_idx) == 4:
                pods = [by_idx[i] for i in range(4)]
                # Check marker /dev/shm/flint-fp8/.fp8_ready on all 4 pods
                all_staged = True
                for p in pods:
                    chk = run_kubectl(
                        ["exec", p["pod"], "-c", "flint-tpu", "--", "test", "-f", "/dev/shm/flint-fp8/.fp8_ready"],
                        check=False,
                        timeout=15,
                    )
                    if chk.returncode != 0:
                        all_staged = False
                        break
                if all_staged:
                    return pods
                print(f"[orchestrator] 4 pods Running ({[p['node'] for p in pods]}), waiting for /dev/shm staging...", flush=True)
            else:
                print(f"[orchestrator] Waiting for 4 Running pods (currently {len(by_idx)}/4)...", flush=True)
        time.sleep(10)
    raise TimeoutError("Timed out waiting for 4 Ready flint-v7x-16-spot-runner pods")


def sync_scripts_to_pods(pods: list[dict[str, str]]) -> None:
    def sync_one(p: dict[str, str]) -> None:
        pod = p["pod"]
        for s in SCRIPTS:
            src = LOCAL_DIR / s
            if src.exists():
                for attempt in range(3):
                    res = run_kubectl(["cp", str(src), f"{pod}:/workspace/{s}", "-c", "flint-tpu"], check=False, timeout=60)
                    if res.returncode == 0:
                        break
                    time.sleep(1.0)
        run_kubectl(
            [
                "exec",
                pod,
                "-c",
                "flint-tpu",
                "--",
                "bash",
                "-lc",
                "gcc -shared -fPIC -O2 -o /workspace/metadata_hook.so /workspace/metadata_hook.c -ldl",
            ],
            check=False,
            timeout=30,
        )

    for p in pods:
        sync_one(p)


def launch_cluster(pods: list[dict[str, str]], args: argparse.Namespace) -> dict:
    dp_size = args.dp_size if args.dp_size > 0 else (2 if args.tp_size == 1 else 1)
    cores_per_srv = args.tp_size * dp_size
    replicas_per_host = args.replicas_per_host if args.replicas_per_host > 0 else (8 // cores_per_srv)

    def start_host(p: dict[str, str]) -> dict:
        pod = p["pod"]
        remote_cmd = [
            "python3",
            "/workspace/launch_host_replicas.py",
            "--mode",
            "start",
            "--tag",
            args.run_name,
            "--model-path",
            args.model_path,
            "--tp-size",
            str(args.tp_size),
            "--dp-size",
            str(dp_size),
            "--ep-size",
            str(args.ep_size),
            "--num-replicas",
            str(replicas_per_host),
            "--base-port",
            "30000",
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--attention-backend",
            args.attention_backend,
            "--moe-runner-backend",
            args.moe_runner_backend,
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
            "--torch-compile-max-bs",
            str(args.torch_compile_max_bs),
            "--page-size",
            str(args.page_size),
            "--swa-full-tokens-ratio",
            str(args.swa_full_tokens_ratio),
            "--router-mode",
            args.router_mode,
            "--wait-timeout",
            str(args.wait_timeout),
        ]
        if args.disable_torch_compile:
            remote_cmd.append("--disable-torch-compile")
        if args.extra_args.strip():
            remote_cmd.extend(["--extra-args", args.extra_args])

        res = run_kubectl(["exec", pod, "-c", "flint-tpu", "--", *remote_cmd], check=False, timeout=args.wait_timeout + 60)
        lines = [ln.strip() for ln in res.stdout.splitlines() if ln.strip().startswith("{")]
        parsed = json.loads(lines[-1]) if lines else {"status": "raw_error", "stdout": res.stdout[-500:], "stderr": res.stderr[-500:]}
        parsed["pod"] = pod
        parsed["ip"] = p["ip"]
        parsed["returncode"] = res.returncode
        return parsed

    host_List = pods[: args.num_hosts]
    print(
        f"[orchestrator] Launching {args.run_name}: {len(host_List)} hosts x {replicas_per_host} servers "
        f"(TP={args.tp_size}, DP_per_srv={dp_size}, EP={args.ep_size}, model={args.model_path}, kv={args.kv_cache_dtype}, router={args.router_mode})...",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(host_List)) as pool:
        host_results = list(pool.map(start_host, host_List))

    for hr in host_results:
        if hr.get("status") != "ready":
            print(f"[ERROR] Host {hr.get('pod')} failed startup:\n{json.dumps(hr, indent=2)}", flush=True)
            sys.exit(1)

    worker_urls: list[str] = []
    for hr in host_results:
        ip = hr["ip"]
        for r in range(replicas_per_host):
            worker_urls.append(f"http://{ip}:{30000 + r}")

    # Start gateway on Pod 0 (:8000)
    pod0 = pods[0]["pod"]
    gw_res = run_kubectl(
        [
            "exec",
            pod0,
            "-c",
            "flint-tpu",
            "--",
            "python3",
            "/workspace/gateway_flint_v7x.py",
            "--port",
            "8000",
            "--workers",
            ",".join(worker_urls),
            "--daemon",
            "--log-file",
            f"/workspace/logs/gateway_{args.run_name}.log",
        ],
        timeout=30,
    )
    time.sleep(2.0)
    health_res = run_kubectl(
        ["exec", pod0, "-c", "flint-tpu", "--", "curl", "-sf", "http://127.0.0.1:8000/health"],
        timeout=20,
    )
    print(f"[orchestrator] Gateway READY ({len(worker_urls)} workers): {health_res.stdout.strip()}", flush=True)
    return {"hosts": host_results, "worker_urls": worker_urls}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["deploy", "bench", "deploy_and_bench", "stop"], default="deploy_and_bench")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--model-path", default="/dev/shm/flint")
    ap.add_argument("--weights-dtype", default="bf16")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--dp-size", type=int, default=0, help="0 = auto (2 when tp_size==1 on v7x, else 1)")
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--num-hosts", type=int, default=4)
    ap.add_argument("--replicas-per-host", type=int, default=0)
    ap.add_argument("--kv-cache-dtype", default="fp8_e4m3")
    ap.add_argument("--attention-backend", default="pallas_tpu_rpa_v3")
    ap.add_argument("--moe-runner-backend", default="auto")
    ap.add_argument("--router-mode", default="fused", choices=["fused", "stock"])
    ap.add_argument("--context-length", type=int, default=135168)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--chunked-prefill-size", type=int, default=8192)
    ap.add_argument("--max-prefill-tokens", type=int, default=16384)
    ap.add_argument("--max-running-requests", type=int, default=64)
    ap.add_argument("--torch-compile-max-bs", type=int, default=64)
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--swa-full-tokens-ratio", type=float, default=-1.0)
    ap.add_argument("--disable-torch-compile", action="store_true")
    ap.add_argument("--extra-args", default="")
    ap.add_argument("--wait-timeout", type=int, default=1800)
    # Benchmark args
    ap.add_argument("--shapes", default="4096:1024,8192:1024,16384:1024")
    ap.add_argument("--hit-rates", default="0.0,0.8,0.95")
    ap.add_argument("--concurrencies", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--req-mult", type=int, default=2)
    ap.add_argument("--max-reqs", type=int, default=512)
    ap.add_argument("--fixed-num-requests", type=int, default=0)
    ap.add_argument("--max-total-seq-tokens", type=int, default=0)
    ap.add_argument("--total-replicas-override", type=int, default=0)
    ap.add_argument("--verify-quality", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    pods = discover_ready_pods()
    if args.action == "stop":
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(
                pool.map(
                    lambda p: run_kubectl(
                        ["exec", p["pod"], "-c", "flint-tpu", "--", "python3", "/workspace/launch_host_replicas.py", "--mode", "stop"],
                        check=False,
                    ),
                    pods,
                )
            )
        print("[orchestrator] Stopped all replicas across 4 hosts.")
        return

    sync_scripts_to_pods(pods)

    dp_size = args.dp_size if args.dp_size > 0 else (2 if args.tp_size == 1 else 1)
    cores_per_srv = args.tp_size * dp_size
    replicas_per_host = args.replicas_per_host if args.replicas_per_host > 0 else (8 // cores_per_srv)
    total_replicas = args.total_replicas_override if args.total_replicas_override > 0 else (replicas_per_host * dp_size * args.num_hosts)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    deploy_meta_path = RESULTS_DIR / f"{args.run_name}_deploy.json"

    if args.action in ("deploy", "deploy_and_bench"):
        deploy_info = launch_cluster(pods, args)
        deploy_meta_path.write_text(json.dumps(deploy_info, indent=2) + "\n")
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(deploy_meta_path),
                f"gs://ank-usc1-bucket-314837540096/flint-v7x-serving-results/{args.run_name}_deploy.json",
            ],
            env=k_env(),
            check=False,
        )

    if args.action in ("bench", "deploy_and_bench"):
        pod0 = pods[0]["pod"]
        remote_out = f"/workspace/results/{args.run_name}.json"
        gcs_mount_out = f"/models/flint-v7x-serving-results/{args.run_name}.json"
        bench_cmd = [
            "kubectl",
            CTX,
            "exec",
            pod0,
            "-c",
            "flint-tpu",
            "--",
            "python3",
            "/workspace/bench_flint_v7x.py",
            "--run-name",
            args.run_name,
            "--base-url",
            "http://127.0.0.1:8000",
            "--model-name",
            args.model_path,
            "--weights-dtype",
            args.weights_dtype,
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--tp-size",
            str(args.tp_size),
            "--ep-size",
            str(args.ep_size),
            "--num-replicas",
            str(total_replicas),
            "--attention-backend",
            args.attention_backend,
            "--notes",
            args.notes,
            "--shapes",
            args.shapes,
            "--hit-rates",
            args.hit_rates,
            "--concurrencies",
            args.concurrencies,
            "--req-mult",
            str(args.req_mult),
            "--max-reqs",
            str(args.max_reqs),
            "--fixed-num-requests",
            str(args.fixed_num_requests),
            "--max-total-seq-tokens",
            str(args.max_total_seq_tokens),
            "--output",
            remote_out,
            "--gcs-output",
            gcs_mount_out,
        ]
        if args.verify_quality:
            bench_cmd.append("--verify-quality")
        if args.resume:
            bench_cmd.append("--resume")

        subprocess.run(bench_cmd, env=k_env(), check=True)
        local_json = RESULTS_DIR / f"{args.run_name}.json"
        run_kubectl(["cp", f"{pod0}:{remote_out}", str(local_json), "-c", "flint-tpu"], timeout=60)
        subprocess.run(
            [
                "gcloud",
                "storage",
                "cp",
                str(local_json),
                f"gs://ank-usc1-bucket-314837540096/flint-v7x-serving-results/{args.run_name}.json",
            ],
            env=k_env(),
            check=False,
        )
        print(f"[orchestrator] Saved {local_json} and uploaded to GCS!", flush=True)


if __name__ == "__main__":
    main()
