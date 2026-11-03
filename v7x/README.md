# `RecursiveSI/flint` TorchTPU Serving & Optimization on TPU `v7x-16`

Production serving stack, sub-host VFIO replica orchestrator, parallel `[128, 128]` Block-FP8 checkpoint converter, fused Pallas `router_topk` (`256`-expert) kernel patch, cache-aware prefix-affinity gateway, and matched-workload benchmark suite for **`RecursiveSI/flint`** (`40B-A2.5B` sparse MoE coding-agent LLM) on a 4-host **Google Cloud TPU `v7x-16`** slice (`2x2x4`).

---

## 1. Model & Hardware Architecture Fit

### `RecursiveSI/flint` (`40B-A2.5B`)
- **Total / Active Parameters**: `39.8B` total (`79.6 GB` BF16 / `40.43 GB` Block-FP8 `[128, 128]`), `~2.5B` active per token.
- **Layer Stack (`49` layers)**:
  - Layers `0, 1`: Dense SwiGLU MLP (`intermediate_size = 6144`).
  - Layers `2..48` (`47` layers): Sparse MoE with **`256` routed experts** (`moe_intermediate_size = 512`, **sigmoid top-6** routing with `expert_bias` + renormalization) plus **1 ungated shared expert** (`shared_expert_intermediate_size = 1024`).
- **Hybrid Attention (`262,144` max context, GQA `32Q / 4KV`, `head_dim = 128`)**:
  - **35 Sliding-Window RoPE layers** (`sliding_window = 513` tokens, `rope_theta = 10,000`).
  - **14 Full-Attention NoPE layers** (every 4th layer `[3, 7, ..., 47]` plus final layer `48`; zero positional encoding for length generalization).
  - **Learned Per-Head Attention Sinks**: `self_attn.sinks` (`softmax_offset`, `[32]` per layer), natively supported by `TpuPallasRpaV3AttnBackend` (`sinks=self.softmax_offset`) in `sglang-torchtpu`.
  - **Hybrid SWA KV Pool (`FlintConfig.is_hybrid_swa = True`)**: Separate paged KV pools for the 14 full-attention layers (`358.4 KiB/token` BF16, `179.2 KiB/token` FP8 across the model) vs the 35 sliding-window layers (`513`-token window).

### TPU `v7x-16` (`TPU v7x`, Topology `2x2x4`)
- **Slice Geometry**: `4 hosts × 4 physical v7x chips = 16 chips / 32 TensorCores`.
- **HBM3e Capacity**: `94.74 GiB` usable HBM per TensorCore (`189.48 GiB` per physical `v7x` chip, `757.9 GiB` per host, `3,031.7 GiB` usable / `3,072 GiB` nominal across the 16-chip slice).
- **Host Memory**: `1,512 GiB` DDR5 RAM per host (`1,000 GiB` `/dev/shm` tmpfs holds both BF16 `79.6 GB` and Block-FP8 `40.43 GB` checkpoints simultaneously in RAM at `12%` utilization).
- **Sub-Host VFIO Isolation**: Every `v7x` host exposes **8 independent VFIO groups (`/dev/vfio/0..7`)** — one per TensorCore — enabling clean process-isolated multi-replica serving at `TP=1`, `TP=2`, `TP=4`, and `TP=8` on the same host without Ray overhead.

---

## 2. Parallelism & Quantization Matrix on `v7x-16` (32 TensorCores)

| Config | Weights | KV Cache | TensorCores / Replica | Physical Chips / Replica | HBM / Replica | Replicas / Host | Total Slice Replicas (`DP`) | Inter-Chip ICI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TP=1 × DP=32`** | Block-FP8 `[128,128]` | FP8 (`e4m3`) | 1 TC | 0.5 chip | `94.74 GiB` | 8 | **32** | **Zero** (single TC) |
| **`TP=2 × DP=16`** | BF16 or Block-FP8 | FP8 / BF16 | 2 TCs | **1 chip** | `189.48 GiB` | 4 | **16** | **Zero** (intra-package die-to-die) |
| **`TP=4 × DP=8`** | BF16 or Block-FP8 | FP8 / BF16 | 4 TCs | 2 chips | `378.96 GiB` *(matches 1 `v5p` host)* | 2 | **8** | 2-chip intra-host ICI |
| **`TP=8 × DP=4`** | BF16 or Block-FP8 | FP8 / BF16 | 8 TCs | 4 chips (1 host) | `757.92 GiB` | 1 | **4** | 4-chip `2x2` intra-host ICI |
| **`TP=8, EP=8 × DP=4`** | BF16 or Block-FP8 | FP8 | 8 TCs | 4 chips (1 host) | `757.92 GiB` | 1 | **4** (`32` experts/TC) | 4-chip `2x2` intra-host ICI |

---

## 3. Key TPU Serving Optimizations Implemented

1. **Sub-Host Multi-Replica PJRT & Metadata Isolation (`scripts/launch_host_replicas.py`, `scripts/mock_metadata.py`, `scripts/metadata_hook.c`)**:
   - Uses `SGLANG_TPU_PD_COLOCATED=1` + `--base-gpu-id <r * tp_size>` so `sglang`'s `monkey_patch_torch_tpu_env` maps `LOCAL_RANK` directly to physical VFIO group `0..7`.
   - Intercepts `169.254.169.254:80` via `LD_PRELOAD=/workspace/metadata_hook.so` -> `127.0.0.1:80` (`mock_metadata.py`), eliminating `libtpu 0.0.50`'s 30x retry delay on missing GKE `tpu-env` attributes and isolating single-host/sub-host `CHIPS_PER_HOST_BOUNDS=2,2,1` topology from the 4-host slice metadata.
   - Assigns disjoint `MASTER_PORT`, `nccl_port`, `TORCH_TPU_SLICEBUILDER_ADDRESSES`, `TPU_RUNTIME_METRICS_PORTS`, and `SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR` per replica to prevent port contention and concurrent TorchInductor cache races.

2. **4.6-Second Parallel BF16 -> `[128, 128]` Block-FP8 Converter (`scripts/convert_flint_fp8.py`)**:
   - Every 2D weight tensor in `RecursiveSI/flint` outside `lm_head`, `model.embed_tokens`, and `mlp.gate` is an exact multiple of `128` across `TP=1, 2, 4` (`q_proj [4096, 2048]`, `k/v_proj [512, 2048]`, `o_proj [2048, 4096]`, dense `[6144, 2048]`, shared expert `[1024, 2048]`, routed experts `[512, 2048]`).
   - Converts all 15 safetensors shards in parallel (`ProcessPoolExecutor(15)` -> `40.43 GB`, `73,173` tensors in `4.6s` inside `/dev/shm/flint-fp8`), unlocking `TP=1 × DP=32` (`37.66 GiB` weights/TC) and accelerating `gmm_v2` MoE grouped matmuls on TPU v7x MXUs.

3. **Fused VMEM Pallas `router_topk` (`256` Experts) + Argmax-Mask Prefill Routing (`scripts/patch_flint_router_topk.py`)**:
   - Stock `flint_sglang` passes a Python `custom_routing_function` (`torch.topk` + `scores.gather`), forcing XLA across all 47 MoE layers to emit a 256-column sort + general dynamic gather.
   - `patch_flint_router_topk.py --mode enable` adds `256` (`2 × 128`, exact TPU VREG lane multiple) to `VALIDATED_NUM_EXPERTS` in `sglang_tpu/moe/router_topk_tpu.py` and wires `FlintSparseMoeBlock` to `TopK(..., score_func="sigmoid", correction_bias=self.gate.expert_bias, renormalize=True)`, activating both:
     - `sglang_tpu_pallas::router_topk` (fused VMEM sigmoid + bias + top-6 + renormalize) for decode (`batch <= 64`), and
     - `_large_batch_topk` (6 `argmax`-and-mask + one-hot reduce rounds, zero XLA sort/gather) for prefill (`batch > 64`).

4. **Disjoint Trunk-Affinity + Least-Inflight Async Gateway (`scripts/gateway_flint_v7x.py`)**:
   - Routes across all `4 / 8 / 16 / 32` replicas across the 4 hosts in a single zero-buffering `aiohttp` SSE hop.
   - Partitions replicas into disjoint trunk-affinity pools (`X-Prewarm: 1` parallel fan-out + intra-pool least-inflight selection) so `80%` and `95%` prefix-cache benchmarks achieve **100% replica utilization AND zero redundant 128K trunk duplication/eviction**, while `0%` prefix-cache benchmarks use pure slice-wide least-inflight load balancing.

---

## 4. Directory Layout

```text
v7x/
├── README.md
├── k8s/
│   └── v7x-16-spot-runner.yaml       # Self-bootstrapping 4-host Indexed Job + headless Service
├── scripts/
│   ├── orchestrate_v7x_slice.py      # Master 4-host dynamic discovery, deploy & benchmark runner
│   ├── launch_host_replicas.py       # Per-host sub-slice (TP=1/2/4/8) isolated SGLang launcher
│   ├── gateway_flint_v7x.py          # Cache-aware trunk-affinity + least-inflight async HTTP gateway
│   ├── bench_flint_v7x.py            # Matched v7x-16 vs v5p-16 benchmark & 32K needle quality suite
│   ├── convert_flint_fp8.py          # 4.6s parallel BF16 -> [128, 128] Block-FP8 converter
│   ├── patch_flint_router_topk.py    # Idempotent fused VMEM Pallas router_topk (256 experts) patcher
│   ├── patch_flint_sglang_tpu.py     # Runtime patches for SGLang TorchTPU Flint integration
│   ├── launch_pod0_ablations.py      # Pod 0 background ablation runner
│   ├── launch_pod2_ablations.py      # Pod 2 background ablation runner
│   ├── mock_metadata.py              # Local GCE tpu-env metadata proxy on 127.0.0.1:80
│   └── metadata_hook.c               # Thread-safe LD_PRELOAD connect() redirector for 169.254.169.254
├── docs/
│   └── FLINT_V7X_SERVING_RESULTS.md  # Comprehensive empirical serving & optimization report
└── results/                          # Complete empirical benchmark JSON runs (16 files, 532+ points)
    ├── fullslice_bf16_tp2_dp16.json  # Full 16-replica 4-host BF16 slice sweep (33,421.5 tok/s)
    ├── fullslice_fp8_tp2_dp16_swa010.json # Full 16-replica Block-FP8 swa=0.10 slice sweep (8K/64K/128K)
    ├── smoke_fp8_dpattn_1chip.json   # Block-FP8 attn_tp=1 x DP=2 benchmark (75 points)
    ├── smoke_bf16_tp8_1rep.json      # Inter-host TP=8 1-replica benchmark (96 points)
    ├── smoke_bf16_tp4_1rep.json      # TP=4 1-replica benchmark (45 points)
    ├── smoke_bf16_tp2_1rep.json      # TP=2 1-chip baseline (12 points)
    ├── singlechip_bf16_tp2_longctx.json # 64K/128K single-chip long-context sweep (33 points)
    ├── ablation_fp8_tp2_1chip.json   # Full FP8 KV baseline (51 points)
    ├── ablation_bf16kv_tp2_1chip.json# BF16 KV cache ablation (48 points)
    ├── ablation_stock_router_tp2_1chip.json # Stock PyTorch router ablation (26 points)
    └── ablation_swa010_tp2_1chip.json# SWA ratio 0.10 ablation (45 points)
```

---

## 5. Benchmark Report

See the full report in [`docs/FLINT_V7X_SERVING_RESULTS.md`](docs/FLINT_V7X_SERVING_RESULTS.md) (also available on GCS at `gs://YOUR_GCS_BUCKET/flint-v7x-serving-results/FLINT_V7X_SERVING_RESULTS.md`).

### Comparison: TPU v7x-16 vs TPU v5p-16 (Matched 16-Chip Slices)

| Metric / Scenario | TPU v5p-16 (`TP=4 × DP=4`) | TPU v7x-16 (`TP=2 × DP=16`) | Delta / Speedup | Primary Driver |
| :--- | :---: | :---: | :---: | :---: |
| **Throughput (`4K:1024`, c=512, 95% cache)** | 10,750.8 tok/s | **33,421.5 tok/s** | **3.11×** | 4× Replicas (DP=16 vs DP=4) + zero inter-chip ICI |
| **Throughput (`4K:1024`, c=512, 0% cache)** | 8,912.4 tok/s | **29,833.1 tok/s** | **3.35×** | HBM3e bandwidth + MXU compute |
| **Throughput (`16K:1024`, c=128, 95% cache)** | 7,120.5 tok/s | **26,632.1 tok/s** | **3.74×** | Prefix caching + DP=16 replica concurrency |
| **Throughput (`16K:1024`, c=128, 0% cache)** | 5,430.1 tok/s | **16,860.3 tok/s** | **3.10×** | Intra-package die-to-die TP=2 |
| **Throughput (`64K:1024`, c=32, 95% cache)** | 2,140.2 tok/s | **6,643.2 tok/s** | **3.10×** | SWA hybrid KV pool + 3.03 TB slice HBM |
| **TTFT (`64K:1024`, c=1, 95% cache)** | 2,840 ms | **985 ms** | **2.88× faster** | Prefix lookup + Pallas attention |
| **Throughput (`128K:1024`, c=16, 95% cache)** | 1,480.0 tok/s | **4,864.2 tok/s** | **3.29×** | 3.03 TB aggregate HBM3e capacity |
| **TTFT (`128K:1024`, c=1, 95% cache)** | 6,420 ms | **2,268 ms** | **2.83× faster** | Trunk cache hits bypass 128K prefill compute |
| **TTFT (`128K:1024`, c=1, 0% cache)** | 48,200 ms | **20,172 ms** | **2.39× faster** | MXU prefill matmul throughput |
| **Cold Boot / JIT Compilation Time** | 1,256s – 1,572s (~26m) | **88s – 120s** (~1.8m) | **11.4× faster** | OpenXLA / PJRT compiler pipeline |
| **Total Slice Usable HBM** | 1,515.8 GiB (94.7 GiB/chip) | **3,031.7 GiB** (189.5 GiB/chip) | **2.00×** | Dual-die packaging with HBM3e per physical chip |
| **Total Concurrently Served Replicas** | 4 replicas | **16 replicas** | **4.00×** | Intra-package TP=2 enables 1 replica per chip |

---

## 6. Sizing Recommendations

| Workload Profile | Target SLA | Recommended Topology | Weights / KV Dtype | SWA Full Ratio | Max Replicas (`v7x-16`) | Concurrency Capacity | Key Rationale |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Interactive Coding Agent** (`4K`–`16K`, c=128–512) | TTFT < 200ms, TPOT < 25ms | `TP=2 × DP=16` | BF16 / FP8 KV | 0.10 | 16 replicas | up to 512 concurrent streams | Reaches **33,421 tok/s** with zero inter-chip ICI latency. |
| **Long Context** (`64K`–`128K`, c=16–64) | TTFT < 3.0s (cached), TPOT < 30ms | `TP=2 × DP=16` | Block-FP8 / FP8 KV | 0.10 | 16 replicas | up to 64 concurrent streams | 189.5 GB/chip holds weights + 7.36M tokens/replica. |
| **High-Throughput Batch Processing** (`4K`–`16K`, c=1024+) | High Tokens / Second | `TP=1 × DP=32` | Block-FP8 / FP8 KV | 0.10 | 32 replicas | up to 1024 concurrent streams | Disables TP communication; 1 replica per TC. |
| **Low-Latency Long Prefill** (`128K`–`256K`, c=1–8) | TTFT < 10.0s (uncached) | `TP=8 × DP=4` | BF16 / FP8 KV | 0.10 | 4 replicas | up to 32 concurrent streams | TP=8 shards prefill across 8 TCs (1 host) via ICI. |

---

## 7. Quickstart Reproduction

```bash
CTX="--context=gke_YOUR_GCP_PROJECT_ID_us-central1_tpu-v7x-cluster"

# 1. Stage scripts to GCS bucket mounted by all pods and deploy the 4-host runner
gcloud storage cp v7x/scripts/* gs://YOUR_GCS_BUCKET/flint-v7x-scripts/
kubectl $CTX apply -f v7x/k8s/v7x-16-spot-runner.yaml

# 2. Deploy Intra-Chip TP=2 x DP=16 (16 replicas across 4 hosts, zero inter-chip ICI) & run matched sweep
python3 v7x/scripts/orchestrate_v7x_slice.py \
  --action deploy_and_bench \
  --run-name fullslice_bf16_tp2_dp16 \
  --model-path /dev/shm/flint \
  --weights-dtype bf16 \
  --tp-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --swa-full-tokens-ratio 0.10 \
  --verify-quality \
  --shapes "4096:1024,8192:1024,16384:1024,65536:1024,131072:1024" \
  --hit-rates "0.0,0.8,0.95" \
  --concurrencies "1,16,64,128,256,512" \
  --resume
```
