# `RecursiveSI/flint` TorchTPU Serving & Optimization on TPU `v7x-16` (`Ironwood`)

Production serving stack, sub-host VFIO replica orchestrator, parallel `[128, 128]` Block-FP8 checkpoint converter, fused Pallas `router_topk` (`256`-expert) kernel patch, cache-aware prefix-affinity gateway, and apples-to-apples benchmark suite for **`RecursiveSI/flint`** (`40B-A2.5B` sparse MoE coding-agent LLM) on a 4-host **Google Cloud TPU `v7x-16` (`Ironwood`)** slice (`2x2x4`).

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

### TPU `v7x-16` (`Ironwood`, Topology `2x2x4`)
- **Slice Geometry**: `4 hosts × 4 physical v7x chips = 16 chips / 32 TensorCores`.
- **HBM3e Capacity**: `94.74 GiB` usable HBM per TensorCore (`189.48 GiB` per physical `v7x` chip, `757.9 GiB` per host, `3,031.7 GiB` usable / `3,072 GiB` nominal across the 16-chip slice).
- **Host Memory**: `1,512 GiB` DDR5 RAM per host (`1,000 GiB` `/dev/shm` tmpfs holds both BF16 `79.6 GB` and Block-FP8 `40.43 GB` checkpoints simultaneously in RAM at `12%` utilization).
- **Sub-Host VFIO Isolation**: Every `v7x` host exposes **8 independent VFIO groups (`/dev/vfio/0..7`)** — one per TensorCore — enabling clean process-isolated multi-replica serving at `TP=1`, `TP=2`, `TP=4`, and `TP=8` on the same host without Ray overhead.

---

## 2. Parallelism & Quantization Matrix on `v7x-16` (32 TensorCores)

| Config | Weights | KV Cache | TensorCores / Replica | Physical Chips / Replica | HBM / Replica | Replicas / Host | Total Slice Replicas (`DP`) | Inter-Chip ICI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TP=1 × DP=32`** | Block-FP8 `[128,128]` | FP8 (`e4m3`) | 1 TC | 0.5 chip | `94.74 GiB` | 8 | **32** | **Zero** (single TC!) |
| **`TP=2 × DP=16`** | BF16 or Block-FP8 | FP8 / BF16 | 2 TCs | **1 chip** | `189.48 GiB` | 4 | **16** | **Zero** (intra-package die-to-die!) |
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
   - Converts all 15 safetensors shards in parallel (`ProcessPoolExecutor(15)` -> `40.43 GB`, `73,173` tensors in `4.6s` inside `/dev/shm/flint-fp8`), unlocking `TP=1 × DP=32` (`37.66 GiB` weights/TC) and accelerating `gmm_v2` MoE grouped matmuls on Ironwood MXUs.

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
│   ├── mock_metadata.py              # Local GCE tpu-env metadata proxy on 127.0.0.1:80
│   └── metadata_hook.c               # Thread-safe LD_PRELOAD connect() redirector for 169.254.169.254
├── docs/
└── results/
```

---

## 5. Quickstart Reproduction

```bash
export CLOUDSDK_METRICS_ENVIRONMENT=datacloud.antigravity
CTX="--context=gke_diesel-patrol-382622_us-central1_ank-tpu-v5p-usc1"

# 1. Stage scripts to GCS bucket mounted by all pods and deploy the 4-host runner
gcloud storage cp v7x/scripts/* gs://ank-usc1-bucket-314837540096/flint-v7x-scripts/
kubectl $CTX apply -f v7x/k8s/v7x-16-spot-runner.yaml

# 2. Deploy Intra-Chip TP=2 x DP=16 (16 replicas across 4 hosts, zero inter-chip ICI) & run matched sweep
python3 v7x/scripts/orchestrate_v7x_slice.py \
  --action deploy_and_bench \
  --run-name pass1_bf16_fp8kv_tp2_dp16 \
  --model-path /dev/shm/flint \
  --weights-dtype bf16 \
  --tp-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --swa-full-tokens-ratio 0.10 \
  --verify-quality \
  --shapes "4096:1024,8192:1024,16384:1024" \
  --hit-rates "0.0,0.8,0.95" \
  --concurrencies "1,2,4,8,16,32,64,128,256" \
  --resume
```
