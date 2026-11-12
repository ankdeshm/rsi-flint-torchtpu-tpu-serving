# `RecursiveSI/flint` TorchTPU Serving & Optimization on TPU `v7x-16`

Serving stack, sub-host VFIO replica orchestrator, parallel `[128, 128]` Block-FP8 checkpoint converter, fused Pallas `router_topk` (`256`-expert) kernel patch, cache-aware prefix-affinity gateway, and matched-workload benchmark suite for **`RecursiveSI/flint`** (`40B-A2.5B` sparse MoE coding-agent LLM) on a 4-host **Google Cloud TPU `v7x-16`** slice (`2x2x4`).

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
  - **Learned Per-Head Attention Sinks**: `self_attn.sinks` (`softmax_offset`, `[32]` per layer), supported by `TpuPallasRpaV3AttnBackend` (`sinks=self.softmax_offset`) in `sglang-torchtpu`.
  - **Hybrid SWA KV Pool (`FlintConfig.is_hybrid_swa = True`)**: Separate paged KV pools for the 14 full-attention layers (`358.4 KiB/token` BF16, `179.2 KiB/token` FP8 across the model) vs the 35 sliding-window layers (`513`-token window).

### TPU `v7x-16` (`TPU v7x`, Topology `2x2x4`)
- **Slice Geometry**: `4 hosts × 4 physical v7x chips = 16 chips / 32 TensorCores`.
- **HBM3e Capacity**: `94.74 GiB` usable HBM per TensorCore (`189.48 GiB` per physical `v7x` chip, `757.9 GiB` per host, `3,031.7 GiB` usable / `3,072 GiB` nominal across the 16-chip slice).
- **Host Memory**: `1,512 GiB` DDR5 RAM per host (`1,000 GiB` `/dev/shm` tmpfs holds both BF16 `79.6 GB` and Block-FP8 `40.43 GB` checkpoints simultaneously in RAM at `12%` utilization).
- **Sub-Host VFIO Isolation**: Every `v7x` host exposes **8 independent VFIO groups (`/dev/vfio/0..7`)** — one per TensorCore — enabling process-isolated multi-replica serving at `TP=2`, `TP=4`, and `TP=8` on the same host without Ray.

---

## 2. Supported Parallelism & Quantization Matrix on `v7x-16` (32 TensorCores)

| Config | Weights | KV Cache | TensorCores / Replica | Physical Chips / Replica | HBM / Replica | Replicas / Host | Total Slice Replicas (`DP`) | Inter-Chip ICI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`TP=2 × DP-Attn=32`** | Block-FP8 `[128,128]` | FP8 (`e4m3`) | 2 TCs (`attn_tp=1 × dp=2`) | **1 chip** | `189.48 GiB` | 4 (`8` attn DP) | **16** (`32` attn DP) | **Zero** (intra-package die-to-die) |
| **`TP=2 × DP=16`** | BF16 or Block-FP8 | FP8 / BF16 | 2 TCs | **1 chip** | `189.48 GiB` | 4 | **16** | **Zero** (intra-package die-to-die) |
| **`TP=4 × DP=8`** | BF16 or Block-FP8 | FP8 / BF16 | 4 TCs | 2 chips | `378.96 GiB` | 2 | **8** | 2-chip intra-host ICI |
| **`TP=8 × DP=4`** | BF16 | FP8 / BF16 | 8 TCs | 4 chips (1 host) | `757.92 GiB` | 1 | **4** | 4-chip `2x2` intra-host ICI |
| **`TP=8, EP=8 × DP=4`** | Block-FP8 `[128,128]` | FP8 | 8 TCs | 4 chips (1 host) | `757.92 GiB` | 1 | **4** (`32` experts/TC) | 4-chip `2x2` intra-host ICI |

*(Note: Single-TensorCore `TP=1` sub-chip meshes in `sglang-torchtpu` are unsupported by `torch_tpu` on dual-core `v7x`, and `Block-FP8 TP=8` without `EP=8` violates `block_n=128` because `512 / 8 = 64 < 128`; for `TP=1 × DP=32 × EP=32` across `v7x-16`, see [`../vllm/README.md`](../vllm/README.md).)*

---

## 3. Modifications Implemented

1. **Sub-Host Multi-Replica PJRT & Metadata Isolation (`scripts/launch_host_replicas.py`, `scripts/mock_metadata.py`, `scripts/metadata_hook.c`)**:
   - Uses `SGLANG_TPU_PD_COLOCATED=1` + `--base-gpu-id <r * tp_size>` so `sglang`'s `monkey_patch_torch_tpu_env` maps `LOCAL_RANK` directly to physical VFIO group `0..7`.
   - Redirects `169.254.169.254:80` via `LD_PRELOAD=/workspace/metadata_hook.so` to `127.0.0.1:80` (`mock_metadata.py`), serving local `CHIPS_PER_HOST_BOUNDS=2,2,1` topology metadata without multi-host slice discovery delays.
   - Assigns disjoint `MASTER_PORT`, `--nccl-port`, `--port <30000 + r*20>` (avoiding `PortArgs.init_new` `+233..239` ZMQ TCP port collisions under `--enable-dp-attention`), `TORCH_TPU_SLICEBUILDER_ADDRESSES`, `TPU_RUNTIME_METRICS_PORTS`, and `SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR` per replica.
2. **Parallel BF16 -> `[128, 128]` Block-FP8 Converter (`scripts/convert_flint_fp8.py`)**:
   - Every 2D weight tensor in `RecursiveSI/flint` outside `lm_head`, `model.embed_tokens`, and `mlp.gate` is an exact multiple of `128` across `TP=1, 2, 4` (`q_proj [4096, 2048]`, `k/v_proj [512, 2048]`, `o_proj [2048, 4096]`, dense `[6144, 2048]`, shared expert `[1024, 2048]`, routed experts `[512, 2048]`).
   - Converts all 15 safetensors shards in parallel (`ProcessPoolExecutor(15)` -> `40.43 GB`, `73,173` tensors in `4.6s` inside `/dev/shm/flint-fp8`), enabling intra-chip `TP=2 + DP-Attention=32` (**`37,321.6` tok/s** `0`-fail @ `4K, 95%, c=768`; **`34,317.3` tok/s** `0`-fail @ `4K, 0%, c=768`) and `117.8M`-token slice KV pools on `TP=2 × DP=16`.
3. **Fused VMEM Pallas `router_topk` (`256` Experts) + Argmax-Mask Prefill Routing (`scripts/patch_flint_router_topk.py`)**:
   - Adds `256` (`2 × 128`) to `VALIDATED_NUM_EXPERTS` in `sglang_tpu/moe/router_topk_tpu.py` and wires `FlintSparseMoeBlock` to `TopK(..., score_func="sigmoid", correction_bias=self.gate.expert_bias, renormalize=True)`, using `sglang_tpu_pallas::router_topk` for decode (`batch <= 64`) and 6-round `argmax`-and-mask routing (`_large_batch_topk`) for prefill (`batch > 64`).
4. **Disjoint Trunk-Affinity + Least-Inflight Async Gateway (`scripts/gateway_flint_v7x.py`)**:
   - Routes across all `4 / 8 / 16` server endpoints across the 4 hosts in a single `aiohttp` SSE hop, using `X-Trunk-Index` client affinity headers (`X-Prewarm: 1` parallel trunk pre-warm + intra-pool least-inflight selection) on `50%`/`80%`/`95%` prefix-cache points and slice-wide least-inflight routing on `0%` uncached points.

---

## 4. Directory Layout

```text
v7x/
├── README.md
├── k8s/
│   └── v7x-16-spot-runner.yaml                      # 4-host Indexed Job + headless Service
├── scripts/
│   ├── orchestrate_v7x_slice.py                     # 4-host dynamic discovery, deploy & benchmark runner
│   ├── launch_host_replicas.py                      # Per-host sub-slice (TP=2/4/8 & DP-Attention) SGLang launcher
│   ├── gateway_flint_v7x.py                         # Trunk-affinity + least-inflight async HTTP gateway
│   ├── bench_flint_v7x.py                           # Matched v7x-16 benchmark & needle quality suite
│   ├── convert_flint_fp8.py                         # Parallel BF16 -> [128, 128] Block-FP8 converter
│   ├── patch_flint_router_topk.py                   # Fused VMEM Pallas router_topk (256 experts) patcher
│   ├── patch_flint_sglang_tpu.py                    # Runtime patches for SGLang TorchTPU Flint integration
│   ├── launch_singlechip_ablations_a.py                     # Single-chip ablation runner (part A)
│   ├── launch_singlechip_ablations_b.py                     # Single-chip ablation runner (part B)
│   ├── mock_metadata.py                             # Local GCE tpu-env metadata proxy on 127.0.0.1:80
│   └── metadata_hook.c                              # Thread-safe LD_PRELOAD connect() redirector for 169.254.169.254
└── results/                                         # Empirical benchmark JSON archives (20 files, 774 benchmark points)
    ├── fullslice_fp8_tp2_dpattn32_swa010.json       # 16-engine (32-rank DP-Attn) Block-FP8 sweep (72 pts)
    ├── fullslice_bf16_tp2_dp16_mrr96_swa010.json    # 16-replica BF16 mrr=96 swa=0.10 sweep (91 pts)
    ├── fullslice_bf16_tp2_dp16.json                 # 16-replica BF16 mrr=64 swa=0.80 sweep (87 pts)
    ├── fullslice_fp8_tp2_dp16_swa010.json           # 16-replica Block-FP8 swa=0.10 sweep (93 pts)
    ├── smoke_fp8_dpattn_1chip.json                  # Block-FP8 attn_tp=1 x DP=2 single-chip sweep (75 pts)
    ├── smoke_bf16_tp8_1rep.json                     # Single-host TP=8 1-replica sweep (96 pts)
    ├── smoke_bf16_tp4_1rep.json                     # TP=4 1-replica sweep (45 pts)
    ├── smoke_bf16_tp2_1rep.json                     # TP=2 1-chip baseline (12 pts)
    ├── singlechip_bf16_tp2_longctx.json             # 64K/128K single-chip long-context sweep (33 pts)
    ├── ablation_fp8_tp2_1chip.json                  # Block-FP8 single-chip ablation (51 pts)
    ├── ablation_bf16kv_tp2_1chip.json               # BF16 KV cache ablation (48 pts)
    ├── ablation_stock_router_tp2_1chip.json         # Unfused PyTorch router ablation (26 pts)
    └── ablation_swa010_tp2_1chip.json               # SWA ratio 0.10 ablation (45 pts)
```

---

## 5. Comparison: TPU `v7x-16` vs TPU `v5p-16` (Measured 16-Chip Slices)

See [`../APPENDIX.md`](../APPENDIX.md) for full concurrency tables and [Methodology Notes](../APPENDIX.md) for warm-up, `X-Trunk-Index` routing, and `n`-per-point conventions.

| Metric / Scenario | TPU `v5p-16` (`TP=2 × DP=8`, `mrr=96`) | TPU `v7x-16` (`TP=2 × DP=16` / `DP-Attn=32`) | Ratio (`v7x / v5p`) | Primary Driver |
| :--- | :---: | :---: | :---: | :--- |
| **Peak `0`-Fail Throughput (`4K:1K`, `95%` cache)** | `18,940.7` tok/s *(`c=768`, `n=1536`)* | **`37,321.6` tok/s** *(`DP-Attn=32`, `c=768`, `n=1536`)*¹ | **1.97×** | 32 DP-Attention ranks + HBM3e bandwidth |
| **Peak `0`-Fail Throughput (`4K:1K`, `0%` cache)** | `10,051.8` tok/s *(`c=256`, `n=512`; `15,016.0` @ `c=768` w/ `1` fail)* | **`34,317.3` tok/s** *(`FP8 DP-Attn=32`, `c=768`, `n=1536`)* / **`33,334.6`** *(`BF16 mrr=96`, `c=768`)*¹ | **3.41×** *(2.29× vs `c=768`)* | HBM3e bandwidth + MXU compute + `DP-Attn=32` |
| **Peak `0`-Fail Throughput (`8K:1K`, `95%` cache)** | `10,323.1` tok/s *(`c=256`, `n=512`; `13,689.2` @ `c=512` w/ `1` fail)* | **`32,269.7` tok/s** *(`BF16 mrr=96`, `c=1024`)* / **`31,432.8`** *(`FP8 TP=2×DP=16`, `c=512`, `n=512`)*¹ | **3.13×** *(2.36× vs `c=512`)* | Intra-chip `TP=2` (`0` inter-chip ICI) + HBM3e |
| **Throughput (`64K:1K`, `95%` cache, `c=64`)** | `3,511.0` tok/s *(`n=128`)* | **`7,225.0` tok/s** *(`FP8`, `n=64`)* / **`6,617.3`** *(`BF16 mrr=96`, `n=64`)* | **2.06×** | SWA hybrid KV pool + 3.03 TiB slice HBM |
| **Throughput (`128K:1K`, `95%` cache)** | `1,867.8` tok/s *(`c=32`, `n=32`)* | **`5,172.5` tok/s** *(`FP8`, `c=64`, `n=64`)* / **`4,823.8`** *(`BF16 mrr=96`, `c=64`, `n=64`)* | **2.77×** | 3.03 TiB aggregate HBM3e (`117.8M` KV) |
| **Cold Boot / JIT Compilation Time** | `1,256s – 1,572s` (~21–26m) | **`88s – 120s`** (~1.5–2.0m) | **11.4× faster** | OpenXLA / PJRT compiler pipeline |
| **Total Slice Usable HBM** | `1,515.8 GiB` (`94.7 GiB`/chip) | **`3,031.7 GiB`** (`189.5 GiB`/chip) | **2.00×** | Dual-die packaging with HBM3e per physical chip |

¹ *Higher-concurrency `v7x-16` points with non-zero HTTP 500s reached `39,520.7` tok/s (`DP-Attn=32`, `4K, 95%, c=1024`, `17/1536` fail), `39,035.7` tok/s (`DP-Attn=32`, `8K, 95%, c=1024`, `4/1536` fail), and `35,004.7` tok/s (`BF16 mrr=96`, `4K, 0%, c=768`, `4/1536` fail).*

---

## 6. Sizing Recommendations

| Workload Profile | Latency Target | Recommended Topology | Weights / KV Dtype | SWA Full Ratio | Max Replicas (`v7x-16`) | Concurrency Capacity | Key Rationale |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Interactive & Uncached Coding Agent** (`4K`–`16K`, `mrr=96`) | TTFT < 350ms, TPOT < 15ms | `TP=2 × DP=16` (`mrr=96`) | BF16 / FP8 KV | 0.10 | 16 replicas | up to 1024 concurrent streams | **`35,581.0` tok/s** (`95%`, `0` fail @ `c=768`), **`33,334.6` tok/s** (`0%`, `0` fail @ `c=768`), **`6.47–6.74 ms`** `4K c=1` TPOT. |
| **High-Concurrency Short/Medium Context** (`4K`–`8K`, `c=512–768`) | Max Slice Tokens / Second | `TP=2 × DP-Attn=32` | Block-FP8 / FP8 KV | 0.10 | 16 engines (`32` attn DP) | up to 768 (`0`-fail) | **`37,321.6` tok/s** (`4K, 95%`, `0` fail @ `c=768`) & **`34,317.3` tok/s** (`4K, 0%`, `0` fail @ `c=768`). |
| **Long Context** (`64K`–`128K`) | TTFT < 2.5s (cached), TPOT < 16ms | `TP=2 × DP=16` | Block-FP8 / FP8 KV | 0.10 | 16 replicas | measured at `c=64` (`n=64`) | `189.5 GiB`/chip holds weights + `7.36M` tokens/replica (`117.8M`/slice); **`7,225.0` / `5,172.5` tok/s** at `64K` / `128K` (`95%`). |
| **Low-Latency Long Prefill** (`64K`–`128K`, `c=1–16`) | Lowest Uncached TTFT | `TP=8 × EP=8 × DP=4` | Block-FP8 / FP8 KV | 0.10 | 4 replicas | up to 32 concurrent streams | Shards prefill across 8 TCs (1 host) via ICI (`1,225.3 ms` uncached `64K` TTFT). |

---

## 7. Quickstart Reproduction

```bash
CTX="--context=gke_YOUR_GCP_PROJECT_ID_YOUR_REGION_YOUR_GKE_CLUSTER"

# 1. Stage scripts to GCS bucket mounted by all pods and deploy the 4-host runner
gcloud storage cp v7x/scripts/* gs://YOUR_GCS_BUCKET/flint-v7x-scripts/
kubectl $CTX apply -f v7x/k8s/v7x-16-spot-runner.yaml

# 2. Deploy Intra-Chip TP=2 x DP=16 (16 replicas across 4 hosts, zero inter-chip ICI) & run matched sweep
python3 v7x/scripts/orchestrate_v7x_slice.py \
  --action deploy_and_bench \
  --run-name fullslice_bf16_tp2_dp16_mrr96_swa010 \
  --model-path /dev/shm/flint \
  --weights-dtype bf16 \
  --tp-size 2 \
  --kv-cache-dtype fp8_e4m3 \
  --swa-full-tokens-ratio 0.10 \
  --verify-quality \
  --shapes "4096:1024,8192:1024,65536:1024,131072:1024" \
  --hit-rates "0.0,0.5,0.95" \
  --concurrencies "1,16,64,256,512,768,1024" \
  --resume
```
