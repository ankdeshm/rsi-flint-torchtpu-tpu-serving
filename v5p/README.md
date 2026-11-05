# `RecursiveSI/flint` (`40B-A2.5B` MoE) Serving Benchmarks on TPU `v5p-16` (`2x2x4`)

This directory (`v5p/`) contains serving benchmark scripts, `[128, 128]` Block-FP8 conversion tooling, TPU `v5p` GCE metadata shims, multi-replica prefix-affinity HTTP gateway, Kubernetes manifests, and raw JSONL benchmark archives for **`RecursiveSI/flint`** on a 4-host **TPU `v5p-16`** slice (`4 × ct5p-hightpu-4t = 16 chips`, `95 GiB` HBM2e/chip = `380 GiB`/host = `1,520 GiB`/slice).

## Model & Runtime Summary

- **Model**: `RecursiveSI/flint` (`40B-A2.5B` sparse MoE coding-agent LLM; `39.8B` total / `~2.5B` active params; 49 layers [layers `0–1` dense `intermediate_size=6144`, layers `2–48` MoE with `256` routed experts `top-6` sigmoid routing + ungated shared expert `1024`, `moe_intermediate_size=512`]; GQA `32Q/4KV`, `head_dim=128`, QK-RMSNorm, learned attention sinks `softmax_offset`; `35` sliding-window `513`-token RoPE layers + `14` full-attention NoPE layers; `262,144` max context).
- **Serving Stack**: `sglang` + `sglang_tpu` (`YOUR_REGION-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0`, `torch==2.13.0+cpu`, `torch-tpu==0.2.0`, `jax==0.11.2`, `libtpu==0.0.50`) + `flint_sglang==0.1.4` model/parser plugin (`--grammar-backend flint --reasoning-parser flint --tool-call-parser flint --enable-fp32-lm-head`).
- **Attention & MoE Kernels on TPU v5p**:
  - Attention: `--attention-backend pallas_tpu_rpa_v3` (ragged paged attention v3 with sliding-window `513` + `sinks` `softmax_offset` support and `--kv-cache-dtype fp8_e4m3`, `page_size=64`).
  - MoE: Pallas `gmm_v2` grouped matrix multiplication (`fp8_e4m3fn` blockwise `[128, 128]` in FP8 mode, `bfloat16` in BF16 mode).

## Modifications on TPU `v5p-16`

1. **Static HBM Allocation (`--mem-fraction-static 0.82–0.85`)**:
   - Increases `--mem-fraction-static` from `sglang_tpu`'s `0.40` default (`~38 GiB/chip` on 95 GiB `v5p`) to `0.82–0.85`, allocating `40–44 GiB/chip` of paged FP8 KV cache alongside `BF16 TP=2` (`37.26 GiB/chip` weights) or `FP8 TP=1` (`38.27 GiB/chip` weights).
2. **`[128, 128]` Dynamic Block-FP8 Checkpoint (`quantize_flint_fp8.py`)**:
   - Every 2D linear & routed-expert dimension in Flint (`hidden=2048`, `q=4096`, `kv=512`, `dense=6144`, `shared=1024`, `moe_intermediate_size=512`) is an exact multiple of `128` across `TP=1, 2, 4` (`512 / 4 = 128`), enabling `Fp8Config` (`weight_block_size=[128, 128]`) without padding and shrinking weights from `79.57 GB` (`74.11 GiB`) to `40.44 GB` (`37.66 GiB`) so a full replica fits on 1 `v5p` chip (`FP8 TP=1`, `DP=16` across 16 chips).
3. **Sub-Host & Multi-Host Independent Replica Serving (`v5p_mock_metadata.py` + `flint_v5p_gateway.py`)**:
   - `v5p_mock_metadata.py` + `LD_PRELOAD` `metadata_hook.so` presents each `ct5p-hightpu-4t` node of the `2x2x4` slice as an independent `2,2,1` (`v5p-8`) 4-chip host, allowing `1× TP=4`, `2× TP=2`, or `4× TP=1` SGLang servers per host (`DP=4`, `DP=8`, or `DP=16` across the 16-chip slice) with deterministic `X-Prefix-Trunk` affinity routing for `80%` and `95%` cache hit benchmarks.

## Directory Layout

- [`scripts/quantize_flint_fp8.py`](scripts/quantize_flint_fp8.py): Multi-process CPU converter from BF16 safetensors to `[128, 128]` dynamic Block-FP8 safetensors + parallel GCS uploader.
- [`scripts/v5p_mock_metadata.py`](scripts/v5p_mock_metadata.py): Local GCE metadata shim isolating each 4-chip `v5p` host from `2x2x4` multi-host Megascale auto-discovery.
- [`scripts/flint_v5p_gateway.py`](scripts/flint_v5p_gateway.py): Async FastAPI/HTTPX load balancer supporting least-inflight (`0%` uncached) and deterministic `X-Prefix-Trunk` affinity (`80%` / `95%` cached).
- [`scripts/bench_flint_v5p.py`](scripts/bench_flint_v5p.py): Smoke test, 4-host single-host ablation matrix, and 16-chip slice-wide benchmark driver with per-point JSONL persistence.
- [`k8s/v5p-flint-smoke.yaml`](k8s/v5p-flint-smoke.yaml): Indexed Job (`4 × 4 chips`) for smoke & code-generation verification across `BF16 TP=4/2` and `FP8 TP=1/2/4`.
- [`k8s/v5p-flint-sweep.yaml`](k8s/v5p-flint-sweep.yaml): Indexed Job (`4 × 4 chips`) for single-host ablation sweeps + baseline 16-chip empirical slice runs (`761` rows).
- [`k8s/v5p-flint-slice16-opt.yaml`](k8s/v5p-flint-slice16-opt.yaml): Indexed Job (`4 × 4 chips`, `emptyDir: Memory` `/dev/shm` + `/tmp`) for high-concurrency (`c=768`, `mrr=96`, `swa_ratio=0.10`) & long-context 16-chip slice sweeps (`60` rows).
- [`results/raw/`](results/raw/): Raw JSONL benchmark archives (`831` total rows: `10` smoke rows in `v5p16_smoke_r2_*.jsonl`, `761` sweep rows in `v5p16_sweep_r1_sweep_pod0..3.jsonl`, and `60` optimized slice rows in `v5p16_slice16_opt_r2_slice16_pod0.jsonl`).

---

## TPU `v5p-16` (`2x2x4`, 16 Chips) Serving Benchmark Results

### A. Measured 16-Chip Slice Benchmarks (`slice_16chip_empirical`, `chips_measured=16`)

*Note: Observed prefix-cache hit rates on `95%`-target rows were `93.75%` (`4K`), `94.53%` (`8K`), `94.92%` (`64K`), and `94.97%` (`128K`). At `128K`, `c=32` points are single-wave (`n=c=32`); in the `mrr=96`/`mrr=64` slice run, `64K` and `128K` were measured at `95%` prefix only.*

| Config Tag | Weights / KV Dtype | Parallelism (`16` Chips) | 16-Chip Output tok/s (`0` Failures unless noted) | `p50` `TPOT` (`c=1..8`) | `64K` (`65,536:1024`, `95%` Hit) | `128K` (`131,072:1024`, `95%` Hit) | Peak HBM / Chip |
|---|---|---|---:|---:|---:|---:|---:|
| **`slice16_bf16_tp2_dp8_mrr96_swa010`** | **`BF16` / `fp8_e4m3` (`mrr=96`, `swa=0.10`)** | **`TP=2 × DP=8` (`8` Replicas, `42.29M` KV)** | **`18,940.7` tok/s** *(`4K:1K`, `95%`, `c=768`, `n=1536`, `0` fail)*; **`10,051.8`** *(`4K:1K`, `0%`, `c=256`, `0` fail; `15,016.0` @ `c=768` w/ `1` fail)*; **`10,323.1`** *(`8K:1K`, `95%`, `c=256`, `0` fail; `13,689.2` @ `c=512` w/ `1` fail)* | **`7.48 ms`** *(`c=1`, `0%`)* | **`3,511.0` tok/s** *(`c=64`, `n=128`, `921.5ms` TTFT)* | **`1,867.8` tok/s** *(`c=32`, `n=32`; `8.52ms` TPOT @ `c=8`)* | `82.02 GiB` |
| **`slice16_bf16_tp2_dp8`** | **`BF16` / `fp8_e4m3`** | **`TP=2 × DP=8` (`8` Replicas)** | **`10,216.2` tok/s** *(`8K:1K`, `95%`, `c=256`, `n=512`, `0` fail)* | **`7.57 ms`** | `2,441.7` tok/s *(`c=64`, `n=128`)* | `1,748.8` tok/s *(`c=32`, `n=32`)* | `80.10 GiB` |
| **`slice16_bf16_tp4_dp4`** | **`BF16` / `fp8_e4m3`** | **`TP=4 × DP=4` (`4` Replicas)** | **`10,163.1` tok/s** *(`8K:1K`, `95%`, `c=256`, `n=512`, `0` fail)* | **`7.36 ms`** | `2,658.4` tok/s *(`c=64`, `n=128`)* | `1,692.1` tok/s *(`c=32`, `n=32`)* | `81.97 GiB` |
| **`slice16_fp8_tp1_dp16_mrr64_swa010`** | **`Block-FP8` / `fp8_e4m3` (`mrr=64`, `swa=0.10`)** | **`TP=1 × DP=16` (`16` Single-Chip Replicas, `40.40M` KV)** | **`9,024.6` tok/s** *(`4K:1K`, `95%`, `c=768`, `n=1536`, `0` fail)*; **`7,877.8`** *(`4K:1K`, `0%`, `c=768`, `0` fail)*; **`5,010.7`** *(`8K:1K`, `95%`, `c=256`, `0` fail; `7,292.2` @ `c=512` w/ `2` fail)* | `12.96 ms` *(`c=1`, `0%`)* | **`2,049.2` tok/s** *(`c=64`, `n=128`)* | **`1,369.8` tok/s** *(`c=32`, `n=32`)* | `81.34 GiB` |
| **`slice16_fp8_tp1_dp16`** | **`Block-FP8` / `fp8_e4m3`** | **`TP=1 × DP=16` (`16` Single-Chip Replicas)** | **`5,166.3` tok/s** *(`8K:1K`, `95%`, `c=256`, `n=512`, `0` fail)* | `13.04 ms` | `1,412.6` tok/s *(`c=64`, `n=128`)* | `912.4` tok/s *(`c=32`, `n=32`)* | **`79.42 GiB`** |
| **`slice16_fp8_tp2_dp8`** | `Block-FP8` / `fp8_e4m3` | `TP=2 × DP=8` (`8` Replicas) | `4,810.4` tok/s *(`8K:1K`, `95%`, `c=256`, `n=512`, `0` fail)* | `11.03 ms` | `1,924.4` tok/s *(`95%`; `791.1` @ `0%`)* | `1,312.3` tok/s *(`95%`; `385.8` @ `0%`)* | `81.21 GiB` |
| **`slice16_fp8_tp4_dp4`** | `Block-FP8` / `fp8_e4m3` | `TP=4 × DP=4` (`4` Replicas) | `4,217.1` tok/s *(`8K:1K`, `95%`, `c=256`, `n=512`, `0` fail)* | `9.82 ms` | `1,511.0` tok/s *(`95%`; `547.9` @ `0%`)* | **`953.6` tok/s** *(`95%`; `274.0` @ `0%`)* | `82.11 GiB` |

### B. Single-Host (`4` `v5p` Chips) Ablation Matrix (Measured 1-Host Values Only)

| Config Tag | Precision & Kernel Config | Replicas / Host | Measured 4-Chip Output tok/s | Single-Stream TPOT (`c=1`) | Peak HBM / Chip |
|---|---|---:|---:|---:|---:|
| **`bf16_tp2_dp2_fp8kv_c8k`** | **`BF16` + `FP8` KV + Fused Router (`c=8K`)** | `2 × TP=2` | **`4,429.8` tok/s/host** *(`4K:1K`, `95%`, `c=128`)* | **`7.48 ms`** *(`133.65` tok/s)* | `80.10 GiB` |
| **`bf16_tp4_fp8kv_c8k`** | **`BF16` + `FP8` KV + Fused Router (`c=8K`)** | `1 × TP=4` | **`2,963.3` tok/s/host** *(`4K:1K`, `95%`, `c=128`)* | **`7.23 ms`** *(`138.24` tok/s)* | `81.97 GiB` |
| **`bf16_tp4_bf16kv_c8k`** | `BF16` + `BF16` KV Ablation (`c=8K`) | `1 × TP=4` | `2,513.1` tok/s/host *(`8K:1K`, `80%`, `c=64`)* | `7.31 ms` *(`136.89` tok/s)* | `81.96 GiB` |
| **`fp8_tp1_dp4_fp8kv_c8k`** | **`Block-FP8` `[128,128]` Single-Chip (`TP=1`)** | `4 × TP=1` | **`2,121.3` tok/s/host** *(`4K:1K`, `95%`, `c=128`)* | `12.97 ms` *(`77.08` tok/s)* | **`79.42 GiB`** |
| **`fp8_tp2_dp2_fp8kv_c8k`** | `Block-FP8` `[128,128]` (`TP=2`) | `2 × TP=2` | `1,919.2` tok/s/host *(`4K:1K`, `95%`, `c=128`)* | `10.94 ms` *(`91.41` tok/s)* | `81.21 GiB` |
| **`fp8_tp4_fp8kv_c8k`** | `Block-FP8` `[128,128]` + Fused Router (`c=8K`) | `1 × TP=4` | `1,129.7` tok/s/host *(`4K:1K`, `95%`, `c=64`)* | `9.78 ms` *(`102.27` tok/s)* | `82.11 GiB` |
| **`fp8_tp4_fp8kv_stock_router`** | `Block-FP8` + Unfused PyTorch Router Ablation | `1 × TP=4` | `891.4` tok/s/host *(`4K:1K`, `0%`, `c=64`)* | `10.12 ms` *(`98.80` tok/s)* | `82.16 GiB` |

### C. High-Concurrency 16-Chip Slice Sweep (`slice16_bf16_tp2_dp8_mrr96_swa010`)

Configured with `--max-running-requests 96 --torch-compile-max-bs 96` per `TP=2` replica (`768` decode slots across `8` replicas), `--swa-full-tokens-ratio 0.10` (`5,286,016` full KV tokens/replica = **`42.29M` full KV tokens across the 16-chip slice**), and `256`-expert fused Pallas `router_topk` ([`v5p16_slice16_opt_r2_slice16_pod0.jsonl`](results/raw/v5p16_slice16_opt_r2_slice16_pod0.jsonl)):

| `ISL : OSL` | Target / Observed Cache Hit | Concurrency | Completed / Total (`Failed`) | 16-Chip Output tok/s | 16-Chip Total tok/s | `p50` `TTFT` | `p50` `TPOT` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `4096 : 1024` | `0%` / `0.0%` | `1` | `3 / 3` (`0`) | `124.7` | `623.6` | `730.32 ms` | **`7.48 ms`** |
| `4096 : 1024` | `0%` / `0.0%` | `8` | `16 / 16` (`0`) | `1,001.1` | `5,005.7` | `191.64 ms` | **`7.48 ms`** |
| `4096 : 1024` | `0%` / `0.0%` | `256` | `512 / 512` (`0`) | **`10,051.8`** | **`50,259.1`** | `2,345.03 ms` | `22.32 ms` |
| `4096 : 1024` | `0%` / `0.0%` | `512` | `1022 / 1024` (`2`) | `13,734.0` | `68,670.1` | `4,634.82 ms` | `31.74 ms` |
| `4096 : 1024` | `0%` / `0.0%` | `768` | `1535 / 1536` (`1`) | `15,016.0` | `75,080.0` | `6,560.79 ms` | `43.59 ms` |
| `4096 : 1024` | `80%` / `79.69%` | `512` | `1022 / 1024` (`2`) | `15,484.1` | `77,420.6` | `1,045.95 ms` | `29.17 ms` |
| `4096 : 1024` | `80%` / `79.69%` | `768` | `1536 / 1536` (`0`) | **`18,678.3`** | **`93,391.4`** | `2,179.20 ms` | `38.43 ms` |
| `4096 : 1024` | `95%` / `93.75%` | `128` | `256 / 256` (`0`) | `8,230.9` | `41,154.4` | **`163.77 ms`** | `15.08 ms` |
| `4096 : 1024` | `95%` / `93.75%` | `512` | `1022 / 1024` (`2`) | `15,319.3` | `76,596.5` | `614.83 ms` | `29.42 ms` |
| `4096 : 1024` | `95%` / `93.75%` | `768` | `1536 / 1536` (`0`) | **`18,940.7`** | **`94,703.6`** | `1,513.51 ms` | `38.16 ms` |
| `8192 : 1024` | `0%` / `0.0%` | `512` | `1024 / 1024` (`0`) | **`10,256.4`** | **`92,307.5`** | `9,087.76 ms` | `40.35 ms` |
| `8192 : 1024` | `95%` / `94.53%` | `256` | `512 / 512` (`0`) | **`10,323.1`** | **`92,907.9`** | `368.02 ms` | `22.79 ms` |
| `8192 : 1024` | `95%` / `94.53%` | `512` | `1023 / 1024` (`1`) | `13,689.2` | `123,203.2` | `699.44 ms` | `33.17 ms` |
| `65536 : 1024` | `95%` / `94.92%` | `64` | `128 / 128` (`0`) | **`3,511.0`** | **`228,212.4`** | `921.47 ms` | `16.87 ms` |
| `131072 : 1024` | `95%` / `94.97%` | `32` | `32 / 32` (`0`, `n=c`) | **`1,867.8`** | **`240,943.6`** | `2,269.53 ms` | `14.52 ms` |
