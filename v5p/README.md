# RecursiveSI/flint (`40B-A2.5B` MoE) Serving Benchmarks on TPU `v5p-16` (`2x2x4`)

This directory (`v5p/`) contains all serving benchmark scripts, `[128, 128]` Block-FP8 conversion tooling, dynamic TPU `v5p` GCE metadata shims, multi-replica prefix-affinity HTTP gateway, Kubernetes manifests, and raw + summarized results for **`RecursiveSI/flint`** on a 4-host **TPU `v5p-16`** slice (`4 × ct5p-hightpu-4t = 16 chips`, `95 GiB` HBM2e/chip = `380 GiB`/host = `1,520 GiB`/slice).

## Model & Runtime Summary

- **Model**: `RecursiveSI/flint` (`40B-A2.5B` sparse MoE coding-agent LLM; `39.8B` total / `~2.5B` active params; 49 layers [layers `0–1` dense `intermediate_size=6144`, layers `2–48` MoE with `256` routed experts `top-6` sigmoid routing + ungated shared expert `1024`, `moe_intermediate_size=512`]; GQA `32Q/4KV`, `head_dim=128`, QK-RMSNorm, learned attention sinks `softmax_offset`; `35` sliding-window `513`-token RoPE layers + `14` full-attention NoPE layers; `262,144` max context).
- **Serving Stack**: `sglang` + `sglang_tpu` (`us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0`, `torch==2.13.0+cpu`, `torch-tpu==0.2.0`, `jax==0.11.2`, `libtpu==0.0.50`) + `flint_sglang==0.1.4` external model/parser plugin (`--grammar-backend flint --reasoning-parser flint --tool-call-parser flint --enable-fp32-lm-head`).
- **Attention & MoE Kernels on TPU v5p**:
  - Attention: `--attention-backend pallas_tpu_rpa_v3` (ragged paged attention v3 with native sliding-window `513` + `sinks` `softmax_offset` support and `--kv-cache-dtype fp8_e4m3`, `page_size=64`).
  - MoE: Pallas `gmm_v2` grouped matrix multiplication (`fp8_e4m3fn` blockwise `[128, 128]` in FP8 mode, `bfloat16` in BF16 mode).

## Key Engineering Highlights on TPU `v5p-16`

1. **Overriding `sglang_tpu`'s `mem_fraction_static=0.40` Trap**:
   - `sglang_tpu/tpu_platform.py` defaults `--mem-fraction-static` to `0.40` (`~38 GiB/chip` on 95 GiB `v5p`), which starves KV cache or OOMs `BF16 TP=2` (`37.05 GiB/chip` weights) and `FP8 TP=1` (`37.66 GiB/chip` weights). Explicitly configuring `--mem-fraction-static 0.82`–`0.85` unlocks `40–62 GiB/chip` of paged FP8 KV cache.
2. **Zero-Padding `[128, 128]` Dynamic Block-FP8 Checkpoint (`quantize_flint_fp8.py`)**:
   - Every 2D linear & routed-expert dimension in Flint (`hidden=2048`, `q=4096`, `kv=512`, `dense=6144`, `shared=1024`, `moe_inter=512`) is an exact multiple of `128` across `TP=1, 2, 4` (`512 / 4 = 128`), enabling native `Fp8Config` (`weight_block_size=[128, 128]`) on TPU without padding waste and shrinking weights from `79.57 GB` (`74.11 GiB`) to `40.44 GB` (`37.66 GiB`) so a full replica fits on **1 v5p chip (`FP8 TP=1`, `DP=16` across 16 chips)**!
3. **Sub-Host & Multi-Host Independent Replica Serving (`v5p_mock_metadata.py` + `flint_v5p_gateway.py`)**:
   - `v5p_mock_metadata.py` + `LD_PRELOAD` `metadata_hook.so` presents each `ct5p-hightpu-4t` node of the `2x2x4` slice as an independent `2,2,1` (`v5p-8`) 4-chip host, allowing `1× TP=4`, `2× TP=2`, or `4× TP=1` SGLang servers per host (`DP=4`, `DP=8`, or `DP=16` across the 16-chip slice) with deterministic prefix-trunk affinity routing for `80%` and `95%` multi-turn coding-agent cache hit regimes.

## Directory Layout

- [`scripts/quantize_flint_fp8.py`](scripts/quantize_flint_fp8.py): Multi-process CPU converter from BF16 safetensors to `[128, 128]` dynamic Block-FP8 safetensors + parallel GCS uploader.
- [`scripts/v5p_mock_metadata.py`](scripts/v5p_mock_metadata.py): Local GCE metadata shim isolating each 4-chip `v5p` host from `2x2x4` multi-host Megascale auto-discovery.
- [`scripts/flint_v5p_gateway.py`](scripts/flint_v5p_gateway.py): Async FastAPI/HTTPX load balancer supporting least-inflight (`0%` uncached) and deterministic `X-Prefix-Trunk` affinity (`80%` / `95%` cached).
- [`scripts/bench_flint_v5p.py`](scripts/bench_flint_v5p.py): End-to-end Step 1 smoke test & Step 2 4-host parallel matrix + 16-chip slice-wide benchmark driver with per-point GCS JSONL persistence.
- [`k8s/v5p-flint-smoke.yaml`](k8s/v5p-flint-smoke.yaml): Indexed Job (`4 × 4 chips`) for Step 1 smoke & coherent code-generation verification across `BF16 TP=4/2` and `FP8 TP=1/2/4`.
- [`k8s/v5p-flint-sweep.yaml`](k8s/v5p-flint-sweep.yaml): Indexed Job (`4 × 4 chips`) for Step 2 full matrix (`4K/8K/16K/64K/128K` × `0%/80%/95%` cache hit × concurrency sweep + scheduler/KV/kernel ablations + 16-chip empirical slice runs).
- `results/` & `docs/`: Raw JSONL/CSV benchmark outputs and full analysis report (`FLINT_SERVING_RESULTS.md`).
