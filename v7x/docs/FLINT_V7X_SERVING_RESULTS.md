# RecursiveSI/flint (40B-A2.5B MoE) Serving Benchmark on Cloud TPU v7x-16

**Platform:** Cloud TPU `v7x-16` (`4` hosts × `4` physical chips = `16` dual-core chips / `32` TensorCores, `2×2×4` 3D Torus)  
**Cluster:** GKE (`gke_YOUR_GCP_PROJECT_ID_YOUR_REGION_YOUR_GKE_CLUSTER`)  
**Model Architecture:** `RecursiveSI/flint` (`39.8B` total parameters, `2.52B` active per token, `hidden_size=2048`, `vocab_size=150,144`, `49`-layer architecture: `2` dense SwiGLU + `47` sparse MoE layers with `256` routed experts [`top-6` sigmoid] + `1` shared expert; `35` sliding-window RoPE layers [`W=513`, `71.4%`] + `14` full-attention NoPE layers [`28.6%`] + per-head learned sinks, `262,144` max position embeddings; benchmarked up to `131,072` context)  
**Software Stack:** PyTorch `2.15` / TorchTPU `0.2.0` / `libtpu` `0.0.50` / `sglang-torchtpu` / Pallas TPU RPA Attention (`pallas_tpu_rpa_v3`) / Fused `256`-Expert Pallas Router (`patch_flint_router_topk.py`)  
**Benchmark Artifacts:** `gs://YOUR_GCS_BUCKET/flint-v7x-serving-results/`  

---

## 1. Summary of Findings

1. **Serving Throughput on TPU `v7x-16` (`0`-Failure Primary Operating Points):**
   - **`Block-FP8 TP=2 + DP-Attention=32` (`fullslice_fp8_tp2_dpattn32_swa010.json`, `32` `attn_tp=1` attention DP ranks, `116.42M` KV tokens across the slice):** Reaches **`37,321.6` output tok/s (`186,607.8` total tok/s, `0` failures, `629.4 ms` p50 `TTFT`)** at `4096:1024`, `95%` target prefix, `c=768, n=1536` (`39,520.7` output tok/s at `c=1024, n=1536` with `17/1536` HTTP 500 failures); **`34,317.3` output tok/s (`171,586.4` total tok/s, `0` failures)** uncached (`0%` prefix, `c=768, n=1536`); and **`30,995.4` output tok/s (`278,958.6` total tok/s, `0` failures)** at `8192:1024`, `95%` target prefix, `c=512, n=1024` (`35,969.7` tok/s with `3/1536` failures at `c=768`; `39,035.7` tok/s with `4/1536` failures at `c=1024`).
   - **`BF16 TP=2 × DP=16, mrr=96, swa=0.10` (`fullslice_bf16_tp2_dp16_mrr96_swa010.json`, `82.97M` KV tokens across the slice):** Reaches **`35,581.0` output tok/s (`0` failures)** at `4096:1024, 95%` target prefix, `c=768, n=1536` (`2`-pass range: `32,517.7` [`1/1536` fail] to `35,581.0` [`0` fail]); **`33,334.6` output tok/s (`0` failures)** at `4096:1024, 0%` uncached, `c=768, n=1536` (`2`-pass range: `33,334.6` [`0` fail] to `35,004.7` [`4/1536` fail]); and **`32,269.7` output tok/s (`0` failures)** at `8192:1024, 95%` target prefix, `c=1024, n=1536`.
   - **Standard Intra-Chip `TP=2 × DP=16` (`attn_tp=2`, `mrr=64`):** Reaches **`33,421.5` output tok/s** (`BF16`, `swa=0.80`) / **`33,200.3` output tok/s** (`Block-FP8`, `swa=0.10`) at `4096:1024, 95%` target prefix, `c=512, n=512` (single wave), and leads long-context serving (`64K–128K`, effective `c=64, n=64`) by sharding attention prefill across both intra-package TensorCores: **`7,119.8–7,225.0` output tok/s** (`95%` target) / **`3,646.1–3,695.0` output tok/s** (`0%`) at `65536:1024` and **`5,113.8–5,172.5` output tok/s** (`95%` target) / **`1,765.5–1,770.4` output tok/s** (`0%`) at `131072:1024` on `Block-FP8`.

2. **Startup & JIT Compilation Time vs TPU `v5p`:**
   - Server startup and ahead-of-time graph compilation on TPU `v7x` completes in **`88.2s – 120.4s`**, compared to **`1,256.7s – 1,600.5s`** (`21–27` minutes) on TPU `v5p-16` (**`~11×` faster cold start**).

3. **Prefix-Cache (`RadixCache`) Impact:**
   - At `128K` context (`131072:1024`, `n=64`), moving from `0%` uncached to `95%` pre-warmed prefix reduces median `TTFT` from **`20,172.8 ms` to `2,268.1 ms`** (**`8.89×` reduction**, excluding `51.4 s` trunk pre-warm) while increasing aggregate output throughput from **`1,745.5–1,746.7` tok/s to `4,838.5–4,864.2` tok/s** (`BF16, swa=0.80`).

4. **Intra-Chip `TP=2` Placement:**
   - Because each physical TPU `v7x` chip contains two TensorCores connected via intra-package die-to-die interconnect, a `TP=2` replica fits within a single physical socket, avoiding inter-chip ICI collectives across **`16` independent serving replicas per `v7x-16` slice**.

5. **KV Capacity Tuning for Hybrid SWA:**
   - Default SGLang allocation (`--swa-full-tokens-ratio 0.80`) yields **`2,050,432–2,163,968` full KV tokens/chip** (`32.8M–34.6M` tokens across 16 `BF16` replicas).
   - Tuning `--swa-full-tokens-ratio 0.10` to match Flint's `35:14` SWA-to-full layer split increases capacity to **`5,185,408` full KV tokens/chip (`83.0M` across slice)** in `BF16` and **`7,362,688` full KV tokens/chip (`117.8M` across slice)** in `Block-FP8`.

6. **Fused Router Kernel:**
   - Replacing unfused PyTorch `sigmoid + topk` routing with the fused `256`-expert Pallas VMEM kernel (`router_topk=256`) increases single-chip generation throughput by **`1.05× – 1.20×`**.

---

## 2. Quality & Correctness Verification

Functional sanity checks were executed before each benchmark run against the live serving gateway:

| Quality Verification Test | Specification | Observed Result | Latency | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Functional Coding (`coding_merge_intervals`)** | `merge_intervals(intervals: list[list[int]])` generation + runtime unit-test execution | Syntactically valid Python passing interval sorting and overlap-merge assertions | `1.48s – 2.68s` | **PASS** |
| **Long-Context Needle Retrieval (`long_context_32k_needle`)** | `51,366–51,368` prompt tokens with embedded passkey `FLINT-TPUV7X-947261` | Exact passkey recovered (`FLINT-TPUV7X-947261`) | `2.18s – 2.24s` | **PASS** |

*Scope note:* These checks verify runtime numerical stability and long-context attention sink/SWA indexing on 1 coding test and 1 passkey needle per run. Multi-sample task evaluations (`GSM8K`, `MMLU-Pro`, `HumanEval`) comparing `BF16`, `Block-FP8`, and `FP8` KV cache were not executed in this campaign (see [Section 8](#8-methodology-notes--known-limitations)).

---

## 3. `v7x-16` Benchmark Results (`BF16 TP=2 × DP=16`, `mrr=64`, `swa_ratio=0.80`)

**Hardware & Archive ([`fullslice_bf16_tp2_dp16.json`](../results/fullslice_bf16_tp2_dp16.json)):** Full 4-host TPU `v7x-16` slice (`16` physical chips / `32` TensorCores, `16` independent `BF16 TP=2` replicas = `DP=16`, `mrr=64`, `swa_full_tokens_ratio=0.80`, `2,163,968` full KV tokens/replica = `34.62M` across slice).  
**Protocol Notes:** `/v1/completions` (`stream=True`, `ignore_eos=True`), `flush_cache` before each point, `64`-token aligned shared prefix trunks pre-warmed outside `wall_sec` and routed via `X-Trunk-Index` headers. On `4K`/`8K`/`16K`, `c=512` points dispatch `n=512` requests (single wave). On `64K`/`128K`, all points with `c >= 64` dispatch `n=64` requests (`effective concurrency = 64`).

### 3.1 Input: 4,096 | Output: 1,024 (`n = max(c, 4)` up to `n = 512` at `c = 512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 144.2 | 9.0 | 130.9 | 133.4 | 6.65 | 7.10 | 0 |
| **0%** | 16 | 2,085.6 | 130.4 | 133.2 | 142.1 | 6.84 | 7.15 | 0 |
| **0%** | 64 | 7,590.6 | 474.4 | 744.1 | 829.4 | 7.02 | 7.95 | 0 |
| **0%** | 128 | 14,598.9 | 912.4 | 490.1 | 871.2 | 7.85 | 8.52 | 0 |
| **0%** | 256 | 22,154.5 | 1,384.7 | 870.8 | 1,120.4 | 9.88 | 10.98 | 0 |
| **0%** | 512 *(n=512)* | **29,833.1** | **1,864.6** | 1,542.3 | 2,104.5 | 14.66 | 15.28 | 0 |
| **80%** | 1 | 148.2 | 9.3 | 69.1 | 70.8 | 6.67 | 6.91 | 0 |
| **80%** | 16 | 2,171.8 | 135.7 | 73.6 | 77.2 | 6.78 | 7.02 | 0 |
| **80%** | 64 | 7,115.4 | 444.7 | 140.9 | 185.3 | 6.95 | 7.26 | 0 |
| **80%** | 128 | 12,468.2 | 779.3 | 125.9 | 240.1 | 7.42 | 7.72 | 0 |
| **80%** | 256 | 18,987.0 | 1,186.7 | 258.6 | 412.3 | 9.12 | 9.58 | 0 |
| **80%** | 512 *(n=512)* | **33,410.8** | **2,088.2** | 1,126.0 | 1,684.2 | 13.59 | 13.62 | 0 |
| **95%** | 1 | 147.2 | 9.2 | 49.2 | 51.4 | 6.67 | 6.98 | 0 |
| **95%** | 16 | 2,228.9 | 139.3 | 50.3 | 54.7 | 6.72 | 6.94 | 0 |
| **95%** | 64 | 8,025.7 | 501.6 | 54.0 | 62.1 | 6.81 | 7.02 | 0 |
| **95%** | 128 | 14,522.0 | 907.6 | 92.2 | 145.8 | 7.21 | 7.47 | 0 |
| **95%** | 256 | 20,982.2 | 1,311.4 | 134.5 | 224.6 | 8.84 | 9.18 | 0 |
| **95%** | 512 *(n=512)* | **33,421.5** | **2,088.8** | **317.3** | 612.4 | 13.60 | 12.57 | 0 |

### 3.2 Input: 8,192 | Output: 1,024 (`n = 512` at `c = 512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 142.6 | 8.9 | 228.6 | 231.2 | 6.83 | 7.22 | 0 |
| **0%** | 16 | 2,168.3 | 135.5 | 228.3 | 245.8 | 6.85 | 7.25 | 0 |
| **0%** | 64 | 7,643.4 | 477.7 | 609.1 | 742.3 | 7.12 | 7.91 | 0 |
| **0%** | 128 | 13,228.4 | 826.8 | 969.0 | 1,340.2 | 8.24 | 9.40 | 0 |
| **0%** | 256 | 18,698.1 | 1,168.6 | 1,716.3 | 2,410.8 | 10.95 | 12.93 | 0 |
| **0%** | 512 *(n=512)* | **23,847.4** | **1,490.5** | 3,215.4 | 4,520.1 | 17.56 | 19.70 | 0 |
| **80%** | 1 | 143.2 | 9.0 | 93.9 | 96.2 | 6.87 | 7.04 | 0 |
| **80%** | 16 | 2,198.3 | 137.4 | 94.6 | 101.4 | 6.81 | 7.07 | 0 |
| **80%** | 64 | 8,190.6 | 511.9 | 107.6 | 142.5 | 6.89 | 7.16 | 0 |
| **80%** | 128 | 13,843.6 | 865.2 | 151.9 | 240.8 | 7.55 | 7.88 | 0 |
| **80%** | 256 | 19,383.0 | 1,211.4 | 306.3 | 512.4 | 9.21 | 9.73 | 0 |
| **80%** | 512 *(n=512)* | **30,451.6** | **1,903.2** | 762.2 | 1,240.6 | 14.98 | 14.10 | 0 |
| **95%** | 1 | 146.6 | 9.2 | 60.9 | 63.4 | 6.72 | 6.96 | 0 |
| **95%** | 16 | 2,173.1 | 135.8 | 62.9 | 67.8 | 6.79 | 7.02 | 0 |
| **95%** | 64 | 8,197.6 | 512.4 | 90.5 | 114.2 | 6.85 | 7.10 | 0 |
| **95%** | 128 | 14,105.6 | 881.6 | 90.6 | 138.4 | 7.34 | 7.61 | 0 |
| **95%** | 256 | 20,007.8 | 1,250.5 | 192.7 | 310.2 | 8.94 | 9.34 | 0 |
| **95%** | 512 *(n=512)* | **29,587.4** | **1,849.2** | **570.0** | 985.4 | 15.31 | 13.70 | 1 |

### 3.3 Input: 16,384 | Output: 1,024 (`n = 512` at `c = 512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 134.3 | 8.4 | 460.8 | 464.2 | 7.04 | 7.51 | 0 |
| **0%** | 16 | 2,036.7 | 127.3 | 459.5 | 478.1 | 6.92 | 7.54 | 0 |
| **0%** | 64 | 6,831.2 | 426.9 | 1,267.4 | 1,620.3 | 7.54 | 8.99 | 0 |
| **0%** | 128 | 10,900.2 | 681.3 | 2,080.8 | 2,890.4 | 9.15 | 11.44 | 0 |
| **0%** | 256 | 14,182.9 | 886.4 | 3,595.1 | 5,120.6 | 12.84 | 16.73 | 0 |
| **0%** | 512 *(n=512)* | **16,860.3** | **1,053.8** | 6,939.3 | 9,840.2 | 22.61 | 26.87 | 1 |
| **80%** | 1 | 146.9 | 9.2 | 152.7 | 156.4 | 6.68 | 7.09 | 0 |
| **80%** | 16 | 2,178.2 | 136.1 | 156.4 | 168.2 | 6.81 | 7.13 | 0 |
| **80%** | 64 | 7,903.6 | 494.0 | 190.5 | 264.1 | 6.95 | 7.30 | 0 |
| **80%** | 128 | 13,024.9 | 814.1 | 399.1 | 582.4 | 7.84 | 8.42 | 0 |
| **80%** | 256 | 18,033.2 | 1,127.1 | 503.9 | 820.6 | 9.72 | 10.49 | 0 |
| **80%** | 512 *(n=512)* | **25,432.0** | **1,589.5** | 1,568.3 | 2,640.1 | 17.66 | 16.43 | 0 |
| **95%** | 1 | 147.9 | 9.2 | 78.2 | 81.3 | 6.62 | 6.98 | 0 |
| **95%** | 16 | 2,146.9 | 134.2 | 78.9 | 84.7 | 6.79 | 7.03 | 0 |
| **95%** | 64 | 8,212.9 | 513.3 | 85.2 | 118.4 | 6.85 | 7.10 | 0 |
| **95%** | 128 | 13,259.6 | 828.7 | 155.9 | 242.0 | 7.61 | 7.94 | 0 |
| **95%** | 256 | 19,978.6 | 1,248.7 | 284.1 | 480.2 | 9.15 | 9.67 | 0 |
| **95%** | 512 *(n=512)* | **26,632.1** | **1,664.5** | **961.2** | 1,620.4 | 16.72 | 15.25 | 0 |

### 3.4 Long Context: 65,536 (64K) | Output: 1,024 *(Note: `n=64` for all `c >= 64` -> effective `c=64`; `80%`/`95%` exclude `15.0–19.1 s` pre-warm)*

| Prefix Cache | Target `c` *(Prompts `n`)* | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 *(n=2)* | 107.3 | 6.7 | 2,486.8 | 2,490.2 | 6.90 | 9.47 | 0 |
| **0%** | 16 *(n=16)* | 1,622.8 | 101.4 | 2,577.6 | 2,680.4 | 6.85 | 9.58 | 0 |
| **0%** | 64 *(n=64)* | **3,554.1** | **222.1** | 6,020.1 | 7,840.2 | 11.74 | 15.48 | 0 |
| **0%** | 128..512 *(n=64, 3 repeats)* | 3,548.1 – 3,571.9 | 221.8 – 223.2 | 6,021.2 – 7,186.8 | 8,118.2 – 9,420.1 | 10.67 – 11.83 | 15.46 – 16.57 | 0 |
| **80%** | 1 *(n=2)* | 132.4 | 8.3 | 755.3 | 759.1 | 6.82 | 7.70 | 0 |
| **80%** | 16 *(n=16)* | 1,959.0 | 122.4 | 833.3 | 890.4 | 6.82 | 7.82 | 0 |
| **80%** | 64 *(n=64)* | **5,578.5** | **348.7** | 2,168.2 | 2,890.1 | 8.99 | 11.25 | 0 |
| **80%** | 128..512 *(n=64, 3 repeats)* | 5,560.7 – 5,562.4 | 347.5 – 347.6 | 2,238.1 – 2,253.4 | 2,980.4 – 2,994.7 | 8.71 – 8.91 | 11.24 – 11.35 | 0 |
| **95%** | 1 *(n=2)* | 142.5 | 8.9 | 239.4 | 242.0 | 6.79 | 7.19 | 0 |
| **95%** | 16 *(n=16)* | 1,820.3 | 113.8 | 303.3 | 366.1 | 6.95 | 7.44 | 0 |
| **95%** | 64 *(n=64)* | **6,590.1** | **411.9** | 979.6 | 1,157.9 | 8.36 | 9.45 | 0 |
| **95%** | 128..512 *(n=64, 3 repeats)* | 6,607.7 – 6,643.2 | 413.0 – 415.2 | 971.7 – 985.8 | 1,137.3 – 1,219.5 | 8.30 – 8.33 | 9.40 – 9.46 | 0 |

### 3.5 Long Context: 131,072 (128K) | Output: 1,024 *(Note: `n=64` for all `c >= 64` -> effective `c=64`; `80%`/`95%` exclude `39.7–51.4 s` pre-warm)*

| Prefix Cache | Target `c` *(Prompts `n`)* | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 *(n=2)* | 73.1 | 4.6 | 6,884.4 | 6,885.8 | 6.97 | 14.02 | 0 |
| **0%** | 16 *(n=16)* | 1,126.4 | 70.4 | 7,103.8 | 7,186.0 | 6.88 | 14.19 | 0 |
| **0%** | 64 *(n=64)* | **1,745.5** | **109.1** | 20,138.7 | 27,404.4 | 16.44 | 37.02 | 0 |
| **0%** | 128..256 *(n=64, 2 repeats)* | 1,745.8 – 1,746.7 | 109.1 – 109.2 | 16,896.2 – 20,172.8 | 27,032.0 – 27,074.1 | 16.45 – 19.59 | 36.99 – 37.02 | 0 |
| **80%** | 1 *(n=2)* | 112.0 | 7.0 | 2,076.7 | 2,076.7 | 6.89 | 9.12 | 1 |
| **80%** | 16 *(n=16)* | 1,677.6 | 104.9 | 2,239.6 | 2,310.4 | 6.91 | 9.33 | 0 |
| **80%** | 64 *(n=64)* | **3,387.9** | **211.8** | 5,786.4 | 8,909.9 | 12.29 | 18.74 | 0 |
| **80%** | 128..256 *(n=64, 2 repeats)* | 3,377.3 – 3,402.9 | 211.1 – 212.7 | 5,795.7 – 6,036.2 | 8,838.2 – 8,869.7 | 11.54 – 12.20 | 18.67 – 18.71 | 0 |
| **95%** | 1 *(n=2)* | 138.0 | 8.6 | 625.6 | 627.0 | 6.64 | 7.42 | 0 |
| **95%** | 16 *(n=16)* | 1,965.2 | 122.8 | 790.7 | 898.8 | 6.85 | 7.80 | 0 |
| **95%** | 64 *(n=64)* | **4,849.2** | **303.1** | 2,274.7 | 3,089.6 | 10.29 | 12.90 | 0 |
| **95%** | 128..256 *(n=64, 2 repeats)* | 4,838.5 – 4,864.2 | 302.4 – 304.0 | 2,268.1 – 2,319.1 | 3,050.0 – 3,126.7 | 10.27 – 10.31 | 12.92 – 12.97 | 0 |

*Note: Under `swa_ratio=0.80` (`2.16M` full KV tokens/replica), `128K` sweep points were stopped at `c=256` (`n=64`); tuning `--swa-full-tokens-ratio 0.10` in Sections `3B` and `3D` expands per-replica KV capacity to `5.19M–7.36M` tokens.*

---

## 3B. `v7x-16` Benchmark Results (`Block-FP8 TP=2 × DP=16`, `mrr=64`, `swa_ratio=0.10`)

**Hardware & Configuration ([`fullslice_fp8_tp2_dp16_swa010.json`](../results/fullslice_fp8_tp2_dp16_swa010.json)):** Full 4-host TPU `v7x-16` slice (`16` physical chips / `32` TensorCores, `16` independent `[128, 128]` `Block-FP8 TP=2` replicas = `DP=16`, `FP8` `e4m3` KV cache, `--swa-full-tokens-ratio 0.10`, **`7,362,688` KV tokens/replica = `117,803,008` (`117.80M`) KV tokens across the slice**).  
**Protocol Notes:** On `4K`/`8K`/`16K`, `c=512` rows send a single wave (`n=512`, `0` points tested above `c=512` in this file). On `64K`/`128K`, all rows with `c >= 64` send `n=64` requests (`effective concurrency = 64`). Where two passes exist at the same `(shape, hit, c)` in the JSON archive, both passes are listed explicitly.

### 3B.1 Input: 4,096 | Output: 1,024 (`Block-FP8 TP=2 × DP=16`, `n=512` at `c=512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 144.2 | 9.0 | 137.5 | 654.2 | 6.66 | 6.99 | 0 |
| **0%** | 16 | 2,100.2 | 131.3 | 139.1 | 713.8 | 7.03 | 7.52 | 0 |
| **0%** | 64 | 7,786.4 | 486.6 | 758.8 | 1,016.2 | 7.37 | 8.23 | 0 |
| **0%** | 128 | 14,228.3 | 889.3 | 512.5 | 826.5 | 8.31 | 8.99 | 0 |
| **0%** | 256 | 21,666.7 | 1,354.2 | 901.6 | 1,607.3 | 10.59 | 11.71 | 0 |
| **0%** | 512 *(n=512)* | **29,923.9** | **1,870.2** | 1,732.3 | 3,087.2 | 14.89 | 16.90 | 0 |
| **80%** | 1 | 152.9 | 9.6 | 69.7 | 73.3 | **6.49** | 6.70 | 0 |
| **80%** | 16 | 2,130.8 | 133.2 | 72.7 | 77.0 | 6.94 | 7.17 | 0 |
| **80%** | 64 | 7,031.8 | 439.5 | 99.9 | 928.4 | 7.54 | 8.20 | 0 |
| **80%** | 128 | 12,004.8 | 750.3 | 129.3 | 797.2 | 9.11 | 9.62 | 0 |
| **80%** | 256 | 17,941.0 | 1,121.3 | 264.5 | 1,124.3 | 12.11 | 12.89 | 0 |
| **80%** | 512 *(n=512)* | **33,091.3** | **2,068.2** | 907.8 | 1,537.7 | 13.88 | 15.03 | 0 |
| **95%** | 1 | 150.0 | 9.4 | **49.1** | 51.2 | 6.59 | 6.79 | 0 |
| **95%** | 16 | 2,259.3 | 141.2 | 50.2 | 55.1 | 6.73 | 6.94 | 0 |
| **95%** | 64 | 8,328.5 | 520.5 | 66.8 | 138.5 | 6.96 | 7.22 | 0 |
| **95%** | 128 | 14,760.8 | 922.5 | 89.3 | 277.9 | 8.20 | 8.50 | 0 |
| **95%** | 256 | 20,809.5 | 1,300.6 | 159.6 | 537.9 | 10.33 | 10.72 | 0 |
| **95%** | 512 *(n=512)* | **33,200.3** | **2,075.0** | **317.3** | 896.4 | 13.90 | 14.79 | 0 |

### 3B.2 Input: 8,192 | Output: 1,024 (`Block-FP8 TP=2 × DP=16`, `n=512` at `c=512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 144.1 | 9.0 | 232.7 | 234.7 | 6.55 | 6.94 | 0 |
| **0%** | 16 | 2,145.2 | 134.1 | 234.6 | 257.1 | 6.79 | 7.19 | 0 |
| **0%** | 64 | 7,829.9 | 489.4 | 543.7 | 848.1 | 7.19 | 7.87 | 0 |
| **0%** | 128 | 12,630.9 | 789.4 | 995.7 | 1,661.0 | 8.93 | 10.08 | 0 |
| **0%** | 256 | 18,121.9 | 1,132.6 | 1,776.9 | 3,271.8 | 12.05 | 14.03 | 0 |
| **0%** | 512 *(n=512)* | **23,530.5** | **1,470.7** | 3,318.0 | 6,287.8 | 17.84 | 21.51 | 0 |
| **80%** | 1 | 150.6 | 9.4 | 94.2 | 98.1 | 6.58 | 6.82 | 0 |
| **80%** | 16 | 2,146.8 | 134.2 | 98.1 | 107.5 | 6.77 | 7.02 | 0 |
| **80%** | 64 | 8,292.8 | 518.3 | 102.7 | 324.7 | 7.15 | 7.47 | 0 |
| **80%** | 128 | 13,912.1 | 869.5 | 156.1 | 483.4 | 8.56 | 8.99 | 0 |
| **80%** | 256 | 20,341.6 | 1,271.3 | 337.6 | 752.1 | 11.26 | 11.88 | 0 |
| **80%** | 512 *(n=512, Pass 1)* | 29,696.9 | 1,856.1 | 798.6 | 1,328.9 | 15.11 | 16.25 | 1 |
| **80%** | 512 *(n=512, Pass 2)* | **27,339.3** | **1,708.7** | 2,249.0 | 3,612.4 | 15.60 | 18.23 | **0** |
| **95%** | 1 | 150.9 | 9.4 | **60.6** | 63.4 | 6.58 | 6.79 | 0 |
| **95%** | 16 | 2,226.9 | 139.2 | 61.8 | 74.3 | 6.76 | 6.98 | 0 |
| **95%** | 64 | 8,420.1 | 526.3 | 82.4 | 178.3 | 6.98 | 7.22 | 0 |
| **95%** | 128 | 13,838.5 | 864.9 | 91.3 | 238.2 | 8.37 | 8.68 | 0 |
| **95%** | 256 | 20,887.9 | 1,305.5 | 195.5 | 507.6 | 10.99 | 11.47 | 0 |
| **95%** | 512 *(n=512)* | **31,432.8** | **1,964.5** | **490.6** | 1,035.8 | 14.82 | 15.71 | 0 |

### 3B.3 Input: 16,384 | Output: 1,024 (`Block-FP8 TP=2 × DP=16`, `n=512` at `c=512`)

| Prefix Cache | Concurrency (`c`) | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 | 136.8 | 8.6 | 471.6 | 475.7 | 6.89 | 7.52 | 0 |
| **0%** | 16 | 2,066.7 | 129.2 | 472.2 | 511.4 | 6.86 | 7.50 | 0 |
| **0%** | 64 | 6,945.8 | 434.1 | 1,111.0 | 1,785.6 | 7.66 | 8.93 | 0 |
| **0%** | 128 | 10,416.9 | 651.1 | 2,131.5 | 3,535.3 | 10.10 | 12.30 | 0 |
| **0%** | 256 | 13,786.9 | 861.7 | 3,776.4 | 7,036.5 | 14.45 | 18.49 | 0 |
| **0%** | 512 *(n=512, Pass 1)* | 16,591.8 | 1,037.0 | 6,754.8 | 13,509.7 | 22.92 | 30.41 | 4 |
| **0%** | 512 *(n=512, Pass 2)* | **15,882.3** | **992.6** | 8,344.1 | 15,210.4 | 23.24 | 32.14 | **0** |
| **80%** | 1 | 141.3 | 8.8 | 157.0 | 164.0 | 6.90 | 7.21 | 0 |
| **80%** | 16 | 2,202.9 | 137.7 | 162.7 | 208.5 | 6.84 | 7.16 | 0 |
| **80%** | 64 | 7,473.0 | 467.1 | 340.4 | 853.9 | 7.58 | 8.19 | 0 |
| **80%** | 128 | 11,329.5 | 708.1 | 398.1 | 966.0 | 9.76 | 10.55 | 0 |
| **80%** | 256 *(Pass 1 / Pass 2)* | 14,624.7 / 15,186.7 | 914.0 / 949.2 | 886.8 / 760.6 | 2,813.1 / 2,410.2 | 14.49 / 13.87 | 15.82 / 15.01 | **0** / 10 |
| **80%** | 512 *(n=512)* | **25,147.1** | **1,571.7** | 1,601.1 | 2,788.3 | 17.97 | 20.09 | 0 |
| **95%** | 1 | **151.1** | **9.4** | **77.5** | 79.4 | **6.55** | 6.78 | 0 |
| **95%** | 16 | 2,197.8 | 137.4 | 82.1 | 101.7 | 6.83 | 7.06 | 0 |
| **95%** | 64 | 7,622.2 | 476.4 | 139.8 | 340.0 | 7.28 | 7.64 | 0 |
| **95%** | 128 | 13,350.5 | 834.4 | 193.9 | 398.0 | 8.99 | 9.38 | 0 |
| **95%** | 256 | 16,585.6 | 1,036.6 | 304.4 | 1,062.7 | 12.31 | 12.94 | 0 |
| **95%** | 512 *(n=512)* | **25,737.5** | **1,608.6** | **889.0** | 1,509.4 | 17.27 | 18.67 | 0 |

### 3B.4 Long Context: 65,536 (64K) | Output: 1,024 (`Block-FP8 TP=2 × DP=16`; `n=64` for `c >= 64`, `80%`/`95%` exclude `15.4–19.6 s` pre-warm)

| Prefix Cache | Target `c` *(Prompts `n`)* | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 *(n=2)* | 110.1 | 6.9 | 2,536.2 | 2,536.9 | **6.61** | 9.30 | 0 |
| **0%** | 16 *(n=16)* | 1,621.8 | 101.4 | 2,627.8 | 2,665.7 | 6.80 | 9.58 | 0 |
| **0%** | 64 *(n=64)* | **3,680.1** | **230.0** | 6,150.7 | 9,874.8 | 11.07 | 17.49 | 0 |
| **0%** | 128..512 *(n=64, 3 repeats)* | 3,646.1 – 3,695.0 | 227.9 – 230.9 | 6,165.9 – 7,371.4 | 9,838.7 – 9,943.9 | 10.00 – 11.14 | 17.43 – 17.48 | 0 |
| **80%** | 1 *(n=2)* | 135.6 | 8.5 | 766.7 | 768.0 | **6.63** | 7.55 | 0 |
| **80%** | 16 *(n=16)* | 1,984.0 | 124.0 | 837.3 | 890.1 | 6.77 | 7.77 | 0 |
| **80%** | 64 *(n=64)* | **5,971.4** | **373.2** | 2,246.5 | 3,083.9 | 8.09 | 10.50 | 0 |
| **80%** | 128..512 *(n=64, 3 repeats)* | 5,882.6 – 6,013.0 | 367.7 – 375.8 | 2,227.8 – 2,305.4 | 3,020.1 – 3,077.4 | 8.02 – 8.21 | 10.43 – 10.48 | 0 |
| **95%** | 1 *(n=2)* | 142.4 | 8.9 | 244.1 | 245.8 | 6.79 | 7.19 | 0 |
| **95%** | 16 *(n=16)* | 1,820.3 | 113.8 | 303.3 | 366.1 | 6.95 | 7.44 | 2 |
| **95%** | 64 *(n=64)* | **7,225.0** | **451.6** | **985.2** | 1,160.3 | 7.63 | 8.74 | 0 |
| **95%** | 128..512 *(n=64, 3 repeats)* | 7,119.8 – 7,172.0 | 445.0 – 448.2 | 995.4 – 1,006.5 | 1,150.4 – 1,254.5 | 7.60 – 7.64 | 8.71 – 8.76 | 0 |

### 3B.5 Long Context: 131,072 (128K) | Output: 1,024 (`Block-FP8 TP=2 × DP=16`; `n=64` for `c >= 64`, `80%`/`95%` exclude `40.4–52.2 s` pre-warm)

| Prefix Cache | Target `c` *(Prompts `n`)* | Slice Output (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 1 *(n=2)* | 72.0 | 4.5 | 6,998.0 | 6,999.7 | 7.07 | 14.23 | 0 |
| **0%** | 16 *(n=16)* | 1,122.6 | 70.2 | 7,153.5 | 7,202.3 | 6.91 | 14.23 | 0 |
| **0%** | 64 *(n=64)* | **1,770.4** | **110.7** | 20,541.5 | 27,757.2 | 15.69 | 36.60 | 0 |
| **0%** | 128..512 *(n=64, 3 repeats)* | 1,765.5 – 1,768.8 | 110.3 – 110.5 | 17,171.5 – 20,544.5 | 27,375.5 – 27,524.7 | 15.73 – 18.94 | 36.54 – 36.61 | 0 |
| **80%** | 1 *(n=2)* | 115.5 | 7.2 | 2,104.2 | 2,105.6 | **6.61** | 8.87 | 0 |
| **80%** | 16 *(n=16)* | 1,670.4 | 104.4 | 2,271.3 | 2,368.2 | 6.92 | 9.41 | 0 |
| **80%** | 64 *(n=64)* | **3,547.8** | **221.7** | 5,722.9 | 8,959.4 | 11.41 | 17.93 | 0 |
| **80%** | 128..512 *(n=64, 3 repeats)* | 3,531.7 – 3,559.9 | 220.7 – 222.5 | 5,866.6 – 6,118.1 | 8,901.4 – 9,100.3 | 10.68 – 11.33 | 17.84 – 17.95 | 0 |
| **95%** | 1 *(n=2)* | 134.7 | 8.4 | **631.8** | 635.4 | 6.81 | 7.60 | 0 |
| **95%** | 16 *(n=16)* | 1,903.7 | 119.0 | 802.9 | 884.0 | 6.90 | 7.89 | 0 |
| **95%** | 64 *(n=64)* | **5,172.5** | **323.3** | **2,301.8** | 3,146.4 | 9.47 | 12.10 | 0 |
| **95%** | 128..512 *(n=64, 3 repeats)* | 5,113.8 – 5,151.1 | 319.6 – 321.9 | 2,315.6 – 2,518.8 | 3,203.1 – 3,398.3 | 9.33 – 9.41 | 12.02 – 12.29 | 0 |

---

## 3C. `v7x-16` Benchmark Results (`Block-FP8 TP=2 + DP-Attention=32`, `swa_ratio=0.10`)

**Hardware & Configuration ([`fullslice_fp8_tp2_dpattn32_swa010.json`](../results/fullslice_fp8_tp2_dpattn32_swa010.json)):** Full 4-host TPU `v7x-16` slice (`16` physical chips / `32` TensorCores, `16` intra-package `[128, 128]` `Block-FP8 TP=2` engines configured with `--dp-size 2 --enable-dp-attention --swa-full-tokens-ratio 0.10`, exposing **`32` independent `attn_tp=1` attention DP ranks** across the slice with **`3,638,272` KV tokens/rank = `116,424,704` (`116.42M`) KV tokens across the slice**). Spacing intra-host HTTP `--port` by `20` (`30000 + r*20`) and `--nccl-port` (`31500 + r*20`) in [`launch_host_replicas.py`](../scripts/launch_host_replicas.py) isolates ZMQ TCP IPC port offsets (`port + 233..239`).  
**Regime Crossover:** At **`4K`–`8K` high concurrency**, `DP-Attention=32` reaches **`37,321.6` tok/s (`0` failures, `c=768, n=1536`)** at `4K, 95%`, **`34,317.3` tok/s (`0` failures, `c=768, n=1536`)** at `4K, 0%`, and **`30,995.4` tok/s (`0` failures, `c=512, n=1024`)** at `8K, 95%` (`39,520.7` tok/s at `4K, 95%, c=1024` with `17/1536` failures; `39,035.7` tok/s at `8K, 95%, c=1024` with `4/1536` failures). At **`64K`–`128K` long context** (`effective c=64, n=64`), standard `TP=2 × DP=16` (`attn_tp=2`, Section `3B.4–3B.5`) outperforms `DP-Attention=32` (**`7,119.8–7,225.0` vs `4,471.8–4,561.6` tok/s** at `64K, 95%`; **`5,113.8–5,172.5` vs `2,400.3–2,411.7` tok/s** at `128K, 95%`) because `attn_tp=2` shards `64K–128K` prefill across both intra-package TensorCores.

### 3C.1 High-Concurrency `4K` & `8K` + Long-Context `64K` & `128K` (`72` Points Summary)

| Shape (`ISL:OSL`) | Prefix Cache | `c` *(Prompts `n`)* | Slice Output (tok/s) | Slice Total (tok/s) | Per-Rank (tok/s) | TTFT p50 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4096:1024** | **0%** | 512 *(n=1024)* | 31,060.4 | 155,302.2 | 970.6 | 1,593.8 | 14.51 | 16.42 | **0** |
| **4096:1024** | **0%** | 768 *(n=1536)* | **34,317.3** | **171,586.4** | **1,072.4** | 2,568.8 | 19.34 | 22.41 | **0** |
| **4096:1024** | **80%** | 1 *(n=4)* | 136.8 | 683.8 | 4.3 | 167.7 | **7.16** | 7.47 | **0** |
| **4096:1024** | **80%** | 768 *(n=1536)* | **36,602.9** | **183,014.8** | **1,143.8** | 761.8 | 19.27 | 20.37 | **0** |
| **4096:1024** | **80%** | 1024 *(n=1536)* | 37,771.0 | 188,855.2 | 1,180.4 | 1,074.7 | 18.42 | 19.98 | 7 |
| **4096:1024** | **95%** | 1 *(n=4)* | 133.0 | 665.0 | 4.2 | **157.3** | 7.40 | 7.73 | **0** |
| **4096:1024** | **95%** | 768 *(n=1536)* | **37,321.6** | **186,607.8** | **1,166.3** | **629.4** | 18.30 | 19.40 | **0** |
| **4096:1024** | **95%** | 1024 *(n=1536)* | 39,520.7 | 197,603.3 | 1,235.0 | 825.4 | 17.55 | 18.81 | 17 |
| **8192:1024** | **0%** | 768 *(n=1536)* | **26,572.9** | **239,156.0** | **830.4** | 5,041.4 | 23.09 | 28.66 | **0** |
| **8192:1024** | **80%** | 768 *(n=1536)* | **33,132.7** | **298,194.3** | **1,035.4** | 1,077.7 | 20.02 | 21.64 | **0** |
| **8192:1024** | **80%** | 1024 *(n=1536)* | 36,500.2 | 328,501.6 | 1,140.6 | 1,668.9 | 19.55 | 22.13 | 4 |
| **8192:1024** | **95%** | 512 *(n=1024)* | **30,995.4** | **278,958.6** | **968.6** | **527.6** | 14.86 | 15.89 | **0** |
| **8192:1024** | **95%** | 768 *(n=1536)* | 35,969.7 | 323,727.4 | 1,124.1 | 762.2 | 18.93 | 20.09 | 3 |
| **8192:1024** | **95%** | 1024 *(n=1536)* | 39,035.7 | 351,321.0 | 1,219.9 | 1,052.9 | 18.76 | 20.25 | 4 |
| **65536:1024** | **0% / 50% / 95%** | 64 *(n=64; 3 repeats c=64..256)* | **3,154.4–3,310.3 / 3,744.1–3,749.8 / 4,471.8–4,561.6** | 205.0K–215.2K / 243.4K–243.7K / 290.7K–296.5K | 98.6–103.5 / 117.0–117.2 / 139.7–142.6 | 7,931–9,995 / 6,233–6,326 / **3,361–3,563** | 9.29–11.60 / 10.52–10.69 / 9.94–10.06 | 19.37 / 16.70 / 13.93 | **0** |
| **131072:1024** | **0% / 50% / 95%** | 64 *(n=64; 3 repeats c=64..256)* | **1,501.2–1,638.6 / 1,771.2–1,778.0 / 2,400.3–2,411.7** | 193.7K–211.4K / 228.5K–229.4K / 309.6K–311.1K | 46.9–51.2 / 55.4–55.6 / 75.0–75.4 | 21,431–28,931 / 18,109–18,212 / **9,052–9,166** | 11.55–17.47 / 17.78–17.82 / 15.60–15.82 | 39.51 / 36.21 / 26.46 | **0** |

*(At `64K` and `128K`, `50%` and `95%` rows exclude `15.5–36.0 s` and `38.4–99.2 s` of trunk pre-warm time, respectively.)*

---

## 3D. `v7x-16` Benchmark Results (`BF16 TP=2 × DP=16`, `mrr=96`, `swa_ratio=0.10` — `91` Points)

**Hardware & Configuration ([`fullslice_bf16_tp2_dp16_mrr96_swa010.json`](../results/fullslice_bf16_tp2_dp16_mrr96_swa010.json)):** Full 4-host TPU `v7x-16` slice (`16` physical chips / `32` TensorCores, `16` intra-package `BF16 TP=2` engines = `DP=16` with `--max-running-requests 96 --torch-compile-max-bs 96 --swa-full-tokens-ratio 0.10`, **`5,185,408` KV tokens/replica = `82,966,528` (`82.97M`) KV tokens across the slice**).  
**Summary:** Raising compiled decode batch capacity to `96`/replica (`1,536` slice decode slots) + `swa_ratio=0.10` reaches **`35,581.0` tok/s (`0` failures)** at `4K:1K, 95%, c=768, n=1536` (Pass 1: `32,517.7` tok/s, `1` failure; Pass 2: `35,581.0` tok/s, `0` failures), **`33,334.6` tok/s (`0` failures)** at `4K:1K, 0%, c=768, n=1536` (Pass 1: `35,004.7` tok/s, `4` failures; Pass 2: `33,334.6` tok/s, `0` failures), **`32,269.7` tok/s (`0` failures)** at `8K:1K, 95%, c=1024, n=1536`, **`6.39–6.74 ms` `c=1` `TPOT`**, and **`6,549.8–6,617.3` tok/s** (`64K, 95%`) / **`4,810.9–4,823.8` tok/s** (`128K, 95%`) across `5` `n=64` repeats (`c=64..1024`) with **`0` failures**.

| Shape (`ISL:OSL`) | Prefix Cache | `c` *(Prompts `n`)* | Slice Output (tok/s) | Slice Total (tok/s) | Per-Replica (tok/s) | TTFT p50 (ms) | TPOT p50 (ms) | E2E p50 (s) | Failures |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4096:1024** | **0%** | 512 *(n=1024)* | 30,785.0 | 153,925.0 | 1,924.1 | 1,625.2 | 14.43 | 16.37 | **0** |
| **4096:1024** | **0%** | 768 *(n=1536, Pass 1)* | 35,004.7 | 175,023.7 | 2,187.8 | 2,359.7 | 19.14 | 21.94 | 4 |
| **4096:1024** | **0%** | 768 *(n=1536, Pass 2)* | **33,334.6** | **166,673.1** | **2,083.4** | 3,037.1 | 19.38 | 22.84 | **0** |
| **4096:1024** | **50%** | 1 *(n=4)* | 153.1 | 765.4 | 9.6 | 89.5 | **6.47** | 6.71 | **0** |
| **4096:1024** | **50%** | 1024 *(n=1536)* | **33,936.3** | **169,681.7** | **2,121.0** | 1,408.7 | 20.62 | 22.42 | **0** |
| **4096:1024** | **95%** | 1 *(n=4)* | 150.2 | 751.0 | 9.4 | **49.0** | 6.62 | 6.82 | **0** |
| **4096:1024** | **95%** | 768 *(n=1536, Pass 1)* | 32,517.7 | 162,588.6 | 2,032.4 | 385.8 | 19.75 | 20.62 | 1 |
| **4096:1024** | **95%** | 768 *(n=1536, Pass 2)* | **35,581.0** | **177,905.2** | **2,223.8** | 824.4 | 18.86 | 20.01 | **0** |
| **8192:1024** | **0%** | 768 *(n=1536)* | **26,514.7** | **238,632.2** | **1,657.2** | 4,763.1 | 23.61 | 28.92 | **0** |
| **8192:1024** | **50%** | 1024 *(n=1536)* | **30,588.2** | **275,293.9** | **1,911.8** | 2,061.2 | 25.29 | 29.14 | **0** |
| **8192:1024** | **95%** | 1024 *(n=1536)* | **32,269.7** | **290,427.1** | **2,016.9** | **693.4** | 23.90 | 25.07 | **0** |
| **65536:1024** | **95%** | 1 *(n=2)* | 151.1 | 9,820.8 | 9.4 | **239.0** | **6.39** | 6.78 | **0** |
| **65536:1024** | **0% / 50% / 95%** | 64 *(n=64; 5 repeats c=64..1024)* | **3,537.8–3,556.9 / 4,485.6–4,498.6 / 6,549.8–6,617.3** | 230.0K–231.2K / 291.6K–292.4K / 425.7K–430.1K | 221.1–222.3 / 280.4–281.2 / 409.4–413.6 | 6,038–7,201 / 3,587–4,277 / **942–1,010** | 10.71–11.85 / 9.83–10.36 / 8.33–8.41 | 18.13 / 14.26 / 9.46 | **0** |
| **131072:1024** | **0% / 50% / 95%** | 64 *(n=64; 5 repeats c=64..1024)* | **1,743.0–1,747.3 / 2,274.7–2,279.7 / 4,810.9–4,823.8** | 224.8K–225.4K / 293.4K–294.1K / 620.6K–622.3K | 108.9–109.2 / 142.2–142.5 / 300.7–301.5 | 16,866–20,232 / 11,327–12,982 / **2,315–2,437** | 16.45–19.59 / 14.05–16.09 / 10.20–10.30 | 36.93 / 27.89 / 12.99 | **0** |

*(At `64K` and `128K`, `50%` and `95%` rows exclude `8.2–19.1 s` and `20.0–51.4 s` of trunk pre-warm time, respectively.)*

---

## 4. Scale-Up Scaling: Single-Chip to Full Slice

Empirical scaling comparison across TPU `v7x` layouts (`ISL=4096`, `OSL=1024`):
1. **Single-Chip (`BF16 TP=2`, 2 TensorCores):** `1` chip, `189.5 GiB` usable HBM.
2. **2-Chip (`BF16 TP=4`, 4 TensorCores):** `2` chips, `379 GiB` usable HBM.
3. **4-Chip Full Host (`BF16 TP=8`, 8 TensorCores):** `4` chips, `758 GiB` usable HBM, Pallas `DISTRIBUTED_TOP_K`.
4. **Full Slice (`16` Chips, `32` TensorCores, `TP=2 × DP=16` & `TP=2 + DP-Attn=32`):** `16` chips, `3.03 TiB` usable HBM, `16` intra-chip engines (`16` or `32` attention DP ranks).

### 4.1 Throughput & Latency Scaling (`ISL=4096`, `OSL=1024`, `0`-Failure Points)

| Layout / Scale | Physical Chips | TensorCores | Concurrency (`c, n`) | Output Throughput (`0%` / `95%` Hit) | 1-Stream `TPOT` (`c=1`) | 1-Stream Speed | `TTFT` p50 @ `c=1` (`0%` / `95%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1-Chip Baseline (`BF16 TP=2`)** | 1 | 2 | `c=64` | **3,058.4 / 3,285.9 tok/s** | 6.84 ms | 146.2 tok/s | 134.4 ms / — |
| **2-Chip Replica (`BF16 TP=4`)** | 2 | 4 | `c=64` | **3,212.8 tok/s** *(0%)* | 6.82 ms | 146.6 tok/s | 133.1 ms |
| **4-Chip Full Host (`BF16 TP=8`)** | 4 | 8 | `c=64` | **3,380.2 tok/s** *(0%)* | 6.88 ms | 145.3 tok/s | 131.8 ms |
| **Full Slice (`BF16 TP=2 × DP=16`, `mrr=96`)** | 16 | 32 | `c=768, n=1536` | **33,334.6 / 35,581.0 tok/s** *(0 fail)* | **6.47–6.74 ms** | **148.4–154.6 tok/s** | 132.3 / 49.0 ms |
| **Full Slice (`Block-FP8 TP=2 + DP-Attn=32`)** | 16 | 32 | `c=768, n=1536` | **34,317.3 / 37,321.6 tok/s** *(0 fail)*\* | 7.16–7.75 ms | 129.0–139.7 tok/s | 226.5 / 157.3 ms |

*\*At `c=1024, n=1536` (`95%` target prefix), `Block-FP8 TP=2 + DP-Attn=32` reaches `39,520.7` tok/s with `17/1536` HTTP 500 failures.*

### 4.2 Key Scale-Up Takeaways
- **Data-Parallel Scaling Across 16 Chips (`10.9×–11.4×` `0`-Failure Scaling vs Single-Chip Peak):** Scaling from 1 chip (`3,058.4` tok/s `0%` / `3,285.9` tok/s `95%`) to 16 chips reaches **`34,317.3` tok/s (`0%` cache, `0` fail)** and **`37,321.6` tok/s (`95%` cache, `0` fail)** without cross-host ICI collectives.
- **Single-Stream Decode Speed:** Across `TP=2`, `TP=4`, and `TP=8`, single-stream `TPOT` remains at **`6.47 ms – 6.88 ms`** (`145–155` tok/s per stream).

---

## 5. Architectural & Optimization Ablations

All ablations were executed on single physical Cloud TPU `v7x` chips (`TP=2`) under matched configurations.

### 5.1 KV Cache Dtype: `BF16` vs `FP8` (`fp8_e4m3`)

*Evaluated on 1 physical `v7x` chip (`TP=2`, `BF16` weights; archive: [`ablation_bf16kv_tp2_1chip.json`](../results/ablation_bf16kv_tp2_1chip.json)).*

| Metric | `BF16` KV Cache | `FP8` (`e4m3`) KV Cache | Delta |
| :--- | :---: | :---: | :---: |
| **Total Full KV Tokens / Chip** | 1,080,256 | 2,050,432 | **+89.8% capacity (1.90×)** |
| **4K Output Tok/s (`c=64, 0%` cache)** | 2,461.5 tok/s | 3,058.4 tok/s | **+24.2% throughput** |
| **4K Output Tok/s (`c=64, 95%` cache)** | 3,034.1 tok/s | 3,285.9 tok/s | **+8.3% throughput** |
| **16K Output Tok/s (`c=16, 0%` cache)** | 842.0 tok/s | 888.7 tok/s | **+5.5% throughput** |
| **64K Output Tok/s (`c=1, 0%` cache)** | 106.5 tok/s | 128.9 tok/s | **+21.0% throughput** |
| **128K Output Tok/s (`c=1, 0%` cache)** | 73.2 tok/s | 86.1 tok/s | **+17.6% throughput** |

---

### 5.2 Model Weights: `BF16` vs `Block-FP8` `[128, 128]`

*Evaluated on 1 physical `v7x` chip (`TP=2`, `FP8` KV cache, `swa_ratio=0.80`; archive: [`ablation_fp8_tp2_1chip.json`](../results/ablation_fp8_tp2_1chip.json)).*

| Metric | `BF16` Weights (`TP=2`) | `Block-FP8` `[128,128]` (`TP=2`) | Delta |
| :--- | :---: | :---: | :---: |
| **Weight HBM Footprint / Chip** | ~75.4 GiB | ~38.1 GiB | **-49.5% weight memory** |
| **Full KV Cache Capacity / Chip (`swa=0.80`)** | 2,050,432 tokens | 3,067,776 tokens | **+49.6% KV capacity** |
| **4K Output Tok/s (`c=64, 0%` cache)** | 3,058.4 tok/s | 3,315.9 tok/s | **+8.4% throughput** |
| **4K TTFT p50 (`c=1, 0%` cache)** | 134.4 ms | 128.9 ms | **-4.1% TTFT** |
| **16K Output Tok/s (`c=64, 95%` cache)** | 2,574.5 tok/s | 2,841.2 tok/s | **+10.4% throughput** |
| **Long-Context Needle Check** | PASS | PASS | Exact passkey match |

---

### 5.3 Router Kernel: Fused `256`-Expert Pallas Kernel vs Unfused PyTorch Router

*Evaluated on 1 physical `v7x` chip (`TP=2`, `BF16` weights, `FP8` KV; archive: [`ablation_stock_router_tp2_1chip.json`](../results/ablation_stock_router_tp2_1chip.json)).*

| Concurrency & Shape | Unfused PyTorch Router (tok/s) | Fused Pallas Router (tok/s) | Speedup Factor |
| :--- | :---: | :---: | :---: |
| **4K:1K, `c=1` (`0%` cache)** | 132.5 tok/s | 139.4 tok/s | **1.05×** |
| **4K:1K, `c=16` (`0%` cache)** | 1,342.8 tok/s | 1,403.6 tok/s | **1.05×** |
| **4K:1K, `c=64` (`0%` cache)** | 2,549.1 tok/s | 3,058.4 tok/s | **1.20×** |
| **4K:1K, `c=64` (`80%` cache)** | 3,130.2 tok/s | 3,315.9 tok/s | **1.06×** |
| **16K:1K, `c=64` (`80%` cache)** | 2,065.4 tok/s | 2,251.6 tok/s | **1.09×** |

---

### 5.4 Hybrid Sliding Window Attention (SWA) Pool Tuning

Flint's `49`-layer transformer architecture comprises:
- **`35` layers (`71.4%`)** using Sliding Window Attention (`window_size = 513` tokens, RoPE).
- **`14` layers (`28.6%`)** using Full Global Attention (NoPE) with learned attention sinks.

Default SGLang allocation (`--swa-full-tokens-ratio 0.80`) sizes the SWA pool near parity with full attention. Setting `FlintConfig.is_hybrid_swa = True` and tuning `--swa-full-tokens-ratio 0.10` (`BF16 TP=2`, 1 chip; archive: [`ablation_swa010_tp2_1chip.json`](../results/ablation_swa010_tp2_1chip.json)):

| Configuration | Full KV Token Pool / Chip | SWA KV Token Pool / Chip | Total KV Pool Slots / Chip |
| :--- | :---: | :---: | :---: |
| **Default (`ratio 0.80`)** | 2,050,432 | 410,086 | 2,460,518 |
| **Tuned (`ratio 0.10`)** | **5,185,408** | 4,666,867 | **9,852,275 (`2.53×` full KV)** |

**Throughput Impact:**
- At `4K:1K` (`c=16, 0%` cache): `1,403.6` tok/s (`ratio 0.80`) vs `1,403.6` tok/s (`ratio 0.10`).
- At `128K:1K` (`c=16, 95%` cache): `497.8` tok/s (`ratio 0.80`) vs `534.8` tok/s (`ratio 0.10`, **`+7.4%`**).

---

## 6. Side-by-Side Comparison: 16-Chip TPU `v7x-16` vs 16-Chip TPU `v5p-16` (`sglang-torchtpu`)

Measured comparison between 4-host `16`-chip slices (`v7x-16` vs `v5p-16`, `sglang-torchtpu`, `mrr=96, swa=0.10`):

| Feature / Metric | Cloud TPU `v5p-16` ([`v5p/`](../../v5p/README.md)) | Cloud TPU `v7x-16` ([`v7x/`](../../README.md)) | `v7x-16` vs `v5p-16` |
| :--- | :---: | :---: | :--- |
| **Physical Chips / TensorCores** | 16 single-core chips / 16 TCs | 16 dual-core chips / 32 TCs | **2.00× TensorCores** |
| **Usable HBM per Chip / Slice** | 94.7 GiB / 1,515.8 GiB | 189.5 GiB / 3,031.7 GiB | **2.00× HBM** |
| **Primary `BF16` Slice Layout** | `BF16 TP=2 × DP=8` (`8` replicas) | `BF16 TP=2 × DP=16` (`16` replicas) | **2.00× `BF16` replicas (zero inter-chip ICI on `v7x`)** |
| **Cold Startup & JIT Compilation** | 1,256.7s – 1,600.5s (`~22 min`) | 88.2s – 120.4s (`~1.8 min`) | **~11× faster cold start** |
| **Single-Stream `c=1` `TPOT` p50 (`BF16 TP=2`)** | 7.48 ms (`133.7` tok/s) | 6.47–6.74 ms (`148.4–154.6` tok/s) | **10%–13% lower `TPOT`** |
| **16-Chip `4K:1K` Output (`0%` Hit, `0` Fail)** | 10,051.8 tok/s *(`c=256`; `15,016.0` @ `c=768, 1 fail`)* | **33,334.6 tok/s** *(`BF16`)* / **34,317.3 tok/s** *(`FP8 DP-Attn=32, c=768`)* | **3.32×–3.41× (`0`-fail) / 2.22×–2.29× (`c=768`)** |
| **16-Chip `4K:1K` Output (`95%` Hit, `0` Fail)** | 18,940.7 tok/s *(`c=768, n=1536`)* | **35,581.0 tok/s** *(`BF16`)* / **37,321.6 tok/s** *(`FP8 DP-Attn=32, c=768`)* | **1.88× (`BF16`) – 1.97× (`DP-Attn=32`)** |
| **Slice Full KV Pool (`swa=0.10`)** | 42.29M tokens (`BF16`) | 82.97M (`BF16`) – 117.80M (`FP8`) tokens | **1.96× – 2.79× KV capacity** |

---

## 7. Measured Deployment Configurations & Sizing Matrix

### 7.1 Validated `v7x-16` Slice Topologies (`sglang-torchtpu`)

1. **`Block-FP8 TP=2 + DP-Attention=32, swa=0.10` (Highest `4K–8K` Throughput):**
   - `16` intra-chip `Block-FP8 TP=2` servers with `--dp-size 2 --enable-dp-attention --swa-full-tokens-ratio 0.10` (`32` `attn_tp=1` DP ranks, `116.42M` slice KV tokens).
   - **Measured (`0` fail):** `37,321.6` tok/s (`4K, 95%, c=768`), `34,317.3` tok/s (`4K, 0%, c=768`), `30,995.4` tok/s (`8K, 95%, c=512`).
2. **`BF16 TP=2 × DP=16, mrr=96, swa=0.10` (Unquantized Weights + Balanced Short/Long Context):**
   - `16` intra-chip `BF16 TP=2` servers with `--max-running-requests 96 --torch-compile-max-bs 96 --swa-full-tokens-ratio 0.10` (`82.97M` slice KV tokens).
   - **Measured (`0` fail):** `35,581.0` tok/s (`4K, 95%, c=768`), `33,334.6` tok/s (`4K, 0%, c=768`), `32,269.7` tok/s (`8K, 95%, c=1024`), `6,549.8–6,617.3` tok/s (`64K, 95%, c=64, n=64`), `4,810.9–4,823.8` tok/s (`128K, 95%, c=64, n=64`).
3. **`Block-FP8 TP=2 × DP=16, mrr=64, swa=0.10` (Maximum `64K–128K` Long-Context Capacity & Throughput):**
   - `16` intra-chip `Block-FP8 TP=2` servers (`attn_tp=2`, `117.80M` slice KV tokens = `7,362,688` tokens/replica).
   - **Measured (`0` fail):** `33,200.3` tok/s (`4K, 95%, c=512, n=512`), `7,119.8–7,225.0` tok/s (`64K, 95%, c=64, n=64`), `5,113.8–5,172.5` tok/s (`128K, 95%, c=64, n=64`).

### 7.2 Measured Capacity & Throughput Sizing Matrix

| Deployment Target | TPU Hardware | TensorCores | Parallelism Layout | Engines / Ranks | Full KV Tokens / Unit | Total Full KV Tokens | Measured `4K` Output (`0` Fail) | Archive |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Single Chip (`BF16`, `swa=0.80`)** | 1× `v7x` chip | 2 | `BF16 TP=2` | 1 | 2,050,432 | 2.05M | 3,058.4 *(0%)* / 3,285.9 *(95%, c=64)* | [`ablation_bf16kv`](../results/ablation_bf16kv_tp2_1chip.json) |
| **Single Chip (`Block-FP8`, `swa=0.80`)** | 1× `v7x` chip | 2 | `Block-FP8 TP=2` | 1 | 3,067,776 | 3.07M | 3,315.9 tok/s *(0%, c=64)* | [`ablation_fp8`](../results/ablation_fp8_tp2_1chip.json) |
| **Single Host (`4` Chips, `BF16`, `swa=0.10`)** | 1× `v7x` host | 8 | `BF16 TP=2 × DP=4` | 4 | 5,185,408 | 20.74M | 8,853.4 tok/s *(95%, c=128)* | [`C1_bf16_tp2_dp4`](../../multilora_singlehost_opt/v7x/C1_bf16_tp2_dp4_fp8kv.json) |
| **Full Slice (`BF16`, `mrr=64`, `swa=0.80`)** | `v7x-16` (4 hosts) | 32 | `BF16 TP=2 × DP=16` | 16 | 2,163,968 | 34.62M | 29,833.1 *(0%)* / 33,421.5 *(95%, c=512, n=512)* | [`fullslice_bf16`](../results/fullslice_bf16_tp2_dp16.json) |
| **Full Slice (`BF16`, `mrr=96`, `swa=0.10`)** | `v7x-16` (4 hosts) | 32 | `BF16 TP=2 × DP=16` | 16 | 5,185,408 | **82.97M** | **33,334.6 *(0%)* / 35,581.0 *(95%, c=768, n=1536)*** | [`fullslice_bf16_mrr96`](../results/fullslice_bf16_tp2_dp16_mrr96_swa010.json) |
| **Full Slice (`Block-FP8`, `mrr=64`, `swa=0.10`)** | `v7x-16` (4 hosts) | 32 | `Block-FP8 TP=2 × DP=16` | 16 | 7,362,688 | **117.80M** | 29,923.9 *(0%)* / **33,200.3 *(95%, c=512, n=512)*** | [`fullslice_fp8`](../results/fullslice_fp8_tp2_dp16_swa010.json) |
| **Full Slice (`Block-FP8`, `DP-Attn=32`, `swa=0.10`)** | `v7x-16` (4 hosts) | 32 | `Block-FP8 TP=2 + DP-Attn=32` | 16 srv / 32 ranks | 3,638,272 / rank | **116.42M** | **34,317.3 *(0%)* / 37,321.6 *(95%, c=768, n=1536)*** | [`fullslice_dpattn32`](../results/fullslice_fp8_tp2_dpattn32_swa010.json) |

---

## 8. Methodology Notes & Known Limitations

- **Prefix Pre-Warming & Oracle Trunk Routing**: On cached points (`50%`, `80%`, `95%` target shared prefix), [`bench_flint_v7x.py`](../scripts/bench_flint_v7x.py) pre-warms `64`-token aligned prefix trunks on designated replicas before starting `wall_sec` (`0.1–3.7 s` at `4K–16K`; `8.2–99.2 s` at `64K–128K`) and passes `X-Trunk-Index` headers to [`gateway_flint_v7x.py`](../scripts/gateway_flint_v7x.py) for deterministic replica affinity. Server-side `observed_cache_hit_pct` was not scraped on `v7x` (`null` in `v7x/results/*.json`; verified at `93.75%–94.97%` on `v5p-16` using the same `64`-token trunk alignment).
- **Prompts per Point (`n`) & Effective Long-Context Concurrency**: `mrr=96` and `DP-Attention=32` `4K`/`8K` sweeps use `n = 2c` at `c=512..768` (`n=1024..1536`) and `n = 1536` at `c=1024`, whereas `mrr=64` sweeps (`fullslice_bf16_tp2_dp16.json`, `fullslice_fp8_tp2_dp16_swa010.json`) use `n = c = 512` (single wave) at `c=512`. All `64K` and `128K` points with `c >= 64` send `n = 64` requests (`effective concurrency = 64`).
- **Unsupported `sglang-torchtpu` Configurations on `v7x`**: Single-TensorCore `TP=1` (`No TPU topology found for count: 1`), `Block-FP8 TP=8` without `EP=8` (`512 / 8 = 64 < 128` block size), `NGRAM` speculative decoding (`NotImplementedError` in `sglang_tpu`), `--quantization w8a8_int8` (Triton kernel dependency), and `chunked_prefill_size=16384 + mrr=128` (`SMEM` overflow) are not supported; see [`SINGLEHOST_AND_MULTILORA_OPTIMIZATION.md` §5](../../docs/SINGLEHOST_AND_MULTILORA_OPTIMIZATION.md#5-supported-configurations--known-limitations) and [`README.md` §3–§4](../../README.md#3-methodology-notes).
