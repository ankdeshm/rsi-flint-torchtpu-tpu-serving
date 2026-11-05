# Multi-Host Slice Serving Optimization for `RecursiveSI/flint` (`40B-A2.5B` MoE) on TPU `v7x-16` & `v5p-16`

Engineering analysis and benchmark comparison between **slice-wide Expert-Parallel serving (`vllm-torchtpu` `v0.32.0`, `EP=32` / `EP=16`)** and **sub-host zero-ICI replica serving (`sglang-torchtpu` + TorchTPU `0.2.0`)** for **`RecursiveSI/flint`** (`39.8B` total / `~2.5B` active parameters, `49` layers, `256` routed experts `top-6` sigmoid + shared expert, `35` SWA `W=513` RoPE + `14` full NoPE layers, learned attention sinks) on 4-host **TPU `v7x-16` (`2x2x4`, `3,031.7 GiB` usable HBM3e)** and **TPU `v5p-16` (`2x2x4`, `1,515.8 GiB` usable HBM2e)**.

---

## 1. Architectural Regimes on 4-Host TPU Slices (`2x2x4`)

Unlike `675B–744B` MoEs whose checkpoints require multi-host sharding to fit in HBM, `RecursiveSI/flint` occupies `79.57 GB` (`74.11 GiB`) in `BF16` and `40.43 GB` (`37.66 GiB`) in `[128, 128]` `Block-FP8`. On 16-chip TPU slices, this creates two distinct serving regimes:

1. **Slice-Wide `DP × EP` Single-Endpoint Serving (`vllm-torchtpu` `v0.32.0`, [`vllm/`](../vllm/README.md))**:
   - Runs a **single native `vllm serve` OpenAI endpoint** over a 4-host Ray cluster (`--data-parallel-size 32 --enable-expert-parallel` on `v7x-16`; `--data-parallel-size 16 --enable-expert-parallel` on `v5p-16`).
   - Replicates only non-MoE weights (`~5.4 GiB` `BF16` / `~2.9 GiB` `FP8`) per rank while sharding the `256` routed experts across all `32` TensorCores (`8` experts/TC = **`1.25 GiB` `BF16`** / **`0.63 GiB` `FP8`**) on `v7x-16` or all `16` chips (`16` experts/chip = **`2.50 GiB` `BF16`**) on `v5p-16`.
   - Enables **Pure `BF16` at `TP=1`** (`vllm_bf16_tp1_dp32_ep32_fp8kv` on `v7x-16` and `vllm_bf16_tp1_dp16_ep16_fp8kv` on `v5p-16`) behind one `vllm serve` process.
2. **Sub-Host Zero-ICI Independent Replica Serving (`sglang-torchtpu`, [`v7x/`](../v7x/README.md) & [`v5p/`](../v5p/README.md))**:
   - Fits each full replica inside 1 physical `v7x` chip (`TP=2 × DP=16` or `TP=2 + DP-Attn=32`, intra-package die-to-die only, **zero inter-chip ICI**) or 2 `v5p` chips (`TP=2 × DP=8`), paired with `FlintConfig.is_hybrid_swa = True` (`swa_full_tokens_ratio=0.10`, separate `513`-token SWA KV pool for `35/49` layers -> **`116.4M–117.8M` KV tokens/slice** on `v7x-16`, **`42.29M` KV tokens/slice** on `v5p-16`) and an HTTP routing gateway.

---

## 2. `vllm-torchtpu` Multi-Host `EP=32` (`v7x-16`) & `EP=16` (`v5p-16`) Results

### A. Modifications Implemented for `RecursiveSI/flint` in `vllm-torchtpu`

| Component | Requirement | Modification | Code Anchor |
|---|---|---|---|
| **1. Model & Config Registration** | Register `model_type="flint"` across `transformers` and `vllm` in all Ray workers (including containers with `/usr/lib/python3.12/sitecustomize.py`) | Standalone `flint_hf_config.py` registered in `transformers/.../configuration_auto.py` + `FlintForCausalLM` with hybrid `layer_types`, `mlp_only_layers=[0,1]`, top-6 sigmoid `FusedMoE`, and TP-sharded `sinks` | [`install_flint_vllm.py`](../vllm/scripts/install_flint_vllm.py) |
| **2. Pallas RPA v3 `head_dim=128` Attention Sinks** | Support per-head learned attention sinks on `head_dim=128` attention layers | Extended `attention_interface.py` and `ragged_paged_attention/v3/kernel.py` to accept a VMEM-resident `[kv_heads, q_per_kv, 128]` sink tile and initialize softmax accumulators `m_ref = sink`, `l_ref = 1.0` prior to the KV page loop | [`install_flint_vllm.py`](../vllm/scripts/install_flint_vllm.py) |
| **3. TPU `v5p` SparseCore `top-6` Routing** | Match SparseCore lane width across TPU `v7x` (`16` lanes) and TPU `v5p` (`8` lanes) | Added `sc.num_lanes != LANES` check in `ensure_v5p_sparsecore_fallback()` to route `v5p` through standard TPU `topk` | [`run_flint_vllm_multihost_suite.py`](../vllm/scripts/run_flint_vllm_multihost_suite.py) |
| **4. Speculative Decoding Scheduling** | Run host-side `ngram` (`NgramProposer`) with synchronous step verification | Sets `--no-async-scheduling` when `ngram` speculative decoding is enabled | [`run_flint_vllm_multihost_suite.py`](../vllm/scripts/run_flint_vllm_multihost_suite.py) |
| **5. Node Ephemeral Disk Allocation** | Accommodate Ray `/tmp/ray`, TorchInductor, and XLA compilation caches on 43 GiB COS boot disks | Mounted `emptyDir: medium: Memory` at `/tmp` and `/dev/shm` across multi-host manifests | [`v7x-16-flint-vllm-ep32.yaml`](../vllm/k8s/v7x-16-flint-vllm-ep32.yaml) · [`v5p-16-flint-vllm-ep16.yaml`](../vllm/k8s/v5p-16-flint-vllm-ep16.yaml) |

### B. TPU `v7x-16` (`EP=32`, 32 TensorCores, `0` Failed Requests)

| Experiment | Parallelism & Precision | `4K:1K` `0%` Hit | `4K:1K` `80%` Hit | `4K:1K` `95%` Hit | Peak Total tok/s (Incl. Cached Prefix) | `c=1` `p50` `TPOT` *(Best Swept)* | `64K` / `128K` (`0%` Hit, `c=8, n=16`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`vllm_bf16_tp1_dp32_ep32_fp8kv`** | `BF16`, `TP=1 × DP=32 × EP=32` | **`21,650.19` tok/s** *(`c=1024, n=1024`)* | **`27,091.93` tok/s** *(`c=1024, n=1024`)* | **`26,912.29` tok/s** *(`c=1024, n=1024`)* | **`130,432.81` tok/s** *(`80%`)* | `11.51 ms` *(`10.92 ms` @ `c=32`)* | — |
| **`vllm_fp8_tp1_dp32_ep32_fp8kv`** | `Block-FP8`, `TP=1 × DP=32 × EP=32` | `19,153.33` tok/s *(`c=1024, n=1024`)* | `25,415.79` tok/s *(`c=1024, n=1024`)* | `26,145.93` tok/s *(`c=1024, n=1024`)* | `125,827.28` tok/s *(`95%`)* | `10.95 ms` *(`10.65 ms` @ `c=32`)* | **`308.56` / `189.67` tok/s** |
| **`vllm_fp8_tp2_dp16_ep32_fp8kv`** | `Block-FP8`, `TP=2 × DP=16 × EP=32` | `15,006.82` tok/s *(`c=1024, n=1024`)* | — | `20,336.83` tok/s *(`c=1024, n=1024`)* | `97,871.00` tok/s *(`95%`)* | **`10.37 ms`** *(`96.4` tok/s)* | — |
| **`vllm_fp8_tp4_dp8_ep32_ngram5`** | `Block-FP8`, `TP=4 × DP=8 × EP=32` + `ngram` `k=5` (`--no-async-scheduling`) | `805.36` tok/s *(`c=16, n=32`)* | — | `812.63` tok/s *(`c=16, n=32`)* | `3,910.80` tok/s *(`95%`)* | `19.62 ms` *(`18.91 ms` @ `c=16`)* | — |

**Key `v7x-16` Findings**:
- **Pure `BF16` vs `Block-FP8` under `EP=32`**: Because `EP=32` places `8` routed experts per TensorCore (`1.25 GiB` `BF16` vs `0.63 GiB` `FP8`), avoiding dynamic input activation quantization makes **`BF16 TP=1×DP=32×EP=32` `3.0%–13.0%` faster** than `FP8 TP=1×DP=32×EP=32` at matched `c=1024, n=1024` single-wave points (`26,912.29` vs `26,145.93` out tok/s at `95%` target hit; `27,091.93` vs `25,415.79` at `80%`; `21,650.19` vs `19,153.33` at `0%`).
- **Concurrency Crossover (`TP=2×DP=16×EP=32` vs `TP=1×DP=32×EP=32`)**: At `c=1..256`, `TP=2×DP=16×EP=32` achieves lower latency and higher throughput (`549.6 ms` vs `856.7 ms` `TTFT` and `10.37 ms` vs `10.95 ms` `TPOT` at `c=1`; `5,440.8` vs `5,025.5` out tok/s at `4K, 95%, c=64, n=128`; `12,264.5` vs `11,376.2` out tok/s at `4K, 95%, c=256, n=512`), whereas `DP=32` overtakes at `c >= 512`.
- **Host-Side `ngram` (`k=5`) Latency Tradeoff**: `vllm_fp8_tp4_dp8_ep32_ngram5` requires `--no-async-scheduling`, introducing a host-device synchronization step per verification pass. Measured `p50` `TPOT` is `18.91–19.62 ms` on synthetic `4K:1K` (`0%`/`95%` hit) and `43.37–80.99 ms` on `coding_edit` (`4K:512, 95%`), which is `~1.8×` slower per output token than non-speculative async `TP=2×DP=16×EP=32` (`10.37 ms`). Acceptance statistics are not recorded in the result JSON (`0%–100%` across server log intervals, with high-acceptance intervals driven by greedy `ignore_eos` repetition), and no matched `TP=4×DP=8×EP=32` `--no-async-scheduling` non-speculative baseline was run.

### C. TPU `v5p-16` (`EP=16`, 16 Chips, `0` Failed Requests)

| Experiment | Parallelism & Precision | `4K:1K` `0%` Hit (`c=512, n=1024`) | `4K:1K` `95%` Hit | `8K:1K` `95%` Hit (`c=256, n=512`) | Peak Total tok/s | `c=1` `p50` `TPOT` *(Best Swept)* | `4K` `0%` `TTFT` (`c=1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`vllm_bf16_tp1_dp16_ep16_fp8kv`** | `BF16`, `TP=1 × DP=16 × EP=16` | **`6,256.04` tok/s** | **`9,470.66` tok/s** *(`c=512, n=1024`)* | **`8,390.24` tok/s** | **`72,382.22` tok/s** *(`8K, 95%`)* | **`12.34 ms`** | `1,261.45 ms` |
| **`vllm_w8a16_tp1_dp16_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=1 × DP=16 × EP=16` | `6,000.31` tok/s | `9,018.25` tok/s *(`c=512, n=1024`)* | **`7,501.46` tok/s** | `64,714.78` tok/s *(`8K, 95%`)* | `16.25 ms` *(`15.09 ms` @ `c=16`)* | `1,293.54 ms` |
| **`vllm_int8_w8a16_tp1_dp16_ep16_fp8kv`** | `INT8 W8A16`, `TP=1 × DP=16 × EP=16` | `5,728.53` tok/s | `7,244.99` tok/s *(`c=256, n=512`)* | — | `34,866.53` tok/s *(`4K, 95%`)* | `17.39 ms` | `1,292.92 ms` |
| **`vllm_w8a16_tp2_dp8_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=2 × DP=8 × EP=16` | `4,628.94` tok/s | `6,680.05` tok/s *(`c=512, n=1024`)* | — | `32,147.74` tok/s *(`4K, 95%`)* | `14.60 ms` | `825.40 ms` |
| **`vllm_w8a16_tp4_dp4_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=4 × DP=4 × EP=16` | `3,218.98` tok/s *(`c=256, n=512`)* | `3,694.72` tok/s *(`c=256, n=512`)* | — | `17,780.83` tok/s *(`4K, 95%`)* | `19.01 ms` *(`16.24 ms` @ `c=4`)* | **`439.46 ms`** |

**Key `v5p-16` Findings**:
- **`+49.7%` Higher `FP8/W8A16` `TP=1×DP=16` Throughput vs Non-EP `TP=1×DP=16` at Matched `8K:1K, 95%, c=256`**: Because TPU `v5p` MXUs execute `BF16` inputs (dequantizing `W8A16` weights in software), non-EP `sglang-torchtpu` `slice16_fp8_tp1_dp16_mrr64_swa010` (`v5p/results/raw/v5p16_slice16_opt_r2_slice16_pod0.jsonl`) dequantizes all `256` experts per chip (`5,010.73` out tok/s at `8K:1K, 95%` target [`94.53%` observed], `c=256, n=512`), whereas `vllm-torchtpu` `vllm_w8a16_tp1_dp16_ep16_fp8kv` (`16` experts/chip) reaches **`7,501.46` out tok/s (`+49.7%`)** under the same `(8K:1K, 95%, c=256, n=512)` workload. Pure `BF16` `vllm_bf16_tp1_dp16_ep16_fp8kv` removes `W8A16` dequantization completely to reach **`8,390.24` out tok/s** at `8K` and **`9,470.66` out tok/s** at `4K` (`c=512, n=1024`).

---

## 3. High-Concurrency `sglang-torchtpu` Multi-Host Optimization (`DP-Attention=32` & `mrr=96, swa=0.10`)

1. **Intra-Chip `TP=2 + DP-Attention=32` on TPU `v7x-16` ([`fullslice_fp8_tp2_dpattn32_swa010.json`](../v7x/results/fullslice_fp8_tp2_dpattn32_swa010.json))**:
   - Launches `16` intra-package `TP=2` `Block-FP8` servers (`4` per host, zero inter-chip ICI) with `--dp-size 2 --enable-dp-attention --swa-full-tokens-ratio 0.10`, exposing **`32` independent `attn_tp=1` attention/KV DP ranks** (`3,638,272` KV tokens/rank = **`116,424,704` (`116.42M`) KV tokens across the slice**) while keeping MoE `gmm_v2` sharded across both intra-package TensorCores (`TP=2`).
   - **Multi-Replica ZMQ TCP Port Spacing ([`launch_host_replicas.py`](../v7x/scripts/launch_host_replicas.py))**: Under `--enable-dp-attention`, SGLang's `PortArgs.init_new` allocates 7 consecutive TCP ports `port + 233..239`; spacing intra-host `--port` by `20` (`30000, 30020, 30040, 30060`) with disjoint `--nccl-port` prevents port collisions across all 4 engines per host.
   - **Measured `v7x-16` Throughput**:
     - **`4K:1K, 95%` target prefix**: **`37,321.6` out tok/s (`186,607.8` tot tok/s, `0` failures, `629.4 ms` p50 `TTFT`)** at `c=768, n=1536`. Pushing to `c=1024, n=1536` reaches `39,520.7` out tok/s (`197,603.3` tot tok/s) with `17/1536` HTTP 500 failures.
     - **`4K:1K, 0%` uncached**: **`34,317.3` out tok/s (`171,586.4` tot tok/s, `0` failures)** at `c=768, n=1536`.
     - **`8K:1K, 95%` target prefix**: **`30,995.4` out tok/s (`278,958.6` tot tok/s, `0` failures)** at `c=512, n=1024`; `35,969.7` out tok/s (`3/1536` failures) at `c=768, n=1536`; and `39,035.7` out tok/s (`4/1536` failures) at `c=1024, n=1536`. At `8K:1K, 0%` uncached (`c=768, n=1536`), throughput is **`26,572.9` out tok/s (`0` failures)**.
     - **`64K–128K` Long-Context Crossover (`DP-Attention=32` vs `TP=2 × DP=16`, effective `c=64, n=64`)**: At `64K:1K` (`3,154.4–3,310.3` / `3,744.1–3,749.8` / `4,471.8–4,561.6` out tok/s at `0%` / `50%` / `95%`) and `128K:1K` (`1,501.2–1,638.6` / `1,771.2–1,778.0` / `2,400.3–2,411.7` out tok/s at `0%` / `50%` / `95%`), standard intra-chip `TP=2 × DP=16` (`attn_tp=2`: `7,119.8–7,225.0` tok/s at `64K, 95%` and `5,113.8–5,172.5` tok/s at `128K, 95%`) outperforms `DP-Attention=32` (`attn_tp=1`) by **`1.58×–2.14×`** because `attn_tp=2` shards `64K–128K` prefill across both intra-package TensorCores and maintains a unified per-chip RadixCache.
2. **Intra-Chip `BF16 TP=2 × DP=16` (`mrr=96, swa=0.10`) on TPU `v7x-16` ([`fullslice_bf16_tp2_dp16_mrr96_swa010.json`](../v7x/results/fullslice_bf16_tp2_dp16_mrr96_swa010.json))**:
   - Raising `--max-running-requests` and `--torch-compile-max-bs` to `96` across `16` pure `BF16 TP=2` replicas (`82.97M` KV tokens across the slice) reaches **`35,581.0` out tok/s (`177,905.2` tot tok/s, `0` failures)** at `4K:1K, 95%, c=768, n=1536` (`2`-pass range: `32,517.7` [`1/1536` fail] to `35,581.0` [`0` fail]), **`33,334.6` out tok/s (`0` failures)** at `4K:1K, 0%, c=768, n=1536` (`2`-pass range: `33,334.6` [`0` fail] to `35,004.7` [`4/1536` fail]), **`32,269.7` out tok/s (`0` failures)** at `8K:1K, 95%, c=1024, n=1536`, **`6.39–6.74 ms` `c=1` `TPOT`**, and **`6,549.8–6,617.3` out tok/s** (`64K, 95%`) / **`4,810.9–4,823.8` out tok/s** (`128K, 95%`) at effective `c=64, n=64` with `0` failures.
3. **`BF16 TP=2 × DP=8` (`mrr=96, swa=0.10`) vs `FP8 TP=1 × DP=16` (`mrr=64, swa=0.10`) on TPU `v5p-16` ([`v5p16_slice16_opt_r2_slice16_pod0.jsonl`](../v5p/results/raw/v5p16_slice16_opt_r2_slice16_pod0.jsonl))**:
   - Raising `--max-running-requests` and `--torch-compile-max-bs` to `96` per `TP=2` replica (`768` active decode slots across `8` replicas) with `--swa-full-tokens-ratio 0.10` (`42.29M` full KV tokens across the slice) and fused `256`-expert Pallas `router_topk` reaches **`18,940.7` out tok/s (`94,703.6` tot tok/s, `0` failures)** at `4K:1K, 95%` target (`93.75%` observed), `c=768, n=1536`; **`10,051.8` out tok/s (`0` failures, `c=256, n=512`)** and `15,016.0` out tok/s (`1/1536` failed, `c=768, n=1536`) at `4K:1K, 0%`; **`10,323.1` out tok/s (`0` failures, `c=256, n=512`)** and `13,689.2` out tok/s (`1/1024` failed, `c=512, n=1024`) at `8K:1K, 95%`; **`3,511.0` out tok/s (`0` failures, `c=64, n=128`)** at `64K:1K, 95%`; and **`1,867.8` out tok/s (`0` failures, `c=32, n=32`)** at `128K:1K, 95%`.
   - Under the same 16-chip harness, `slice16_fp8_tp1_dp16_mrr64_swa010` (`16` single-chip `Block-FP8 TP=1` replicas, `40.40M` full KV tokens) reaches **`9,024.6` out tok/s (`0` failures)** at `4K, 95%, c=768`, **`7,877.8` out tok/s (`0` failures)** at `4K, 0%, c=768`, **`5,010.7` out tok/s (`0` failures, `c=256`)** / `7,292.2` out tok/s (`2/1024` failed, `c=512`) at `8K, 95%`, **`2,049.2` out tok/s (`0` failures, `c=64, n=128`)** at `64K, 95%`, and **`1,369.8` out tok/s (`0` failures, `c=32, n=32`)** at `128K, 95%` — confirming `BF16 TP=2 × DP=8` is **`2.10×` faster** on `v5p` because `v5p` MXUs execute native `BF16`.

---

## 4. Cross-Engine Summary (`sglang-torchtpu` vs `vllm-torchtpu`, `0`-Failure Primary Points)

| Operational Objective | Stack & Topology | TPU `v7x-16` (`2x2x4`) | TPU `v5p-16` (`2x2x4`) | Key Tradeoff |
|---|---|---|---|---|
| **Maximum High-Concurrency Slice Throughput** | **`sglang-torchtpu` Intra-Chip `TP=2 + DP-Attn=32` (`v7x`) / `BF16 TP=2 × DP=8, mrr=96` (`v5p`)** | **`37,321.6` out tok/s** *(`4K, 95%, c=768, 0 fail`; `39,520.7` @ `c=1024, 17 fail`)* · **`34,317.3`** *(`4K, 0%, c=768, 0 fail`)* · **`30,995.4`** *(`8K, 95%, c=512, 0 fail`; `39,035.7` @ `c=1024, 4 fail`)* | **`18,940.7` out tok/s** *(`4K, 95%, c=768, 0 fail`)* · **`10,051.8`** *(`4K, 0%, c=256, 0 fail`; `15,016.0` @ `c=768, 1 fail`)* · **`3,511.0`** *(`64K, 95%, c=64`)* · **`1,867.8`** *(`128K, 95%, c=32, n=32`)* | Requires external prefix-affinity HTTP gateway; highest `4K–8K` throughput and `116.4M` (`v7x`) / `42.3M` (`v5p`) slice KV capacity |
| **Low-Latency Interactive & `64K–128K` Long Context** | **`sglang-torchtpu` Intra-Chip `TP=2 × DP=16`** (`mrr=96` `BF16` / `swa=0.10` `Block-FP8`) | **`35,581.0` out tok/s** *(`BF16, 4K, 95%, c=768, 0 fail`)* / **`33,334.6`** *(`0%, 0 fail`)* · **`7,225.0`** *(`FP8, 64K, 95%, c=64, n=64`)* · **`5,172.5`** *(`FP8, 128K, 95%, c=64, n=64`)* · **`6.39–6.74 ms` `c=1` `TPOT`** | **`7.36 ms` `c=1` `TPOT`** *(`BF16 TP=4×DP=4` slice)* · **`7.48 ms` `c=1` `TPOT`** *(`BF16 TP=2×DP=8` slice)* | `attn_tp=2` shards `64K–128K` prefill across 2 TensorCores/chips (`1.58×–2.14×` faster than `DP-Attn=32` at `64K–128K`) |
| **Single Native OpenAI Endpoint (`0` Gateways) & `TP=1` `BF16`** | **`vllm-torchtpu` Multi-Host Ray `TP=1 × DP × EP`** (`vllm serve --enable-expert-parallel`) | **`26,912.3` out tok/s** *(`95%`)* / **`27,091.9`** *(`80%`)* / **`21,650.2`** *(`0%`, `BF16 TP=1×DP=32×EP=32, c=1024, n=1024`)* · **`10.37 ms` `c=1` `TPOT`** *(`TP=2`)* | **`9,470.7` out tok/s** *(`4K, 95%, c=512, n=1024`)* · **`6,256.0`** *(`4K, 0%`)* · **`8,390.2`** *(`8K, 95%, c=256`)* · **`12.34 ms` `c=1` `TPOT`** | Single native `vllm serve` endpoint across 4 hosts; `EP` shrinks per-device experts `16–32×` so `BF16` runs at `TP=1` |
| **`TP=1 × DP=16` Quantized Serving on `v5p-16`** | **`vllm-torchtpu` `W8A16 TP=1 × DP=16 × EP=16`** | — | **`9,018.3` out tok/s** *(`4K, 95%, c=512`)* / **`6,000.3`** *(`4K, 0%, c=512`)* / **`7,501.5`** *(`8K, 95%, c=256`, **`+49.7%`** vs non-EP `5,010.7`)* | `EP=16` cuts software `W8A16` expert dequantization `16×` per chip |

---

## 5. Methodology Notes & Known Limitations

- **Prefix Pre-Warming & Routing**: Cached points (`50%`/`80%`/`95%` target hit) pre-warm shared prefix trunks prior to starting the timed window (`wall_sec` excludes pre-warm time, up to `51–99 s` at `128K`). On `v7x-16` SGLang slices, [`bench_flint_v7x.py`](../v7x/scripts/bench_flint_v7x.py) supplies `X-Trunk-Index` headers consumed by [`gateway_flint_v7x.py`](../v7x/scripts/gateway_flint_v7x.py) (oracle prefix-affinity routing), and server-side observed cache hit % was recorded on `v5p-16` (`93.75%–94.97%` at `95%` target) but not on `v7x-16` or `vLLM` (where Ray DPLB is not prefix-aware).
- **Single-Wave Points & `64K`/`128K` Request Caps**: `vLLM` `c=1024` rows (`n=1024`), `sglang` `fullslice_fp8_tp2_dp16_swa010` `c=512` rows (`n=512`), and `v5p-16` `128K` `c=32` rows (`n=32`) run a single wave (`n = c`). Every `v7x-16` `64K` and `128K` point with `c >= 64` dispatches `n = 64` requests (`effective concurrency = 64`). On `v5p-16`, `64K`/`128K` slice rows were measured at `95%` target prefix only.
- **Quality Evaluation Scope**: Each run verifies functional coding (`merge_intervals`) and passkey needle retrieval (`51,368` prompt tokens), and `vLLM` `v7x-16` vs `v5p-16` runs passed greedy output-equivalence gating; full task accuracy benchmarks (`GSM8K`, `MMLU-Pro`, `HumanEval`) across precisions remain future work. See [`README.md` §3–§4](../README.md#3-methodology-notes) for complete methodology and limitations.
