# Multi-Host Slice Serving Optimization for `RecursiveSI/flint` (`40B-A2.5B` MoE) on TPU `v7x-16` & `v5p-16`

Engineering analysis and benchmark comparison between **slice-wide Expert-Parallel serving (`vllm-torchtpu` `v0.32.0`, `EP=32` / `EP=16`)** and **sub-host zero-ICI replica serving (`sglang-torchtpu` + TorchTPU `0.2.0`)** for **`RecursiveSI/flint`** (`39.8B` total / `~2.5B` active parameters, `49` layers, `256` routed experts `top-6` sigmoid + shared expert, `35` SWA `W=513` RoPE + `14` full NoPE layers, learned attention sinks) on 4-host **TPU `v7x-16` (`2x2x4`, `3,031.7 GiB` usable HBM3e)** and **TPU `v5p-16` (`2x2x4`, `1,515.8 GiB` usable HBM2e)**.

---

## 1. Architectural Regimes on 4-Host TPU Slices (`2x2x4`)

Unlike `675B–744B` frontier MoEs (`Mistral-Large-3`, `GLM-5.3`) whose `400–700+ GB` checkpoints require multi-host sharding simply to fit in HBM, `RecursiveSI/flint` occupies `79.57 GB` (`74.11 GiB`) in BF16 and `40.43 GB` (`37.66 GiB`) in `[128, 128]` Block-FP8. On 16-chip TPU slices, this creates two distinct serving regimes:

1. **Slice-Wide `DP × EP` Single-Endpoint Serving (`vllm-torchtpu` `v0.32.0`, [`vllm/`](../vllm/README.md))**:
   - Runs a **single native `vllm serve` OpenAI endpoint** over a 4-host Ray cluster (`--data-parallel-size 32 --enable-expert-parallel` on `v7x-16`; `--data-parallel-size 16 --enable-expert-parallel` on `v5p-16`).
   - Replicates only non-MoE weights (`~5.4 GiB` BF16 / `~2.9 GiB` FP8) per rank while sharding the `256` routed experts across all `32` TensorCores (`8` experts/TC = **`1.25 GiB` BF16** / **`0.63 GiB` FP8**) on `v7x-16` or all `16` chips (`16` experts/chip = **`2.50 GiB` BF16**) on `v5p-16`.
   - Unlocks **Pure `BF16` at `TP=1`** (`vllm_bf16_tp1_dp32_ep32_fp8kv` on `v7x-16` and `vllm_bf16_tp1_dp16_ep16_fp8kv` on `v5p-16`) without external HTTP gateways or metadata shims.
2. **Sub-Host Zero-ICI Independent Replica Serving (`sglang-torchtpu`, [`v7x/`](../v7x/README.md) & [`v5p/`](../v5p/README.md))**:
   - Fits each full replica inside 1 physical `v7x` chip (`TP=2 × DP=16`, intra-package die-to-die only, **zero inter-chip ICI**) or 2 `v5p` chips (`TP=2 × DP=8`), paired with `FlintConfig.is_hybrid_swa = True` (`swa_full_tokens_ratio=0.10`, separate `513`-token SWA KV pool for 35/49 layers -> **`117.8M` KV tokens/slice** on `v7x-16`) and a prefix-trunk affinity gateway.

---

## 2. `vllm-torchtpu` Multi-Host `EP=32` (`v7x-16`) & `EP=16` (`v5p-16`) Results

### A. Enablement Patches Required for `RecursiveSI/flint` in `vllm-torchtpu`

| Component | Bottleneck / Blocker in Stock `vllm-torchtpu` (`v0.32.0`) | Fix Implemented | Code Anchor |
|---|---|---|---|
| **1. Model & Config Registration** | `model_type="flint"` unknown to both `transformers` and `vllm`; `/usr/lib/python3.12/sitecustomize.py` shadows `site-packages/sitecustomize.py` in Ray workers | Standalone `flint_hf_config.py` registered in `transformers/.../configuration_auto.py` + `FlintForCausalLM` with hybrid `layer_types`, `mlp_only_layers=[0,1]`, top-6 sigmoid `FusedMoE`, and TP-sharded `sinks` | [`install_flint_vllm.py#L21-L751`](../vllm/scripts/install_flint_vllm.py#L21-L751) |
| **2. Pallas RPA v3 `head_dim=128` Attention Sinks** | `attention_interface.py` raised `NotImplementedError("Attention sink support is only available when head_dim==64")` | Removed `head_dim==64` restriction, passed VMEM-resident `[kv_heads, q_per_kv, 128]` sink tile into `ragged_paged_attention/v3/kernel.py`, and seeded `m_ref = sink`, `l_ref = 1.0` before the KV page loop | [`install_flint_vllm.py#L753-L912`](../vllm/scripts/install_flint_vllm.py#L753-L912) |
| **3. TPU `v5p` SparseCore `top-6` Guard** | `sparsecore_topk.py` hardcoded `LANES = 16` (`v7x` SparseCore), failing compilation on `v5p` (`sc.num_lanes == 8`) | Added `sc.num_lanes != LANES` guard in `ensure_v5p_sparsecore_fallback()` to route `v5p` cleanly through standard TPU top-k | [`run_flint_vllm_multihost_suite.py#L929-L960`](../vllm/scripts/run_flint_vllm_multihost_suite.py#L929-L960) |
| **4. Speculative Decoding Scheduling** | Host-side `ngram` (`NgramProposer`) rejected under default `--async-scheduling` | Automatically disables async scheduling (`--no-async-scheduling`) when `ngram` speculative decoding is enabled | [`run_flint_vllm_multihost_suite.py#L510-L530`](../vllm/scripts/run_flint_vllm_multihost_suite.py#L510-L530) |
| **5. Node Ephemeral Disk Protection** | Ray `/tmp/ray` + TorchInductor + XLA caches filled 43 GiB COS rootfs (`DiskPressure`) | Mounted `emptyDir: medium: Memory` at `/tmp` and `/dev/shm` across all multi-host manifests | [`v7x-16-flint-vllm-ep32.yaml`](../vllm/k8s/v7x-16-flint-vllm-ep32.yaml) · [`v5p-16-flint-vllm-ep16.yaml`](../vllm/k8s/v5p-16-flint-vllm-ep16.yaml) |

### B. TPU `v7x-16` (`EP=32`, 32 TensorCores) Empirical Matrix

| Experiment | Parallelism & Precision | `4K:1K` `0%` Hit Peak | `4K:1K` `80%` Hit Peak | `4K:1K` `95%` Hit Peak | Peak Total tok/s | Best `p50` `TPOT` | `64K` / `128K` (`0%` Hit) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`vllm_bf16_tp1_dp32_ep32_fp8kv`** | `BF16`, `TP=1 × DP=32 × EP=32` | **`21,650.19` tok/s** *(`c=1024`)* | **`27,091.93` tok/s** *(`c=1024`)* | **`26,912.29` tok/s** *(`c=1024`)* | **`130,432.81` tok/s** | `10.92 ms` *(`92.0` tok/s)* | — |
| **`vllm_fp8_tp1_dp32_ep32_fp8kv`** | `Block-FP8`, `TP=1 × DP=32 × EP=32` | `19,153.33` tok/s *(`c=1024`)* | `25,415.79` tok/s *(`c=1024`)* | `26,145.93` tok/s *(`c=1024`)* | `125,827.28` tok/s | `10.65 ms` *(`94.2` tok/s)* | **`308.56` / `189.67` tok/s** |
| **`vllm_fp8_tp2_dp16_ep32_fp8kv`** | `Block-FP8`, `TP=2 × DP=16 × EP=32` | `14,358.15` tok/s *(`c=512`)* | — | `20,336.83` tok/s *(`c=1024`)* | `97,871.00` tok/s | **`10.37 ms`** *(`96.4` tok/s)* | — |
| **`vllm_fp8_tp4_dp8_ep32_ngram5`** | `Block-FP8`, `TP=4 × DP=8 × EP=32` + `ngram` `k=5` | `805.36` tok/s *(`c=16`)* | — | `812.63` tok/s *(`c=16`)* | `3,910.80` tok/s | `18.91 ms` *(`87.6–100%` accept)* | — |

**Key `v7x-16` Insights**:
- **Pure `BF16` vs `Block-FP8` under `EP=32`**: Because `EP=32` leaves only `8` routed experts per TensorCore (`1.25 GiB` BF16 vs `0.63 GiB` FP8), expert HBM traffic is already tiny; avoiding dynamic input activation quantization overhead makes **`BF16 TP=1×DP=32×EP=32` `3.6%–13.0%` faster** than `FP8 TP=1×DP=32×EP=32` at `c=1024` (`27,091.93` vs `26,145.93` out tok/s cached; `21,650.19` vs `19,153.33` uncached).
- **Concurrency Crossover (`TP=2×DP=16×EP=32` vs `TP=1×DP=32×EP=32`)**: At `c=1..256`, `TP=2×DP=16×EP=32` outperforms `TP=1×DP=32×EP=32` (`549.6 ms` vs `856.7 ms` TTFT at `c=1`; `5,440.8` vs `5,154.0` out tok/s at `c=64`; `12,264.5` vs `11,293.0` out tok/s at `c=256` with `19.12 ms` p50 TPOT), while `DP=32` overtakes above `c=512`.
- **Host-Side `ngram` (`k=5`) Tradeoff on TPU**: On structured prompts and `coding_edit`, `NgramProposer` achieves **`87.6%–100%` draft acceptance (`5.38–6.00` accepted tokens/step)**, but requiring `--no-async-scheduling` introduces a synchronous host-device barrier per `6`-token verification step (`18.91 ms` TPOT vs `10.37 ms` async `TP=2`), demonstrating why async scheduling remains superior until device-resident TPU draft proposers are used.

### C. TPU `v5p-16` (`EP=16`, 16 Chips) Empirical Matrix

| Experiment | Parallelism & Precision | `4K:1K` `0%` Hit Peak | `4K:1K` `95%` Hit Peak | `8K:1K` `95%` Hit (`c=256`) | Peak Total tok/s | Best `p50` `TPOT` | `4K` `0%` `TTFT` (`c=1`) |
|---|---|---:|---:|---:|---:|---:|---:|
| **`vllm_bf16_tp1_dp16_ep16_fp8kv`** | `BF16`, `TP=1 × DP=16 × EP=16` | **`6,256.04` tok/s** *(`c=512`)* | **`9,470.66` tok/s** *(`c=512`)* | **`8,390.24` tok/s** | **`72,382.22` tok/s** | **`12.34 ms`** *(`81.1` tok/s)* | `1,261.45 ms` |
| **`vllm_w8a16_tp1_dp16_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=1 × DP=16 × EP=16` | `5,693.19` tok/s *(`c=512`)* | `9,018.25` tok/s *(`c=512`)* | **`7,501.46` tok/s** | `64,714.78` tok/s | `15.09 ms` *(`66.7` tok/s)* | `1,293.54 ms` |
| **`vllm_int8_w8a16_tp1_dp16_ep16_fp8kv`** | `INT8 W8A16`, `TP=1 × DP=16 × EP=16` | `5,728.53` tok/s *(`c=512`)* | `7,244.99` tok/s *(`c=256`)* | — | `34,866.53` tok/s | `17.39 ms` *(`57.5` tok/s)* | `1,292.92 ms` |
| **`vllm_w8a16_tp2_dp8_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=2 × DP=8 × EP=16` | `4,297.27` tok/s *(`c=256`)* | `6,680.05` tok/s *(`c=512`)* | — | `32,147.74` tok/s | `14.60 ms` *(`68.5` tok/s)* | `825.40 ms` |
| **`vllm_w8a16_tp4_dp4_ep16_fp8kv`** | `Block-FP8 W8A16`, `TP=4 × DP=4 × EP=16` | `2,934.09` tok/s *(`c=256`)* | `3,694.72` tok/s *(`c=256`)* | — | `17,780.83` tok/s | `16.24 ms` *(`61.4` tok/s)* | **`439.46 ms`** |

**Key `v5p-16` Insights**:
- **`+45.2%` Higher `FP8/W8A16` `TP=1×DP=16` Throughput vs Non-EP `TP=1×DP=16`**: Because TPU `v5p` MXUs execute BF16 inputs (requiring software dequantization for W8A16), non-EP `TP=1×DP=16` dequantizes all `256` experts locally (`5,166.3` out tok/s @ `8K:1K, c=256, 95%`), whereas `vllm-torchtpu` `EP=16` (`16` experts/chip) reaches **`7,501.46` out tok/s (`+45.2%`)** at `8K` and **`9,018.25` out tok/s** at `4K` — and **`vllm_bf16_tp1_dp16_ep16_fp8kv`** eliminates W8A16 dequantization completely to reach **`9,470.66` out tok/s** (`4K`) and **`8,390.24` out tok/s / `72,382.22` total tok/s** (`8K`)!

---

## 3. High-Concurrency `sglang-torchtpu` Multi-Host Optimization (`DP-Attention=32` & `mrr=96, swa=0.10`)

To push 16-chip slice throughput past `c=512` up to `c=768–1024`, two engine-level optimizations were executed across `v7x-16` and `v5p-16`:

1. **Intra-Chip `TP=2 + DP-Attention=32` on TPU `v7x-16` ([`fullslice_fp8_tp2_dpattn32_swa010.json`](../v7x/results/fullslice_fp8_tp2_dpattn32_swa010.json))**:
   - Launches `16` intra-package `TP=2` Block-FP8 servers (`4` per host, zero inter-chip ICI) with `--dp-size 2 --enable-dp-attention --swa-full-tokens-ratio 0.10`, giving **`32` independent `attn_tp=1` attention/KV DP ranks** (`3,638,272` KV tokens/rank = **`116.42M` KV tokens across the slice**) while keeping MoE `gmm_v2` sharded across both intra-package TensorCores (`TP=2`).
   - **Multi-Replica ZMQ TCP Port Isolation Fix ([`launch_host_replicas.py`](../v7x/scripts/launch_host_replicas.py))**: Under `--enable-dp-attention`, SGLang's `PortArgs.init_new` switches IPC from Unix sockets to 7 consecutive TCP ports `port + 233..239`; spacing intra-host `--port` by `20` (`30000, 30020, 30040, 30060`) + disjoint `--nccl-port` eliminates `ZMQError: Address already in use` across all 4 engines per host.
   - **Empirical `v7x-16` Gains**:
     - **`4K:1K, 95%` cache**: **`39,520.7` out tok/s (`197,603.3` tot tok/s)** at `c=1024` and **`37,321.6` out tok/s (`186,607.8` tot tok/s, `0` errors, `629.4 ms` p50 TTFT)** at `c=768` (**`+18.3%`** peak / **`+11.7%`** `0`-error over prior `33,421.5` tok/s record).
     - **`4K:1K, 0%` uncached**: **`34,317.3` out tok/s (`171,586.4` tot tok/s, `0` errors)** at `c=768` (**`+14.7%`** over prior `29,923.9` tok/s uncached record — surpassing the previous `95%`-cached record even with zero cache).
     - **`8K:1K, 95%` cache**: **`39,035.7` out tok/s (`351,321.0` tot tok/s)** at `c=1024`, **`35,969.7` out tok/s** at `c=768`, and **`30,995.4` out tok/s** (`0` errors) at `c=512` (**`+24.2%`** peak over prior `31,432.8` tok/s).
     - **`64K–128K` Long-Context Crossover (`DP-Attention=32` vs `TP=2 × DP=16`)**: At `64K:1K` (`3,310.3` / `3,749.8` / `4,561.6` out tok/s at `0%` / `50%` / `95%`) and `128K:1K` (`1,638.6` / `1,778.0` / `2,411.7` out tok/s at `0%` / `50%` / `95%`), standard intra-chip `TP=2 × DP=16` (`attn_tp=2`: `7,225.0` tok/s at `64K` and `5,172.5` tok/s at `128K`, `631.8 ms` `c=1` TTFT) outperforms `DP-Attention=32` (`attn_tp=1`) by **`1.58×–2.14×`** because `attn_tp=2` shards `64K–128K` prefill across both intra-package TensorCores and keeps a unified per-chip RadixCache instead of splitting HTTP worker traffic across 2 internal DP ranks.
2. **`BF16 TP=2 × DP=8` (`mrr=96, swa=0.10`) vs `FP8 TP=1 × DP=16` (`mrr=64, swa=0.10`) on TPU `v5p-16` ([`v5p16_slice16_opt_r2_slice16_pod0.jsonl`](../v5p/results/raw/v5p16_slice16_opt_r2_slice16_pod0.jsonl))**:
   - Raising `--max-running-requests` and `--torch-compile-max-bs` to `96` per `TP=2` replica (`768` active decode slots across `8` replicas) with `--swa-full-tokens-ratio 0.10` (`42.29M` full KV tokens across the slice) + `256`-expert fused Pallas `router_topk` drives **empirical 16-chip `v5p-16` output throughput to `18,940.7` out tok/s (`94,703.6` tot tok/s, `1536/1536` ok)** at `4K:1K, 95%, c=768` (**`+85.4%`** over prior `10,216.2` tok/s empirical peak), **`15,016.0` out tok/s** at `0%` uncached, **`13,689.2` out tok/s** at `8K:1K, 95%, c=512`, **`3,511.0` out tok/s** at `64K:1K, 95%, c=64`, and **`1,867.8` out tok/s** at `128K:1K, 95%, c=32`.
   - Under identical `c=768` high-concurrency load, `slice16_fp8_tp1_dp16_mrr64_swa010` (`16` single-chip `Block-FP8 TP=1` replicas, `40.40M` full KV tokens) reaches **`9,024.6` out tok/s** (`4K, 95%`), **`7,877.8` out tok/s** (`4K, 0%`), **`7,292.2` out tok/s** (`8K, 95%`), **`2,049.2` out tok/s** (`64K, 95%`), and **`1,369.8` out tok/s** (`128K, 95%`) — `+74.7%` higher than stock `mrr=32` `FP8 TP=1`, while confirming `BF16 TP=2 × DP=8` is **`2.10×` faster** on `v5p` because `v5p` MXUs execute native `BF16` without hardware `FP8` tensor cores.

---

## 4. Cross-Engine Pareto Summary (`sglang-torchtpu` vs `vllm-torchtpu`)

| Operational Objective | Recommended Stack & Topology | TPU `v7x-16` (`2x2x4`) | TPU `v5p-16` (`2x2x4`) | Why It Wins |
|---|---|---|---|---|
| **Maximum High-Concurrency Slice Throughput (`c=768–1024`)** | **`sglang-torchtpu` Intra-Chip `TP=2 × DP-Attn=32` (`v7x`) / `TP=2 × DP=8, mrr=96` (`v5p`)** | **`39,520.7` out tok/s** *(`4K`, `c=1024`)* / **`37,321.6`** *(`c=768`, `0` err)*; **`39,035.7`** *(`8K`, `c=1024`)*; **`34,317.3`** *(`4K` `0%` hit)* | **`18,940.7` out tok/s** *(`4K`, `c=768`, `1536/1536` ok)* / **`15,016.0`** *(`0%`)*; **`13,689.2`** *(`8K`)*; **`3,511.0`** *(`64K`)*; **`1,867.8`** *(`128K`)* | 32 attention DP ranks (`v7x`) / `mrr=96` compiled batching (`v5p`) + zero inter-host ICI + `swa_ratio=0.10` hybrid KV pool |
| **Low-Latency Interactive & `64K–128K` Long Context** | **`sglang-torchtpu` Intra-Chip `TP=2 × DP=16`** (`swa_ratio=0.10`, fused Pallas `router_topk`) | **`33,421.5` out tok/s** (`BF16`) / **`33,200.3`** (`FP8`); **`7,225.0`** (`64K`), **`5,172.5`** (`128K`); **`6.49 ms` TPOT** | **`7.23 ms` TPOT** (`BF16 TP=4×DP=4`) / **`7.48 ms` TPOT** (`BF16 TP=2×DP=8`) | Fastest `c=1..128` TPOT (`6.49–7.48 ms`) + `117.8M` (`v7x`) / `42.3M` (`v5p`) slice KV tokens |
| **Single Native OpenAI Endpoint (`0` Shims / Gateways) & `TP=1` BF16** | **`vllm-torchtpu` Multi-Host Ray `TP=1 × DP × EP`** (`vllm serve --enable-expert-parallel`) | **`27,091.9` out tok/s (`130,432.8` tot tok/s)** (`BF16 TP=1×DP=32×EP=32`); **`10.37 ms` TPOT** (`TP=2`) | **`9,470.7` out tok/s (`72,382.2` tot tok/s @ `8K`)** (`BF16 TP=1×DP=16×EP=16`); **`12.34 ms` TPOT** | Native single-process `vllm serve` + Ray DPLB across 4 hosts; `EP` shrinks per-device experts `16–32×` so pure `BF16` runs at `TP=1` |
| **`TP=1 × DP=16` Quantized Serving on `v5p-16`** | **`vllm-torchtpu` `W8A16 TP=1 × DP=16 × EP=16`** | — | **`9,018.3` out tok/s** (`4K`) / **`7,501.5` out tok/s** (`8K`, **`+45.2%`** over non-EP `5,166.3`) | `EP=16` cuts software W8A16 expert dequantization overhead `16×` per chip |
