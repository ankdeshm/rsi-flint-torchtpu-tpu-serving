# Single-Host Optimization Matrix (`Axes A–E`) & Static-Shape Compiled Multi-LoRA Serving (`TPU v7x` & `TPU v5p`)

Comprehensive single-host parallelism (`TP=1..8`, `DP=1..8`, `EP=8`), weight & KV-cache precision (`Block-FP8` vs `BF16`, `FP8` vs `BF16` KV), Pallas `router_topk` kernel, `RadixCache` (`On` vs `Off`), hybrid `swa_ratio` (`0.05` vs `0.10`), speculative-decoding capability, `SGLang-TorchTPU` vs `vLLM-TorchTPU` engine comparison, and **static-shape XLA-compiled Multi-LoRA / QLoRA-style serving** study for **`RecursiveSI/flint` (`39.8B` total / `2.52B` active `256`-expert MoE)** across:
- **Single-Host TPU `v7x` (`4` Dual-Core Chips = `8` TensorCores, `768 GB` HBM):** [`multilora_singlehost_opt/scripts/run_v7x_sglang_worker.py`](../multilora_singlehost_opt/scripts/run_v7x_sglang_worker.py) & [`multilora_singlehost_opt/v7x/`](../multilora_singlehost_opt/v7x/)
- **4-Host TPU `v5p` Wave B (`4 × 4`-Chip Hosts = `16` `v5p` Chips, `95 GB` HBM/Chip):** [`multilora_singlehost_opt/scripts/run_v5p_wave_b_worker.py`](../multilora_singlehost_opt/scripts/run_v5p_wave_b_worker.py) & [`multilora_singlehost_opt/v5p/`](../multilora_singlehost_opt/v5p/)
- **Static-Shape TPU Multi-LoRA Compiler Patch (`TorchNativeLoRABackend`):** [`multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py`](../multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) & [`multilora_singlehost_opt/scripts/bench_flint_multilora_sglang.py`](../multilora_singlehost_opt/scripts/bench_flint_multilora_sglang.py)
- **Companion LoRA & QLoRA Fine-Tuning Repository (`TorchTitan` on `TPU v5p-16`):** [`ankdeshm/rsi-flint-tpu-lora-qlora-finetuning`](https://github.com/ankdeshm/rsi-flint-tpu-lora-qlora-finetuning) (`11` runs across `BF16 LoRA`, `INT8 QLoRA`, `NF4 QLoRA`, `r=64`, `attn-only`, and `16K/32K/64K` `PallasCrossEntropyLoss` producing the served Hugging Face `PEFT` adapters).

---

## 1. TPU `v7x` Single-Host (`4` Chips / `8` TensorCores) Optimization Matrix (`Axes A–D`)

| Config ID | Status | Points | `c=1` `TPOT p50` | `1K` Peak Host (`4`-Host Equiv) | `4K` Peak Host | `64K` Peak Host | `128K` Peak Host | JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`A1_fp8_tp2_dp4`** | `completed` | `36` | `6.49 ms` (`154.1` tok/s) | **`8,657.5`** (`34,629.9`) | **`8,763.6` tok/s** | `3,289.4` tok/s | `2,078.1` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/A1_fp8_tp2_dp4.json) |
| **`A2_fp8_tp1_dp8`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/A2_fp8_tp1_dp8.json) |
| **`A3_fp8_tp4_dp2`** | `completed` | `28` | `6.81 ms` (`146.8` tok/s) | **`6,679.9`** (`26,719.7`) | **`6,534.4` tok/s** | `2,496.9` tok/s | `1,487.8` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/A3_fp8_tp4_dp2.json) |
| **`A4_fp8_tp8_dp1`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/A4_fp8_tp8_dp1.json) |
| **`A5_fp8_tp8_ep8`** | `completed` | `22` | `7.29 ms` (`137.3` tok/s) | **`4,678.1`** (`18,712.6`) | **`4,665.7` tok/s** | `1,517.2` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/A5_fp8_tp8_ep8.json) |
| **`A6_fp8_dpattn8_ep8`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/A6_fp8_dpattn8_ep8.json) |
| **`B1_fp8_tp2_dp4_ngram4`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/B1_fp8_tp2_dp4_ngram4.json) |
| **`B2_fp8_tp8_dp1_ngram6`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/B2_fp8_tp8_dp1_ngram6.json) |
| **`B2_fp8_tp8_ep8_ngram6`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/B2_fp8_tp8_ep8_ngram6.json) |
| **`C1_bf16_tp2_dp4_fp8kv`** | `completed` | `26` | `6.63 ms` (`150.9` tok/s) | **`8,996.8`** (`35,987.3`) | **`8,853.4` tok/s** | `3,338.3` tok/s | `1,313.2` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/C1_bf16_tp2_dp4_fp8kv.json) |
| **`C2_bf16_tp2_dp4_bf16kv`** | `completed` | `22` | `6.51 ms` (`153.6` tok/s) | **`9,534.1`** (`38,136.3`) | **`8,815.1` tok/s** | `2,216.8` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/C2_bf16_tp2_dp4_bf16kv.json) |
| **`C3_fp8_tp2_dp4_bf16kv`** | `completed` | `22` | `6.60 ms` (`151.4` tok/s) | **`9,381.1`** (`37,524.6`) | **`8,432.9` tok/s** | `2,789.3` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/C3_fp8_tp2_dp4_bf16kv.json) |
| **`C4_fp8_tp2_dp4_fused_rs`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/C4_fp8_tp2_dp4_fused_rs.json) |
| **`C5_w8a8_int8_tp2_dp4`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/C5_w8a8_int8_tp2_dp4.json) |
| **`D1_fp8_tp2_dp4_noradix`** | `completed` | `14` | `6.48 ms` (`154.2` tok/s) | — | **`7,617.0` tok/s** | `901.3` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/D1_fp8_tp2_dp4_noradix.json) |
| **`D2_fp8_tp2_dp4_swa005`** | `completed` | `17` | `6.72 ms` (`148.9` tok/s) | — | **`8,553.4` tok/s** | `3,429.6` tok/s | `2,073.4` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/D2_fp8_tp2_dp4_swa005.json) |
| **`D3_fp8_tp2_dp4_swa020`** | `completed` | `13` | `6.79 ms` (`147.3` tok/s) | — | **`8,579.0` tok/s** | `3,281.0` tok/s | `1,405.4` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/D3_fp8_tp2_dp4_swa020.json) |
| **`D4_fp8_tp2_dp4_cp16k_mrr128`** | `unsupported` | `0` | — | — | — | — | — | [`JSON`](../multilora_singlehost_opt/v7x/D4_fp8_tp2_dp4_cp16k_mrr128.json) |
| **`D5_fp8_tp2_dp4_cp4k_mrr32`** | `completed` | `14` | `6.59 ms` (`151.7` tok/s) | — | **`8,528.9` tok/s** | `2,500.7` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/D5_fp8_tp2_dp4_cp4k_mrr32.json) |

### Key Single-Host `v7x` Findings (`Axes A–D`)
1. **Peak Single-Host Throughput (`9,534.1 tok/s/host` = `38,136.3 tok/s` `4`-Host Equiv):** `C2_bf16_tp2_dp4_bf16kv` reaches **`9,534.1 tok/s/host`** at `1K` (`hit=95%`, `c=128`) and **`8,815.1 tok/s/host`** at `4K` (`6.51 ms` `1`-stream `TPOT`, `152.8 tok/s`), while `C1_bf16_tp2_dp4_fp8kv` sustains **`3,338.3 tok/s/host`** at `64K` and `A1_fp8_tp2_dp4` sustains **`2,078.1 tok/s/host`** at `128K` (`6.45 ms` `TPOT`).
2. **Unlocking `TP=8 × EP=8` (`A5_fp8_tp8_ep8`) Halves `64K` TTFT:** Pure `TP=8` (`A4_fp8_tp8_dp1`) hits the `512 / 8 = 64 < 128` Block-FP8 `block_n` divisibility constraint on routed experts; enabling `ep_size=8` (`A5_fp8_tp8_ep8`) keeps full `[512, 2048]` expert matrices per rank (`32` experts/TC) and delivers **`1,225.3 ms` uncached `64K` TTFT (`2.06×` faster than `TP=2×DP=4`)** and **`135.8 ms` cached `64K` TTFT (`1.81×` faster)** at `4,678.2 tok/s/host`.
3. **RadixCache Ablation (`A1` vs `D1`):** Prefix caching on `TPU v7x` yields **`10.3×` lower `64K` TTFT (`246 ms` vs `2,526 ms`)** and **`3.65×` higher `64K` output throughput (`3,289` vs `901 tok/s/host`)**.
4. **Fused VMEM Pallas `router_topk` (`A1` vs `D3`):** Fused `256`-expert Pallas `router_topk` outperforms stock PyTorch `topk` across `47` MoE layers (`8,763.6` vs `8,274.2 tok/s/host` at `4K`, `+5.9%`).

---

## 2. Static-Shape XLA-Compiled Multi-LoRA & QLoRA-Style Serving on TPU `v7x` (`Axis E1 & E2`)

[`patch_sglang_tpu_lora.py`](../multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) equips `SGLang`'s `TorchNativeLoRABackend` on TPU with persistent static device buffers (`_tpu_weight_indices` initialized to `-1` for base/padding tokens and `_tpu_scalings`), `@torch.compiler.assume_constant_result` slice offsets, and `pin_memory=False` offset tensors, compiling `apply_lora` directly into every `EXTEND` and `DECODE` XLA bucket (`--lora-backend torch_native --max-loras-per-batch 5 --max-lora-rank 16`) with **zero runtime graph breaks or recompilations** across arbitrary mixes of `0..4` concurrent adapters (`lora_bf16`, `qlora_int8`, `qlora_nf4`, `lora_attn`).

### `E1_sglang_fp8_multilora` (`fp8_block128` Base + `FP8` KV, `TP=2 × DP=4`) — Dynamic hot-swap: `13.3 ms` unload / `24.8 ms` reload (`0` errors)

| Concurrent Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Slice Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `95.5` tok/s (`10.47 ms`) | **`3,984.6`** (`15,938.5` tok/s) | `3,487.7` tok/s | `434.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `88.2` tok/s (`11.34 ms`) | **`4,133.8`** (`16,535.4` tok/s) | `3,590.7` tok/s | `436.9` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `90.9` tok/s (`11.00 ms`) | **`4,322.8`** (`17,291.4` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `91.3` tok/s (`10.95 ms`) | **`3,634.9`** (`14,539.8` tok/s) | `3,001.4` tok/s | `365.2` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `92.3` tok/s (`10.84 ms`) | **`4,352.3`** (`17,409.3` tok/s) | — | — | `0` |

### `E2_sglang_bf16_multilora` (`bfloat16` Base + `FP8` KV, `TP=2 × DP=4`) — Dynamic hot-swap: `12.4 ms` unload / `24.4 ms` reload (`0` errors)

| Concurrent Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Slice Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `87.7` tok/s (`11.40 ms`) | **`4,120.3`** (`16,481.2` tok/s) | `3,603.9` tok/s | `431.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `87.5` tok/s (`11.43 ms`) | **`4,293.8`** (`17,175.2` tok/s) | `3,573.5` tok/s | `440.0` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `87.7` tok/s (`11.40 ms`) | **`4,175.2`** (`16,700.7` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `90.6` tok/s (`11.03 ms`) | **`3,633.6`** (`14,534.2` tok/s) | `3,055.4` tok/s | `365.4` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `91.2` tok/s (`10.96 ms`) | **`4,162.7`** (`16,650.9` tok/s) | — | — | `0` |

---

## 3. TPU `v5p` Wave B: Compiled Multi-LoRA (`Block-FP8` & `BF16`), `TP=1..4` Scaling, and `128K` RadixCache / `swa_ratio`

### Pod `0`: `v5p_fp8_block128_multilora` (`fp8_block128` Base + `FP8` KV, `TP=4 × DP=1`, `--lora-backend torch_native`) — Dynamic hot-swap: `22 ms` unload / `26 ms` reload

| Concurrent Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Slice Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `66.2` tok/s (`15.12 ms`) | **`604.5`** (`2,418.1` tok/s) | `728.7` tok/s | `244.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `69.1` tok/s (`14.48 ms`) | **`748.0`** (`2,991.8` tok/s) | `952.0` tok/s | `263.4` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `69.0` tok/s (`14.49 ms`) | **`803.7`** (`3,214.8` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `67.3` tok/s (`14.87 ms`) | **`741.5`** (`2,966.2` tok/s) | `860.5` tok/s | `130.6` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `70.3` tok/s (`14.22 ms`) | **`679.7`** (`2,718.6` tok/s) | — | — | `0` |

### Pod `1`: `v5p_bfloat16_multilora` (`bfloat16` Base + `FP8` KV, `TP=4 × DP=1`, `--lora-backend torch_native`) — Dynamic hot-swap: `20 ms` unload / `28 ms` reload

| Concurrent Adapter Schedule | `1K` `c=1` (`TPOT p50`) | `1K` Peak Host (`4`-Host Slice Equiv) | `4K` Peak Host | `64K` Peak Host | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `69.6` tok/s (`14.37 ms`) | **`1,529.1`** (`6,116.4` tok/s) | `1,620.0` tok/s | `385.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `67.0` tok/s (`14.92 ms`) | **`1,540.9`** (`6,163.6` tok/s) | `1,647.6` tok/s | `393.1` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8 QLoRA`)* | `68.3` tok/s (`14.63 ms`) | **`1,655.9`** (`6,623.6` tok/s) | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `68.7` tok/s (`14.55 ms`) | **`1,632.1`** (`6,528.5` tok/s) | `1,580.2` tok/s | `231.8` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `68.7` tok/s (`14.55 ms`) | **`1,577.8`** (`6,311.4` tok/s) | — | — | `0` |

### Pod `2`: Single-Host `v5p` `TP × DP` Parallelism (`TP=1×DP=4`, `TP=2×DP=2`, `TP=4×DP=1`) & SpecDec Probe

| Config | Status | `c=1` `TPOT p50` | `1K` Peak Host (`16`-Chip Equiv) | `4K` (`95%` Hit) Peak Host | `64K` (`95%` Hit) Peak Host | `128K` (`95%` Hit) Peak Host |
|---|---|---:|---:|---:|---:|---:|
| **`v5p_fp8_tp4_ngram4_probe`** | `unsupported` | — | — | — | — | — |
| **`v5p_fp8_tp1_dp4`** | `completed` | `12.87 ms` (`77.7` tok/s) | **`1,277.3`** (`5,109.4`) | **`1,250.4` tok/s** | `316.3` tok/s | — |
| **`v5p_fp8_tp2_dp2`** | `unsupported_or_failed` | — | — | — | — | — |
| **`v5p_bf16_tp2_dp2`** | `completed` | `7.43 ms` (`134.6` tok/s) | **`2,712.9`** (`10,851.4`) | **`2,975.1` tok/s** | `587.6` tok/s | — |
| **`v5p_bf16_tp4_dp1`** | `completed` | `7.22 ms` (`138.5` tok/s) | **`2,720.9`** (`10,883.7`) | **`3,026.9` tok/s** | `687.3` tok/s | — |

### Pod `3`: `v5p` RadixCache (`On` vs `Off`), `swa_ratio` (`0.05` vs `0.10`), & `FP8` vs `BF16` KV (`4K..128K`)

| Config | Status | `c=1` `TPOT p50` | `1K` Peak Host (`16`-Chip Equiv) | `4K` (`95%` Hit) Peak Host | `64K` (`95%` Hit) Peak Host | `128K` (`95%` Hit) Peak Host |
|---|---|---:|---:|---:|---:|---:|
| **`v5p_fp8_tp4_radix_on_swa010`** | `completed` | `9.75 ms` (`102.6` tok/s) | — | **`761.3` tok/s** | `344.9` tok/s | `242.5` tok/s |
| **`v5p_fp8_tp4_radix_off_swa010`** | `completed` | `9.73 ms` (`102.8` tok/s) | — | **`555.3` tok/s** | `79.5` tok/s | `37.6` tok/s |
| **`v5p_fp8_tp4_swa005_longctx`** | `completed` | `9.74 ms` (`102.6` tok/s) | — | **`783.7` tok/s** | `345.7` tok/s | `242.9` tok/s |
| **`v5p_fp8_tp4_bf16kv_longctx`** | `completed` | `9.62 ms` (`104.0` tok/s) | — | **`778.1` tok/s** | `347.3` tok/s | `266.3` tok/s |

---

## 4. Empirical Single-Host Engine Comparison (`Axis E3`): `SGLang-TorchTPU` vs `vLLM-TorchTPU`

- **Result Archive:** [`multilora_singlehost_opt/v7x/E3_vllm_torchtpu_flint_comparison.json`](../multilora_singlehost_opt/v7x/E3_vllm_torchtpu_flint_comparison.json) (scripts: [`bench_flint_vllm_rqkr.py`](../multilora_singlehost_opt/scripts/bench_flint_vllm_rqkr.py), [`install_flint_vllm.py`](../multilora_singlehost_opt/scripts/install_flint_vllm.py))
- **Stock Upstream `vllm-torchtpu` (`v0.32.0+empty`):** `FlintForCausalLM` (`model_type='flint'`) is unregistered in `ModelRegistry` (`391` registered architectures, `flint_for_causal_lm_registered=False`); `--model-impl auto` -> `exited`, `--model-impl transformers` -> `exited`.
- **Plugin-Enabled `vllm-torchtpu` Multi-LoRA & Sub-Slice Constraints:** With `FlintConfig` + `FlintForCausalLM` installed, `VllmConfig` explicitly blocks TPU Multi-LoRA on both `vllm_fp8_tp2_dp4_ep8_lora_probe` and `vllm_fp8_tp1_dp8_ep8_multilora` (`ValidationError: LoRA on TPU does not support tensor_parallel_size > 1; data_parallel_size > 1; MoE models`), and single-host `TP=4 × DP=2 × EP=8` sub-slice launch on a physical `2x2x2` ICI pool without `LD_PRELOAD` metadata isolation exits during `EngineCore` mesh startup (whereas full-slice `4`-host Ray `EP=32` / `EP=16` base & `ngram5` serving succeeds at scale in the multi-slice `vLLM` suite). Consequently, `SGLang-TorchTPU` + [`patch_sglang_tpu_lora.py`](../multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) + `metadata_hook.so` is uniquely capable of both **isolated sub-slice single-host deployment** and **compiled multi-tenant LoRA/QLoRA serving on `RecursiveSI/flint` MoE**.
