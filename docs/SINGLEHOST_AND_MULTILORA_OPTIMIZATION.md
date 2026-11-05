# Single-Host Optimization Matrix (`Axes A–E`) & Static-Shape Compiled Multi-LoRA Serving (`TPU v7x` & `TPU v5p`)

Single-host parallelism (`TP=1..8`, `DP=1..4`, `EP=8`), weight and KV-cache precision (`Block-FP8` vs `BF16`, `FP8` vs `BF16` KV), Pallas `router_topk` kernel, `RadixCache` (`On` vs `Off`), hybrid `swa_ratio` (`0.05` vs `0.10` vs `0.20`), `SGLang-TorchTPU` vs `vLLM-TorchTPU` single-host engine comparison, and **static-shape XLA-compiled Multi-LoRA serving** for **`RecursiveSI/flint` (`39.8B` total / `2.52B` active `256`-expert MoE)** across:
- **Single-Host TPU `v7x` (`4` Dual-Core Chips = `8` TensorCores, `768 GB` HBM):** [`multilora_singlehost_opt/scripts/run_v7x_sglang_worker.py`](../multilora_singlehost_opt/scripts/run_v7x_sglang_worker.py) & [`multilora_singlehost_opt/v7x/`](../multilora_singlehost_opt/v7x/)
- **Single-Host TPU `v5p` Pods (`4` Chips/Host, `95 GB` HBM/Chip):** [`multilora_singlehost_opt/scripts/run_v5p_wave_b_worker.py`](../multilora_singlehost_opt/scripts/run_v5p_wave_b_worker.py) & [`multilora_singlehost_opt/v5p/`](../multilora_singlehost_opt/v5p/)
- **Static-Shape TPU Multi-LoRA Compiler Integration (`TorchNativeLoRABackend`):** [`multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py`](../multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) & [`multilora_singlehost_opt/scripts/bench_flint_multilora_sglang.py`](../multilora_singlehost_opt/scripts/bench_flint_multilora_sglang.py)
- **Companion LoRA & QLoRA Fine-Tuning Repository (`TorchTitan` on `TPU v5p-16`):** [`ankdeshm/rsi-flint-tpu-lora-qlora-finetuning`](https://github.com/ankdeshm/rsi-flint-tpu-lora-qlora-finetuning) (produces the Hugging Face `PEFT` adapters served below).

All throughput figures in Sections 1–3 are **directly measured single-host (`4`-chip) values** (`0` multi-host extrapolations). Configurations that are unsupported or failed validation (`0` measured points) are documented separately in [Section 5](#5-supported-configurations--known-limitations).

---

## 1. TPU `v7x` Single-Host (`4` Chips / `8` TensorCores) Measured Matrix (`Axes A, C, D`)

| Config ID | Precision & Topology | Points | `c=1` `TPOT p50` | `1K:512` (`95%` Hit, `c=128`) | `4K:1K` (`95%` Hit, `c=128`) | `64K:1K` (`95%` Hit, `c=16`) | `128K:1K` (`95%` Hit, `c=8`) | JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`A1_fp8_tp2_dp4`** | `Block-FP8`, `TP=2 × DP=4`, `FP8` KV, `swa=0.10` | `36` | `6.49 ms` (`154.1` tok/s) | **`8,657.5` tok/s** | **`8,763.6` tok/s** | `3,289.4` tok/s | **`2,078.1` tok/s** | [`JSON`](../multilora_singlehost_opt/v7x/A1_fp8_tp2_dp4.json) |
| **`A3_fp8_tp4_dp2`** | `Block-FP8`, `TP=4 × DP=2`, `FP8` KV, `swa=0.10` | `28` | `6.81 ms` (`146.8` tok/s) | **`6,679.9` tok/s** | **`6,534.4` tok/s** | `2,496.9` tok/s | `1,487.8` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/A3_fp8_tp4_dp2.json) |
| **`A5_fp8_tp8_ep8`** | `Block-FP8`, `TP=8 × EP=8`, `FP8` KV, `swa=0.10` | `22` | `7.29 ms` (`137.3` tok/s) | **`4,678.1` tok/s** | **`4,665.7` tok/s** | `1,517.2` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/A5_fp8_tp8_ep8.json) |
| **`C1_bf16_tp2_dp4_fp8kv`** | `BF16`, `TP=2 × DP=4`, `FP8` KV, `swa=0.10` | `26` | `6.63 ms` (`150.9` tok/s) | **`8,996.8` tok/s** | **`8,853.4` tok/s** | `3,338.3` tok/s | `1,313.2` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/C1_bf16_tp2_dp4_fp8kv.json) |
| **`C2_bf16_tp2_dp4_bf16kv`** | `BF16`, `TP=2 × DP=4`, `BF16` KV, `swa=0.10` | `22` | `6.51 ms` (`153.6` tok/s) | **`9,534.1` tok/s** | **`8,815.1` tok/s** | `2,216.8` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/C2_bf16_tp2_dp4_bf16kv.json) |
| **`C3_fp8_tp2_dp4_bf16kv`** | `Block-FP8`, `TP=2 × DP=4`, `BF16` KV, `swa=0.10` | `22` | `6.60 ms` (`151.4` tok/s) | **`9,381.1` tok/s** | **`8,432.9` tok/s** | `2,789.3` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/C3_fp8_tp2_dp4_bf16kv.json) |
| **`D1_fp8_tp2_dp4_noradix`** | `Block-FP8`, `TP=2 × DP=4`, `--disable-radix-cache` | `14` | `6.48 ms` (`154.2` tok/s) | — | **`7,617.0` tok/s** | `901.3` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/D1_fp8_tp2_dp4_noradix.json) |
| **`D2_fp8_tp2_dp4_swa005`** | `Block-FP8`, `TP=2 × DP=4`, `swa_ratio=0.05` | `17` | `6.72 ms` (`148.9` tok/s) | — | **`8,553.4` tok/s** | **`3,429.6` tok/s** | `2,073.4` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/D2_fp8_tp2_dp4_swa005.json) |
| **`D3_fp8_tp2_dp4_swa020`** | `Block-FP8`, `TP=2 × DP=4`, `swa_ratio=0.20` | `13` | `6.79 ms` (`147.3` tok/s) | — | **`8,579.0` tok/s** | `3,281.0` tok/s | `1,405.4` tok/s | [`JSON`](../multilora_singlehost_opt/v7x/D3_fp8_tp2_dp4_swa020.json) |
| **`D5_fp8_tp2_dp4_cp4k_mrr32`** | `Block-FP8`, `TP=2 × DP=4`, `cp=4096, mrr=32` | `14` | `6.59 ms` (`151.7` tok/s) | — | **`8,528.9` tok/s** | `2,500.7` tok/s | — | [`JSON`](../multilora_singlehost_opt/v7x/D5_fp8_tp2_dp4_cp4k_mrr32.json) |

### Key Single-Host `v7x` Findings
1. **Single-Host Throughput & Context Capacity (`TP=2 × DP=4`)**: `C2_bf16_tp2_dp4_bf16kv` reaches **`9,534.1` tok/s/host** at `1K:512` (`95%` target hit, `c=128`) and **`8,815.1` tok/s/host** at `4K:1K` (`6.51 ms` `c=1` `TPOT`), while `C1_bf16_tp2_dp4_fp8kv` reaches **`8,853.4` tok/s/host** at `4K:1K` and **`3,338.3` tok/s/host** at `64K:1K`, and `A1_fp8_tp2_dp4` / `D2_fp8_tp2_dp4_swa005` sustain **`3,429.6` tok/s/host** at `64K:1K` and **`2,078.1` tok/s/host** at `128K:1K`.
2. **`TP=8 × EP=8` (`A5_fp8_tp8_ep8`) Long-Context Prefill Latency**: Enabling `ep_size=8` alongside `tp_size=8` keeps full `[512, 2048]` expert matrices per rank (`32` experts/TC, satisfying `[128, 128]` Block-FP8 block alignment) and lowers single-request `64K` `TTFT` to **`1,225.3 ms` uncached (`2.06×` lower than `TP=2×DP=4`)** and **`135.8 ms` cached (`1.81×` lower)** at `4,678.1` tok/s/host peak output.
3. **RadixCache Ablation (`A1` vs `D1`)**: Prefix caching on `TPU v7x` reduces `64K` `TTFT` (`95%` target prefix) by **`10.3×` (`246 ms` vs `2,526 ms`)** and increases `64K` host output throughput by **`3.65×` (`3,289.4` vs `901.3` tok/s/host)**.

---

## 2. Static-Shape XLA-Compiled Multi-LoRA Serving on TPU `v7x` (`Axis E1 & E2`)

[`patch_sglang_tpu_lora.py`](../multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) extends `SGLang`'s `TorchNativeLoRABackend` on TPU with persistent static device buffers (`_tpu_weight_indices` initialized to `-1` for base/padding tokens and `_tpu_scalings`), `@torch.compiler.assume_constant_result` slice offsets, and `pin_memory=False` offset tensors, compiling `apply_lora` into `EXTEND` and `DECODE` XLA buckets (`--lora-backend torch_native --max-loras-per-batch 5 --max-lora-rank 16`) with zero runtime recompilations across mixes of `0..4` concurrent adapters (`lora_bf16`, `qlora_int8`, `qlora_nf4`, `lora_attn`; archive: [`E_sglang_flint_multilora.json`](../multilora_singlehost_opt/v7x/E_sglang_flint_multilora.json)).

> **Adapter & Measurement Scope (`E1`, `E2`, and `v5p` Pods `0–1`):**
> - **Adapter Architecture & Base Precision**: All four served adapters (`lora_bf16`, `qlora_int8`, `qlora_nf4`, `lora_attn`) are attention-only `r=16` (`q_proj`, `k_proj`, `v_proj`, `o_proj`) PEFT adapters. The `qlora_int8` and `qlora_nf4` adapters were trained against an `INT8` or `NF4` frozen base in TorchTitan and are served here in `BF16` adapter precision on top of the `Block-FP8` (`E1`, `v5p` Pod `0`) or `BF16` (`E2`, `v5p` Pod `1`) base checkpoint (not an `INT8`/`NF4`-quantized serving base).
> - **Single-Wave `c=64` Variance**: On `v7x`, the `1K:512` and `4K:512` `c=64` points send a single wave (`n = c = 64`, `7.5–9.6 s` wall time); `1–3`-adapter throughputs slightly above the `0`-LoRA row (`4,133.8–4,352.3` vs `3,984.6` tok/s/host on `FP8`; `4,162.7–4,293.8` vs `4,120.3` tok/s/host on `BF16`) are within single-wave run-to-run variance.
> - **Early EOS in `4_loras_all_mixed` (`*`)**: On `v7x`, requests in the `4_loras_all_mixed` schedule stopped early at **`75%–88%` of the requested `512` output tokens** (whereas `0..3`-adapter schedules generated `~100%`), reducing amortized decode tokens per prefill; rows marked `*` are therefore not directly throughput-comparable to the `512`-token rows.

### `E1_sglang_fp8_multilora` (`fp8_block128` Base + `FP8` KV, `TP=2 × DP=4`, 1 `v7x` Host) — Dynamic hot-swap: `13.3 ms` unload / `24.8 ms` reload (`0` errors)

| Concurrent Adapter Schedule | `1K:512` (`0%` Hit, `c=1, n=8`) `TPOT p50` | `1K:512` (`0%` Hit, `c=64, n=64`) Host tok/s | `4K:512` (`95%` Hit, `c=64, n=64`) Host tok/s | `64K:512` (`95%` Hit, `c=16, n=32`) Host tok/s | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `10.47 ms` (`95.5` tok/s) | **`3,984.6` tok/s** | `3,487.7` tok/s | `434.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `11.34 ms` (`88.2` tok/s) | **`4,133.8` tok/s** | `3,590.7` tok/s | `436.9` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8-trained QLoRA`)* | `11.00 ms` (`90.9` tok/s) | **`4,322.8` tok/s** | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)*\* | `10.95 ms` (`91.3` tok/s) | `3,634.9` tok/s\* | `3,001.4` tok/s\* | `365.2` tok/s\* | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `10.84 ms` (`92.3` tok/s) | **`4,352.3` tok/s** | — | — | `0` |

### `E2_sglang_bf16_multilora` (`bfloat16` Base + `FP8` KV, `TP=2 × DP=4`, 1 `v7x` Host) — Dynamic hot-swap: `12.4 ms` unload / `24.4 ms` reload (`0` errors)

| Concurrent Adapter Schedule | `1K:512` (`0%` Hit, `c=1, n=8`) `TPOT p50` | `1K:512` (`0%` Hit, `c=64, n=64`) Host tok/s | `4K:512` (`95%` Hit, `c=64, n=64`) Host tok/s | `64K:512` (`95%` Hit, `c=16, n=32`) Host tok/s | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `11.40 ms` (`87.7` tok/s) | **`4,120.3` tok/s** | `3,603.9` tok/s | `431.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `11.43 ms` (`87.5` tok/s) | **`4,293.8` tok/s** | `3,573.5` tok/s | `440.0` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8-trained QLoRA`)* | `11.40 ms` (`87.7` tok/s) | **`4,175.2` tok/s** | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)*\* | `11.03 ms` (`90.6` tok/s) | `3,633.6` tok/s\* | `3,055.4` tok/s\* | `365.4` tok/s\* | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `10.96 ms` (`91.2` tok/s) | **`4,162.7` tok/s** | — | — | `0` |

---

## 3. TPU `v5p` Single-Host (`4` Chips/Host) Measurements: Compiled Multi-LoRA, `TP=1..4`, & `128K` RadixCache

### Pod `0`: `v5p_fp8_block128_multilora` (`fp8_block128` Base + `FP8` KV, `TP=4 × DP=1`, 1 `v5p` Host) — Dynamic hot-swap: `22 ms` unload / `26 ms` reload ([`JSON`](../multilora_singlehost_opt/v5p/pod0_v5p_fp8_block128_multilora.json))

| Concurrent Adapter Schedule | `1K:512` (`0%` Hit, `c=1, n=8`) `TPOT p50` | `1K:512` (`0%` Hit, `c=32, n=64`) Host tok/s | `4K:512` (`95%` Hit, `c=32, n=64`) Host tok/s | `64K:512` (`95%` Hit, `c=8, n=8`) Host tok/s | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `15.12 ms` (`66.2` tok/s) | **`604.5` tok/s** | `728.7` tok/s | `244.4` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `14.48 ms` (`69.1` tok/s) | **`748.0` tok/s** | `952.0` tok/s | `263.4` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8-trained QLoRA`)* | `14.49 ms` (`69.0` tok/s) | **`803.7` tok/s** | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `14.87 ms` (`67.3` tok/s) | **`741.5` tok/s** | `860.5` tok/s | `130.6` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `14.22 ms` (`70.3` tok/s) | **`679.7` tok/s** | — | — | `0` |

### Pod `1`: `v5p_bfloat16_multilora` (`bfloat16` Base + `FP8` KV, `TP=4 × DP=1`, 1 `v5p` Host) — Dynamic hot-swap: `20 ms` unload / `28 ms` reload ([`JSON`](../multilora_singlehost_opt/v5p/pod1_v5p_bfloat16_multilora.json))

| Concurrent Adapter Schedule | `1K:512` (`0%` Hit, `c=1, n=8`) `TPOT p50` | `1K:512` (`0%` Hit, `c=32, n=64`) Host tok/s | `4K:512` (`95%` Hit, `c=32, n=64`) Host tok/s | `64K:512` (`95%` Hit, `c=8, n=8`) Host tok/s | Failed |
|---|---:|---:|---:|---:|---:|
| `0` LoRAs *(Base Only)* | `14.37 ms` (`69.6` tok/s) | **`1,529.1` tok/s** | `1,620.0` tok/s | `385.3` tok/s | `0` |
| `1` LoRA *(`lora_bf16`)* | `14.92 ms` (`67.0` tok/s) | **`1,540.9` tok/s** | `1,647.6` tok/s | `393.1` tok/s | `0` |
| `2` Concurrent *(`BF16 + INT8-trained QLoRA`)* | `14.63 ms` (`68.3` tok/s) | **`1,655.9` tok/s** | — | — | `0` |
| `4` Concurrent *(`BF16 + INT8 + NF4 + Attn`)* | `14.55 ms` (`68.7` tok/s) | **`1,632.1` tok/s** | `1,580.2` tok/s | `231.8` tok/s | `0` |
| Mixed *(`Base + 3 LoRAs`)* | `14.55 ms` (`68.7` tok/s) | **`1,577.8` tok/s** | — | — | `0` |

### Pod `2`: Single-Host `v5p` `TP × DP` Parallelism (`4` Chips/Host, [`JSON`](../multilora_singlehost_opt/v5p/pod2_v5p_tp_dp_and_specdec.json))

| Config | Precision & Topology | `c=1` `TPOT p50` | `1K:512` (`0%` Hit, `c=64, n=128`) Host tok/s | `4K:512` (`95%` Hit, `c=64, n=128`) Host tok/s | `64K:512` (`95%` Hit, `c=16, n=16`) Host tok/s |
|---|---|---:|---:|---:|---:|
| **`v5p_fp8_tp1_dp4`** | `Block-FP8`, `TP=1 × DP=4` | `12.87 ms` (`77.7` tok/s) | **`1,277.3` tok/s** | **`1,250.4` tok/s** | `316.3` tok/s |
| **`v5p_bf16_tp2_dp2`** | `BF16`, `TP=2 × DP=2` | `7.43 ms` (`134.6` tok/s) | **`2,712.9` tok/s** | **`2,975.1` tok/s** | `587.6` tok/s |
| **`v5p_bf16_tp4_dp1`** | `BF16`, `TP=4 × DP=1` | **`7.22 ms`** (`138.5` tok/s) | **`2,720.9` tok/s** | **`3,026.9` tok/s** | **`687.3` tok/s** |

### Pod `3`: Single-Host `v5p` RadixCache (`On` vs `Off`), `swa_ratio`, & `FP8` vs `BF16` KV (`4` Chips/Host, [`JSON`](../multilora_singlehost_opt/v5p/pod3_v5p_radix_swa_longctx.json))

| Config | Configuration | `c=1` `TPOT p50` | `4K:512` (`95%` Hit, `c=32`) Host tok/s | `64K:512` (`95%` Hit, `c=8`) Host tok/s | `128K:512` (`95%` Hit, `c=4`) Host tok/s |
|---|---|---:|---:|---:|---:|
| **`v5p_fp8_tp4_radix_on_swa010`** | `Block-FP8`, `TP=4`, Radix `On`, `swa=0.10`, `FP8` KV | `9.75 ms` (`102.6` tok/s) | **`761.3` tok/s** | `344.9` tok/s | `242.5` tok/s |
| **`v5p_fp8_tp4_radix_off_swa010`** | `Block-FP8`, `TP=4`, Radix `Off`, `swa=0.10`, `FP8` KV | `9.73 ms` (`102.8` tok/s) | **`555.3` tok/s** | `79.5` tok/s | `37.6` tok/s |
| **`v5p_fp8_tp4_swa005_longctx`** | `Block-FP8`, `TP=4`, Radix `On`, `swa=0.05`, `FP8` KV | `9.74 ms` (`102.6` tok/s) | **`783.7` tok/s** | `345.7` tok/s | `242.9` tok/s |
| **`v5p_fp8_tp4_bf16kv_longctx`** | `Block-FP8`, `TP=4`, Radix `On`, `swa=0.10`, `BF16` KV | `9.62 ms` (`104.0` tok/s) | **`778.1` tok/s** | `347.3` tok/s | **`266.3` tok/s** |

---

## 4. Single-Host Engine Comparison (`Axis E3`): `SGLang-TorchTPU` vs `vLLM-TorchTPU`

- **Result Archive:** [`multilora_singlehost_opt/v7x/E3_vllm_torchtpu_flint_comparison.json`](../multilora_singlehost_opt/v7x/E3_vllm_torchtpu_flint_comparison.json) (scripts: [`bench_flint_vllm_singlehost.py`](../multilora_singlehost_opt/scripts/bench_flint_vllm_singlehost.py), [`install_flint_vllm.py`](../multilora_singlehost_opt/scripts/install_flint_vllm.py)).
- **Unpatched `vllm-torchtpu` (`v0.32.0+empty`)**: `FlintForCausalLM` (`model_type='flint'`) is not registered in `ModelRegistry` (`391` built-in architectures; both `--model-impl auto` and `--model-impl transformers` exit during config resolution unless [`install_flint_vllm.py`](../multilora_singlehost_opt/scripts/install_flint_vllm.py) is applied).
- **Plugin-Enabled `vllm-torchtpu` Multi-LoRA & Sub-Slice Constraints**: With `FlintConfig` + `FlintForCausalLM` installed, `VllmConfig` rejects TPU Multi-LoRA on MoE and `TP/DP > 1` (`ValidationError: LoRA on TPU does not support tensor_parallel_size > 1; data_parallel_size > 1; MoE models`), and launching a single-host `TP=4 × DP=2 × EP=8` mesh on a `2x2x2` sub-slice without `LD_PRELOAD` metadata isolation exits during `EngineCore` mesh startup (whereas 4-host Ray `EP=32` / `EP=16` base serving runs across full `2x2x4` slices in [`vllm/README.md`](../vllm/README.md)).

---

## 5. Supported Configurations & Known Limitations

The following single-host configurations were evaluated during `Axes A–D` and `v5p` Wave B and are **not supported** (`0` benchmark points):

| Config ID | Platform & Stack | Technical Cause | Diagnostic Archive |
|---|---|---|---|
| **`A2_fp8_tp1_dp8`** | `v7x` (`sglang-torchtpu`) | `torch_tpu` `get_tpu_topology(1)` raises `RuntimeError: No TPU topology found for count: 1` on dual-TensorCore TPU `v7x` chips (minimum single-engine world size in `sglang-torchtpu` is `2` TensorCores = `1` physical `v7x` chip; use `vllm-torchtpu` `TP=1 × DP=32 × EP=32` for `TP=1` on `v7x-16`) | [`A2_fp8_tp1_dp8.json`](../multilora_singlehost_opt/v7x/A2_fp8_tp1_dp8.json) |
| **`A4_fp8_tp8_dp1`** | `v7x` (`sglang-torchtpu`) | `[128, 128]` Block-FP8 requires expert shard dimension `>= 128`; `TP=8` without `EP` shards `moe_intermediate_size=512` to `512 / 8 = 64 < 128` (supported when `EP=8` is enabled in `A5_fp8_tp8_ep8`) | [`A4_fp8_tp8_dp1.json`](../multilora_singlehost_opt/v7x/A4_fp8_tp8_dp1.json) |
| **`A6_fp8_dpattn8_ep8`** | `v7x` (`sglang-torchtpu`) | Combining `--tp-size 8 --ep-size 8` with `--enable-dp-attention --dp-size 4` requests `32` TensorCores (`> 8` TensorCores available on 1 host) | [`A6_fp8_dpattn8_ep8.json`](../multilora_singlehost_opt/v7x/A6_fp8_dpattn8_ep8.json) |
| **`B1_fp8_tp2_dp4_ngram4`**, **`B2_fp8_tp8_dp1_ngram6`**, **`B2_fp8_tp8_ep8_ngram6`**, **`v5p_fp8_tp4_ngram4_probe`** | `v7x` & `v5p` (`sglang-torchtpu`) | `sglang_tpu` raises `NotImplementedError: Speculative algorithm 'NGRAM' is not supported on TPU yet (only EAGLE, EAGLE3, NEXTN and DFLASH among builtin algorithms)` | [`B1`](../multilora_singlehost_opt/v7x/B1_fp8_tp2_dp4_ngram4.json) · [`B2`](../multilora_singlehost_opt/v7x/B2_fp8_tp8_dp1_ngram6.json) · [`B2_ep8`](../multilora_singlehost_opt/v7x/B2_fp8_tp8_ep8_ngram6.json) · [`v5p_pod2`](../multilora_singlehost_opt/v5p/pod2_v5p_tp_dp_and_specdec.json) |
| **`C4_fp8_tp2_dp4_fused_rs`** | `v7x` (`sglang-torchtpu`) | `--moe-runner-backend pallas_tpu_fused_rs` is rejected by `flint_sglang` as unvalidated with Flint's top-6 sigmoid routing | [`C4_fp8_tp2_dp4_fused_rs.json`](../multilora_singlehost_opt/v7x/C4_fp8_tp2_dp4_fused_rs.json) |
| **`C5_w8a8_int8_tp2_dp4`** | `v7x` (`sglang-torchtpu`) | `--quantization w8a8_int8` dispatches Triton `per_token_quant_int8`, causing a `torch._dynamo` graph break/failure on TPU (`INT8 W8A16` is supported in `vllm-torchtpu` via `CompressedTensorsWNA16`) | [`C5_w8a8_int8_tp2_dp4.json`](../multilora_singlehost_opt/v7x/C5_w8a8_int8_tp2_dp4.json) |
| **`D4_fp8_tp2_dp4_cp16k_mrr128`** | `v7x` (`sglang-torchtpu`) | `chunked_prefill_size=16384` + `max_running_requests=128` exceeds TPU Scalar Memory (`SMEM`: `1,081,856 > 1,048,576` bytes) in `RPAd-p_64` during XLA compilation (`mrr=96, cp=8192` compiles and runs cleanly) | [`D4_fp8_tp2_dp4_cp16k_mrr128.json`](../multilora_singlehost_opt/v7x/D4_fp8_tp2_dp4_cp16k_mrr128.json) |
| **`v5p_fp8_tp2_dp2`** | `v5p` (`sglang-torchtpu`) | Dual-replica `Block-FP8 TP=2 × DP=2` timed out during simultaneous warmup/health check on a 4-chip `v5p` host (`BF16 TP=2 × DP=2` completed in `1,270 s` and is faster on `v5p`) | [`pod2_v5p_tp_dp_and_specdec.json`](../multilora_singlehost_opt/v5p/pod2_v5p_tp_dp_and_specdec.json) |

### Additional Multi-LoRA & Single-Host Limitations
- **Multi-LoRA Scope**: Multi-LoRA was measured on single hosts (`1× v7x` host `TP=2 × DP=4` and `1× v5p` host `TP=4 × DP=1`) with `max_lora_rank=16` on attention projections (`q/k/v/o_proj`) only; full 16-chip slice Multi-LoRA, `r=64` adapters, MoE expert LoRA layers, and downstream per-adapter benchmark evaluations (`GSM8K` / `HumanEval`) were not measured.
- **Caching & Pre-Warm**: Cached (`95%` target hit) single-host rows pre-warm shared prefixes outside `wall_s` and route requests with prefix affinity where `DP > 1`. See [`README.md` §3–§4](../README.md#3-methodology-notes) for repository-wide methodology notes.
