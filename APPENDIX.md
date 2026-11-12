# Appendix: Kernel Modifications, Full Concurrency Sweeps, Ablations, and Caveats (`RecursiveSI/flint` TPU Serving)

This appendix preserves all runtime/kernel modifications, multi-host `sglang-torchtpu` and `vllm-torchtpu` concurrency ladders (`4K` to `128K`), single-chip and single-host ablations, non-zero HTTP 500 saturation points, compiled Multi-LoRA details, unsupported configurations, and methodology notes across all `2,034` top-level benchmark rows (`2,214` including nested phase records). For headline results and reproduction commands, see [`README.md`](./README.md).

---

## 1. Runtime & Kernel Modifications

| Component | Engineering Modification | Code Anchor | Raw Archive |
| :--- | :--- | :--- | :--- |
| **1. Static HBM Allocation (`mem_fraction_static`)** | Raised `--mem-fraction-static` from `0.40` default to `0.82–0.88`, allocating `40–68 GiB/rank` of paged KV cache alongside `BF16 TP=2` or `Block-FP8 TP=1/2` weights | [`launch_host_replicas.py`](./v7x/scripts/launch_host_replicas.py) · [`bench_flint_v5p.py`](./v5p/scripts/bench_flint_v5p.py) | [`smoke_bf16_tp2_1rep.json`](./v7x/results/smoke_bf16_tp2_1rep.json) |
| **2. Fused VMEM Pallas `router_topk` (`256` Experts)** | Added `256` experts to `VALIDATED_NUM_EXPERTS` in `sglang_tpu`, using `sglang_tpu_pallas::router_topk` for decode (`batch <= 64`) and 6-round `argmax`-and-mask routing for prefill (`batch > 64`) across 47 MoE layers | [`patch_flint_router_topk.py`](./v7x/scripts/patch_flint_router_topk.py) | [`ablation_stock_router_tp2_1chip.json`](./v7x/results/ablation_stock_router_tp2_1chip.json) |
| **3. `FP8` (`e4m3`) KV Cache + Hybrid `swa_ratio=0.10`** | Enabled `fp8_e4m3` paged KV (`pallas_tpu_rpa_v3`) and `--swa-full-tokens-ratio 0.10` matching Flint's `35:14` sliding-window (`W=513`) to full-attention layer split (`83.0M–117.8M` KV tokens/slice on `v7x-16`, `42.3M` on `v5p-16`) | [`orchestrate_v7x_slice.py`](./v7x/scripts/orchestrate_v7x_slice.py) | [`ablation_bf16kv_tp2_1chip.json`](./v7x/results/ablation_bf16kv_tp2_1chip.json) · [`ablation_swa010_tp2_1chip.json`](./v7x/results/ablation_swa010_tp2_1chip.json) |
| **4. Parallel `[128, 128]` Block-FP8 Converter (`4.6 s`)** | 15-process safetensors converter (`79.6 GB -> 40.4 GB`, zero padding across `TP=1, 2, 4`), fitting single-chip `TP=1 × DP=16` on `v5p-16` and intra-chip `TP=2 + DP-Attention=32` on `v7x-16` | [`convert_flint_fp8.py`](./v7x/scripts/convert_flint_fp8.py) · [`quantize_flint_fp8.py`](./v5p/scripts/quantize_flint_fp8.py) | [`fullslice_fp8_tp2_dp16_swa010.json`](./v7x/results/fullslice_fp8_tp2_dp16_swa010.json) |
| **5. Sub-Host VFIO Isolation + Trunk-Affinity Gateway** | `LD_PRELOAD` `metadata_hook.so` + `/dev/vfio/0..7` isolates sub-host `TP=2` replicas across 4 hosts behind an async HTTP gateway supporting `X-Trunk-Index` prefix-affinity and least-inflight routing | [`metadata_hook.c`](./v7x/scripts/metadata_hook.c) · [`gateway_flint_v7x.py`](./v7x/scripts/gateway_flint_v7x.py) · [`flint_v5p_gateway.py`](./v5p/scripts/flint_v5p_gateway.py) | [`fullslice_bf16_tp2_dp16.json`](./v7x/results/fullslice_bf16_tp2_dp16.json) |
| **6. Intra-Chip `DP-Attention=32` (`v7x`) & `mrr=96` Batching** | Spaced per-replica HTTP/ZMQ TCP ports (`30000 + r*20`, avoiding `port + 233..239` collisions) for `4`/host `--enable-dp-attention --dp-size 2` (`32` `attn_tp=1` ranks on `v7x-16`), and compiled `max_running_requests=96` on `BF16` replicas | [`launch_host_replicas.py`](./v7x/scripts/launch_host_replicas.py) · [`bench_flint_v5p.py`](./v5p/scripts/bench_flint_v5p.py) | [`fullslice_fp8_tp2_dpattn32_swa010.json`](./v7x/results/fullslice_fp8_tp2_dpattn32_swa010.json) · [`v5p16_slice16_optimized.jsonl`](./v5p/results/raw/v5p16_slice16_optimized.jsonl) |
| **7. `vllm-torchtpu` `FlintForCausalLM` + RPA v3 `hd=128` Sink + `EP=32/16`** | Registered `FlintConfig` + `FlintForCausalLM` in `vllm`, extended Pallas RPA v3 `attention_sink` online-softmax initialization (`m_ref = sink, l_ref = 1.0`) to `head_dim=128`, and added a `v5p` `8`-lane SparseCore top-k fallback | [`install_flint_vllm.py`](./vllm/scripts/install_flint_vllm.py) · [`run_flint_vllm_multihost_suite.py`](./vllm/scripts/run_flint_vllm_multihost_suite.py) | [`vllm_bf16_tp1_dp32_ep32_fp8kv.json`](./vllm/results/v7x-16/vllm_bf16_tp1_dp32_ep32_fp8kv.json) |
| **8. Static-Shape Compiled Multi-LoRA (`TorchNativeLoRABackend`)** | Added persistent device buffers (`_tpu_weight_indices = -1` for base/pad tokens, `_tpu_scalings`) and `@torch.compiler.assume_constant_result` offsets so `apply_lora` compiles into static XLA `EXTEND`/`DECODE` buckets | [`patch_sglang_tpu_lora.py`](./multilora_singlehost_opt/scripts/patch_sglang_tpu_lora.py) | [`E_sglang_flint_multilora.json`](./multilora_singlehost_opt/v7x/E_sglang_flint_multilora.json) |

---

## 2. TPU `v7x-16` (`2x2x4`, 16 Physical Chips / 32 TensorCores) Full-Slice Sweeps & Saturation Points

Cold startup and ahead-of-time PJRT compilation on `v7x-16` completed in **`88.2–120.4 s`**, compared to **`1,256.7–1,600.5 s` (`21–27 min`)** on `v5p-16` (`~11×` faster cold start).

### A. `Block-FP8 TP=2 + DP-Attention=32, swa=0.10` (`116.42M` Slice KV Tokens; [`fullslice_fp8_tp2_dpattn32_swa010.json`](./v7x/results/fullslice_fp8_tp2_dpattn32_swa010.json))

| Shape (`ISL:OSL`) | Prefix Cache | `c` *(Prompts `n`)* | Slice Output (tok/s) | Slice Total (tok/s) | `TTFT p50` (ms) | `TPOT p50` (ms) | `E2E p50` (s) | Failures |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4096:1024` | `0%` | `512` *(1024)* | 31,060.4 | 155,302.2 | 1,593.8 | 14.51 | 16.42 | 0 |
| `4096:1024` | `0%` | `768` *(1536)* | **34,317.3** | **171,586.4** | 2,568.8 | 19.34 | 22.41 | **0** |
| `4096:1024` | `80%` | `1` *(4)* | 136.8 | 683.8 | 167.7 | **7.16** | 7.47 | 0 |
| `4096:1024` | `80%` | `768` *(1536)* | **36,602.9** | **183,014.8** | 761.8 | 19.27 | 20.37 | **0** |
| `4096:1024` | `80%` | `1024` *(1536)* | 37,771.0 | 188,855.2 | 1,074.7 | 18.42 | 19.98 | `7` *(HTTP 500)* |
| `4096:1024` | `95%` | `1` *(4)* | 133.0 | 665.0 | 157.3 | 7.40 | 7.73 | 0 |
| `4096:1024` | `95%` | `768` *(1536)* | **37,321.6** | **186,607.8** | 629.4 | 18.30 | 19.40 | **0** |
| `4096:1024` | `95%` | `1024` *(1536)* | 39,520.7 | 197,603.3 | 825.4 | 17.55 | 18.81 | `17` *(HTTP 500)* |
| `8192:1024` | `0%` | `768` *(1536)* | **26,572.9** | **239,156.0** | 5,041.4 | 23.09 | 28.66 | **0** |
| `8192:1024` | `80%` | `768` *(1536)* | **33,132.7** | **298,194.3** | 1,077.7 | 20.02 | 21.64 | **0** |
| `8192:1024` | `80%` | `1024` *(1536)* | 36,500.2 | 328,501.6 | 1,668.9 | 19.55 | 22.13 | `4` *(HTTP 500)* |
| `8192:1024` | `95%` | `512` *(1024)* | **30,995.4** | **278,958.6** | 527.6 | 14.86 | 15.89 | **0** |
| `8192:1024` | `95%` | `768` *(1536)* | 35,969.7 | 323,727.4 | 762.2 | 18.93 | 20.09 | `3` *(HTTP 500)* |
| `8192:1024` | `95%` | `1024` *(1536)* | 39,035.7 | 351,321.0 | 1,052.9 | 18.76 | 20.25 | `4` *(HTTP 500)* |
| `65536:1024` | `0% / 50% / 95%` | `64` *(n=64; 3 repeats)* | 3,154.4–3,310.3 / 3,744.1–3,749.8 / 4,471.8–4,561.6 | 205K–297K | 7,931 / 6,233 / 3,361 | 9.29–11.60 / 10.52 / 9.94 | 19.37 / 16.70 / 13.93 | 0 |
| `131072:1024` | `0% / 50% / 95%` | `64` *(n=64; 3 repeats)* | 1,501.2–1,638.6 / 1,771.2–1,778.0 / 2,400.3–2,411.7 | 194K–311K | 21,431 / 18,109 / 9,052 | 11.55–17.47 / 17.78 / 15.60 | 39.51 / 36.21 / 26.46 | 0 |

### B. `BF16 TP=2 × DP=16, mrr=96, swa=0.10` (`82.97M` Slice KV Tokens; [`fullslice_bf16_tp2_dp16_mrr96_swa010.json`](./v7x/results/fullslice_bf16_tp2_dp16_mrr96_swa010.json))

| Shape (`ISL:OSL`) | Prefix Cache | `c` *(Prompts `n`)* | Slice Output (tok/s) | Slice Total (tok/s) | `TTFT p50` (ms) | `TPOT p50` (ms) | Failures |
| :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `4096:1024` | `0%` | `512` *(1024)* | 30,785.0 | 153,925.0 | 1,625.2 | 14.43 | 0 |
| `4096:1024` | `0%` | `768` *(1536, Pass 1 / Pass 2)* | 35,004.7 / **33,334.6** | 175,023.7 / 166,673.1 | 2,359.7 / 3,037.1 | 19.14 / 19.38 | `4` / **0** |
| `4096:1024` | `50%` | `1` *(4)* / `1024` *(1536)* | 153.1 / **33,936.3** | 765.4 / 169,681.7 | 89.5 / 1,408.7 | **6.47** / 20.62 | 0 / 0 |
| `4096:1024` | `95%` | `1` *(4)* | 150.2 | 751.0 | **49.0** | 6.62 | 0 |
| `4096:1024` | `95%` | `768` *(1536, Pass 1 / Pass 2)* | 32,517.7 / **35,581.0** | 162,588.6 / 177,905.2 | 385.8 / 824.4 | 19.75 / 18.86 | `1` / **0** |
| `8192:1024` | `0%` / `50%` / `95%` | `768` / `1024` / `1024` *(1536)* | **26,514.7** / **30,588.2** / **32,269.7** | 238.6K / 275.3K / 290.4K | 4,763.1 / 2,061.2 / 693.4 | 23.61 / 25.29 / 23.90 | 0 |
| `65536:1024` | `0% / 50% / 95%` | `64` *(n=64; 5 repeats)* | 3,537.8–3,556.9 / 4,485.6–4,498.6 / **6,549.8–6,617.3** | 230K–430K | 6,038 / 3,587 / 942 | 10.71 / 9.83 / 8.33 | 0 |
| `131072:1024` | `0% / 50% / 95%` | `64` *(n=64; 5 repeats)* | 1,743.0–1,747.3 / 2,274.7–2,279.7 / **4,810.9–4,823.8** | 225K–622K | 16,866 / 11,327 / 2,315 | 16.45 / 14.05 / 10.20 | 0 |

### C. `Block-FP8 TP=2 × DP=16, mrr=64, swa=0.10` (`117.80M` Slice KV Tokens; [`fullslice_fp8_tp2_dp16_swa010.json`](./v7x/results/fullslice_fp8_tp2_dp16_swa010.json)) & `BF16 mrr=64, swa=0.80` ([`fullslice_bf16_tp2_dp16.json`](./v7x/results/fullslice_bf16_tp2_dp16.json))

- **`Block-FP8 TP=2 × DP=16, swa=0.10` (`attn_tp=2`):**
  - `4K:1K` (`c=512, n=512` single wave): `29,923.9` tok/s (`0%`), `33,091.3` tok/s (`80%`), **`33,200.3` tok/s** (`95%`, `317.3 ms` `TTFT p50`, `0` fail); `c=1` `TPOT p50`: `6.49–6.66 ms`.
  - `8K:1K` (`c=512, n=512`): `23,530.5` tok/s (`0%`), `27,339.3` tok/s (`80%`, `0` fail; Pass 1 `29,696.9` w/ `1` fail), **`31,432.8` tok/s** (`95%`, `0` fail).
  - `16K:1K` (`c=512, n=512`): `15,882.3` tok/s (`0%`, `0` fail; Pass 1 `16,591.8` w/ `4` fail), `25,147.1` tok/s (`80%`), **`25,737.5` tok/s** (`95%`, `0` fail).
  - `64K:1K` (`c=64, n=64`): `3,646.1–3,695.0` tok/s (`0%`), `5,882.6–6,013.0` tok/s (`80%`), **`7,119.8–7,225.0` tok/s** (`95%`, `985.2 ms` `TTFT p50`, `0` fail).
  - `128K:1K` (`c=64, n=64`): `1,765.5–1,770.4` tok/s (`0%`, `20.5 s` `TTFT`), `3,531.7–3,559.9` tok/s (`80%`), **`5,113.8–5,172.5` tok/s** (`95%`, `2,301.8 ms` `TTFT p50`, `0` fail).
  - **Why `TP=2 × DP=16` beats `DP-Attention=32` by `1.58–2.14×` at `64K–128K`:** Standard `TP=2` (`attn_tp=2`) shards `64K–128K` prefill across both intra-package TensorCores and keeps a unified per-chip RadixCache, whereas `DP-Attention=32` (`attn_tp=1`) executes long prefill on a single TensorCore.
- **`BF16 TP=2 × DP=16, mrr=64, swa=0.80` (`34.62M` KV):** `29,833.1` (`4K, 0%`), `33,421.5` (`4K, 95%, c=512`), `30,451.6` (`8K, 80%, c=512`; `29,587.4` w/ `1` fail at `95%`), `26,632.1` (`16K, 95%, c=512`), `6,590.1–6,643.2` (`64K, 95%, n=64`), `4,838.5–4,864.2` (`128K, 95%, n=64`; `8.89×` lower `TTFT` vs. `0%` uncached `20,172.8 -> 2,268.1 ms`).

### D. Single-Chip `v7x` (`TP=2`) & Scale-Up Ablations ([`v7x/results/`](./v7x/results/))

1. **KV Cache Dtype (`FP8 e4m3` vs. `BF16` on `BF16 TP=2` 1-Chip, [`ablation_bf16kv_tp2_1chip.json`](./v7x/results/ablation_bf16kv_tp2_1chip.json)):** `FP8` KV increases per-chip full KV slots by **`+89.8%`** (`1,080,256 -> 2,050,432` at `swa=0.80`) and boosts output throughput by **`+24.2%`** at `4K, 0%, c=64` (`2,461.5 -> 3,058.4` tok/s), **`+8.3%`** at `4K, 95%, c=64` (`3,034.1 -> 3,285.9`), and **`+21.0%` / `+17.6%`** at `64K` / `128K` (`c=1, 0%`).
2. **Weights Dtype (`Block-FP8` vs. `BF16` on `TP=2` 1-Chip, [`ablation_fp8_tp2_1chip.json`](./v7x/results/ablation_fp8_tp2_1chip.json)):** Cuts weight HBM by **`-49.5%`** (`75.4 -> 38.1 GiB`), expands KV capacity `+49.6%` (`2.05M -> 3.07M` at `swa=0.80`), and increases `4K, 0%, c=64` throughput `+8.4%` (`3,058.4 -> 3,315.9` tok/s).
3. **Fused Pallas `router_topk` (`256` Experts) vs. Unfused PyTorch Router ([`ablation_stock_router_tp2_1chip.json`](./v7x/results/ablation_stock_router_tp2_1chip.json)):** Speeds up single-chip generation by **`1.05×`** at `c=1..16` (`132.5 -> 139.4` tok/s) and **`1.20×`** at `4K, 0%, c=64` (`2,549.1 -> 3,058.4` tok/s).
4. **Hybrid `swa_full_tokens_ratio` (`0.80 -> 0.10`, [`ablation_swa010_tp2_1chip.json`](./v7x/results/ablation_swa010_tp2_1chip.json)):** Expands full-attention KV pool **`2.53×`** (`2,050,432 -> 5,185,408` tokens/chip on `BF16`; `7,362,688` on `Block-FP8`) and lifts `128K, 95%, c=16` throughput `+7.4%` (`497.8 -> 534.8` tok/s).
5. **Scale-Up (`1 -> 2 -> 4 -> 16` Chips at `4K:1K`):** `1` chip (`BF16 TP=2`): `3,058.4` (`0%`) / `3,285.9` (`95%`) tok/s, `6.84 ms` `TPOT`; `2` chips (`TP=4`): `3,212.8` tok/s, `6.82 ms`; `4` chips (`TP=8`): `3,380.2` tok/s, `6.88 ms`; `16` chips (`DP=16` / `DP-Attn=32`): `34,317.3` (`0%`) / `37,321.6` (`95%`) tok/s (`10.9×–11.4×` zero-failure scaling over 1 chip).

---

## 3. TPU `v5p-16` (`2x2x4`, 16 Chips) `sglang-torchtpu` Sweeps & Saturation Points

Detailed guide and raw archives: [`v5p/README.md`](./v5p/README.md) and [`v5p/results/raw/`](./v5p/results/raw/).
- **`slice16_bf16_tp2_dp8_mrr96_swa010` (`42.29M` KV, `82.02 GiB` peak HBM/chip):**
  - Zero-failure points: **`18,940.7` tok/s** (`4K:1K, 95%` [`93.75%` obs], `c=768, n=1536`), **`18,678.3` tok/s** (`4K:1K, 80%, c=768`), **`10,051.8` tok/s** (`4K:1K, 0%, c=256, n=512`), **`10,256.4` tok/s** (`8K:1K, 0%, c=512, n=1024`), **`10,323.1` tok/s** (`8K:1K, 95%, c=256, n=512`), **`3,511.0` tok/s** (`64K:1K, 95%, c=64, n=128`), **`1,867.8` tok/s** (`128K:1K, 95%, c=32, n=32`), and **`7.48 ms`** `c=1` `TPOT`.
  - Non-zero-failure points: `15,016.0` tok/s (`4K, 0%, c=768, 1/1536` fail), `13,734.0` (`4K, 0%, c=512, 2/1024` fail), `15,319.3` (`4K, 95%, c=512, 2/1024` fail), and `13,689.2` (`8K, 95%, c=512, 1/1024` fail).
- **`slice16_fp8_tp1_dp16_mrr64_swa010` (`40.40M` KV, `81.34 GiB` HBM/chip):** `9,024.6` tok/s (`4K, 95%, c=768, 0` fail), `7,877.8` (`4K, 0%, c=768, 0` fail), `5,010.7` (`8K, 95%, c=256, 0` fail; `7,292.2` @ `c=512` w/ `2/1024` fail), `2,049.2` (`64K, 95%, c=64`), `1,369.8` (`128K, 95%, c=32`), `12.96 ms` `c=1` `TPOT`. Because `v5p` MXUs execute native `BF16` and must dequantize `Block-FP8` weights in software, `BF16 TP=2 × DP=8` is **`2.10×` faster** on `v5p-16`.

---

## 4. Multi-Host `vllm-torchtpu` (`EP=32` on `v7x-16` & `EP=16` on `v5p-16`)

Detailed guide and raw archives (`145` points, `0` failed requests): [`vllm/README.md`](./vllm/README.md) and [`vllm/results/`](./vllm/results/).
- **Why `BF16` Beats `Block-FP8` Under `EP=32` on `v7x-16`:** Sharding `256` experts across `32` TensorCores leaves only `8` experts/TC (`1.25 GiB` `BF16` vs. `0.63 GiB` `FP8`). Avoiding dynamic input activation quantization makes `vllm_bf16_tp1_dp32_ep32_fp8kv` **`3.0%–13.0%` faster** than `vllm_fp8_tp1_dp32_ep32_fp8kv` at `c=1024, n=1024` (`27,091.9` vs. `25,415.8` tok/s at `80%` prefix; `26,912.3` vs. `26,145.9` at `95%`; `21,650.2` vs. `19,153.3` at `0%`). At mid concurrency (`c=1..256`), `TP=2 × DP=16 × EP=32` achieves lower latency (`549.6 ms` `TTFT`, `10.37 ms` `TPOT` at `c=1`).
- **Why `EP=16` Boosts `v5p-16` Quantized `TP=1` Throughput `+49.7%`:** On `v5p-16` at matched `(8K:1K, 95%, c=256, n=512)`, `vllm_w8a16_tp1_dp16_ep16_fp8kv` (`16` experts/chip) reaches **`7,501.5` tok/s** vs. `5,010.7` tok/s on non-EP `sglang` `slice16_fp8_tp1_dp16_mrr64_swa010` (`256` experts dequantized per chip), while pure `BF16` `vllm_bf16_tp1_dp16_ep16_fp8kv` reaches **`8,390.2` tok/s** at `8K` and **`9,470.7` tok/s** at `4K` (`c=512`).
- **Host-Side `ngram` (`k=5`) Speculative Decoding Slowdown (`vllm_fp8_tp4_dp8_ep32_ngram5`):** Requires `--no-async-scheduling` in `vllm-torchtpu v0.32.0`, inserting a host-device barrier per step (`18.91–19.62 ms` `TPOT` on `4K:1K` and `43.37–80.99 ms` on `coding_edit`, `~1.8×` slower per token than async `TP=2` at `10.37 ms`; `812.6` tok/s peak at `c=16`).

---

## 5. Single-Host Optimization Matrix (`Axes A–E`) & Compiled Multi-LoRA Details

All single-host archives live under [`multilora_singlehost_opt/`](./multilora_singlehost_opt/README.md) (`4` chips/host; zero multi-host extrapolations):
- **Single-Host `v7x` (`4` Chips / `8` TCs):**
  - `C2_bf16_tp2_dp4_bf16kv`: `9,534.1` tok/s (`1K:512, 95%, c=128`), `8,815.1` (`4K:1K`), `2,216.8` (`64K:1K, c=16`), `6.51 ms` `c=1` `TPOT`.
  - `C1_bf16_tp2_dp4_fp8kv`: `8,996.8` (`1K:512`), `8,853.4` (`4K:1K`), `3,338.3` (`64K:1K`), `1,313.2` (`128K:1K, c=8`), `6.63 ms` `TPOT`.
  - `A1_fp8_tp2_dp4`: `8,657.5` (`1K:512`), `8,763.6` (`4K:1K`), `3,289.4` (`64K:1K`), **`2,078.1`** (`128K:1K`), `6.49 ms` `TPOT`.
  - `D2_fp8_tp2_dp4_swa005`: **`3,429.6` tok/s** (`64K:1K`), `2,073.4` (`128K:1K`); `D1_fp8_tp2_dp4_noradix` (`--disable-radix-cache`): `901.3` tok/s at `64K` (`3.65×` slower than `A1` with RadixCache, `2,526 ms` vs. `246 ms` `TTFT`).
  - `A5_fp8_tp8_ep8` (`TP=8 × EP=8`): Lowers `64K` `c=1` `TTFT` to **`1,225.3 ms`** uncached (`2.06×` faster than `TP=2×DP=4`) and **`135.8 ms`** cached (`4,678.1` tok/s peak).
- **Single-Host `v5p` (`4` Chips):**
  - `TP × DP` ([`v5p_tp_dp_and_specdec.json`](./multilora_singlehost_opt/v5p/v5p_tp_dp_and_specdec.json)): `v5p_bf16_tp4_dp1` reaches `3,026.9` tok/s (`4K:512, 95%, c=64`) and `7.22 ms` `TPOT`; `v5p_bf16_tp2_dp2` reaches `2,975.1` tok/s (`7.43 ms`); `v5p_fp8_tp1_dp4` reaches `1,277.3` tok/s (`12.87 ms`).
  - `128K` RadixCache ([`v5p_radix_swa_longctx.json`](./multilora_singlehost_opt/v5p/v5p_radix_swa_longctx.json)): `Radix On` delivers `344.9` tok/s (`64K`) and `242.5` tok/s (`128K`) vs. `79.5` and `37.6` tok/s with `Radix Off` (`4.3×` and `6.4×` speedups).
- **Compiled Multi-LoRA Caveats (`E1`, `E2`, `v5p_fp8_block128_multilora`, `v5p_bfloat16_multilora`):**
  - All 4 served adapters (`lora_bf16`, `qlora_int8`, `qlora_nf4`, `lora_attn`, exported from companion repo [`ankdeshm/rsi-flint-tpu-lora-qlora-finetuning`](https://github.com/ankdeshm/rsi-flint-tpu-lora-qlora-finetuning)) are attention-only `r=16` (`q/k/v/o_proj`) PEFT adapters served in `BF16` adapter precision atop the `Block-FP8` or `BF16` base checkpoint (`qlora_*` denotes the frozen base quantization used during training, not `INT8`/`NF4` serving weights).
  - On `v7x`, `c=64` points ran a single wave (`n = c = 64`); in the `4_loras_all_mixed` schedule (`3,634.9` / `3,633.6` tok/s), adapters hit EOS at `75–88%` of `512` tokens, making those rows non-comparable to full `512`-token `0–3` adapter rows.

---

## 6. Unsupported / Rejected Configurations & Methodology Caveats

- **Unsupported Single-Host `sglang-torchtpu` Configurations (`0` Points):**
  - `TP=1` on `v7x` ([`A2_fp8_tp1_dp8.json`](./multilora_singlehost_opt/v7x/A2_fp8_tp1_dp8.json)): `torch_tpu` raises `RuntimeError: No TPU topology found for count: 1` on dual-TensorCore `v7x` (minimum `sglang` replica is `TP=2` = 1 physical chip; use `vLLM EP=32` for `TP=1`).
  - `Block-FP8 TP=8` without `EP` ([`A4_fp8_tp8_dp1.json`](./multilora_singlehost_opt/v7x/A4_fp8_tp8_dp1.json)): Shards `moe_intermediate_size = 512` to `512 / 8 = 64 < 128`, violating `block_n = 128` (`TP=8 × EP=8` in `A5` resolves this).
  - `DP-Attn + EP=8` on 1 host ([`A6_fp8_dpattn8_ep8.json`](./multilora_singlehost_opt/v7x/A6_fp8_dpattn8_ep8.json)): Requests `32 > 8` single-host TensorCores.
  - `NGRAM` speculation on `sglang-torchtpu` ([`B1`](./multilora_singlehost_opt/v7x/B1_fp8_tp2_dp4_ngram4.json), [`B2`](./multilora_singlehost_opt/v7x/B2_fp8_tp8_dp1_ngram6.json)): Raises `NotImplementedError` (`NGRAM` unsupported on TPU; only `EAGLE`/`EAGLE3`/`NEXTN`/`DFLASH`).
  - `pallas_tpu_fused_rs` ([`C4`](./multilora_singlehost_opt/v7x/C4_fp8_tp2_dp4_fused_rs.json)), `--quantization w8a8_int8` ([`C5`](./multilora_singlehost_opt/v7x/C5_w8a8_int8_tp2_dp4.json), Triton kernel dependency), and `cp=16384 + mrr=128` ([`D4`](./multilora_singlehost_opt/v7x/D4_fp8_tp2_dp4_cp16k_mrr128.json), `1.08 MB > 1.00 MB` TPU `SMEM` limit).
  - `vllm-torchtpu v0.32.0` Multi-LoRA ([`E3_vllm_torchtpu_flint_comparison.json`](./multilora_singlehost_opt/v7x/E3_vllm_torchtpu_flint_comparison.json)): Rejects `--enable-lora` on TPU when `TP > 1`, `DP > 1`, or on MoE models.
- **Methodology & Quality Scope:**
  - Cached points (`50%`/`80%`/`95%`) pre-warm shared `64`-token-aligned prefix trunks prior to starting `wall_sec`; `v7x-16` `sglang` uses `X-Trunk-Index` oracle prefix-affinity routing (`observed_cache_hit_pct` was logged on `v5p-16` at `93.75%–94.97%` for `95%` target, not scraped on `v7x-16` or Ray DPLB `vLLM`).
  - `v7x` `64K`/`128K` sweeps cap `n=64` requests (`effective c=64`), and `vLLM c=1024`, `sglang Block-FP8 mrr=64 c=512`, `v5p 128K c=32`, and `v7x Multi-LoRA c=64` points run a single wave (`n = c`).
  - Each run passed functional coding (`merge_intervals`) and `51,368`-token passkey needle checks (`FLINT-TPUV7X-947261`); full benchmark suites (`GSM8K`, `HumanEval`, `MMLU-Pro`) across `BF16` vs. `Block-FP8` / `W8A16` remain future work.
