# RecursiveSI/flint Serving on Google Cloud TPU v7x and v5p

Multi-host slice serving (`4K`–`128K` context) and single-host compiled Multi-LoRA serving benchmarks for [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) on Google Cloud TPU `v7x` and `v5p` with `sglang-torchtpu` and `vllm-torchtpu`.

**Summary:** On a 16-chip TPU `v7x-16` slice, `sglang-torchtpu` with `[128, 128]` `Block-FP8` weights and intra-chip `TP=2 + DP-Attention=32` reaches **37,321.6 output tok/s at `4K:1K` (`95%` prefix cache) and 34,317.3 output tok/s uncached (`0%` prefix) with 0 failures — 1.97× and 3.41× higher than the best zero-failure TPU `v5p-16` slice (`18,940.7` and `10,051.8` tok/s) and 1.39× higher than single-endpoint `vllm-torchtpu` `EP=32` (`26,912.3` tok/s at `95%` prefix)**. At `128K` context (`95%` prefix, `c=64`), `Block-FP8 TP=2 × DP=16` on `v7x-16` sustains **5,172.5 output tok/s (`2.77×` higher than `1,867.8` tok/s on `v5p-16`)**.

---

## 1. Model, Framework, and Hardware

- **Model:** [`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint) is an open-weights sparse Mixture-of-Experts (MoE) coding-agent LLM (`vocab_size = 150,144`, `max_position_embeddings = 262,144`, benchmarked up to `131,072` context).
  - **Parameters & Precision:** **39.8B total parameters** (`79.6 GB` in `bfloat16`; `40.4 GB` in dynamic `[128, 128]` `Block-FP8`) with **2.5B active parameters per token**, served with `fp8_e4m3` or `bfloat16` paged KV cache (`page_size = 64`).
  - **Architecture:** 49 transformer layers (`hidden_size = 2,048`): 2 dense SwiGLU layers (`intermediate_size = 6,144`) + 47 sparse MoE layers (`256` routed experts at `moe_intermediate_size = 512`, sigmoid top-6 routing, plus `1` shared expert at `1,024`). Attention uses 8:1 GQA (`32` Q / `4` KV heads, `head_dim = 128`, QK RMSNorm) with learned per-head attention sinks (`softmax_offset`), splitting into 35 sliding-window RoPE layers (`W = 513`) and 14 full-attention NoPE layers (`swa_full_tokens_ratio = 0.10` allocates separate SWA and full KV pools).
- **Frameworks:**
  - **`sglang-torchtpu`** (`torch_tpu` `0.2.0`, `libtpu` `0.0.50`, `flint_sglang` `0.1.4`): Sub-host VFIO/metadata-isolated replicas behind an async prefix-affinity HTTP gateway, fused VMEM Pallas `router_topk` (`256` experts), Pallas RPA v3 paged attention, and static-shape XLA-compiled `TorchNativeLoRABackend`.
  - **`vllm-torchtpu` (`v0.32.0`)**: Single-endpoint OpenAI server over multi-host Ray (`EP=32` / `EP=16`) with custom `FlintForCausalLM` plugin and Pallas RPA v3 `head_dim = 128` `attention_sink` support.
- **Hardware:** Google Cloud TPU slices on GKE:
  - **TPU `v7x-16`** (`2x2x4`: 4 hosts × 4 dual-TensorCore chips = 16 physical chips / 32 TensorCores, `189.5 GiB` usable HBM3e per chip = `3.03 TiB` slice) and single-host `v7x` (`4` chips / `8` TensorCores).
  - **TPU `v5p-16`** (`2x2x4`: 4 hosts × 4 chips = 16 chips, `95 GiB` HBM2e per chip = `1.52 TiB` slice) and single-host `v5p` (`4` chips).

---

## 2. Main Results

### A. 16-Chip Slice Serving Throughput & Latency (Zero-Failure Operating Points)

Output throughput equals `completed_output_tokens / wall_sec` with prefix pre-warming excluded from `wall_sec`. Cells list `(ISL:OSL, target prefix %, concurrency c)`.

| Slice & Stack | Weights / KV & Parallelism | `4K:1K` Output tok/s (`95%` / `0%` Prefix) | `c=1` `TPOT` (`p50`, `4K`) | `64K:1K` (`95%` Prefix) | `128K:1K` (`95%` Prefix) | Comparison |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| **`v7x-16` `sglang`** | **`Block-FP8` / `FP8` KV, `TP=2 + DP-Attn=32`** (`116.4M` KV) | **37,321.6** (`c=768`) / **34,317.3** (`c=768`) | 7.40 ms / 7.75 ms | 4,561.6 tok/s (`c=64`) | 2,411.7 tok/s (`c=64`) | **1.97×** (`95%`) and **3.41×** (`0%`) higher than `v5p-16` best `0`-fail; **1.39×** higher than `v7x-16` `vLLM` at `95%` |
| **`v7x-16` `sglang`** | **`BF16` / `FP8` KV (`mrr=96`), `TP=2 × DP=16`** (`83.0M` KV) | **35,581.0** (`c=768`) / **33,334.6** (`c=768`) | **6.62 ms** / **6.74 ms** | 6,617.3 tok/s (`c=64`) | 4,823.8 tok/s (`c=64`) | Unquantized weights; **1.88×** higher than `v5p-16` `BF16` (`95%`) and **10–13% lower** `c=1` `TPOT` |
| **`v7x-16` `sglang`** | **`Block-FP8` / `FP8` KV, `TP=2 × DP=16`** (`117.8M` KV) | 33,200.3 (`c=512`) / 29,923.9 (`c=512`) | 6.59 ms / 6.66 ms | **7,225.0 tok/s** (`c=64`) | **5,172.5 tok/s** (`c=64`) | **1.58–2.14×** higher than `DP-Attn=32` at `64K–128K` via intra-chip `attn_tp=2`; **2.77×** higher than `v5p-16` at `128K` |
| `v7x-16` `vllm` | `BF16` / `FP8` KV, Ray `TP=1 × DP=32 × EP=32` | 26,912.3 (`c=1024`) / 21,650.2 (`c=1024`) | 11.51 ms (`0%`) | — (`308.6` FP8 `0%, c=8`) | — (`189.7` FP8 `0%, c=8`) | Single native endpoint (no gateway); **3.0–13.0% higher** than `vLLM` `Block-FP8 EP=32` (`26,145.9` / `19,153.3`) |
| **`v5p-16` `sglang`** | **`BF16` / `FP8` KV (`mrr=96`), `TP=2 × DP=8`** (`42.3M` KV) | **18,940.7** (`c=768`) / **10,051.8** (`c=256`) | **7.48 ms** (`0%`) | **3,511.0 tok/s** (`c=64`) | **1,867.8 tok/s** (`c=32`) | **2.10× higher** than `v5p-16` `sglang` `Block-FP8 TP=1×DP=16` (`4K, 95%`) and **2.00× higher** than `v5p-16` `vLLM` |
| `v5p-16` `vllm` | `BF16` & `W8A16` / `FP8` KV, Ray `TP=1 × DP=16 × EP=16` | 9,470.7 (`BF16, c=512`) / 6,256.0 (`0%`) | 12.34 ms (`BF16`) | 200.8 tok/s (`W8A16 0%, c=8`) | 57.9 tok/s (`W8A16 0%, c=4`) | At `8K:1K, 95%, c=256`, `W8A16 EP=16` (`7,501.5` tok/s) is **49.7% higher** than non-EP `sglang` `FP8 TP=1×DP=16` (`5,010.7`) |
| `v5p-16` `sglang` | `Block-FP8` / `FP8` KV (`mrr=64`), `TP=1 × DP=16` (`40.4M` KV) | 9,024.6 (`c=768`) / 7,877.8 (`c=768`) | 12.96 ms (`0%`) | 2,049.2 tok/s (`c=64`) | 1,369.8 tok/s (`c=32`) | Single-chip replica baseline (`52.4%` lower than `BF16 TP=2×DP=8` on `v5p-16` due to software `W8A16` dequant) |

*Notes:* On `v7x-16` `vLLM BF16 EP=32`, `80%` prefix reached `27,091.9` tok/s (`c=1024, n=1024`). All `v7x` `64K`/`128K` points with `c >= 64` dispatched `n=64` requests (`effective c=64`). Higher-concurrency points with non-zero HTTP 500s (`39,520.7` tok/s on `v7x-16` `c=1024` with `17/1536` errors) are documented in [`APPENDIX.md`](./APPENDIX.md).

### B. Single-Host (`4`-Chip) Optimization & Compiled Multi-LoRA Serving

Directly measured on 1 host (`4` chips; no multi-host extrapolations) using attention-only `r=16` (`q/k/v/o_proj`) PEFT adapters ([`multilora_singlehost_opt/`](./multilora_singlehost_opt/README.md)):

| Single-Host Platform (`4` Chips) | Configuration | `c=1` `TPOT` (`p50`) | Peak Measured Host Output tok/s (`0` Failures) | Comparison / Hot-Swap Latency |
| :--- | :--- | ---: | ---: | :--- |
| **TPU `v7x` (`TP=2 × DP=4`, Base)** | `BF16` (`C2`/`C1`) & `Block-FP8` (`A1`) | **6.49–6.63 ms** | **9,534.1 tok/s** (`1K:512, 95%, c=128`) · **8,853.4** (`4K:1K, 95%`) · **2,078.1** (`128K:1K, 95%`) | RadixCache provides **3.65× higher** `64K` tok/s (`3,289.4` vs. `901.3` with `--disable-radix-cache`) |
| **TPU `v7x` (`TP=2 × DP=4`, Multi-LoRA)** | `Block-FP8` (`E1`) / `BF16` (`E2`) + `0..3` LoRAs | 10.47–11.43 ms | **4,133.8–4,352.3 tok/s** (`FP8`, `1–3` LoRAs) · **4,162.7–4,293.8** (`BF16`, `1–3` LoRAs) | Matches `0`-LoRA base (`3,984.6` FP8 / `4,120.3` BF16 at `1K:512, 0%, c=64, n=64`); `12.4–13.3 ms` unload / `24.4–24.8 ms` reload |
| **TPU `v5p` (`TP=4 × DP=1`, Multi-LoRA)** | `BF16` / `Block-FP8` + `0..4` LoRAs | 14.22–15.12 ms | **1,540.9–1,655.9 tok/s** (`BF16`, `1–4` LoRAs, `c=32`) · **679.7–803.7** (`FP8`, `1–4` LoRAs) | `BF16` base (`1,529.1` tok/s `0`-LoRA) is **2.06–2.53× higher** than `Block-FP8` (`604.5` tok/s `0`-LoRA) on `v5p` |

---

## 3. Reproducing the Main Results

```bash
# 1A. TPU v7x-16 (16 chips): sglang-torchtpu sub-host TP=2 x DP=16 or DP-Attention=32 slice sweep
kubectl apply -f v7x/k8s/v7x-16-spot-runner.yaml
python3 v7x/scripts/convert_flint_fp8.py --src /dev/shm/flint --dst /dev/shm/flint-fp8 --workers 15
python3 v7x/scripts/orchestrate_v7x_slice.py \
  --action deploy_and_bench --run-name fullslice_fp8_tp2_dpattn32_swa010 \
  --model-path /dev/shm/flint-fp8 --weights-dtype fp8 --tp-size 2 --dp-size 2 --enable-dp-attention \
  --kv-cache-dtype fp8_e4m3 --swa-full-tokens-ratio 0.10 --verify-quality

# 1B. TPU v5p-16 (16 chips): sglang-torchtpu high-concurrency BF16 TP=2 x DP=8 (mrr=96) & FP8 TP=1 x DP=16
kubectl apply -f v5p/k8s/v5p-flint-slice16-opt.yaml

# 2. Multi-host vllm-torchtpu single-endpoint EP=32 (v7x-16) & EP=16 (v5p-16) suites
kubectl apply -f vllm/k8s/v7x-16-flint-vllm-ep32.yaml
kubectl apply -f vllm/k8s/v5p-16-flint-vllm-ep16.yaml

# 3. Single-host XLA-compiled Multi-LoRA serving (v7x TP=2 x DP=4 & v5p TP=4 x DP=1)
kubectl apply -f multilora_singlehost_opt/k8s/flint-opt-sglang-v7x.yaml
kubectl apply -f multilora_singlehost_opt/k8s/flint-v5p-multilora.yaml
```

---

## 4. Detailed Ablations, Engineering Notes, and Raw Logs

- **[`APPENDIX.md`](./APPENDIX.md):** Kernel and runtime modifications, full `v7x-16` / `v5p-16` concurrency curves (`4K`–`128K`), single-chip ablations (`FP8` KV, `Block-FP8`, fused `router_topk`, `swa_ratio`), non-zero HTTP 500 saturation points, `vLLM` `EP=32`/`EP=16` & `ngram` speculation analysis, single-host `Axes A–E` & Multi-LoRA caveats, unsupported configurations, and measurement methodology.
- **[`results/`](./results/)** (plus stack guides [`v7x/README.md`](./v7x/README.md), [`v5p/README.md`](./v5p/README.md), [`vllm/README.md`](./vllm/README.md), and [`multilora_singlehost_opt/README.md`](./multilora_singlehost_opt/README.md)): Raw JSON and JSONL outputs across all `2,034` benchmark rows.
