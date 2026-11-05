# `RecursiveSI/flint` (`40B-A2.5B` MoE) Multi-Host `vllm-torchtpu` Serving (`EP=32` on `v7x-16` & `EP=16` on `v5p-16`)

Single-endpoint OpenAI serving stack, `FlintForCausalLM` + `FlintConfig` vLLM model plugin, Pallas Ragged Paged Attention v3 (`head_dim=128`) learned `attention_sink` online-softmax kernel patch, Kubernetes Ray multi-host manifests, and benchmark archives for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.5B` active MoE) across 4-host **TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores)** and **TPU `v5p-16` (`2x2x4`, 16 chips)** using **`vllm-torchtpu` (`v0.32.0`)**.

---

## 1. Benchmark Summary (`145` Benchmark Points, `0` Failed Requests)

### A. TPU `v7x-16` (`2x2x4`, 4 Hosts / 16 Chips / 32 TensorCores, `EP=32`)

| Experiment | Weights / KV Dtype | Parallelism (`32` TensorCores) | Peak Output Throughput | Peak Total Throughput (incl. Cached Prefix) | Single-Stream `TPOT` (`c=1`, `p50`) / Best `TPOT` | `4K` `0%` `TTFT` (`c=1`) | `64K` / `128K` (`0%` Hit, `c=8`, `n=16`) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`vllm_bf16_tp1_dp32_ep32_fp8kv`** | **`BF16` / `FP8`** | **`TP=1 × DP=32 × EP=32`** *(`8` experts/TC = `1.25 GiB`)* | **`27,091.93` tok/s** *(`4K:1K`, `80%`, `c=1024`, `n=1024`)* / **`26,912.29`** *(`95%`)* / **`21,650.19`** *(`0%`)* | **`130,432.81` tok/s** *(`80%`)* | **`11.51 ms`** *(`c=1`)* / `10.92 ms` *(`c=32`)* | `708.97 ms` | — | [`JSON`](./results/v7x-16/vllm_bf16_tp1_dp32_ep32_fp8kv.json) |
| **`vllm_fp8_tp1_dp32_ep32_fp8kv`** | **`Block-FP8` / `FP8`** | **`TP=1 × DP=32 × EP=32`** *(`8` experts/TC = `0.63 GiB`)* | **`26,145.93` tok/s** *(`4K:1K`, `95%`, `c=1024`, `n=1024`)* / **`25,415.79`** *(`80%`)* / **`19,153.33`** *(`0%`)* | **`125,827.28` tok/s** *(`95%`; `102,420.62` @ `8K, 0%, c=512`)* | **`10.95 ms`** *(`c=1`)* / `10.65 ms` *(`c=32`)* | `856.65 ms` | **`308.56` / `189.67` tok/s** *(`11.40 / 11.48 ms` TPOT)* | [`JSON`](./results/v7x-16/vllm_fp8_tp1_dp32_ep32_fp8kv.json) |
| **`vllm_fp8_tp2_dp16_ep32_fp8kv`** | **`Block-FP8` / `FP8`** | **`TP=2 × DP=16 × EP=32`** *(mid-concurrency `c=1..256`)* | **`20,336.83` tok/s** *(`4K:1K`, `95%`, `c=1024`, `n=1024`)* / **`12,264.51`** *(`95%`, `c=256`, `n=512`)* | **`97,871.00` tok/s** *(`95%`)* | **`10.37 ms`** *(`c=1`, `0%`)* | **`549.57 ms`** | — | [`JSON`](./results/v7x-16/vllm_fp8_tp2_dp16_ep32_fp8kv.json) |
| **`vllm_fp8_tp4_dp8_ep32_ngram5`** | **`Block-FP8` / `FP8`** | **`TP=4 × DP=8 × EP=32` + `ngram` (`k=5`, `--no-async-scheduling`)** | **`812.63` tok/s** *(`4K:1K`, `95%`, `c=16`, `n=32`)* / `173.38` *(`4K:512` `coding_edit`, `c=16`)* | **`3,910.80` tok/s** | `19.62 ms` *(`c=1`, `0%`)* / `18.91 ms` *(`c=16`, `0%`; `43.37–80.99 ms` on `coding_edit`)* | `629.33 ms` | — | [`JSON`](./results/v7x-16/vllm_fp8_tp4_dp8_ep32_ngram5.json) |

### B. TPU `v5p-16` (`2x2x4`, 4 Hosts / 16 Chips, `EP=16`)

| Experiment | Weights / KV Dtype | Parallelism (`16` Chips) | Peak Output Throughput | Peak Total Throughput (incl. Cached Prefix) | Single-Stream `TPOT` (`c=1`, `p50`) / Best `TPOT` | `4K` `0%` `TTFT` (`c=1`) | `64K` (`c=8`, `n=16`) / `128K` (`c=4`, `n=8`) (`0%` Hit) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`vllm_bf16_tp1_dp16_ep16_fp8kv`** | **`BF16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`16` experts/chip = `2.50 GiB`)* | **`9,470.66` tok/s** *(`4K:1K`, `95%`, `c=512`, `n=1024`; `6,256.04` @ `0%`)* / **`8,390.24`** *(`8K:1K`, `95%`, `c=256`, `n=512`)* | **`72,382.22` tok/s** *(`8K:1K`, `95%`, `c=256`)* | **`12.34 ms`** *(`c=1`)* | `1,261.45 ms` | — | [`JSON`](./results/v5p-16/vllm_bf16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_w8a16_tp1_dp16_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`+49.7%` vs `slice16_fp8_tp1_dp16_mrr64_swa010` `5,010.7` @ `8K, 95%, c=256`)* | **`9,018.25` tok/s** *(`4K:1K`, `95%`, `c=512`, `n=1024`; `6,000.31` @ `0%`)* / **`7,501.46`** *(`8K:1K`, `95%`, `c=256`, `n=512`)* | **`64,714.78` tok/s** *(`8K:1K`, `95%`, `c=256`)* | **`16.25 ms`** *(`c=1`)* / `15.09 ms` *(`c=16`, `80%`)* | `1,293.54 ms` | **`200.76` / `57.89` tok/s** *(`16.27 / 17.07 ms` TPOT)* | [`JSON`](./results/v5p-16/vllm_w8a16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_int8_w8a16_tp1_dp16_ep16_fp8kv`** | **`INT8 W8A16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`REQUANTIZE_WEIGHT_DTYPE=int8`)* | **`7,244.99` tok/s** *(`4K:1K`, `95%`, `c=256`, `n=512`)* / `5,728.53` *(`0%`, `c=512`, `n=1024`)* | **`34,866.53` tok/s** *(`95%`)* | **`17.39 ms`** *(`c=1`)* | `1,292.92 ms` | — | [`JSON`](./results/v5p-16/vllm_int8_w8a16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_w8a16_tp2_dp8_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=2 × DP=8 × EP=16`** | **`6,680.05` tok/s** *(`4K:1K`, `95%`, `c=512`, `n=1024`; `4,628.94` @ `0%`)* | **`32,147.74` tok/s** *(`95%`)* | **`14.60 ms`** *(`c=1`)* | `825.40 ms` | — | [`JSON`](./results/v5p-16/vllm_w8a16_tp2_dp8_ep16_fp8kv.json) |
| **`vllm_w8a16_tp4_dp4_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=4 × DP=4 × EP=16`** *(low-latency prefill)* | **`3,694.72` tok/s** *(`4K:1K`, `95%`, `c=256`, `n=512`)* / `444.92` *(`4K:512` `coding_edit`, `c=8`)* | **`17,780.83` tok/s** *(`95%`)* | **`19.01 ms`** *(`c=1`, `0%`)* / `16.24 ms` *(`c=4`, `0%`)* | **`439.46 ms`** *(`2.87×` lower vs `TP=1`)* | — | [`JSON`](./results/v5p-16/vllm_w8a16_tp4_dp4_ep16_fp8kv.json) |

---

## 2. Modifications

1. **`FlintForCausalLM` + `FlintConfig` vLLM Plugin ([`scripts/install_flint_vllm.py`](./scripts/install_flint_vllm.py#L21-L657))**:
   - Registers `model_type="flint"` in `transformers` and `vllm` (`FlintConfig`, `FlintForCausalLM`), implementing Flint's hybrid 49-layer stack: dense SwiGLU on `mlp_only_layers=[0, 1]`, 256-expert top-6 sigmoid-routed `FusedMoE` (`scoring_func="sigmoid"`, `e_score_correction_bias`, `renormalize=True`) + ungated `shared_expert` on layers `2..48`, QK-RMSNorm (`q_norm`, `k_norm`), 14 RoPE-free `full_attention` layers + 35 RoPE `sliding_attention` layers (`sliding_window=513`), and TP-sharded learned per-head `sinks` (`softmax_offset`).
2. **Pallas RPA v3 `head_dim=128` Learned `attention_sink` Online-Softmax Support ([`scripts/install_flint_vllm.py#L753-L912`](./scripts/install_flint_vllm.py#L753-L912))**:
   - Extends `vllm_torchtpu` RPA v3 attention sink support from `head_dim==64` to `head_dim==128` in `attention_interface.py`, broadcasts per-KV-group sinks `[num_kv_heads, q_per_kv, 128]` into TPU VMEM (`BlockSpec(memory_space=pltpu.VMEM)`), and initializes the online-softmax running maximum `m_ref[...] = attention_sink_ref[...]` and denominator `l_ref[...] = 1.0` prior to the paged KV loop in `ragged_paged_attention/v3/kernel.py`.
3. **Slice-Wide Expert Parallelism (`EP=32` on `v7x-16`, `EP=16` on `v5p-16`) + `BF16` at `TP=1` ([`scripts/run_flint_vllm_multihost_suite.py`](./scripts/run_flint_vllm_multihost_suite.py#L720-L960))**:
   - Shards the 256 routed experts across all 32 TensorCores on `v7x-16` (`8` experts/TC = `1.25 GiB` BF16 / `0.63 GiB` FP8) or all 16 chips on `v5p-16` (`16` experts/chip = `2.50 GiB` BF16), enabling **`BF16` at `TP=1`** behind a single `vllm serve` endpoint with internal Ray Data-Parallel Load Balancing (DPLB).
   - Includes a TPU `v5p` SparseCore vector-lane guard ([`run_flint_vllm_multihost_suite.py#L929-L960`](./scripts/run_flint_vllm_multihost_suite.py#L929-L960)) so `top-6` routing falls back to standard TPU top-k when `sc.num_lanes == 8` on `v5p`.
4. **Memory-Backed Kubernetes Manifests ([`k8s/v7x-16-flint-vllm-ep32.yaml`](./k8s/v7x-16-flint-vllm-ep32.yaml), [`k8s/v5p-16-flint-vllm-ep16.yaml`](./k8s/v5p-16-flint-vllm-ep16.yaml))**:
   - Mounts `emptyDir: medium: Memory` at `/dev/shm` and `/tmp` for Ray session logs, TorchInductor artifacts, and XLA compilation caches.

---

## 3. Methodology Notes & Known Limitations

- **Warm-Up & Request Counts:** Each configuration runs 1 warm-up request before timed points. On `v7x-16`, `c=1024` points are single-wave (`n=c=1024`), while `c <= 512` points use `n = 2c` (`n = 4` at `c=1`). `total_tok_s` includes prompt tokens from cached prefixes.
- **Prefix Caching under Ray DPLB:** Internal Ray Data-Parallel Load Balancing is not prefix-affinity routed, and server-side observed prefix cache hit counters are not recorded in the JSON rows (`hit_rate` denotes the client's target shared-prefix fraction).
- **Host-Side `ngram` (`k=5`) Speculative Decoding:** Enabling `ngram` in `vllm-torchtpu v0.32.0` requires `--no-async-scheduling`. Per-token latency with `ngram` (`18.91–19.62 ms` `p50` `TPOT` on `4K:1K`; `43.37–80.99 ms` on `coding_edit`) is `~1.8×` higher than async `TP=2` without speculation (`10.37 ms`). Draft acceptance counters are not stored in the result JSON, and no matched `TP=4` non-speculative `--no-async-scheduling` baseline was measured.
- **Multi-LoRA Constraint:** `vllm-torchtpu v0.32.0` does not support LoRA on TPU with `TP > 1`, `DP > 1`, or MoE models.

---

## 4. Directory Layout

```text
vllm/
├── README.md
├── k8s/
│   ├── v5p-16-flint-vllm-ep16.yaml            # 4-host TPU v5p-16 (2x2x4) Ray + vllm-torchtpu EP=16 Job
│   └── v7x-16-flint-vllm-ep32.yaml            # 4-host TPU v7x-16 (2x2x4) Ray + vllm-torchtpu EP=32 Job
├── scripts/
│   ├── install_flint_vllm.py                  # FlintConfig + FlintForCausalLM + Pallas RPA v3 hd128 sink installer
│   └── run_flint_vllm_multihost_suite.py      # Multi-host Ray launcher & matched benchmark suite
└── results/
    ├── v5p-16/                                # 5 v5p-16 EP=16 JSON archives (85 points, 0 failed)
    └── v7x-16/                                # 4 v7x-16 EP=32 JSON archives (60 points, 0 failed)
```
