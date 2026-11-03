# `RecursiveSI/flint` (`40B-A2.5B` MoE) Multi-Host `vllm-torchtpu` Serving (`EP=32` on `v7x-16` & `EP=16` on `v5p-16`)

Native single-endpoint OpenAI serving stack, `FlintForCausalLM` + `FlintConfig` vLLM model plugin, Pallas Ragged Paged Attention v3 (`head_dim=128`) learned `attention_sink` online-softmax kernel patch, Kubernetes Ray multi-host manifests, and benchmark archives for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.5B` active MoE) across 4-host **TPU `v7x-16` (`2x2x4`, 16 chips / 32 TensorCores)** and **TPU `v5p-16` (`2x2x4`, 16 chips)** using **`vllm-torchtpu` (`v0.32.0`)**.

---

## 1. Benchmark Summary (`145` Benchmark Points, `0` Failed Requests)

### A. TPU `v7x-16` (`2x2x4`, 4 Hosts / 16 Chips / 32 TensorCores, `EP=32`)

| Experiment | Weights / KV Dtype | Parallelism (`32` TensorCores) | Peak Output Throughput | Peak Total Throughput | Best `TPOT` (`p50`) | `4K` `0%` `TTFT` (`c=1`) | `64K` / `128K` (`0%` Hit, `c=8`) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`vllm_bf16_tp1_dp32_ep32_fp8kv`** | **`BF16` / `FP8`** | **`TP=1 × DP=32 × EP=32`** *(`8` experts/TC = `1.25 GiB`)* | **`27,091.93` tok/s** *(`4K:1K`, `c=1024`, `80%`)* / **`26,912.29`** *(`95%`)* | **`130,432.81` tok/s** | **`10.92 ms`** *(`92.00` tok/s @ `c=32`)* | `708.97 ms` | — | [`JSON`](./results/v7x-16/vllm_bf16_tp1_dp32_ep32_fp8kv.json) |
| **`vllm_fp8_tp1_dp32_ep32_fp8kv`** | **`Block-FP8` / `FP8`** | **`TP=1 × DP=32 × EP=32`** *(`8` experts/TC = `0.63 GiB`)* | **`26,145.93` tok/s** *(`4K:1K`, `c=1024`, `95%`)* / **`19,153.33`** *(`0%`)* | **`125,827.28` tok/s** *(`102,420.62` @ `8K, 0%`)* | **`10.65 ms`** *(`94.18` tok/s @ `c=32`)* | `856.65 ms` | **`308.56` / `189.67` tok/s** *(`11.40 / 11.48 ms` TPOT)* | [`JSON`](./results/v7x-16/vllm_fp8_tp1_dp32_ep32_fp8kv.json) |
| **`vllm_fp8_tp2_dp16_ep32_fp8kv`** | **`Block-FP8` / `FP8`** | **`TP=2 × DP=16 × EP=32`** *(mid-concurrency winner `c=1..256`)* | **`20,336.83` tok/s** *(`c=1024`)* / **`12,264.51`** *(`c=256`, `19.12 ms`)* | **`97,871.00` tok/s** | **`10.37 ms`** *(`96.44` tok/s @ `c=1`)* | **`549.57 ms`** | — | [`JSON`](./results/v7x-16/vllm_fp8_tp2_dp16_ep32_fp8kv.json) |
| **`vllm_fp8_tp4_dp8_ep32_ngram5`** | **`Block-FP8` / `FP8`** | **`TP=4 × DP=8 × EP=32` + `ngram` (`k=5`, `--no-async-scheduling`)** | **`812.63` tok/s** *(`4K:1K`, `c=16`, `95%`)* / `173.38` *(`coding_edit`)* | **`3,910.80` tok/s** | `18.91 ms` *(`52.99` tok/s; `87.6–100%` accept)* | `629.33 ms` | — | [`JSON`](./results/v7x-16/vllm_fp8_tp4_dp8_ep32_ngram5.json) |

### B. TPU `v5p-16` (`2x2x4`, 4 Hosts / 16 Chips, `EP=16`)

| Experiment | Weights / KV Dtype | Parallelism (`16` Chips) | Peak Output Throughput | Peak Total Throughput | Best `TPOT` (`p50`) | `4K` `0%` `TTFT` (`c=1`) | `64K` (`c=8`) / `128K` (`c=4`) | Result JSON |
|---|---|---|---:|---:|---:|---:|---:|---|
| **`vllm_bf16_tp1_dp16_ep16_fp8kv`** | **`BF16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`16` experts/chip = `2.50 GiB`)* | **`9,470.66` tok/s** *(`4K:1K`, `c=512`, `95%`)* / **`8,390.24`** *(`8K:1K`, `c=256`)* | **`72,382.22` tok/s** *(`8K:1K`, `c=256`, `95%`)* | **`12.34 ms`** *(`81.12` tok/s @ `c=1`)* | `1,261.45 ms` | — | [`JSON`](./results/v5p-16/vllm_bf16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_w8a16_tp1_dp16_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`+45.2%` vs SGLang `FP8 TP=1` @ `8K`)* | **`9,018.25` tok/s** *(`4K:1K`, `c=512`, `95%`)* / **`7,501.46`** *(`8K:1K`, `c=256`)* | **`64,714.78` tok/s** *(`8K:1K`, `c=256`, `95%`)* | `15.09 ms` *(`66.70` tok/s @ `c=16`)* | `1,293.54 ms` | **`200.76` / `57.89` tok/s** *(`16.27 / 17.07 ms` TPOT)* | [`JSON`](./results/v5p-16/vllm_w8a16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_int8_w8a16_tp1_dp16_ep16_fp8kv`** | **`INT8 W8A16` / `FP8`** | **`TP=1 × DP=16 × EP=16`** *(`REQUANTIZE_WEIGHT_DTYPE=int8`)* | **`7,244.99` tok/s** *(`4K:1K`, `c=256`, `95%`)* / `5,728.53` *(`c=512`, `0%`)* | **`34,866.53` tok/s** | `17.39 ms` *(`57.51` tok/s @ `c=1`)* | `1,292.92 ms` | — | [`JSON`](./results/v5p-16/vllm_int8_w8a16_tp1_dp16_ep16_fp8kv.json) |
| **`vllm_w8a16_tp2_dp8_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=2 × DP=8 × EP=16`** | **`6,680.05` tok/s** *(`4K:1K`, `c=512`, `95%`)* | **`32,147.74` tok/s** | **`14.60 ms`** *(`68.48` tok/s @ `c=1`)* | `825.40 ms` | — | [`JSON`](./results/v5p-16/vllm_w8a16_tp2_dp8_ep16_fp8kv.json) |
| **`vllm_w8a16_tp4_dp4_ep16_fp8kv`** | **`Block-FP8 W8A16` / `FP8`** | **`TP=4 × DP=4 × EP=16`** *(low-latency prefill)* | **`3,694.72` tok/s** *(`4K:1K`, `c=256`, `95%`)* / `444.92` *(`coding_edit`)* | **`17,780.83` tok/s** | `16.24 ms` *(`61.40` tok/s @ `c=4`)* | **`439.46 ms`** *(`2.87×` faster)* | — | [`JSON`](./results/v5p-16/vllm_w8a16_tp4_dp4_ep16_fp8kv.json) |

---

## 2. Key Engineering Contributions

1. **Native `FlintForCausalLM` + `FlintConfig` vLLM Plugin ([`scripts/install_flint_vllm.py`](./scripts/install_flint_vllm.py#L21-L657))**:
   - Registers `model_type="flint"` directly in `transformers` and `vllm` (`FlintConfig`, `FlintForCausalLM`), implementing Flint's hybrid 49-layer stack: dense SwiGLU on `mlp_only_layers=[0, 1]`, 256-expert top-6 sigmoid-routed `FusedMoE` (`scoring_func="sigmoid"`, `e_score_correction_bias`, `renormalize=True`) + ungated `shared_expert` on layers `2..48`, QK-RMSNorm (`q_norm`, `k_norm`), 14 RoPE-free `full_attention` layers + 35 RoPE `sliding_attention` layers (`sliding_window=513`), and TP-sharded learned per-head `sinks` (`softmax_offset`).
2. **Pallas RPA v3 `head_dim=128` Learned `attention_sink` Online-Softmax Patch ([`scripts/install_flint_vllm.py#L753-L912`](./scripts/install_flint_vllm.py#L753-L912))**:
   - Stock `vllm_torchtpu` raised `NotImplementedError("Attention sink support is only available when head_dim==64")`.
   - `patch_rpa_v3_attention_sink()` lifts the `head_dim==64` gate in `attention_interface.py`, broadcasts per-KV-group sinks `[num_kv_heads, q_per_kv, 128]` into TPU VMEM (`BlockSpec(memory_space=pltpu.VMEM)`), and initializes the online-softmax running maximum `m_ref[...] = attention_sink_ref[...]` and denominator `l_ref[...] = 1.0` prior to the paged KV loop in `ragged_paged_attention/v3/kernel.py`.
3. **Slice-Wide Expert Parallelism (`EP=32` on `v7x-16`, `EP=16` on `v5p-16`) + Pure `BF16` at `TP=1` ([`scripts/run_flint_vllm_multihost_suite.py`](./scripts/run_flint_vllm_multihost_suite.py#L720-L960))**:
   - Shards the 256 routed experts across all 32 TensorCores on `v7x-16` (`8` experts/TC = `1.25 GiB` BF16 / `0.63 GiB` FP8) or all 16 chips on `v5p-16` (`16` experts/chip = `2.50 GiB` BF16), enabling **Pure `BF16` at `TP=1`** (impossible without `EP`, where `74.1 GiB` BF16 weights + KV exceed single-chip `95 GiB` HBM) behind a **single `vllm serve` endpoint** with internal Ray Data-Parallel Load Balancing (DPLB).
   - Includes a TPU `v5p` SparseCore vector-lane guard ([`run_flint_vllm_multihost_suite.py#L929-L960`](./scripts/run_flint_vllm_multihost_suite.py#L929-L960)) so `top-6` routing cleanly falls back when `sc.num_lanes == 8` on `v5p`.
4. **Zero-Disk-Pressure Memory-Backed Kubernetes Manifests ([`k8s/v7x-16-flint-vllm-ep32.yaml`](./k8s/v7x-16-flint-vllm-ep32.yaml), [`k8s/v5p-16-flint-vllm-ep16.yaml`](./k8s/v5p-16-flint-vllm-ep16.yaml))**:
   - Mounts `emptyDir: medium: Memory` at both `/dev/shm` and `/tmp` so Ray session logs, TorchInductor artifacts, and XLA compilation caches never touch Container-Optimized OS boot disks.

---

## 3. Directory Layout

```text
vllm/
├── README.md
├── k8s/
│   ├── v5p-16-flint-vllm-ep16.yaml            # 4-host TPU v5p-16 (2x2x4) Ray + vllm-torchtpu EP=16 Job
│   └── v7x-16-flint-vllm-ep32.yaml            # 4-host TPU v7x-16 (2x2x4) Ray + vllm-torchtpu EP=32 Job
├── scripts/
│   ├── install_flint_vllm.py                  # FlintConfig + FlintForCausalLM + Pallas RPA v3 hd128 sink installer
│   └── run_flint_vllm_multihost_suite.py      # Idempotent multi-host Ray launcher & matched benchmark suite
└── results/
    ├── v5p-16/                                # 5 complete v5p-16 EP=16 JSON archives (85 points, 0 errors)
    └── v7x-16/                                # 4 complete v7x-16 EP=32 JSON archives (60 points, 0 errors)
```
