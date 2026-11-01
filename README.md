# `RecursiveSI/flint` (`40B-A2.5B` MoE) High-Throughput & Long-Context (`128K`) Serving on Cloud TPU (`v7x-16` Ironwood & `v5p-16`)

End-to-end production serving stack, `[128, 128]` dynamic Block-FP8 checkpoint converter, fused VMEM Pallas `router_topk` (`256`-expert) kernel, sub-host VFIO / metadata replica isolation, prefix-trunk affinity HTTP gateway, and empirical serving benchmarks up to **`131,072` (`128K`) context** for **[`RecursiveSI/flint`](https://huggingface.co/RecursiveSI/flint)** (`39.8B` total / `~2.5B` active MoE) on **Google Cloud TPU `v7x-16` (`Ironwood`, `2x2x4`)** and **TPU `v5p-16` (`2x2x4`)** using **`sglang-torchtpu` + TorchTPU `0.2.0` GA**.

---

## 1. Headline Serving Results (`v7x-16` vs `v5p-16` — 1,307 Empirical Benchmark Points)

| Platform & Slice Topology | Best Parallelism & Precision | Peak 16-Chip Output Throughput | Best Single-Stream Latency (`TPOT`) | `64K` (`65,536:1024`) `95%` Hit Throughput | `128K` (`131,072:1024`) `95%` Hit Throughput | Usable Slice HBM |
|---|---|---:|---:|---:|---:|---:|
| **TPU `v7x-16` (`2x2x4`, 16 Ironwood Chips / 32 TensorCores)** | **`BF16` Weights + `FP8` KV, Intra-Chip `TP=2 × DP=16` (`0` Inter-Chip ICI)** | **`33,421.5` tok/s** *(`4K:1K`, `c=512`, `95%` hit)* / **`29,833.1` tok/s** *(`0%` hit)* | **`~6.1 ms`** *(`>160` tok/s/stream)* | **`6,643.2` tok/s** *(`985 ms` TTFT @ `c=1`)* | **`4,864.2` tok/s** *(`2,268 ms` TTFT @ `c=1`)* | **`3,031.7 GiB`** (`189.5 GiB/chip`) |
| **TPU `v7x-16` (`2x2x4`, 16 Ironwood Chips — Block-FP8 `swa=0.10`)** | **`Block-FP8` `[128,128]` + `FP8` KV (`swa_ratio=0.10`, `117.8M` KV), Intra-Chip `TP=2 × DP=16`** | **`33,200.3` tok/s** *(`4K:1K`, `95%`)* / **`31,432.8` tok/s** *(`8K:1K` #1!)* / **`25,737.5` tok/s** *(`16K:1K`, `95%`)* | **`6.49 ms`** *(`154.1` tok/s/stream)* | **`7,225.0` tok/s** *(`95%` #1!)* / **`6,013.0` tok/s** *(`80%` #1!)* / **`3,695.0` tok/s** *(`0%` #1!)* | **`5,172.5` tok/s** *(`95%` #1!)* / **`3,559.9` tok/s** *(`80%` #1!)* / **`1,770.4` tok/s** *(`0%` #1, `0` fail @ `c=512`!)* | **`3,031.7 GiB`** (`20.2 GiB` weights/TC) |
| **TPU `v5p-16` (`2x2x4`, 16 `v5p` Chips)** | **`BF16` Weights + `FP8` KV, `TP=2 × DP=8` (`bf16_tp2_dp2_fp8kv_c8k`)** | **`17,719.1` tok/s** *(4-host scaled @ `4K:1K` `c=512`)* / **`10,216.2` tok/s** *(empirical 16-chip @ `8K:1K` `c=256`)* | **`7.23 ms`** *(`138.24` tok/s/stream @ `TP=4`)* / **`7.48 ms`** *(`133.65` tok/s @ `TP=2`)* | **`2,658.4` tok/s** *(empirical `TP=4 × DP=4`)* | **`1,748.8` tok/s** *(empirical `BF16`)* / **`953.6` tok/s** *(`FP8 TP=4 × DP=4`, `10.49 ms` TPOT @ `c=4`)* | **`1,515.8 GiB`** (`94.7 GiB/chip`) |
| **TPU `v5p-16` (`2x2x4`, 16 `v5p` Chips — Single-Chip Replica Mode)** | **`Block-FP8` `[128,128]` + `FP8` KV, `TP=1 × DP=16` (`16` Replicas, `0` ICI)** | **`8,485.2` tok/s** *(4-host scaled @ `4K:1K`)* / **`5,166.3` tok/s** *(empirical 16-chip @ `8K:1K` `c=256`)* | **`12.97 ms`** *(`77.08` tok/s/stream, `79.4 GiB` HBM/chip)* | **`1,412.6` tok/s** | **`912.4` tok/s** | **`1,515.8 GiB`** (`37.7 GiB` weights/chip) |

> **Key Architectural Insight (`v7x` vs `v5p` Precision Behavior):**
> - On **TPU `v5p`**, MXUs execute native `BF16` systolic matmuls without hardware Block-FP8 accumulation, while `BF16` weights at `TP=2` (`37.1 GiB/chip`) fit comfortably inside `95 GiB` HBM alongside `43+ GiB` of `FP8` (`e4m3`) paged KV cache — making **`BF16` weights + `FP8` KV cache (`TP=2 × DP=8`) `2.1×` higher throughput and `1.47×` lower TPOT (`7.48 ms` vs `10.94 ms`)** than software-dequantized Block-FP8 on `v5p`!
> - On **TPU `v7x` (`Ironwood`)**, each physical package integrates **2 TensorCores (`189.5 GiB` HBM3e)** over intra-package die-to-die interconnect plus native FP8/BF16 MXUs, enabling **`16` zero-ICI `TP=2` replicas (`33,421.5` tok/s)** or **`32` single-TensorCore `Block-FP8 TP=1` replicas** across a single 16-chip slice (`3.11×–3.74×` higher throughput and `11.4×` faster cold JIT compilation than `v5p-16`).

---

## 2. Repository Structure

- **[`v7x/`](./v7x/README.md)** — Complete **TPU `v7x-16` (`Ironwood`)** serving stack, sub-host VFIO multi-replica launcher (`TP=1/2/4/8`), fused 256-expert Pallas `router_topk` patch, `[128, 128]` Block-FP8 converter, prefix-affinity gateway, [`v7x/docs/FLINT_V7X_SERVING_RESULTS.md`](./v7x/docs/FLINT_V7X_SERVING_RESULTS.md), and `532+` raw benchmark points in [`v7x/results/`](./v7x/results/).
- **[`v5p/`](./v5p/README.md)** — Complete **TPU `v5p-16` (`2x2x4`)** serving stack, single-host metadata isolation shim, async prefix-affinity gateway, Kubernetes `smoke` + `full_sweep` manifests, and **`761/761` (`100%` pass) empirical single-host & 16-chip slice JSONL results** across `4K..128K` context in [`v5p/results/raw/`](./v5p/results/raw/).
- **[`docs/FLINT_V7X_SERVING_RESULTS.md`](./docs/FLINT_V7X_SERVING_RESULTS.md)** — Deep-dive engineering & roofline report.
