# Serving Benchmark Results Index (`results/`)

Raw measured outputs (`2,034` top-level rows / `2,214` including nested phase records) across all serving campaigns in this repository:

| Subdirectory | TPU Hardware & Engine | Contents |
| :--- | :--- | :--- |
| [`../v7x/results/`](../v7x/results/) | TPU `v7x-16` (`2x2x4`, `sglang-torchtpu`) | `20` JSON files (`774` rows): 16-chip slice sweeps (`DP-Attn=32`, `BF16 mrr=96`, `Block-FP8 swa=0.10`, `BF16 swa=0.80`), scale-up (`TP=2/4/8`), and single-chip ablations |
| [`../v5p/results/raw/`](../v5p/results/raw/) | TPU `v5p-16` (`2x2x4`, `sglang-torchtpu`) | `13` JSONL/JSON files (`831` rows): `v5p16_slice16_optimized.jsonl` (`60` high-concurrency `mrr=96`/`mrr=64` slice rows), `v5p16_sweep_host0..3.jsonl` (`761` rows), and `v5p16_smoke_host0..3.jsonl` (`10` rows) |
| [`../vllm/results/`](../vllm/results/) | TPU `v7x-16` (`EP=32`) & `v5p-16` (`EP=16`) (`vllm-torchtpu v0.32.0`) | `9` JSON files (`145` rows, `0` failed requests) under [`../vllm/results/v7x-16/`](../vllm/results/v7x-16/) and [`../vllm/results/v5p-16/`](../vllm/results/v5p-16/) |
| [`../multilora_singlehost_opt/v7x/`](../multilora_singlehost_opt/v7x/) | Single-Host TPU `v7x` (`4` chips / `8` TensorCores) | `21` JSON files: single-host parallelism, precision, RadixCache, SWA sweeps (`A1..D5`), XLA-compiled Multi-LoRA (`E_sglang_flint_multilora.json`), and `vLLM` comparison (`E3`) |
| [`../multilora_singlehost_opt/v5p/`](../multilora_singlehost_opt/v5p/) | Single-Host TPU `v5p` (`4` chips) | `4` JSON files: compiled Multi-LoRA (`BF16` and `Block-FP8`), `TP×DP`, and `128K` RadixCache / SWA sweeps |

For full tables, saturation points, and methodology notes, see [`../APPENDIX.md`](../APPENDIX.md).
