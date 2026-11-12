# Single-Host Optimization Matrix & Compiled Multi-LoRA Serving (`v7x` & `v5p`)

- **Full Technical Details & Ablations:** [`../APPENDIX.md`](../APPENDIX.md)
- **Companion Fine-Tuning Repository:** [`ankdeshm/rsi-flint-tpu-lora-qlora-finetuning`](https://github.com/ankdeshm/rsi-flint-tpu-lora-qlora-finetuning)
- **TPU `v7x` Single-Host & Multi-LoRA JSONs (`21` files):** [`./v7x/`](./v7x/)
- **TPU `v5p` Single-Host Multi-LoRA, `TP × DP`, & `128K` RadixCache JSONs (`4` files):** [`./v5p/`](./v5p/)
- **Benchmark & Compiler Patch Scripts (`6` files):** [`./scripts/`](./scripts/)
- **Kubernetes Job Manifests (`4` files):** [`./k8s/`](./k8s/)
