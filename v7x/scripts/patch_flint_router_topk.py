#!/usr/bin/env python3
"""Surgical idempotent patcher to enable/restore fused TPU Pallas router_topk (256 experts) for RecursiveSI/flint.

Why this matters on TPU v7x (Ironwood):
  - Stock flint_sglang passes a Python custom_routing_function (sigmoid_routing) to SGLang's TopK,
    forcing XLA across all 47 MoE layers to emit a full 256-column TopK sort + dynamic gather.
  - sglang_tpu/moe/router_topk_tpu.py already implements:
      1) sglang_tpu_pallas::router_topk (fused VMEM Pallas sigmoid + bias + topk + renormalize) for batch <= 64
      2) _large_batch_topk (k argmax-and-mask + one-hot reduce rounds, zero XLA sort/gather) for batch > 64 & k <= 8
    gated only on VALIDATED_NUM_EXPERTS = frozenset({288, 896}) and SGLANG's standard biased_topk_impl path!
  - Adding 256 (an exact 2 x 128 TPU VREG multiple!) to VALIDATED_NUM_EXPERTS and routing FlintSparseMoeBlock
    through TopK(..., score_func="sigmoid", correction_bias=self.gate.expert_bias, renormalize=True) activates
    both TPU-optimized paths across all 47 MoE layers!
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def locate_files() -> tuple[Path, Path]:
    import flint_sglang.model as fm
    import sglang_tpu.moe.router_topk_tpu as rt

    return Path(rt.__file__).resolve(), Path(fm.__file__).resolve()


def enable_patch() -> None:
    rt_path, fm_path = locate_files()
    for p in (rt_path, fm_path):
        bak = p.with_suffix(".py.orig_bak")
        if not bak.exists():
            shutil.copy2(p, bak)

    rt_src = rt_path.read_text()
    if "256" not in rt_src.split("VALIDATED_NUM_EXPERTS")[1].splitlines()[0]:
        rt_src = rt_src.replace(
            "VALIDATED_NUM_EXPERTS = frozenset({288, 896})",
            "VALIDATED_NUM_EXPERTS = frozenset({256, 288, 896})",
        )
        rt_path.write_text(rt_src)

    fm_src = fm_path.read_text()
    old_block = """        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=False,  # the routing function renormalizes
            use_grouped_topk=False,
            layer_id=layer_id,
            custom_routing_function=sigmoid_routing,
        )"""
    new_block = """        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=True,
            use_grouped_topk=False,
            correction_bias=self.gate.expert_bias,
            score_func="sigmoid",
            layer_id=layer_id,
        )"""
    if old_block in fm_src:
        fm_src = fm_src.replace(old_block, new_block)
        fm_path.write_text(fm_src)
    elif 'score_func="sigmoid"' not in fm_src:
        raise RuntimeError(f"Could not find expected TopK block in {fm_path}")

    print(f"[router_topk_patch] ENABLED on {rt_path} and {fm_path}")


def restore_patch() -> None:
    rt_path, fm_path = locate_files()
    for p in (rt_path, fm_path):
        bak = p.with_suffix(".py.orig_bak")
        if bak.exists():
            shutil.copy2(bak, p)
    print(f"[router_topk_patch] RESTORED stock files on {rt_path} and {fm_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["enable", "restore"], required=True)
    args = ap.parse_args()
    if args.mode == "enable":
        enable_patch()
    else:
        restore_patch()


if __name__ == "__main__":
    main()
