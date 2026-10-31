#!/usr/bin/env python3
"""Fast parallel BF16 -> Block-FP8 [128, 128] checkpoint converter for RecursiveSI/flint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
BLOCK = 128


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return False
    if ".mlp.gate.weight" in name:
        return False
    n, k = tensor.shape
    assert n % BLOCK == 0 and k % BLOCK == 0, f"Unexpected non-128-aligned weight {name}: {tensor.shape}"
    return True


def convert_shard(args_tuple: tuple[str, str, str]) -> tuple[str, dict[str, str], int]:
    shard_name, src_dir_s, dst_dir_s = args_tuple
    torch.set_num_threads(4)
    src_path = Path(src_dir_s) / shard_name
    dst_path = Path(dst_dir_s) / shard_name

    tensors = load_file(str(src_path), device="cpu")
    out_tensors: dict[str, torch.Tensor] = {}
    shard_map: dict[str, str] = {}
    shard_bytes = 0

    for name, w in tensors.items():
        if should_quantize(name, w):
            n, k = w.shape
            w_f32 = w.float().reshape(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
            amax = w_f32.abs().amax(dim=(1, 3))
            scale_inv = torch.clamp(amax / FP8_MAX, min=1e-12).to(torch.float32).contiguous()
            w_q = (
                torch.clamp(w_f32 / scale_inv[:, None, :, None], min=-FP8_MAX, max=FP8_MAX)
                .to(FP8_DTYPE)
                .reshape(n, k)
                .contiguous()
            )
            scale_name = f"{name}_scale_inv"
            out_tensors[name] = w_q
            out_tensors[scale_name] = scale_inv
            shard_map[name] = shard_name
            shard_map[scale_name] = shard_name
            shard_bytes += w_q.numel() * w_q.element_size() + scale_inv.numel() * scale_inv.element_size()
        else:
            t_c = w.contiguous()
            out_tensors[name] = t_c
            shard_map[name] = shard_name
            shard_bytes += t_c.numel() * t_c.element_size()

    save_file(out_tensors, str(dst_path), metadata={"format": "pt"})
    return shard_name, shard_map, shard_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/dev/shm/flint")
    ap.add_argument("--dst", default="/dev/shm/flint-fp8")
    ap.add_argument("--workers", type=int, default=min(15, os.cpu_count() or 8))
    args = ap.parse_args()

    src_dir = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    marker = dst_dir / ".fp8_ready"
    if marker.exists():
        print(f"[fp8-convert] {dst_dir} already converted (.fp8_ready present).", flush=True)
        return

    t0 = time.time()
    # Copy non-safetensors metadata / tokenizer files
    for p in src_dir.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in (
            "config.json",
            "model.safetensors.index.json",
            ".staged_ok",
        ):
            shutil.copy2(p, dst_dir / p.name)

    shards = sorted(p.name for p in src_dir.glob("*.safetensors"))
    assert shards, f"No .safetensors shards found in {src_dir}"
    print(f"[fp8-convert] Converting {len(shards)} shards ({args.workers} workers): {src_dir} -> {dst_dir}", flush=True)

    full_weight_map: dict[str, str] = {}
    total_bytes = 0
    work = [(s, str(src_dir), str(dst_dir)) for s in shards]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for shard_name, smap, sbytes in pool.map(convert_shard, work):
            full_weight_map.update(smap)
            total_bytes += sbytes
            print(f"[fp8-convert] Finished {shard_name} ({sbytes / 1e9:.2f} GB)", flush=True)

    # Write updated index
    index_obj = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(full_weight_map.items())),
    }
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index_obj, indent=2) + "\n")

    # Write updated config.json with Fp8Config
    cfg = json.loads((src_dir / "config.json").read_text())
    cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [BLOCK, BLOCK],
        "ignored_layers": [
            "lm_head",
            "model.embed_tokens",
            "mlp.gate",
        ],
    }
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    marker.write_text("ok\n")
    dt = max(time.time() - t0, 1e-3)
    print(
        f"[fp8-convert] DONE! Total FP8 size={total_bytes / 1e9:.2f} GB ({len(full_weight_map)} tensors) in {dt:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
