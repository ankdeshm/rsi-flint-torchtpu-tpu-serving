#!/usr/bin/env python3
"""Fast multi-process BF16 -> Block-FP8 [128, 128] converter for RecursiveSI/flint.

Converts the 15 BF16 safetensors shards (~79.57 GB) into SGLang/vLLM-compatible
[128, 128] dynamic block-FP8 safetensors (~40.44 GB) in ~20s across CPU cores,
preserving BF16 on embeddings, lm_head, router gates (mlp.gate), RMSNorms, and
learned attention sinks (softmax_offset). Optionally uploads to regional GCS.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import shutil
import time
import urllib.parse
import urllib.request
import os
from pathlib import Path
from typing import Any

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_num_threads(4)


QUANT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
    ".mlp.shared_expert.gate_proj.weight",
    ".mlp.shared_expert.up_proj.weight",
    ".mlp.shared_expert.down_proj.weight",
)


def should_quantize_tensor(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    if name in ("lm_head.weight", "model.embed_tokens.weight"):
        return False
    if name.endswith(".mlp.gate.weight"):
        return False
    if ".mlp.experts." in name and name.endswith((".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")):
        return True
    return name.endswith(QUANT_SUFFIXES)


def quantize_block_fp8_128(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max  # 448.0
    n, k = w.shape
    if n % 128 != 0 or k % 128 != 0:
        raise ValueError(f"Expected weight shape divisible by 128, got {(n, k)}")
    w_f32 = w.to(torch.float32).view(n // 128, 128, k // 128, 128)
    amax = w_f32.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    scale_inv = amax / fp8_max
    w_q = (w_f32 / scale_inv).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    return (
        w_q.view(n, k).contiguous(),
        scale_inv.squeeze(3).squeeze(1).to(torch.float32).contiguous(),
    )


def _convert_one_shard(args: tuple[str, str, str]) -> tuple[str, dict[str, str], int]:
    torch.set_num_threads(4)
    src_dir_s, dst_dir_s, shard_name = args
    src_path = Path(src_dir_s) / shard_name
    dst_path = Path(dst_dir_s) / shard_name
    out_tensors: dict[str, torch.Tensor] = {}
    shard_map: dict[str, str] = {}
    shard_bytes = 0

    with safe_open(str(src_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if should_quantize_tensor(key, tuple(t.shape)):
                w_q, s_inv = quantize_block_fp8_128(t)
                scale_key = f"{key}_scale_inv"
                out_tensors[key] = w_q
                out_tensors[scale_key] = s_inv
                shard_map[key] = shard_name
                shard_map[scale_key] = shard_name
                shard_bytes += w_q.numel() * w_q.element_size() + s_inv.numel() * s_inv.element_size()
            else:
                t_c = t.contiguous()
                out_tensors[key] = t_c
                shard_map[key] = shard_name
                shard_bytes += t_c.numel() * t_c.element_size()

    save_file(out_tensors, str(dst_path), metadata={"format": "pt"})
    return shard_name, shard_map, shard_bytes


def convert_flint_bf16_to_fp8(src_dir: Path, dst_dir: Path, workers: int = 15) -> Path:
    marker = dst_dir / ".ready_fp8_128"
    if marker.exists() and (dst_dir / "config.json").exists():
        print(f"[fp8-conv] Reusing ready FP8 checkpoint at {dst_dir}", flush=True)
        return dst_dir

    t0 = time.time()
    dst_dir.mkdir(parents=True, exist_ok=True)
    idx_data = json.loads((src_dir / "model.safetensors.index.json").read_text())
    shards = sorted(set(idx_data["weight_map"].values()))
    print(f"[fp8-conv] Converting {len(shards)} shards from {src_dir} -> {dst_dir} (workers={workers})...", flush=True)

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    work = [(str(src_dir), str(dst_dir), s) for s in shards]
    with cf.ProcessPoolExecutor(max_workers=min(workers, len(shards))) as ex:
        for shard_name, smap, sbytes in ex.map(_convert_one_shard, work):
            new_weight_map.update(smap)
            total_bytes += sbytes
            print(f"[fp8-conv] Finished {shard_name} ({sbytes / (1024**3):.2f} GiB)", flush=True)

    # Copy non-safetensor metadata/tokenizer files
    for p in src_dir.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst_dir / p.name)

    cfg: dict[str, Any] = json.loads((src_dir / "config.json").read_text())
    num_layers = int(cfg.get("num_hidden_layers", 49))
    dense_layers = set(cfg.get("mlp_only_layers", [0, 1]))
    ignored = ["lm_head", "model.embed_tokens"]
    for lid in range(num_layers):
        if lid not in dense_layers:
            ignored.append(f"model.layers.{lid}.mlp.gate")

    cfg["quantization_config"] = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": ignored,
    }
    (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    new_idx = {
        "metadata": {"total_size": total_bytes},
        "weight_map": dict(sorted(new_weight_map.items())),
    }
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(new_idx, indent=2) + "\n")
    marker.write_text("ok\n")
    print(
        f"[fp8-conv] Completed FP8 conversion in {time.time() - t0:.1f}s: "
        f"{len(new_weight_map)} tensors, {total_bytes / (1024**3):.2f} GiB",
        flush=True,
    )
    return dst_dir


def _vm_token() -> str:
    req = urllib.request.Request(
        "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())["access_token"]


def upload_dir_to_gcs(local_dir: Path, bucket: str, prefix: str, workers: int = 16) -> None:
    prefix = prefix.strip("/") + "/"
    all_files = sorted(p for p in local_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    non_cfg = [p for p in all_files if p.name != "config.json"]
    cfg_only = [p for p in all_files if p.name == "config.json"]

    def _up(p: Path) -> str:
        obj = prefix + p.name
        for att in range(4):
            try:
                tok = _vm_token()
                url = (
                    f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o"
                    f"?uploadType=media&name={urllib.parse.quote(obj, safe='')}"
                )
                with open(p, "rb") as f:
                    req = urllib.request.Request(
                        url,
                        data=f,
                        method="POST",
                        headers={
                            "Authorization": f"Bearer {tok}",
                            "Content-Type": "application/octet-stream",
                            "Content-Length": str(p.stat().st_size),
                        },
                    )
                    with urllib.request.urlopen(req, timeout=600):
                        return p.name
            except Exception:
                if att == 3:
                    raise
                time.sleep(2 * (att + 1))
        return p.name

    t0 = time.time()
    print(f"[fp8-conv] Uploading {len(all_files)} FP8 files to gs://{bucket}/{prefix}...", flush=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_up, non_cfg))
    for p in cfg_only:
        _up(p)
    print(f"[fp8-conv] Uploaded FP8 checkpoint to gs://{bucket}/{prefix} in {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--gcs-bucket", default="")
    ap.add_argument("--gcs-prefix", default="models/RecursiveSI/flint-fp8")
    args = ap.parse_args()
    convert_flint_bf16_to_fp8(args.src, args.dst, workers=args.workers)
    if args.gcs_bucket:
        upload_dir_to_gcs(args.dst, args.gcs_bucket, args.gcs_prefix)


if __name__ == "__main__":
    main()
