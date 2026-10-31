#!/usr/bin/env python3
"""Aggregate raw Flint TPU v5p-16 JSONL & smoke JSON results into clean CSV + summary tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "timestamp_utc",
    "phase",
    "config_tag",
    "precision",
    "tp_size",
    "replicas_per_host",
    "slice_replicas_dp",
    "num_hosts_measured",
    "chips_measured",
    "attention_backend",
    "kv_cache_dtype",
    "router_mode",
    "mem_fraction_static",
    "max_running_requests",
    "chunked_prefill_size",
    "weight_gib_per_chip",
    "kv_gib_per_chip",
    "peak_hbm_gib_per_chip",
    "max_total_tokens_per_replica",
    "swa_max_total_tokens_per_replica",
    "boot_s",
    "isl",
    "osl",
    "prefix_hit_target_pct",
    "shared_prefix_tokens",
    "observed_cache_hit_pct",
    "concurrency",
    "concurrency_slice_equiv",
    "num_prompts",
    "completed",
    "failed",
    "wall_s",
    "measured_output_tok_s",
    "measured_total_tok_s",
    "per_host_output_tok_s",
    "per_host_total_tok_s",
    "slice_16chip_output_tok_s",
    "slice_16chip_total_tok_s",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_e2e_s",
    "median_e2e_s",
    "p99_e2e_s",
    "median_stream_tok_s",
    "first_error",
]


def load_jsonl_records(raw_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(raw_dir.glob("*.jsonl")):
        for line in p.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Deduplicate by (phase, config_tag, isl, osl, prefix_hit_target_pct, concurrency), keeping latest completed run
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for r in records:
        key = (
            r.get("phase"),
            r.get("config_tag"),
            r.get("isl"),
            r.get("osl"),
            r.get("prefix_hit_target_pct"),
            r.get("concurrency"),
        )
        prev = dedup.get(key)
        if prev is None or (r.get("completed", 0) >= prev.get("completed", 0)):
            dedup[key] = r
    return sorted(
        dedup.values(),
        key=lambda x: (
            str(x.get("phase", "")),
            str(x.get("config_tag", "")),
            int(x.get("isl", 0)),
            int(x.get("prefix_hit_target_pct", 0)),
            int(x.get("concurrency", 0)),
        ),
    )


def write_csv(records: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    args = ap.parse_args()

    recs = load_jsonl_records(args.raw_dir)
    write_csv(recs, args.out_csv)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"Aggregated {len(recs)} unique benchmark points -> {args.out_csv} & {args.out_jsonl}")


if __name__ == "__main__":
    main()
