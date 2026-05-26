"""
Phase 1 — Stratified Train / Val Split

Reads data/processed/train_cot.jsonl, splits by category (stratified),
writes to data/splits/train_cot.jsonl and data/splits/val_cot.jsonl.

Usage:
    python scripts/split_data.py \
        --input  data/processed/train_cot.jsonl \
        --val_ratio 0.15 \
        --seed 42
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def stratified_split(
    rows: list[dict], val_ratio: float, seed: int
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row.get("category", "other")].append(row)

    train_rows, val_rows = [], []
    for cat, items in by_category.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
        print(f"  {cat:<20}: {len(items):>5} total → {len(items)-n_val:>5} train, {n_val:>4} val")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified train/val split")
    parser.add_argument("--input", default="data/processed/train_cot.jsonl")
    parser.add_argument("--train_out", default="data/splits/train_cot.jsonl")
    parser.add_argument("--val_out", default="data/splits/val_cot.jsonl")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    print(f"Loaded {len(rows)} verified triples from {args.input}")
    print(f"Val ratio: {args.val_ratio:.0%}")
    print()

    train_rows, val_rows = stratified_split(rows, args. b, args.seed)

    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.val_out, val_rows)

    print()
    print(f"Train → {args.train_out}  ({len(train_rows)} rows)")
    print(f"Val   → {args.val_out}  ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
