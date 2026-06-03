#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-pass deduplication for latamgpt-gen JSONL output.

Pass 1 — Exact dedup:   hash of the first user message.
Pass 2 — Near dedup:    MinHash LSH on the full conversation text (char n-grams).

Usage:
    python dedup.py data/es_20k.jsonl --out data/es_20k_dedup.jsonl
    python dedup.py data/*.jsonl --out data/all_dedup.jsonl   # merge + dedup
    python dedup.py data/es_20k.jsonl --threshold 0.85 --dry-run
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


# ─── Helpers ──────────────────────────────────────────────────────────────────

def conv_text(messages: list) -> str:
    """Flatten all user+assistant turns to a single string for hashing."""
    return " ".join(
        m["content"] for m in messages if m["role"] in ("user", "assistant")
    )


def first_user_message(messages: list) -> str:
    for m in messages:
        if m["role"] == "user":
            return m["content"]
    return ""


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace for more robust comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def make_minhash(text: str, num_perm: int = 128) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    # Character 5-grams — language-agnostic, robust for short texts
    norm = normalize(text)
    for i in range(max(1, len(norm) - 4)):
        mh.update(norm[i : i + 5].encode("utf-8"))
    return mh


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(input_paths: list[Path], out_path: Path, threshold: float, num_perm: int, dry_run: bool) -> None:
    # Read all records
    records = []
    for path in input_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"  WARN: skipping malformed line in {path}: {e}", file=sys.stderr)

    total = len(records)
    print(f"Loaded {total:,} conversations from {len(input_paths)} file(s)")

    # ── Pass 1: exact dedup on first user message ──────────────────────────────
    seen_exact: set[str] = set()
    after_exact = []
    for rec in records:
        key = hashlib.sha256(normalize(first_user_message(rec["messages"])).encode()).hexdigest()
        if key not in seen_exact:
            seen_exact.add(key)
            after_exact.append(rec)

    exact_removed = total - len(after_exact)
    print(f"Pass 1 (exact):  removed {exact_removed:,}  →  {len(after_exact):,} remain")

    # ── Pass 2: near-dedup with MinHash LSH ────────────────────────────────────
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    near_removed = 0

    for idx, rec in enumerate(tqdm(after_exact, desc="MinHash LSH")):
        text = conv_text(rec["messages"])
        mh = make_minhash(text, num_perm=num_perm)
        key = f"conv_{idx}"

        if lsh.query(mh):
            near_removed += 1
        else:
            lsh.insert(key, mh)
            kept.append(rec)

    print(f"Pass 2 (near):   removed {near_removed:,}  →  {len(kept):,} remain")
    print(f"Total removed:   {total - len(kept):,} / {total:,}  ({(total - len(kept)) / total * 100:.1f}%)")

    if dry_run:
        print("Dry run — nothing written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(kept):,} conversations → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate latamgpt-gen JSONL output")
    p.add_argument("inputs", nargs="+", help="Input JSONL file(s)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--threshold", type=float, default=0.80,
                   help="Jaccard similarity threshold for near-dedup (default: 0.80)")
    p.add_argument("--num-perm", type=int, default=128,
                   help="MinHash permutations — higher = more accurate, slower (default: 128)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing output")
    args = p.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for path in input_paths:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    run(input_paths, Path(args.out), args.threshold, args.num_perm, args.dry_run)


if __name__ == "__main__":
    main()
