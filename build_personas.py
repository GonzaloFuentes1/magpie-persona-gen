#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replace data/personas/ with personas from latam-gpt/personas (HuggingFace).

Detects country from persona text when mentioned; otherwise assigns randomly
weighted by approximate country population share among LATAM Spanish speakers.
Portuguese personas (detected via exclusive chars/words) go to brasil/.

Usage:
    python build_personas.py                    # default: up to 300 per country
    python build_personas.py --per-country 500
    python build_personas.py --dry-run          # print stats, don't write files
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

HF_DATASET = "latam-gpt/personas"

PERSONAS_DIR = Path(__file__).parent / "data" / "personas"

# Canonical country dir names (must match load_personas() expectations)
ES_COUNTRIES = [
    "argentina", "bolivia", "chile", "colombia", "costa_rica", "cuba",
    "ecuador", "el_salvador", "guatemala", "honduras", "méxico", "nicaragua",
    "panamá", "paraguay", "perú", "puerto_rico", "república_dominicana",
    "uruguay", "venezuela",
]

# Approximate Spanish-speaking population weights for random assignment
ES_WEIGHTS = [
    45, 12, 19, 51, 5, 11,    # argentina..cuba
    18, 6, 17, 10, 128, 7,    # ecuador..nicaragua
    4, 7, 33, 3, 11,          # panamá..rep_dominicana
    3, 29,                    # uruguay, venezuela
]

# Map text mentions → canonical dir name
_COUNTRY_MAP = {
    "argentina": "argentina",
    "bolivia": "bolivia",
    "chile": "chile",
    "colombia": "colombia",
    "costa rica": "costa_rica",
    "cuba": "cuba",
    "ecuador": "ecuador",
    "el salvador": "el_salvador",
    "guatemala": "guatemala",
    "honduras": "honduras",
    "méxico": "méxico", "mexico": "méxico",
    "nicaragua": "nicaragua",
    "panamá": "panamá", "panama": "panamá",
    "paraguay": "paraguay",
    "perú": "perú", "peru": "perú",
    "puerto rico": "puerto_rico",
    "república dominicana": "república_dominicana",
    "republica dominicana": "república_dominicana",
    "dominicana": "república_dominicana",
    "uruguay": "uruguay",
    "venezuela": "venezuela",
    "brasil": "brasil", "brazil": "brasil",
}

_COUNTRY_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_COUNTRY_MAP, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_PT_INDICATORS = re.compile(
    r"[ãõçâêô]|\b(você|voce|obrigado|obrigada|também|então|isso|estou|fazer|"
    r"minha?|nosso|nossa|esse|essa|tudo|muito|já|ainda|depois|antes|quando|"
    r"onde|quem)\b",
    re.IGNORECASE,
)


def detect_country(text: str) -> str | None:
    m = _COUNTRY_RE.search(text)
    if m:
        return _COUNTRY_MAP.get(m.group(1).lower())
    return None


def is_portuguese(text: str) -> bool:
    return bool(_PT_INDICATORS.search(text))


def main() -> None:
    p = argparse.ArgumentParser(description="Build personas from latam-gpt/personas HF dataset")
    p.add_argument("--per-country", type=int, default=300,
                   help="Max personas per country directory (default: 300)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing files")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    print(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"Loaded {len(ds):,} personas")

    # Bucket personas by country
    buckets: dict[str, list[str]] = defaultdict(list)

    for row in tqdm(ds, desc="Assigning countries"):
        text = row["persona"].strip()
        if not text:
            continue

        if is_portuguese(text):
            buckets["brasil"].append(text)
            continue

        country = detect_country(text)
        if country is None:
            country = random.choices(ES_COUNTRIES, weights=ES_WEIGHTS, k=1)[0]
        buckets[country].append(text)

    print("\nDistribution before capping:")
    for c in sorted(buckets):
        print(f"  {c}: {len(buckets[c]):,}")

    # Cap and shuffle each bucket
    for country in buckets:
        random.shuffle(buckets[country])
        buckets[country] = buckets[country][: args.per_country]

    total = sum(len(v) for v in buckets.values())
    print(f"\nAfter capping at {args.per_country}/country: {total:,} personas total")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    # Wipe existing persona files and rewrite
    if PERSONAS_DIR.exists():
        for f in PERSONAS_DIR.rglob("*.txt"):
            f.unlink()
        print(f"Cleared existing .txt files in {PERSONAS_DIR}")

    written = 0
    for country, personas in buckets.items():
        country_dir = PERSONAS_DIR / country
        country_dir.mkdir(parents=True, exist_ok=True)
        for idx, content in enumerate(personas):
            (country_dir / f"{country}_{idx:06d}.txt").write_text(content, encoding="utf-8")
            written += 1

    print(f"Written {written:,} persona files to {PERSONAS_DIR}")


if __name__ == "__main__":
    main()
