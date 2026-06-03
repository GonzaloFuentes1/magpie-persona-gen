#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload deduped JSONL files to HuggingFace Hub.

Schema per row:
  messages  — list[{role: str, content: str}]
  hash      — str  (SHA-256 of first user message)
  country   — str  (from metadata.pais)
  language  — str  (from metadata.language)
  scenario  — str  (from metadata.scenario_id)

Usage:
    huggingface-cli login                               # once
    python upload_hf.py --repo-id org/dataset-es --lang es
    python upload_hf.py --repo-id org/dataset-es --lang es --input data/es_dedup.jsonl
    python upload_hf.py --repo-id org/dataset-es --lang es --dry-run
    python upload_hf.py --repo-id org/dataset-es --lang es --description "My SFT dataset"
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo

LANG_NAMES = {"es": "Spanish", "en": "English", "pt": "Portuguese"}

HERE = Path(__file__).parent


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _conv_hash(messages: list[dict]) -> str:
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return hashlib.sha256(first_user.encode("utf-8")).hexdigest()


def to_arrow_table(records: list[dict]) -> pa.Table:
    """Convert records to a PyArrow table."""
    messages_col, hash_col, country_col, language_col, scenario_col = [], [], [], [], []

    for rec in records:
        messages_col.append([
            {"role": m["role"], "content": m["content"]}
            for m in rec["messages"]
        ])
        hash_col.append(_conv_hash(rec["messages"]))
        meta = rec.get("metadata", {})
        country_col.append(meta.get("pais", ""))
        language_col.append(meta.get("language", ""))
        scenario_col.append(meta.get("scenario_id", ""))

    message_type = pa.list_(
        pa.struct([("content", pa.string()), ("role", pa.string())])
    )
    schema = pa.schema([
        pa.field("messages", message_type),
        pa.field("hash", pa.string()),
        pa.field("country", pa.string()),
        pa.field("language", pa.string()),
        pa.field("scenario", pa.string()),
    ])

    n = len(records)
    return pa.table(
        {
            "messages": pa.array(messages_col, type=message_type),
            "hash": pa.array(hash_col, type=pa.string()),
            "country": pa.array(country_col, type=pa.string()),
            "language": pa.array(language_col, type=pa.string()),
            "scenario": pa.array(scenario_col, type=pa.string()),
        },
        schema=schema,
    )


def build_readme(repo_id: str, lang: str, n_rows: int, description: str) -> str:
    lang_code = lang
    lang_name = LANG_NAMES.get(lang, lang)
    dataset_name = repo_id.split("/")[-1]
    desc_block = f"\n{description}\n" if description else ""
    return f"""\
---
language:
  - {lang_code}
license: apache-2.0
tags:
  - sft
  - synthetic
  - magpie
  - conversations
size_categories:
  - 1K<n<10K
dataset_info:
  features:
    - name: messages
      list:
        - name: content
          dtype: string
        - name: role
          dtype: string
    - name: hash
      dtype: string
    - name: country
      dtype: string
    - name: language
      dtype: string
    - name: scenario
      dtype: string
  splits:
    - name: train
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/data.parquet
---

# {dataset_name}
{desc_block}
Synthetic SFT dataset in {lang_name} generated with **[magpie-persona-gen](https://github.com/latam-gpt/magpie-persona-gen)** using the **Magpie technique** (ICLR 2025): the model generates both sides of the conversation by completing its own chat-template prefix.

**Rows:** {n_rows:,}
**Pipeline:** generate → quality filter → exact+near dedup

## Schema

| Column | Description |
|--------|-------------|
| `messages` | List of `{{role, content}}` dicts (user/assistant turns, optional system) |
| `hash` | SHA-256 of the first user message (for dedup) |
| `country` | Persona's country of origin |
| `language` | Conversation language |
| `scenario` | Generation scenario ID |
"""


def upload(repo_id: str, lang: str, jsonl_path: Path, dry_run: bool,
           description: str = "", token: str | None = None) -> None:
    parquet_path = jsonl_path.with_suffix(".parquet")

    print(f"\n{'='*60}")
    print(f"Lang: {lang.upper()}  |  {repo_id}")
    print(f"Input:  {jsonl_path}  ({jsonl_path.stat().st_size // 1024} KB)")

    records = load_records(jsonl_path)
    print(f"Rows:   {len(records):,}")

    table = to_arrow_table(records)
    pq.write_table(table, parquet_path, compression="snappy")
    print(f"Parquet: {parquet_path}  ({parquet_path.stat().st_size // 1024} KB)")

    if dry_run:
        print("Dry run — skipping upload.")
        return

    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=True, token=token)
    print(f"Repo ready: https://huggingface.co/datasets/{repo_id}")

    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/data.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add {len(records):,} conversations ({lang})",
    )
    print(f"Uploaded → {repo_id}/data/data.parquet")

    readme = build_readme(repo_id, lang, len(records), description)
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add README with dataset card",
    )
    print(f"Uploaded → {repo_id}/README.md")


def main() -> None:
    p = argparse.ArgumentParser(description="Upload magpie-persona-gen datasets to HuggingFace Hub")
    p.add_argument("--repo-id", required=True,
                   help="HuggingFace dataset repo ID, e.g. 'org/my-dataset-es'")
    p.add_argument("--lang", choices=list(LANG_NAMES.keys()), required=True,
                   help="Language of the dataset (es / en / pt)")
    p.add_argument("--input", default=None,
                   help="Input deduped JSONL file (default: data/{lang}_dedup.jsonl)")
    p.add_argument("--description", default="",
                   help="Optional one-line description added to the dataset card README")
    p.add_argument("--token", default=None,
                   help="HuggingFace token (overrides HF_TOKEN env var and cached login)")
    p.add_argument("--dry-run", action="store_true",
                   help="Convert to parquet locally but skip HF upload")
    args = p.parse_args()

    token = args.token or __import__("os").environ.get("HF_TOKEN")

    jsonl_path = Path(args.input) if args.input else HERE / "data" / f"{args.lang}_dedup.jsonl"
    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found — run filter.py + dedup.py first", file=sys.stderr)
        sys.exit(1)

    upload(args.repo_id, args.lang, jsonl_path, args.dry_run, args.description, token=token)
    print("\nDone.")


if __name__ == "__main__":
    main()
