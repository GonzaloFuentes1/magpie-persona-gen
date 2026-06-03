#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality filter for magpie-persona-gen JSONL output.

Removes conversations that exhibit known generation artifacts:
  1. Degenerate user turn   — first user message < 8 chars, ends with ':', or no verb/question
  2. Assistant apology      — response starts with "I'm sorry", "lamento", etc.
  3. Very short answer      — any assistant turn < 30 chars
  4. Identity miss          — identity scenario where assistant never mentions identity terms
  5. Markdown lists (ES/PT) — assistant used bullet lists despite system prompt forbidding them

Usage:
    python filter.py data/es_5k_v2.jsonl --out data/es_5k_v2_clean.jsonl
    python filter.py data/en_5k_v2.jsonl --out data/en_5k_v2_clean.jsonl
    python filter.py data/es_5k_v2.jsonl data/en_5k_v2.jsonl --out data/merged_clean.jsonl
    python filter.py data/es_5k_v2.jsonl --out data/es_5k_v2_clean.jsonl --dry-run
    python filter.py data/es.jsonl --out data/es_clean.jsonl --identity-terms "mybot,myorg"
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import config as C

# ─── Filter predicates ────────────────────────────────────────────────────────

_APOLOGY = re.compile(
    r"^(i'?m sorry|unfortunately|lo siento|lamento|disculpa|perdona|"
    r"desafortunadamente|lamentablemente)",
    re.IGNORECASE,
)

_MARKDOWN_LIST = re.compile(r"^\s*[-*]\s+\w", re.MULTILINE)

_AS_AI = re.compile(
    r"\b(as an AI|as an artificial intelligence|"
    r"como (una? )?IA\b|como (una? )?inteligencia artificial|"
    r"como (uma? )?IA\b|como (uma? )?inteligência artificial)\b",
    re.IGNORECASE,
)

_ROLE_ECHO_MID = re.compile(
    r"\n(assistant|asistente|user|usuario|usuário|latamgpt)\s*:",
    re.IGNORECASE,
)

_ROLE_FLOW = re.compile(r"\s*user\s*>\s*(assistant|user)", re.IGNORECASE)

_META_TRAILING = re.compile(
    r"(The assistant (has |)provided|"
    r"La (IA|asistente?) (ha |)(respondido|proporcionado)|"
    r"O assistente (forneceu|respondeu))",
    re.IGNORECASE,
)

_IDENTITY_SCENARIOS = {"casual_discovery", "direct_identity", "origin_story"}

# Identity terms checked in check_identity_miss; overridden via --identity-terms CLI flag
_IDENTITY_TERMS: list[str] = [C.ASSISTANT_NAME.lower(), C.ASSISTANT_ORG.lower()]


def _user_msgs(record):
    return [m for m in record["messages"] if m["role"] == "user"]


def _asst_msgs(record):
    return [m for m in record["messages"] if m["role"] == "assistant"]


def check_degenerate_user(record) -> str | None:
    """Return a reason string if the first user turn is degenerate, else None."""
    users = _user_msgs(record)
    if not users:
        return "no_user_turn"
    first = users[0]["content"].strip()
    if len(first) < 8:
        return f"first_user_too_short ({len(first)} chars)"
    if first.endswith(":"):
        return "first_user_ends_with_colon"
    # Entire conversation has at least one user turn ending with ':'
    if any(m["content"].strip().endswith(":") for m in users):
        return "user_turn_ends_with_colon"
    return None


def check_apology(record) -> str | None:
    for m in _asst_msgs(record):
        if _APOLOGY.match(m["content"].strip()):
            return "assistant_starts_with_apology"
    return None


def check_short_answer(record, min_chars: int = 30) -> str | None:
    for m in _asst_msgs(record):
        if len(m["content"].strip()) < min_chars:
            return f"very_short_answer ({len(m['content'].strip())} chars)"
    return None


def check_identity_miss(record) -> str | None:
    if record["metadata"]["scenario_id"] not in _IDENTITY_SCENARIOS:
        return None
    full = " ".join(m["content"] for m in _asst_msgs(record)).lower()
    if not any(term in full for term in _IDENTITY_TERMS):
        return "identity_scenario_no_identity_mention"
    return None


def check_as_ai_phrase(record) -> str | None:
    for m in _asst_msgs(record):
        if _AS_AI.search(m["content"]):
            return "assistant_as_generic_ai"
    return None


def check_role_echo_mid(record) -> str | None:
    for m in record["messages"]:
        if _ROLE_ECHO_MID.search(m["content"]):
            return "role_echo_mid_content"
    return None


def check_conversation_artifacts(record) -> str | None:
    for m in record["messages"]:
        if m["role"] == "user":
            c = m["content"]
            if _ROLE_FLOW.search(c):
                return "user_role_flow_artifact"
            if _META_TRAILING.search(c):
                return "user_meta_commentary"
    return None


def check_markdown_list(record) -> str | None:
    lang = record["metadata"].get("language", "")
    if lang == "English":
        return None  # EN system prompt allows more formatting flexibility
    for m in _asst_msgs(record):
        if _MARKDOWN_LIST.search(m["content"]):
            return "assistant_markdown_list"
    return None


CHECKS = [
    check_degenerate_user,
    check_conversation_artifacts,
    check_apology,
    check_short_answer,
    check_identity_miss,
    check_as_ai_phrase,
    check_role_echo_mid,
    check_markdown_list,
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(input_paths: list[Path], out_path: Path, dry_run: bool) -> None:
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

    kept = []
    removed_by_reason = defaultdict(int)

    for rec in records:
        reason = None
        for check in CHECKS:
            reason = check(rec)
            if reason:
                break
        if reason:
            removed_by_reason[reason] += 1
        else:
            kept.append(rec)

    removed = total - len(kept)
    print(f"\nRemoved {removed:,} / {total:,}  ({removed / total * 100:.1f}%)")
    print(f"Kept    {len(kept):,} / {total:,}  ({len(kept) / total * 100:.1f}%)")
    print("\nBreakdown:")
    for reason, count in sorted(removed_by_reason.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}  ({count / total * 100:.1f}%)")

    if dry_run:
        print("\nDry run — nothing written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(kept):,} conversations → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Quality filter for magpie-persona-gen JSONL output")
    p.add_argument("inputs", nargs="+", help="Input JSONL file(s)")
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--dry-run", action="store_true", help="Print stats without writing output")
    p.add_argument(
        "--identity-terms",
        default=None,
        help=(
            "Comma-separated terms the assistant must mention in identity scenarios "
            f"(default: '{','.join(_IDENTITY_TERMS)}')"
        ),
    )
    args = p.parse_args()

    if args.identity_terms:
        global _IDENTITY_TERMS
        _IDENTITY_TERMS = [t.strip().lower() for t in args.identity_terms.split(",") if t.strip()]

    input_paths = [Path(i) for i in args.inputs]
    for path in input_paths:
        if not path.exists():
            print(f"ERROR: file not found: {path}", file=sys.stderr)
            sys.exit(1)

    run(input_paths, Path(args.out), args.dry_run)


if __name__ == "__main__":
    main()
