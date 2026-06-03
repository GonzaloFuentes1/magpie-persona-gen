#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point for magpie-persona-gen.

Usage:
    python run.py --total 200 --lang es --out data/out.jsonl
    python run.py --total 500 --lang pt --turns-min 4 --turns-max 12 --batch-size 64
    python run.py --total 100 --lang en --model /path/to/other/model --assistant-name "MyBot"
    python run.py --total 30  --lang es --tp 2 --gpu-ids 1,2
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import config as C

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate synthetic conversations using the Magpie technique."
    )
    p.add_argument("--total", type=int, default=100,
                   help="Number of conversations to generate (default: 100)")
    p.add_argument("--lang", choices=["es", "en", "pt"], default=None,
                   help="Force language (es=Spanish, en=English, pt=Portuguese). Default: mixed.")
    p.add_argument("--turns-min", type=int, default=C.TURNS_MIN,
                   help=f"Minimum turns per conversation (default: {C.TURNS_MIN})")
    p.add_argument("--turns-max", type=int, default=C.TURNS_MAX,
                   help=f"Maximum turns per conversation (default: {C.TURNS_MAX})")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Number of conversations processed per vLLM batch (default: 50)")
    p.add_argument("--out", default="data/conversations.jsonl",
                   help="Output JSONL file path (default: data/conversations.jsonl)")
    p.add_argument("--model", default=None,
                   help=f"Override model path (default: {C.MODEL_PATH})")
    p.add_argument("--assistant-name", default=C.ASSISTANT_NAME,
                   help=f"Name of the assistant shown to the interviewer (default: {C.ASSISTANT_NAME})")
    p.add_argument("--tp", type=int, default=None,
                   help="Tensor parallel size — number of GPUs for vLLM (default: from config)")
    p.add_argument("--gpu-ids", default=None,
                   help="Comma-separated GPU IDs to use, e.g. '1,2'. Sets CUDA_VISIBLE_DEVICES.")
    args = p.parse_args()

    if args.turns_min > args.turns_max:
        p.error("--turns-min cannot be greater than --turns-max")

    # Set CUDA_VISIBLE_DEVICES BEFORE importing vllm — device selection is locked at import time
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        logging.info("CUDA_VISIBLE_DEVICES=%s", args.gpu_ids)

    # Apply tensor parallel override
    if args.tp is not None:
        C.ENGINE_KWARGS["tensor_parallel_size"] = args.tp

    # Lazy-import vllm and pipeline so CUDA_VISIBLE_DEVICES is already set
    from transformers import GenerationConfig
    from vllm import LLM
    from pipeline import run_pipeline

    model_path = args.model or C.MODEL_PATH
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Model:          %s", model_path)
    logging.info("Assistant name: %s", args.assistant_name)
    logging.info("Language:       %s", args.lang or "mixed")
    logging.info("Total:          %d conversations", args.total)
    logging.info("Turns:          %d – %d", args.turns_min, args.turns_max)
    logging.info("Batch size:     %d", args.batch_size)
    logging.info("TP size:        %d", C.ENGINE_KWARGS["tensor_parallel_size"])
    logging.info("Output:         %s", out_path)

    # Derive stop token IDs from the model's generation_config
    try:
        gen_cfg = GenerationConfig.from_pretrained(model_path)
        raw = gen_cfg.eos_token_id
        stop_ids = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    except Exception:
        stop_ids = []
    stop_ids = [i for i in stop_ids if i is not None]
    logging.info("Stop token IDs: %s", stop_ids)

    if stop_ids:
        C.INTERVIEWER_SAMPLING["stop_token_ids"] = stop_ids
        C.RESPONDER_SAMPLING["stop_token_ids"] = stop_ids

    llm = LLM(model=model_path, **C.ENGINE_KWARGS)

    run_pipeline(
        llm=llm,
        total=args.total,
        lang=args.lang,
        turns_min=args.turns_min,
        turns_max=args.turns_max,
        batch_size=args.batch_size,
        out_path=out_path,
        assistant_name=args.assistant_name,
    )


if __name__ == "__main__":
    main()
