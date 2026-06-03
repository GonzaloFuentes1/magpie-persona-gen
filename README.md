# magpie-persona-gen

Synthetic conversation generator for SFT fine-tuning using the **Magpie technique** (ICLR 2025) with regional persona injection.

The model generates both sides of each conversation by feeding its own chat-template prefix up to each role header, letting it complete each turn naturally — no separate datasets needed.

## Structure

```
magpie-persona-gen/
├── config.py                    # Model path, vLLM params, sampling, scenarios
├── pipeline.py                  # Magpie core: message builders + generation loop
├── run.py                       # CLI entry point
├── filter.py                    # Quality filtering
├── dedup.py                     # Exact + near deduplication
├── upload_hf.py                 # Upload to HuggingFace Hub
├── build_personas.py            # Refresh personas from HF dataset
├── requirements.txt
├── config/
│   ├── system_prompt_es.txt     # Responder system prompt — Spanish   ← edit here
│   ├── system_prompt_en.txt     # Responder system prompt — English   ← edit here
│   ├── system_prompt_pt.txt     # Responder system prompt — Portuguese ← edit here
│   └── curricula/               # Per-scenario knowledge excerpts (injected at runtime)
└── data/
    └── personas/                # User personas organised by country
```

## Quick start

```bash
pip install -r requirements.txt

# Spanish — 500 conversations, default model
python run.py --total 500 --lang es --out data/es.jsonl

# 2-GPU run (e.g. on GPUs 1 and 2)
python run.py --total 500 --lang es --tp 2 --gpu-ids 1,2 --out data/es.jsonl

# Custom assistant name (shown to the simulated user)
python run.py --total 200 --lang es --assistant-name "MyBot" --out data/es.jsonl

# Use a different model
python run.py --total 100 --lang en --model /path/to/model --out data/en.jsonl
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--total` | 100 | Number of conversations to generate |
| `--lang` | mixed | Force language: `es`, `en`, `pt` |
| `--turns-min` | 1 | Minimum turns per conversation |
| `--turns-max` | 5 | Maximum turns per conversation |
| `--batch-size` | 50 | vLLM batch size |
| `--out` | `data/conversations.jsonl` | Output JSONL path |
| `--model` | (see config.py) | Override model path |
| `--assistant-name` | `LatamGPT` | Name shown to the simulated user |
| `--tp` | 4 | Tensor parallel size (number of GPUs) |
| `--gpu-ids` | all visible | Comma-separated GPU IDs, e.g. `1,2` |

### Resume support

If a run is interrupted, re-run the exact same command. The pipeline counts existing lines in the output file and generates only what's missing:

```bash
# Run crashed at 1 267 / 20 000 — just re-run:
python run.py --total 20000 --lang es --out data/es_20k.jsonl
# → Resuming: 1267 conversations already in data/es_20k.jsonl, need 18733 more.
```

## Full pipeline

```bash
# 1. Generate raw conversations
python run.py --total 5000 --lang es --out data/es_raw.jsonl

# 2. Filter for quality
python filter.py data/es_raw.jsonl --out data/es_clean.jsonl

# 3. Deduplicate
python dedup.py data/es_clean.jsonl --out data/es_dedup.jsonl

# 4. Upload to HuggingFace
huggingface-cli login
python upload_hf.py --repo-id myorg/my-dataset-es --lang es --input data/es_dedup.jsonl
```

## Output format

Each line is a JSON object:

```json
{
  "messages": [
    {"role": "user", "content": "¿Quién te creó?"},
    {"role": "assistant", "content": "Fui creado por CENIA..."}
  ],
  "metadata": {
    "scenario_id": "origin_story",
    "pais": "argentina",
    "language": "Spanish",
    "turns": 5
  }
}
```

The `system` message is included in ~10% of conversations so the model learns to follow it when present but does not depend on it at inference time.

## Customising the assistant profile

1. **System prompts** — edit `config/system_prompt_es.txt` (or `_en.txt` / `_pt.txt`). The `{curriculum}` placeholder is replaced at runtime with a scenario-specific excerpt.

2. **Curricula** — edit files in `config/curricula/`. Each scenario maps to one file; the content is injected into the responder's system prompt. Add, rename, or replace files freely — just update the `curriculum` key in `config.py` scenarios.

3. **Scenarios** — defined in `config.py`. Each entry has a `weight`, `curriculum`, `objective`, `first_turn_hint`, and `personalities` list. Add custom scenarios without touching the generation code.

4. **Personas** — place `.txt` files under `data/personas/<country>/`. One persona per file. Run `python build_personas.py` to refresh from the HuggingFace dataset.

## Filtering

`filter.py` removes known generation artifacts. For identity scenarios, it checks that the assistant mentions at least one configured term:

```bash
# Default terms: "latamgpt,cenia"
python filter.py data/es_raw.jsonl --out data/es_clean.jsonl

# Custom terms for a different assistant
python filter.py data/es_raw.jsonl --out data/es_clean.jsonl --identity-terms "mybot,myorg"
```

## Uploading to HuggingFace

`--repo-id` is required (no default to avoid accidental uploads):

```bash
python upload_hf.py --repo-id org/my-dataset-es --lang es
python upload_hf.py --repo-id org/my-dataset-es --lang es --description "LatamGPT self-identity SFT"
python upload_hf.py --repo-id org/my-dataset-es --lang es --dry-run   # local parquet only
```

## Model requirements

Any instruction-tuned model served via vLLM. The model generates both user and assistant turns, so it works best with a chat-tuned model that follows instructions reliably.

Default: `Llama-3.1-70B-LatamGPT-SFT-1.0` with 4-GPU tensor parallelism.
