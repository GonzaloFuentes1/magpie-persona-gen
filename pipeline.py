# -*- coding: utf-8 -*-
"""
Magpie-style synthetic conversation generator.

Uses llm.chat() — vLLM applies the model's own chat template and returns
only the generated assistant content. No manual prefix building or cleanup needed.

  build_interviewer_messages()  → messages list ending ready for the model to play "user"
  build_responder_messages()    → messages list ending with a user turn for the assistant to answer
"""

import json
import logging
import random
import re
from pathlib import Path
from typing import Optional

from vllm import LLM, SamplingParams
from tqdm import tqdm

import config as C


# ─── Message builders ─────────────────────────────────────────────────────────

def build_interviewer_messages(system: str, history: list) -> list:
    """Messages for the model acting as a human user (interviewer role)."""
    msgs = [{"role": "system", "content": system}]
    for q, a in history:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    return msgs


def build_responder_messages(system: str, history: list, question: str) -> list:
    """Messages for LatamGPT to answer the latest user question."""
    msgs = [{"role": "system", "content": system}]
    for q, a in history:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    return msgs


# ─── Validation ───────────────────────────────────────────────────────────────

_META_PATTERNS = re.compile(
    r"^(For your (first|second|third|follow[\s-]?up|next|\d+\w*) message|"
    r"\[End of conversation\]|"
    r"^\[.*\]$)",
    re.IGNORECASE,
)
_ROLE_FLOW = re.compile(r"\s*user\s*>\s*(assistant|user)", re.IGNORECASE)
_META_TRAILING = re.compile(
    r"(The assistant (has |)provided|"
    r"La (IA|asistente?) (ha |)(respondido|proporcionado)|"
    r"O assistente (forneceu|respondeu))",
    re.IGNORECASE,
)


def is_valid_question(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    if _META_PATTERNS.search(text.strip()):
        return False
    if _ROLE_FLOW.search(text):
        return False
    if _META_TRAILING.search(text):
        return False
    return True


_ES_INDICATORS = re.compile(
    r"[áéíóúüñ¿¡]"
    r"|\b(el|la|los|las|una?|del|con|por|para|que|qué|cómo|cuál|cuándo|"
    r"dónde|quién|porque|pero|también|además|según|aunque|desde|hasta|"
    r"sobre|entre|durante|después|antes|esto|esta|estos|estas|ese|esa|"
    r"algo|alguien|nada|nadie|todo|todos|siempre|nunca|puedes|puedo|"
    r"podría|podrías|tengo|tienes|tiene|hace|hacen|soy|eres|somos|son|"
    r"hola|gracias|favor|ayuda|necesito)\b",
    re.IGNORECASE,
)
_PT_INDICATORS = re.compile(
    r"[ãõçâêô]"
    r"|\b(você|voce|obrigado|obrigada|também|então|porque|isso|aqui|"
    r"preciso|estou|fazer|ajuda|minha?|nosso|nossa|esse|essa|esses|essas|"
    r"aquele|aquela|tudo|nada|muito|pouco|já|ainda|sempre|nunca|agora|"
    r"depois|antes|quando|onde|quem|como|qual|quais|quanto)\b",
    re.IGNORECASE,
)
_MIN_WORDS_FOR_LANG_CHECK = 4


def is_lang_ok(text: str, lang: Optional[str]) -> bool:
    if lang not in ("es", "pt"):
        return True
    words = text.split()
    if len(words) < _MIN_WORDS_FOR_LANG_CHECK:
        return True
    if lang == "es":
        return bool(_ES_INDICATORS.search(text))
    if lang == "pt":
        return bool(_PT_INDICATORS.search(text))
    return True


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_personas(lang: Optional[str] = None) -> list:
    BRAZIL_DIRS = {"brasil"}
    personas = []
    for country_dir in sorted(C.PERSONAS_DIR.iterdir()):
        if not country_dir.is_dir():
            continue
        is_brazil = country_dir.name.lower() in BRAZIL_DIRS
        if lang == "pt" and not is_brazil:
            continue
        if lang == "es" and is_brazil:
            continue
        for txt in sorted(country_dir.glob("*.txt")):
            try:
                content = txt.read_text(encoding="utf-8").strip()
                if content:
                    personas.append({"content": content, "country": country_dir.name, "is_brazil": is_brazil})
            except OSError:
                pass
    if not personas:
        raise ValueError(f"No personas found for lang={lang!r} in {C.PERSONAS_DIR}")
    logging.info("Loaded %d personas (lang=%s)", len(personas), lang)
    return personas


def load_curriculum(name: str) -> str:
    path = C.CURRICULA_DIR / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.strip().startswith("#"))


# ─── Scenario / system-prompt helpers ─────────────────────────────────────────

def pick_scenario() -> dict:
    weights = [s["weight"] for s in C.SCENARIOS]
    return random.choices(C.SCENARIOS, weights=weights, k=1)[0]


def resolve_language(persona: dict, lang_override: Optional[str]) -> str:
    if lang_override == "es":
        return "Spanish"
    if lang_override == "pt":
        return "Portuguese"
    if lang_override == "en":
        return "English"
    return "Portuguese" if persona["is_brazil"] else "Spanish"


def build_systems(persona: dict, scenario: dict, lang: Optional[str], assistant_name: str = C.ASSISTANT_NAME) -> tuple:
    """Return (interviewer_system_prompt, responder_system_prompt)."""
    language = resolve_language(persona, lang)
    curriculum = load_curriculum(scenario["curriculum"])
    personality = random.choice(scenario["personalities"])

    raw_persona = persona["content"][:600]
    if len(persona["content"]) > 600 and not persona["content"][600:601].isspace():
        cut = raw_persona.rfind(" ")
        if cut > 0:
            raw_persona = raw_persona[:cut]

    interviewer_sys = C.INTERVIEWER_SYSTEM.format(
        assistant_name=assistant_name,
        language=language,
        persona=raw_persona,
        personality=personality,
        objective=scenario["objective"],
    )

    if language == "Portuguese":
        responder_sys = C.RESPONDER_SYSTEM_PT.format(curriculum=curriculum)
    elif language == "English":
        responder_sys = C.RESPONDER_SYSTEM_EN.format(curriculum=curriculum)
    else:
        responder_sys = C.RESPONDER_SYSTEM_ES.format(curriculum=curriculum)

    return interviewer_sys, responder_sys


# ─── Generation ───────────────────────────────────────────────────────────────

def batch_chat(llm: LLM, message_batches: list, sampling_params: SamplingParams) -> list:
    """Run llm.chat() on a batch of message lists. Returns stripped text outputs."""
    if not message_batches:
        return []
    outputs = llm.chat(message_batches, sampling_params)
    return [out.outputs[0].text.strip() if out.outputs else "" for out in outputs]


# ─── Main pipeline ────────────────────────────────────────────────────────────

_LANG_NAME = {"es": "Spanish", "pt": "Portuguese", "en": "English"}


def _interviewer_system(base_prompt: str, scenario: dict, lang: Optional[str], turn: int) -> str:
    if turn == 0:
        hint = scenario.get("first_turn_hint", "")
        if hint:
            lang_name = _LANG_NAME.get(lang or "es", "Spanish")
            return base_prompt + f"\n\nFor your FIRST message (write in {lang_name}): {hint}"
    return base_prompt


def run_pipeline(
    llm: LLM,
    total: int,
    lang: Optional[str],
    turns_min: int,
    turns_max: int,
    batch_size: int,
    out_path: Path,
    assistant_name: str = C.ASSISTANT_NAME,
) -> None:
    """Generate `total` synthetic conversations and append them to `out_path` as JSONL."""
    personas = load_personas(lang)
    int_sp = SamplingParams(**C.INTERVIEWER_SAMPLING)
    res_sp = SamplingParams(**C.RESPONDER_SAMPLING)

    already_done = 0
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            already_done = sum(1 for line in f if line.strip())
    if already_done >= total:
        logging.info("Output already contains %d conversations (>= --total %d). Nothing to do.", already_done, total)
        return
    if already_done:
        logging.info("Resuming: %d conversations already in %s, need %d more.", already_done, out_path, total - already_done)
    remaining = total - already_done

    completed = 0
    with open(out_path, "a", encoding="utf-8") as fout, tqdm(total=remaining, desc="Conversations") as pbar:
        while completed < remaining:
            current_batch = min(batch_size, remaining - completed)

            # ── Initialise batch state ─────────────────────────────────────────
            batch_personas  = [random.choice(personas) for _ in range(current_batch)]
            batch_scenarios = [pick_scenario() for _ in range(current_batch)]
            batch_systems   = [build_systems(p, s, lang, assistant_name) for p, s in zip(batch_personas, batch_scenarios)]
            batch_histories: list[list] = [[] for _ in range(current_batch)]
            batch_turns     = [random.randint(turns_min, turns_max) for _ in range(current_batch)]
            batch_alive     = [True] * current_batch
            max_turns       = max(batch_turns)

            # ── Turn loop ─────────────────────────────────────────────────────
            for turn in range(max_turns):
                active_idx = [i for i in range(current_batch) if batch_alive[i] and turn < batch_turns[i]]
                if not active_idx:
                    break

                # -- Interviewer: model plays human user -----------------------
                int_msgs = [
                    build_interviewer_messages(
                        _interviewer_system(batch_systems[i][0], batch_scenarios[i], lang, turn),
                        batch_histories[i],
                    )
                    for i in active_idx
                ]
                questions = batch_chat(llm, int_msgs, int_sp)

                valid_q: dict[int, str] = {}
                for i, q in zip(active_idx, questions):
                    if not is_valid_question(q):
                        batch_alive[i] = False
                        logging.warning("Conv %d turn %d: invalid question (%r) — dropping", i, turn + 1, q[:60])
                    elif not is_lang_ok(q, lang):
                        batch_alive[i] = False
                        logging.warning("Conv %d turn %d: wrong language (%r) — dropping", i, turn + 1, q[:60])
                    else:
                        valid_q[i] = q

                if not valid_q:
                    continue

                # -- Responder: LatamGPT answers -------------------------------
                res_msgs = [
                    build_responder_messages(batch_systems[i][1], batch_histories[i], valid_q[i])
                    for i in valid_q
                ]
                answers = batch_chat(llm, res_msgs, res_sp)

                for i, a in zip(valid_q, answers):
                    if a and len(a) >= 5:
                        batch_histories[i].append((valid_q[i], a))
                    else:
                        batch_alive[i] = False
                        logging.warning("Conv %d turn %d: empty answer — dropping conversation", i, turn + 1)

            # ── Save results ──────────────────────────────────────────────────
            for i in range(current_batch):
                hist = batch_histories[i]
                if len(hist) < turns_min:
                    continue

                persona  = batch_personas[i]
                scenario = batch_scenarios[i]
                language = resolve_language(persona, lang)
                _, res_sys = batch_systems[i]

                messages = []
                if random.random() < 0.10:
                    messages.append({"role": "system", "content": res_sys})
                for q, a in hist:
                    messages.append({"role": "user", "content": q})
                    messages.append({"role": "assistant", "content": a})

                record = {
                    "messages": messages,
                    "metadata": {
                        "scenario_id": scenario["id"],
                        "pais": persona["country"],
                        "language": language,
                        "turns": len(hist),
                    },
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                pbar.update(1)

    logging.info("Done. Saved %d new conversations to %s (%d total)", completed, out_path, already_done + completed)
