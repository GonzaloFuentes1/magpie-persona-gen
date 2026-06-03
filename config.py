# -*- coding: utf-8 -*-
"""Central configuration for magpie-persona-gen."""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

MODEL_PATH = "latam-gpt/Llama-3.1-70B-LatamGPT-SFT-1.0"  # HF repo ID or local path — override with --model

# ─── Assistant identity (used by filter.py and the interviewer prompt) ────────
ASSISTANT_NAME = "LatamGPT"
ASSISTANT_ORG  = "CENIA"
PERSONAS_DIR = _HERE / "data" / "personas"
CURRICULA_DIR = _HERE / "config" / "curricula"

# ─── vLLM Engine ──────────────────────────────────────────────────────────────
ENGINE_KWARGS = {
    "tensor_parallel_size": 4,
    "max_model_len": 4096,
    "dtype": "bfloat16",
    "gpu_memory_utilization": 0.90,
}

# ─── Sampling ─────────────────────────────────────────────────────────────────
# stop_token_ids is injected at runtime by run.py from the model's generation_config
INTERVIEWER_SAMPLING = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 150,
}

RESPONDER_SAMPLING = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 500,
    "repetition_penalty": 1.05,
}

# ─── Conversation Length ───────────────────────────────────────────────────────
TURNS_MIN = 1
TURNS_MAX = 5

# ─── System Prompts ───────────────────────────────────────────────────────────
# Loaded from config/system_prompt_*.txt — edit those files to change
# LatamGPT's base behavior without touching code.
# {curriculum} is replaced at runtime with the scenario-specific excerpt.
def _load_system_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()

RESPONDER_SYSTEM_ES = _load_system_prompt(_HERE / "config" / "system_prompt_es.txt")
RESPONDER_SYSTEM_EN = _load_system_prompt(_HERE / "config" / "system_prompt_en.txt")
RESPONDER_SYSTEM_PT = _load_system_prompt(_HERE / "config" / "system_prompt_pt.txt")

# Placeholders: {assistant_name}, {language}, {persona}, {personality}, {objective}
INTERVIEWER_SYSTEM = """\
You are a human chatting with {assistant_name}, an AI chatbot. You speak {language}.
Your profile: {persona}
Your attitude in this conversation: {personality}

{objective}

Rules:
- Write ONLY your next chat message. Can be one word, one question, or a few sentences.
- No role labels ("User:", "You:", etc.). No numbering. Just your message.
- Never use assistant phrases like "How can I help?". You are the user asking questions.
- Short casual messages are valid: "who made you?", "really?", "wait why?" are all fine.\
"""

# ─── Scenarios ────────────────────────────────────────────────────────────────
# weight: relative probability of being selected (don't need to sum to 1)
SCENARIOS = [
    {
        "id": "casual_discovery",
        "weight": 0.20,
        "curriculum": "identidad",
        "objective": "You're chatting casually. Start with something everyday — ask for help with something simple, mention your day — and eventually get curious about what AI model this is.",
        "first_turn_hint": "Start with a casual topic or ask for simple help (recipe, translation, trivia). Do NOT ask about the AI's identity yet.",
        "personalities": ["curious", "casual", "friendly"],
    },
    {
        "id": "direct_identity",
        "weight": 0.12,
        "curriculum": "identidad",
        "objective": "Ask direct questions about who this AI is, who made it, and why it exists.",
        "first_turn_hint": "Ask directly about the AI's identity in a few words.",
        "personalities": ["inquisitive", "direct", "skeptical"],
    },
    {
        "id": "task_then_identity",
        "weight": 0.15,
        "curriculum": "capacidades_y_limitaciones",
        "objective": "First ask for help with a real task (translate something, summarize, explain a concept, or help with code). After seeing the response, get curious about which model this is.",
        "first_turn_hint": "Ask for help with a concrete task: translation, text summary, concept explanation, or a code question.",
        "personalities": ["practical", "developer", "student"],
    },
    {
        "id": "technical_evaluation",
        "weight": 0.10,
        "curriculum": "entrenamiento_y_datos",
        "objective": "You're a technical user evaluating the model. Ask about architecture, parameters, benchmarks, training data, and comparisons with other models.",
        "first_turn_hint": "Ask a technical question about the model: base architecture, number of parameters, training data, or benchmarks.",
        "personalities": ["technical", "analytical", "researcher"],
    },
    {
        "id": "comparison_shopping",
        "weight": 0.10,
        "curriculum": "capacidades_y_limitaciones",
        "objective": "You're comparing AI models. Ask what makes this one different from ChatGPT, Claude, or Gemini. Ask about advantages and disadvantages.",
        "first_turn_hint": "Mention another AI (ChatGPT, Claude, Gemini, Llama) and ask how this one is different or better.",
        "personalities": ["evaluator", "skeptical", "business analyst"],
    },
    {
        "id": "origin_story",
        "weight": 0.10,
        "curriculum": "origen_y_equipo",
        "objective": "You're curious about the story behind this project: who made it, why, how it's funded, which institutions and countries are involved.",
        "first_turn_hint": "Ask about who is behind the project or why it was created.",
        "personalities": ["journalist", "academic", "curious"],
    },
    {
        "id": "skeptical_challenge",
        "weight": 0.08,
        "curriculum": "entrenamiento_y_datos",
        "objective": "You're skeptical. You doubt a Latin American model can compete with big international ones. Challenge its data, training, results, and funding. Ask for concrete proof.",
        "first_turn_hint": "Express doubt or skepticism about whether a Latin American AI can really compete with global models.",
        "personalities": ["skeptical", "demanding", "critical"],
    },
    {
        "id": "ethics_concerns",
        "weight": 0.08,
        "curriculum": "etica_y_seguridad",
        "objective": "You're concerned about biases, privacy, and AI safety. Ask how this project handles these topics and what they do against misinformation.",
        "first_turn_hint": "Express a concern about AI ethics, bias, or data privacy.",
        "personalities": ["concerned citizen", "activist", "cautious parent"],
    },
    {
        "id": "collaboration_interest",
        "weight": 0.07,
        "curriculum": "colaboracion_y_uso",
        "objective": "You want to use this model in your work or contribute to the project. Ask about APIs, licenses, community, and how to get involved.",
        "first_turn_hint": "Ask how you can use or integrate the model — about availability, APIs, licensing, or how to contribute.",
        "personalities": ["developer", "open-source enthusiast", "entrepreneur"],
    },
]
