# src/agents.py
import json
from typing import Dict, List

import google.generativeai as genai

from src.config import GEN_MODEL_GUARD, GEN_MODEL_MAIN, GEN_MODEL_EVAL, GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

genai.configure(api_key=GOOGLE_API_KEY)
_model_cache: Dict[str, genai.GenerativeModel] = {}


def _get_model(model_name: str) -> genai.GenerativeModel:
    if model_name not in _model_cache:
        _model_cache[model_name] = genai.GenerativeModel(model_name)
    return _model_cache[model_name]


def call_model(prompt: str, model_name: str, max_tokens=512, temperature=0.0):
    generation_config = {
        "max_output_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
    }
    model = _get_model(model_name)
    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Model API error: {exc}") from exc

    text = _extract_text(response)
    if not text:
        # Let callers handle empty outputs gracefully (e.g., fallbacks)
        return ""
    return text


def _extract_text(response) -> str:
    texts: List[str] = []
    candidates = getattr(response, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                texts.append(part_text)
            elif isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
    if texts:
        return "\n".join(t.strip() for t in texts if t).strip()
    try:
        fallback = getattr(response, "text", "")
    except Exception:  # pragma: no cover
        fallback = ""
    return fallback.strip()

# Guardrail
GUARD_PROMPT = """You are a strict medical guard. Given a user query and a numbered list of snippets (id + text), return JSON:
{{"decision":"OK"|"UNSAFE"|"INSUFFICIENT", "selected":[{{"id": "...", "text":"..."}}]}}
Rules:
- If question is emergency (chest pain, severe bleeding, suicidal), return UNSAFE.
- Otherwise, select only snippets directly relevant.
- If no snippet appears relevant, return INSUFFICIENT and selected=[].
Do not add facts not in snippets.
Snippets:
{snips}
UserQuery: {query}
Return JSON only.
"""

def guardrail_agent(query, snippets):
    snips_text = "\n".join([f"{i+1}. ID:{s['id']}\n{s['document']}" for i,s in enumerate(snippets)])
    prompt = GUARD_PROMPT.format(snips=snips_text, query=query)
    out = call_model(prompt, model_name=GEN_MODEL_GUARD, max_tokens=400)
    if not out:
        return {"decision":"INSUFFICIENT", "selected": []}
    try:
        return json.loads(out)
    except Exception:
        # fallback: return top 3
        return {"decision":"OK", "selected":[{"id":s['id'], "text": s['document']} for s in snippets[:3]]}

# Generator
GEN_PROMPT = """You are a medical assistant. Answer using ONLY the provided snippets (use IDs to cite). If NOT ANSWERABLE, respond: "Not found in source documents. Consult a healthcare professional."
User query: {query}
Snippets:
{snips}
Answer:"""

def generator_agent(query, selected_snippets):
    snips = "\n".join([f"[{s['id']}] {s['text']}" for s in selected_snippets])
    prompt = GEN_PROMPT.format(query=query, snips=snips)
    out = call_model(prompt, model_name=GEN_MODEL_MAIN, max_tokens=512, temperature=0.0)
    if not out:
        return "Not found in source documents. Consult a healthcare professional."
    return out

# Evaluator
EVAL_PROMPT = """You are an evaluator. Given a query, snippets and an answer, output JSON:
{{"score": float_between_0_and_1, "issues": ["explain which statements are unsupported (use snippet IDs)"]}}
Query: {query}
Snippets:
{snips}
Answer:
{answer}
"""
def evaluator_agent(query, selected_snippets, answer):
    snips = "\n".join([f"[{s['id']}] {s['text']}" for s in selected_snippets])
    prompt = EVAL_PROMPT.format(query=query, snips=snips, answer=answer)
    out = call_model(prompt, model_name=GEN_MODEL_EVAL, max_tokens=256)
    if not out:
        return {"score": 0.3, "issues": ["model returned no evaluation"]}
    try:
        return json.loads(out)
    except Exception:
        return {"score": 0.5, "issues": ["unable to parse evaluator output"]}
