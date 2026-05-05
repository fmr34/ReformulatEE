"""
Tratabilidade(Q) -- zero-shot via Claude API com prompt caching e cache local.

Saida:
  {
    prob_tractable: float,
    trajectory: 'rising' | 'plateau' | 'declining' | 'absent',
    confidence: float,
    nearest_resolved_question: str,
  }

Otimizações (Onda 1):
  - Prompt caching via cache_control ephemeral (reduz custo ~70%)
  - Cache in-memory por sessão (dict)
  - Cache persistente SQLite cross-session (via src.db.historico)
"""

from __future__ import annotations

import json
import os
import re
import threading

import anthropic
from dotenv import load_dotenv

load_dotenv(override=True)

_SYSTEM = """\
You are an expert in philosophy of science and research methodology.
Given a research question Q, assess whether it belongs to a domain with an active research paradigm.
An active paradigm means: there exist established methodologies, instruments or datasets, and a community \
actively producing empirical results that could answer or make progress on this question.

Respond ONLY with a valid JSON object (no markdown, no explanation) with these exact keys:
{
  "prob_tractable": <float 0.0-1.0>,
  "trajectory": <"rising"|"plateau"|"declining"|"absent">,
  "confidence": <float 0.0-1.0>,
  "nearest_resolved_question": <string, the closest well-resolved scientific question you know>
}

Criteria:
- prob_tractable=1.0: clear methodology exists, community is active, results expected in <10 years
- prob_tractable=0.5: methodology partially exists, unclear if empirically resolvable
- prob_tractable=0.0: no methodology conceivable, ontologically blocked (e.g. "what is the essence of X")
- trajectory "rising": publication rate increasing last 5 years
- trajectory "absent": no meaningful research community exists for this question
- confidence: how certain you are given your knowledge; lower if the question is highly specialized
"""

# Cache in-memory (por sessão): query → resultado
_mem_cache: dict[str, dict] = {}
_cache_lock = threading.Lock()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY nao configurada no .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _parse_response(text: str) -> dict:
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    return json.loads(text)


def tratabilidade(query: str, model: str = "claude-haiku-4-5-20251001") -> dict:
    """
    Avalia tratabilidade de uma questão de pesquisa.
    Ordem de consulta:
      1. Modelo local (se treinado) — ~5ms, custo R$0,00
      2. Cache in-memory
      3. Cache SQLite cross-session
      4. Claude API (fallback)
    """
    # 1. Modelo local — caminho preferido quando disponível
    try:
        from src.classifier.tractability_local import is_trained, predict_local
        if is_trained():
            return predict_local(query)
    except Exception:
        pass

    key = query.strip().lower()

    # 2. Cache in-memory
    with _cache_lock:
        if key in _mem_cache:
            return _mem_cache[key]

    # 2. Cache SQLite (cross-session)
    try:
        from src.db.historico import get_trat_cache, set_trat_cache
        cached = get_trat_cache(query)
        if cached:
            with _cache_lock:
                _mem_cache[key] = cached
            return cached
    except Exception:
        pass  # se o DB ainda não está disponível, segue para a API

    # 3. Chamada à API com prompt caching
    client = _get_client()
    message = client.messages.create(
        model=model,
        max_tokens=256,
        system=[{
            "type": "text",
            "text": _SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": f"Research question: {query}"}],
    )
    raw = message.content[0].text
    result = _parse_response(raw)

    # Sanitiza valores
    result["prob_tractable"] = max(0.0, min(1.0, float(result["prob_tractable"])))
    result["confidence"]     = max(0.0, min(1.0, float(result["confidence"])))
    if result["trajectory"] not in ("rising", "plateau", "declining", "absent"):
        result["trajectory"] = "absent"

    # Persiste em ambos os caches
    with _cache_lock:
        _mem_cache[key] = result
    try:
        from src.db.historico import set_trat_cache
        set_trat_cache(query, result)
    except Exception:
        pass

    return result


def tratabilidade_score(query: str, **kwargs) -> float:
    """Atalho escalar: retorna prob_tractable * confidence."""
    out = tratabilidade(query, **kwargs)
    return out["prob_tractable"] * out["confidence"]


if __name__ == "__main__":
    questions = [
        "What are the molecular mechanisms of CRISPR-Cas9 gene editing?",
        "What is the essence of life?",
        "How does transformer architecture scale with parameter count?",
        "What is the true nature of consciousness?",
        "What methodologies exist for measuring scientific paradigm shifts?",
    ]
    for q in questions:
        result = tratabilidade(q)
        score = result["prob_tractable"] * result["confidence"]
        print(f"  [{score:.2f}] traj={result['trajectory']} | {q[:60]}")
        print(f"        nearest: {result['nearest_resolved_question'][:70]}")
