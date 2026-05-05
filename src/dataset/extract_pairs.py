"""
Extrai pares (Q_bad, Q_good) de textos candidatos usando Claude.

Para cada candidato (abstract de paper, laudação Nobel, excerto SEP),
pede ao Claude para identificar se há uma transição explícita de questão
mal-formulada -> questão melhor-formulada, e extrair o par.

Saída por par:
  {
    "q_bad":      str,   # questao original/abandonada
    "q_good":     str,   # questao reformulada/tratavel
    "context":    str,   # trecho do texto que documenta a transicao
    "domain":     str,   # dominio cientifico
    "source":     str,   # arxiv | nobel | sep
    "source_id":  str,   # id do candidato de origem
    "year":       int,
    "confidence": float, # 0-1, auto-avaliado pelo modelo
    "extractable": bool, # False se o texto nao contem par identificavel
  }
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

_SYSTEM = """\
You are an expert in philosophy of science and the history of scientific ideas.

Your task: given a text excerpt, determine whether it contains an explicit transition \
from a poorly-formulated research question to a better-formulated one.

A "poorly-formulated" question is one that is vague, unanswerable in principle, or lacks \
operational methodology (e.g., "What is the essence of life?").
A "well-formulated" question has clear methodology, measurable outcomes, and an active \
research community (e.g., "What are the molecular mechanisms of self-replication?").

Respond ONLY with a valid JSON object. No markdown, no explanation.

If the text contains such a transition:
{
  "extractable": true,
  "q_bad": "<the original poorly-formulated question, as close to verbatim as possible>",
  "q_good": "<the reformulated, better question>",
  "context": "<exact quote or close paraphrase from the text showing the transition>",
  "domain": "<scientific domain, e.g. biology, physics, neuroscience>",
  "confidence": <float 0.0-1.0 — how clearly documented this transition is in the text>
}

If the text does NOT contain a clear documented transition:
{
  "extractable": false,
  "reason": "<brief explanation>"
}

Rules:
- q_bad and q_good must be actual questions (end with "?")
- q_good must be genuinely more tractable than q_bad, not just a paraphrase
- confidence > 0.7 only if the transition is explicitly stated in the text, not inferred
- Extract at most ONE pair per text (the clearest one if multiple exist)
"""


def _get_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY nao configurada")
    return anthropic.Anthropic(api_key=api_key)


def _extract_one(client: anthropic.Anthropic, candidate: dict, model: str) -> dict:
    text = candidate.get("abstract", "")[:3000]
    title = candidate.get("title", "")

    prompt = f"Source: {title}\n\nText:\n{text}"

    try:
        msg = client.messages.create(
            model=model,
            max_tokens=512,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        result = json.loads(raw)
    except json.JSONDecodeError:
        return {"extractable": False, "reason": "parse_error"}
    except Exception as e:
        return {"extractable": False, "reason": str(e)}

    result["source"] = candidate.get("source", "unknown")
    result["source_id"] = candidate.get("id", "")
    result["year"] = candidate.get("year", 0)
    return result


def extract_pairs(
    candidates: list[dict],
    output_path: Path,
    model: str = "claude-haiku-4-5-20251001",
    delay: float = 0.4,
    min_confidence: float = 0.5,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Retoma de onde parou se output ja existe
    done_ids: set[str] = set()
    pairs: list[dict] = []
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done_ids.add(r.get("source_id", ""))
                if r.get("extractable"):
                    pairs.append(r)
        print(f"Retomando: {len(done_ids)} candidatos ja processados, {len(pairs)} pares extraidos")

    client = _get_client()
    remaining = [c for c in candidates if c.get("id", "") not in done_ids]

    with output_path.open("a", encoding="utf-8") as f:
        for candidate in tqdm(remaining, desc="Extraindo pares"):
            result = _extract_one(client, candidate, model)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if result.get("extractable") and result.get("confidence", 0) >= min_confidence:
                pairs.append(result)
            time.sleep(delay)

    print(f"\nPares extraidos (confidence >= {min_confidence}): {len(pairs)}")
    return pairs


def load_pairs(output_path: Path, min_confidence: float = 0.5) -> list[dict]:
    if not output_path.exists():
        return []
    pairs = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("extractable") and r.get("confidence", 0) >= min_confidence:
            pairs.append(r)
    return pairs


if __name__ == "__main__":
    # Carrega apenas o que ja foi buscado — nao dispara novos fetches
    candidate_files = [
        Path("data/pairs/arxiv_candidates.jsonl"),
        Path("data/pairs/nobel_candidates.jsonl"),
        Path("data/pairs/sep_candidates.jsonl"),
        Path("data/pairs/s2_candidates.jsonl"),
        Path("data/pairs/pubmed_candidates.jsonl"),
    ]

    all_candidates: list[dict] = []
    for f in candidate_files:
        if f.exists():
            batch = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            print(f"  {f.name:<35} {len(batch):>4} candidatos")
            all_candidates += batch
        else:
            print(f"  {f.name:<35}    0 (nao encontrado)")

    print(f"\nTotal candidatos: {len(all_candidates)}")

    pairs = extract_pairs(all_candidates, Path("data/pairs/extracted_pairs.jsonl"))

    print(f"\n=== Amostra de pares extraidos ===")
    for p in pairs[:5]:
        print(f"\n  [{p['source']}] [{p['domain']}] conf={p['confidence']:.2f}")
        print(f"  BAD:  {p['q_bad']}")
        print(f"  GOOD: {p['q_good']}")
