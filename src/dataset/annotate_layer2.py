"""
Layer 2 — Anotação multi-annotator com famílias divergentes (Patch B).

Para cada par (q_bad, q_good) da Layer 1, três anotadores independentes
de arquiteturas distintas avaliam se a reformulação é genuinamente melhor.

Anotadores suportados:
  - Claude Haiku   (Anthropic)  — ANTHROPIC_API_KEY  [obrigatório]
  - Llama 3.3 70B  (Meta/Groq)  — GROQ_API_KEY       [recomendado]
  - GPT-4o-mini    (OpenAI)     — OPENAI_API_KEY      [opcional]

Saída: data/pairs/pairs_layer2.jsonl
  Mantém apenas pares onde todos os anotadores ativos concordam
  na direção (q_good > q_bad) E variância de magnitude < VAR_THRESHOLD.
  Pares com divergência ficam em pairs_layer2_divergent.jsonl para auditoria.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

# Pares com variância de magnitude acima deste threshold vão para divergentes
VAR_THRESHOLD = 0.04   # stdev < 0.2 entre anotadores
MIN_ANNOTATORS = 2     # mínimo para considerar um par validado


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class Annotation:
    model: str
    direction: bool      # True = q_good é genuinamente melhor
    magnitude: float     # 0-1: quanto melhor
    reasoning: str
    error: str = ""


@dataclass
class AnnotatedPair:
    # Campos originais da Layer 1
    q_bad: str
    q_good: str
    context: str
    domain: str
    confidence: float
    source: str
    source_id: str
    year: int
    # Anotações da Layer 2
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def valid_annotations(self) -> list[Annotation]:
        return [a for a in self.annotations if not a.error]

    @property
    def agreement_direction(self) -> bool:
        va = self.valid_annotations
        return len(va) >= MIN_ANNOTATORS and all(a.direction for a in va)

    @property
    def magnitude_variance(self) -> float:
        va = self.valid_annotations
        mags = [a.magnitude for a in va]
        if len(mags) < 2:
            return 0.0
        return statistics.variance(mags)

    @property
    def magnitude_mean(self) -> float:
        va = self.valid_annotations
        if not va:
            return 0.0
        return statistics.mean(a.magnitude for a in va)

    @property
    def passes_layer2(self) -> bool:
        return (
            self.agreement_direction
            and self.magnitude_variance <= VAR_THRESHOLD
            and len(self.valid_annotations) >= MIN_ANNOTATORS
        )

    def to_dict(self) -> dict:
        return {
            "q_bad": self.q_bad,
            "q_good": self.q_good,
            "context": self.context,
            "domain": self.domain,
            "confidence": self.confidence,
            "source": self.source,
            "source_id": self.source_id,
            "year": self.year,
            "annotations": [
                {
                    "model": a.model,
                    "direction": a.direction,
                    "magnitude": a.magnitude,
                    "reasoning": a.reasoning,
                    **({"error": a.error} if a.error else {}),
                }
                for a in self.annotations
            ],
            "n_annotators": len(self.valid_annotations),
            "agreement_direction": self.agreement_direction,
            "magnitude_mean": round(self.magnitude_mean, 4),
            "magnitude_variance": round(self.magnitude_variance, 4),
            "passes_layer2": self.passes_layer2,
        }


# ---------------------------------------------------------------------------
# Prompt compartilhado
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in philosophy of science and research methodology.

Your task: evaluate whether a question reformulation is genuinely epistemically productive.

A reformulation is genuinely better (not just superficially different) when:
1. The new question has clearer methodology — there exist known tools, experiments, or formal frameworks to make progress on it.
2. Answering the new question makes meaningful progress toward answering the original question (inferential connection).
3. The new question is not merely a paraphrase, synonym substitution, or vocabulary upgrade of the original.

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON.

{
  "direction": <true if q_good is genuinely more epistemically tractable than q_bad, else false>,
  "magnitude": <float 0.0-1.0 — how much better: 0=no improvement, 0.5=moderate, 1.0=transformative>,
  "reasoning": "<one sentence explaining the key epistemic improvement, or why it fails>"
}
"""

_USER_TEMPLATE = """\
Context (from source text):
{context}

Original question (q_bad):
{q_bad}

Reformulated question (q_good):
{q_good}

Domain: {domain}
"""


def _parse_annotation(raw: str, model: str) -> Annotation:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        data = json.loads(raw)
        return Annotation(
            model=model,
            direction=bool(data["direction"]),
            magnitude=float(max(0.0, min(1.0, data["magnitude"]))),
            reasoning=str(data.get("reasoning", "")),
        )
    except Exception as e:
        return Annotation(model=model, direction=False, magnitude=0.0,
                          reasoning="", error=str(e))


# ---------------------------------------------------------------------------
# Anotadores
# ---------------------------------------------------------------------------

class Annotator(Protocol):
    name: str
    def annotate(self, pair: dict) -> Annotation: ...


class ClaudeAnnotator:
    name = "claude-haiku-4-5-20251001"

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def annotate(self, pair: dict) -> Annotation:
        prompt = _USER_TEMPLATE.format(**pair)
        try:
            msg = self._client.messages.create(
                model=self.name,
                max_tokens=256,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return _parse_annotation(msg.content[0].text, self.name)
        except Exception as e:
            return Annotation(model=self.name, direction=False, magnitude=0.0,
                              reasoning="", error=str(e))


class GroqAnnotator:
    name = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        from groq import Groq
        self._client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def annotate(self, pair: dict) -> Annotation:
        prompt = _USER_TEMPLATE.format(**pair)
        try:
            resp = self._client.chat.completions.create(
                model=self.name,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            return _parse_annotation(resp.choices[0].message.content, self.name)
        except Exception as e:
            return Annotation(model=self.name, direction=False, magnitude=0.0,
                              reasoning="", error=str(e))


class OpenAIAnnotator:
    name = "gpt-4o-mini"

    def __init__(self) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def annotate(self, pair: dict) -> Annotation:
        prompt = _USER_TEMPLATE.format(**pair)
        try:
            resp = self._client.chat.completions.create(
                model=self.name,
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return _parse_annotation(resp.choices[0].message.content, self.name)
        except Exception as e:
            return Annotation(model=self.name, direction=False, magnitude=0.0,
                              reasoning="", error=str(e))


class GeminiAnnotator:
    name = "gemini-2.0-flash"

    def __init__(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def annotate(self, pair: dict) -> Annotation:
        from google.genai import types
        prompt = _USER_TEMPLATE.format(**pair)
        for attempt in range(4):
            try:
                resp = self._client.models.generate_content(
                    model=self.name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=_SYSTEM,
                        temperature=0.1,
                        max_output_tokens=256,
                    ),
                )
                return _parse_annotation(resp.text, self.name)
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 15 * (attempt + 1)
                    time.sleep(wait)
                    continue
                return Annotation(model=self.name, direction=False, magnitude=0.0,
                                  reasoning="", error=str(e))
        return Annotation(model=self.name, direction=False, magnitude=0.0,
                          reasoning="", error="max retries exceeded")


def _build_annotators() -> list:
    annotators = []

    if os.getenv("ANTHROPIC_API_KEY"):
        annotators.append(ClaudeAnnotator())
        print(f"  [OK] Claude Haiku (Anthropic)")
    else:
        print(f"  [--] Claude — ANTHROPIC_API_KEY nao definida")

    if os.getenv("GROQ_API_KEY"):
        annotators.append(GroqAnnotator())
        print(f"  [OK] Llama 3.3 70B (Groq)")
    else:
        print(f"  [--] Llama/Groq — GROQ_API_KEY nao definida (recomendado: console.groq.com)")

    if os.getenv("GEMINI_API_KEY"):
        annotators.append(GeminiAnnotator())
        print(f"  [OK] Gemini 1.5 Flash (Google)")
    else:
        print(f"  [--] Gemini — GEMINI_API_KEY nao definida (gratuito: aistudio.google.com)")

    if os.getenv("OPENAI_API_KEY"):
        annotators.append(OpenAIAnnotator())
        print(f"  [OK] GPT-4o-mini (OpenAI)")
    else:
        print(f"  [--] GPT — OPENAI_API_KEY nao definida (opcional)")

    return annotators


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def annotate_pairs(
    input_path: Path,
    output_path: Path,
    divergent_path: Path,
    delay: float = 0.3,
) -> tuple[list[dict], list[dict]]:

    pairs_raw = [
        json.loads(l)
        for l in input_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    print(f"\n=== Layer 2 — Multi-annotator ===")
    print(f"Pares de entrada: {len(pairs_raw)}")
    print(f"\nAnotadores ativos:")
    annotators = _build_annotators()

    if len(annotators) < MIN_ANNOTATORS:
        raise RuntimeError(
            f"Minimo de {MIN_ANNOTATORS} anotadores necessarios. "
            f"Configure GROQ_API_KEY (gratuito em console.groq.com)."
        )

    # Retoma de onde parou
    done_ids: set[str] = set()
    agreed: list[dict] = []
    divergent: list[dict] = []

    for path, bucket in [(output_path, agreed), (divergent_path, divergent)]:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec.get("source_id", ""))
                    bucket.append(rec)

    if done_ids:
        print(f"\nRetomando: {len(done_ids)} pares ja anotados "
              f"({len(agreed)} concordantes, {len(divergent)} divergentes)")

    pending = [p for p in pairs_raw if p.get("source_id", "") not in done_ids]
    print(f"Pares pendentes: {len(pending)}\n")

    out_f = output_path.open("a", encoding="utf-8")
    div_f = divergent_path.open("a", encoding="utf-8")

    try:
        for raw in tqdm(pending, desc="Anotando pares"):
            ap = AnnotatedPair(
                q_bad=raw["q_bad"],
                q_good=raw["q_good"],
                context=raw.get("context", ""),
                domain=raw.get("domain", ""),
                confidence=raw.get("confidence", 0.0),
                source=raw.get("source", ""),
                source_id=raw.get("source_id", ""),
                year=raw.get("year", 0),
            )

            for ann in annotators:
                ap.annotations.append(ann.annotate(raw))
                time.sleep(delay)

            record = ap.to_dict()
            if ap.passes_layer2:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                agreed.append(record)
            else:
                div_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                div_f.flush()
                divergent.append(record)

    finally:
        out_f.close()
        div_f.close()

    return agreed, divergent


def print_summary(agreed: list[dict], divergent: list[dict]) -> None:
    total = len(agreed) + len(divergent)
    print(f"\n{'='*55}")
    print(f"  Total anotados:      {total}")
    print(f"  Concordantes:        {len(agreed)} ({100*len(agreed)//max(total,1)}%)")
    print(f"  Divergentes:         {len(divergent)} ({100*len(divergent)//max(total,1)}%)")

    if agreed:
        mags = [p["magnitude_mean"] for p in agreed]
        print(f"  Magnitude media:     {statistics.mean(mags):.3f}")
        from collections import Counter
        sources = Counter(p["source"] for p in agreed)
        print(f"\n  Por fonte (concordantes):")
        for s, n in sources.most_common():
            print(f"    {s}: {n}")

    # Motivos de divergência
    if divergent:
        no_dir = sum(1 for p in divergent if not p["agreement_direction"])
        hi_var = sum(1 for p in divergent if p["agreement_direction"]
                     and p["magnitude_variance"] > VAR_THRESHOLD)
        print(f"\n  Motivo divergencia:")
        print(f"    Direcao discordante:  {no_dir}")
        print(f"    Variancia alta:       {hi_var}")
    print(f"{'='*55}")


if __name__ == "__main__":
    output_path   = Path("data/pairs/pairs_layer2.jsonl")
    divergent_path = Path("data/pairs/pairs_layer2_divergent.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Com Gemini (15 RPM free tier): delay de 5s entre pares garante < 12 RPM
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    agreed, divergent = annotate_pairs(
        input_path=Path("data/pairs/pairs_layer1.jsonl"),
        output_path=output_path,
        divergent_path=divergent_path,
        delay=5.0 if has_gemini else 0.3,
    )
    print_summary(agreed, divergent)

    print(f"\n=== Amostra concordantes ===")
    for p in agreed[:3]:
        models = [a["model"].split("-")[0] for a in p["annotations"]]
        mags = [a["magnitude"] for a in p["annotations"]]
        print(f"\n  [{p['source']}] [{p['domain'][:40]}]")
        print(f"  BAD:  {p['q_bad'][:80]}")
        print(f"  GOOD: {p['q_good'][:80]}")
        print(f"  Anotadores: {list(zip(models, [round(m,2) for m in mags]))}")
