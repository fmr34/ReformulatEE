"""
Funcao de recompensa composta conforme Source.txt + patches v3.

EE(Q, Q0) = b1*Respondibilidade + b2*Tratabilidade + b3*Nao-trivialidade
Score(Q, Q0, alpha) = alpha*EE + (1-alpha)*Prox

Tratabilidade eh ponderada por confidence (Patch A):
  tratabilidade_ponderada = prob_tractable * confidence

Filtro de Estagio 1 (Patch B):
  rejeitar Q se EE(Q) <= EE(Q0) + epsilon
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field

from dotenv import load_dotenv

load_dotenv(override=True)

from src.corpus.index import CorpusIndex
from src.ee.nao_trivialidade import cosine_similarity
from src.ee.nao_trivialidade import nao_trivialidade
from src.ee.respondibilidade import respondibilidade
from src.ee.tratabilidade import tratabilidade

# Pesos calibrados (Fase 2 — grid search sobre 163 pares Layer 2)
_B1 = float(os.getenv("BETA1", "0.05"))
_B2 = float(os.getenv("BETA2", "0.05"))
_B3 = float(os.getenv("BETA3", "0.90"))
_ALPHA = float(os.getenv("ALPHA", "0.5"))
_EPSILON = float(os.getenv("EPSILON", "0.05"))
# Parametros da funcao sino (nao_trivialidade) — calibrados
_BELL_CENTER = float(os.getenv("BELL_CENTER", "0.50"))
_BELL_WIDTH = float(os.getenv("BELL_WIDTH", "0.25"))

# Aliases publicos para testes e uso externo
BETA1 = _B1
BETA2 = _B2
BETA3 = _B3
ALPHA = _ALPHA
EPSILON = _EPSILON


class _NullIndex:
    """Corpus index stub para testes — retorna scores zero sem precisar de corpus."""

    def search(self, query: str, top_k: int = 10):  # noqa: ARG002
        return []

    def __len__(self):
        return 0


@dataclass
class EEResult:
    query: str
    respondibilidade: float
    tratabilidade: float
    tratabilidade_confidence: float
    nao_trivialidade: float
    ee: float
    prox: float | None = None
    score: float | None = None
    trajectory: str = ""
    nearest_resolved: str = ""
    betas: tuple[float, float, float] = field(default_factory=lambda: (_B1, _B2, _B3))


def compute_ee(
    query: str,
    q0: str,
    index: CorpusIndex,
    beta1: float = _B1,
    beta2: float = _B2,
    beta3: float = _B3,
    bell_center: float = _BELL_CENTER,
    bell_width: float = _BELL_WIDTH,
    top_k: int = 10,
) -> EEResult:
    resp = respondibilidade(query, index, top_k=top_k)

    tract_out = tratabilidade(query)
    tract = tract_out["prob_tractable"] * tract_out["confidence"]

    nt = nao_trivialidade(query, q0, center=bell_center, width=bell_width)

    ee = beta1 * resp + beta2 * tract + beta3 * nt
    prox = cosine_similarity(query, q0)

    return EEResult(
        query=query,
        respondibilidade=resp,
        tratabilidade=tract_out["prob_tractable"],
        tratabilidade_confidence=tract_out["confidence"],
        nao_trivialidade=nt,
        ee=ee,
        prox=prox,
        trajectory=tract_out.get("trajectory", ""),
        nearest_resolved=tract_out.get("nearest_resolved_question", ""),
        betas=(beta1, beta2, beta3),
    )


def compute_score(result: EEResult, alpha: float = _ALPHA) -> float:
    if result.prox is None:
        raise ValueError("EEResult.prox nao calculado.")
    score = alpha * result.ee + (1 - alpha) * result.prox
    result.score = score
    return score


def passes_stage1_filter(
    q_result: EEResult,
    q0_result: EEResult,
    epsilon: float = _EPSILON,
) -> bool:
    """Estagio 1 do Patch B: rejeitar se EE(Q) <= EE(Q0) + epsilon."""
    return q_result.ee > q0_result.ee + epsilon


if __name__ == "__main__":
    from pathlib import Path

    from src.corpus.fetch import fetch_corpus
    from src.corpus.index import build_index

    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    fetch_corpus(corpus_dir)
    idx = build_index(corpus_dir)

    q0 = "What is the essence of life?"
    candidates = [
        "What are the molecular mechanisms of self-replication in living cells?",
        "What is the true nature of life?",
        "How do ribosomes synthesize proteins from mRNA sequences?",
        "What is the meaning of consciousness?",
    ]

    print(f"\nQ0: {q0}")
    r0 = compute_ee(q0, q0, idx)
    compute_score(r0)
    print(f"  EE={r0.ee:.3f} | Score={r0.score:.3f}\n")

    for q in candidates:
        r = compute_ee(q, q0, idx)
        compute_score(r)
        passed = passes_stage1_filter(r, r0)
        print(f"  [{'PASS' if passed else 'FAIL'}] EE={r.ee:.3f} Score={r.score:.3f} | {q[:60]}")
