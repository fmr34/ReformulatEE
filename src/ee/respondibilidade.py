"""
Respondibilidade(Q) -- densidade de evidencias no corpus para a questao Q.

Pipeline:
  1. BM25 recupera top-k candidatos
  2. Cross-encoder re-pontua cada candidato contra Q
  3. Score final = media dos top-k scores do cross-encoder

Retorna float em [0, 1].
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from src.corpus.index import CorpusIndex

_cross_encoder: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_MODEL_NAME)
    return _cross_encoder


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def respondibilidade(query: str, index: CorpusIndex, top_k: int = 10) -> float:
    candidates = index.search(query, top_k=top_k)
    if not candidates:
        return 0.0

    ce = _get_cross_encoder()
    pairs = [(query, c["title"] + " " + c["abstract"][:512]) for c in candidates]
    raw_scores = ce.predict(pairs)

    # cross-encoder ms-marco returns logits; sigmoid maps to [0,1]
    scores = [_sigmoid(float(s)) for s in raw_scores]
    return float(np.mean(scores))


if __name__ == "__main__":
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from src.corpus.fetch import fetch_corpus
    from src.corpus.index import build_index

    load_dotenv()
    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    fetch_corpus(corpus_dir)
    idx = build_index(corpus_dir)

    test_pairs = [
        ("What are the molecular mechanisms of CRISPR-Cas9 gene editing?", "alta"),
        ("What is the essence of life?", "baixa"),
        ("How does transformer architecture scale with parameter count?", "alta"),
        ("What is the true nature of consciousness?", "baixa"),
    ]
    for q, label in test_pairs:
        score = respondibilidade(q, idx)
        print(f"  [{label}] {score:.3f} | {q[:60]}")
