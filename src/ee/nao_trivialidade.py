"""
Nao-trivialidade(Q, Q0) -- distancia semantica com funcao sino.

Logica: maxima para distancias intermediarias entre Q e Q0.
  - Muito proxima de Q0 -> trivial (parafrase)
  - Muito distante de Q0 -> desconectada da intencao original
  - Distancia intermediaria -> reformulacao genuina

Parametros (calibraveis):
  center: distancia cosine-otima (default 0.45)
  width:  largura da gaussiana   (default 0.20)

Retorna float em [0, 1].
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_embedder: SentenceTransformer | None = None
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder


def embed(text: str) -> np.ndarray:
    return _get_embedder().encode(text, normalize_embeddings=True)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings already normalized -> dot product = cosine similarity
    sim = float(np.dot(a, b))
    return 1.0 - sim


def bell(distance: float, center: float = 0.45, width: float = 0.20) -> float:
    return float(np.exp(-((distance - center) ** 2) / (2 * width**2)))


def nao_trivialidade(
    q: str,
    q0: str,
    center: float = 0.45,
    width: float = 0.20,
) -> float:
    eq = embed(q)
    eq0 = embed(q0)
    dist = cosine_distance(eq, eq0)
    return bell(dist, center=center, width=width)


def cosine_similarity(q: str, q0: str) -> float:
    eq = embed(q)
    eq0 = embed(q0)
    return float(np.dot(eq, eq0))


if __name__ == "__main__":
    q0 = "What is the essence of life?"
    pairs = [
        ("What is the essence of life?", "identica"),
        ("What is the true nature of life?", "parafrase"),
        (
            "What are the molecular mechanisms of self-replication in living cells?",
            "boa reformulacao",
        ),
        ("What is the optimal trading strategy for cryptocurrency markets?", "desconectada"),
    ]
    for q, label in pairs:
        score = nao_trivialidade(q, q0)
        print(f"  [{label}] {score:.3f} | {q[:65]}")
