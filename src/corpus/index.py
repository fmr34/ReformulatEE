"""
Índice de corpus com BM25 e busca semântica híbrida opcional.

Quando o índice semântico (semantic_index.npy) existe ao lado do BM25,
search() automaticamente faz re-ranking semântico dos candidatos.
Respondibilidade não precisa de nenhuma alteração.

Para construir o índice semântico (uma vez):
  python -m src.corpus.build_semantic_index
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class CorpusIndex:
    def __init__(self, papers: list[dict]):
        self.papers = papers
        docs = [_tokenize(p["title"] + " " + p.get("abstract", "")) for p in papers]
        self.bm25 = BM25Okapi(docs)
        self._embeddings: np.ndarray | None = None  # lazy / carregado externamente

    # ── Busca BM25 (base) ────────────────────────────────────────────

    def _search_bm25(self, query: str, top_k: int) -> list[dict]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {**self.papers[i], "bm25_score": float(score), "_paper_idx": i}
            for i, score in ranked
            if score > 0
        ]

    # ── Busca híbrida (BM25 + semântica) ────────────────────────────

    def _search_hybrid(
        self, query: str, top_k: int, alpha: float = 0.5, bm25_pool: int = 50
    ) -> list[dict]:
        """
        Recupera candidatos via BM25 e re-rankeia com similaridade semântica.
        alpha=1.0 → apenas BM25 · alpha=0.0 → apenas semântica
        """
        pool = self._search_bm25(query, top_k=bm25_pool)
        if not pool:
            return pool

        from src.ee.nao_trivialidade import embed

        q_emb = embed(query)

        bm25_max = max(r["bm25_score"] for r in pool) or 1.0

        for r in pool:
            idx = r.get("_paper_idx", -1)
            if idx >= 0 and self._embeddings is not None:
                sem = float(np.dot(q_emb, self._embeddings[idx]))
                sem = max(sem, 0.0)
            else:
                sem = 0.0
            r["hybrid_score"] = alpha * r["bm25_score"] / bm25_max + (1.0 - alpha) * sem

        ranked = sorted(pool, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
        for r in ranked:
            r.pop("_paper_idx", None)
        return ranked

    # ── Interface pública ────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Busca no corpus. Usa busca híbrida automaticamente quando o índice
        semântico está carregado, caso contrário usa apenas BM25.
        """
        if self._embeddings is not None:
            return self._search_hybrid(query, top_k=top_k)
        results = self._search_bm25(query, top_k=top_k)
        for r in results:
            r.pop("_paper_idx", None)
        return results

    # ── Índice semântico ─────────────────────────────────────────────

    def build_semantic_index(self, save_path: Path | None = None) -> None:
        """
        Computa embeddings para todos os papers e carrega em memória.
        Se save_path for fornecido, persiste em disco (.npy).
        """
        from src.ee.nao_trivialidade import embed

        texts = [p.get("title", "") + " " + p.get("abstract", "")[:300] for p in self.papers]
        print(f"  Computando embeddings para {len(texts)} papers...")
        self._embeddings = np.array([embed(t) for t in texts], dtype=np.float32)
        if save_path is not None:
            np.save(str(save_path), self._embeddings)
            print(f"  Índice semântico salvo em: {save_path}")

    # ── Persistência ─────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Salva sem embeddings (mantém pkl pequeno e compatível)
        emb_backup = self._embeddings
        self._embeddings = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self._embeddings = emb_backup

    @classmethod
    def load(cls, path: Path) -> "CorpusIndex":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not hasattr(obj, "_embeddings"):
            obj._embeddings = None
        # Tenta carregar índice semântico se existir
        sem_path = Path(str(path).replace(".pkl", "_semantic.npy"))
        if sem_path.exists():
            obj._embeddings = np.load(str(sem_path))
            print(f"  Índice semântico carregado ({obj._embeddings.shape[0]} papers).")
        return obj


# ── Builder ──────────────────────────────────────────────────────────────────


def build_index(corpus_dir: Path) -> CorpusIndex:
    index_path = corpus_dir / "bm25_index.pkl"

    if index_path.exists():
        print("Índice BM25 carregado do cache.")
        return CorpusIndex.load(index_path)

    papers_file = corpus_dir / "papers.jsonl"
    if not papers_file.exists():
        raise FileNotFoundError(f"Corpus não encontrado em {papers_file}. Rode fetch.py primeiro.")

    papers = [
        json.loads(l) for l in papers_file.read_text(encoding="utf-8").splitlines() if l.strip()
    ]
    print(f"Construindo índice BM25 sobre {len(papers)} papers...")
    index = CorpusIndex(papers)
    index.save(index_path)
    print(f"Índice salvo em {index_path}")
    return index


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    idx = build_index(corpus_dir)
    has_sem = idx._embeddings is not None
    print(f"Índice semântico: {'✅ ativo' if has_sem else '⚠ não disponível'}")
    results = idx.search("how are research questions formulated in science", top_k=5)
    for r in results:
        score_key = "hybrid_score" if "hybrid_score" in r else "bm25_score"
        print(f"  [{r[score_key]:.3f}] {r['title'][:80]}")
