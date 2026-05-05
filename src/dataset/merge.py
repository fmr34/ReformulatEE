"""
Consolida, deduplica e valida todos os pares extraídos.

Saída final: data/pairs/pairs_layer1.jsonl
  — Camada 1 do Patch B: pares atestados historicamente, alta qualidade

Deduplicacao: remove pares onde q_bad ou q_good sao semanticamente
duplicatas (cosine similarity > 0.92) de um par ja incluido.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.dataset.extract_pairs import load_pairs

_DEDUP_THRESHOLD = 0.92
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def _embed_batch(texts: list[str]) -> np.ndarray:
    return _get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _is_duplicate(new_q: str, existing_embeddings: np.ndarray, threshold: float) -> bool:
    if len(existing_embeddings) == 0:
        return False
    new_emb = _embed_batch([new_q])[0]
    sims = existing_embeddings @ new_emb
    return bool(np.max(sims) > threshold)


def merge_and_deduplicate(
    extracted_path: Path,
    output_path: Path,
    min_confidence: float = 0.5,
) -> list[dict]:
    pairs = load_pairs(extracted_path, min_confidence=min_confidence)
    print(f"Pares brutos carregados: {len(pairs)}")

    good_embeddings = np.zeros((0, 384))
    bad_embeddings = np.zeros((0, 384))
    kept: list[dict] = []
    n_dup = 0

    for p in pairs:
        q_bad = p.get("q_bad", "").strip()
        q_good = p.get("q_good", "").strip()

        if not q_bad or not q_good:
            continue
        if not q_bad.endswith("?") or not q_good.endswith("?"):
            continue
        if q_bad.lower() == q_good.lower():
            continue
        if len(q_bad) < 15 or len(q_good) < 15:
            continue

        if (_is_duplicate(q_bad, bad_embeddings, _DEDUP_THRESHOLD) or
                _is_duplicate(q_good, good_embeddings, _DEDUP_THRESHOLD)):
            n_dup += 1
            continue

        new_bad = _embed_batch([q_bad])
        new_good = _embed_batch([q_good])
        bad_embeddings = np.vstack([bad_embeddings, new_bad]) if len(bad_embeddings) else new_bad
        good_embeddings = np.vstack([good_embeddings, new_good]) if len(good_embeddings) else new_good

        kept.append(p)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Duplicatas removidas:  {n_dup}")
    print(f"Pares finais (Layer 1): {len(kept)}")
    print(f"Salvos em: {output_path}")
    return kept


def print_summary(pairs: list[dict]) -> None:
    from collections import Counter
    sources = Counter(p["source"] for p in pairs)
    domains = Counter(p.get("domain", "unknown") for p in pairs)
    confs = [p.get("confidence", 0) for p in pairs]

    print(f"\n=== Resumo Layer 1 ===")
    print(f"Total pares: {len(pairs)}")
    print(f"Confianca media: {np.mean(confs):.2f} | min: {min(confs):.2f} | max: {max(confs):.2f}")
    print(f"\nPor fonte:")
    for src, n in sources.most_common():
        print(f"  {src}: {n}")
    print(f"\nPor dominio (top 10):")
    for dom, n in domains.most_common(10):
        print(f"  {dom}: {n}")


if __name__ == "__main__":
    extracted = Path("data/pairs/extracted_pairs.jsonl")
    output = Path("data/pairs/pairs_layer1.jsonl")

    if not extracted.exists():
        print("Arquivo extracted_pairs.jsonl nao encontrado. Rode extract_pairs.py primeiro.")
        raise SystemExit(1)

    pairs = merge_and_deduplicate(extracted, output)
    print_summary(pairs)

    print(f"\n=== Amostra ===")
    for p in pairs[:8]:
        print(f"\n  [{p['source']}] [{p.get('domain','?')}] conf={p['confidence']:.2f} ({p['year']})")
        print(f"  BAD:  {p['q_bad']}")
        print(f"  GOOD: {p['q_good']}")
