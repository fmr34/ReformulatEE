"""
Busca papers no arXiv que contêm marcadores explícitos de reformulação de questões.

Estratégia: queries direcionadas a frases que documentam a transição Q_mal -> Q_bem.
Retorna candidatos para extracao de pares via Claude (extract_pairs.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import arxiv
from tqdm import tqdm

# Marcadores linguísticos de reformulação documentada
REFORMULATION_QUERIES = [
    'abs:"the right question is not" OR abs:"we reformulate the question"',
    'abs:"instead of asking" AND abs:"we ask"',
    'abs:"the question should be" AND abs:"rather than"',
    'abs:"reframe the question" OR abs:"reframing the question"',
    'abs:"wrong question" AND abs:"right question"',
    'abs:"the question is not X but" OR abs:"not the question of"',
    'abs:"shift from asking" OR abs:"shift the question"',
    'abs:"more productive question" OR abs:"more tractable question"',
    # Historia e filosofia da ciencia com reformulacoes explicitas
    'cat:physics.hist-ph AND abs:"the question" AND abs:"reformulat"',
    'cat:q-bio AND abs:"reformulate" AND abs:"question"',
    # Revisoes sistematicas com refinamento de PICO
    'abs:"research question" AND abs:"refined" AND abs:"systematic review"',
    'abs:"PICO" AND abs:"reformulation" OR abs:"PICO" AND abs:"revised question"',
    # Paradigm shifts documentados
    'abs:"paradigm shift" AND abs:"question" AND abs:"replaced"',
    'abs:"prior question" AND abs:"we now ask" OR abs:"previous question" AND abs:"now ask"',
    # Filosofia e historia da ciencia
    'cat:physics.hist-ph AND abs:"the question" AND abs:"replaced"',
    'abs:"conceptual clarification" AND abs:"question" AND abs:"tractable"',
    'abs:"epistemically" AND abs:"question" AND abs:"reformulat"',
    'abs:"ill-posed question" OR abs:"ill-formed question"',
    'abs:"better formulated" AND abs:"question" AND abs:"research"',
    'abs:"untractable" AND abs:"question" AND abs:"replaced"',
    # Ciencias cognitivas e IA
    'abs:"question answering" AND abs:"reformulat" AND abs:"original question"',
    'abs:"query reformulation" AND abs:"epistemic" OR abs:"query rewriting" AND abs:"semantic"',
    # Biologia e medicina com reformulacoes explicitas
    'cat:q-bio AND abs:"the question becomes" AND abs:"instead"',
    'abs:"PICO" AND abs:"refined" AND abs:"question"',
    'abs:"systematic review" AND abs:"research question" AND abs:"reformulat"',
    # Ciencias sociais e economicas
    'cat:econ AND abs:"reformulat" AND abs:"question" AND abs:"tractable"',
    'abs:"the question is not whether" AND abs:"but rather"',
    # Matematica e logica
    'abs:"the problem is not" AND abs:"the right problem is"',
    'abs:"reformulate the problem" AND abs:"equivalent" AND abs:"tractable"',
]

MAX_PER_QUERY = 80


def fetch_arxiv_candidates(
    output_path: Path,
    max_per_query: int = MAX_PER_QUERY,
    force_refresh: bool = False,
) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.with_suffix(".jsonl")

    if cache.exists() and not force_refresh:
        candidates = [
            json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        # Se o número de queries cresceu, re-fetch automático para queries ausentes
        cached_queries = {c.get("query", "") for c in candidates}
        new_queries = [q for q in REFORMULATION_QUERIES if q not in cached_queries]
        if not new_queries:
            print(f"Candidatos arXiv carregados do cache: {len(candidates)}")
            return candidates
        print(
            f"Cache existente ({len(candidates)} candidatos). {len(new_queries)} queries novas — buscando incrementalmente."
        )
    else:
        candidates = []
        new_queries = REFORMULATION_QUERIES

    seen_ids = {c["id"] for c in candidates}
    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)

    # Abre em modo append — grava após cada query, preserva progresso se interrompido
    with cache.open("a", encoding="utf-8") as f:
        for query in tqdm(new_queries, desc="Buscando arXiv (pares)"):
            search = arxiv.Search(
                query=query,
                max_results=max_per_query,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            new_this_query: list[dict] = []
            try:
                for result in client.results(search):
                    if result.entry_id in seen_ids:
                        continue
                    seen_ids.add(result.entry_id)
                    entry = {
                        "id": result.entry_id,
                        "title": result.title,
                        "abstract": result.summary.replace("\n", " "),
                        "year": result.published.year,
                        "categories": result.categories,
                        "source": "arxiv",
                        "query": query,
                    }
                    new_this_query.append(entry)
                    candidates.append(entry)
            except KeyboardInterrupt:
                # Salva o que chegou antes do Ctrl+C e propaga
                for e in new_this_query:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
                f.flush()
                raise
            except Exception as e:
                print(f"  Erro na query: {e}")

            for e in new_this_query:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(1.0)

    print(f"Candidatos arXiv salvos: {len(candidates)}")
    return candidates


if __name__ == "__main__":
    out = Path("data/pairs/arxiv_candidates.jsonl")
    candidates = fetch_arxiv_candidates(out)
    print(f"\nTotal: {len(candidates)} papers candidatos")
    for c in candidates[:5]:
        print(f"  [{c['year']}] {c['title'][:70]}")
