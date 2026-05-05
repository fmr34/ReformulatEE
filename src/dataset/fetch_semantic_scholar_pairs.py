"""
Busca papers no Semantic Scholar com marcadores explícitos de reformulação.

API pública gratuita (sem key: 100 req/5 min; com S2_API_KEY: 1000 req/5 min).
Cobre arXiv + ACL Anthology + NeurIPS + AAAI + revistas de filosofia.

Referência: https://api.semanticscholar.org/api-docs/
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "paperId,title,abstract,year,fieldsOfStudy,externalIds"

# A S2 API requer queries curtas (<=3 termos relevantes); frases longas retornam zero.
REFORMULATION_QUERIES = [
    "question reformulation epistemic",
    "research question reformulation",
    "question reframing science",
    "scientific question tractable",
    "paradigm shift question",
    "question replaced productive",
    "reformulation problem tractable",
    "research question refinement",
    "question reframing methodology",
    "epistemic question productive",
    "question replaced methodology",
    "conceptual clarification question",
    "question reformulation philosophy",
    "ill-posed question science",
    "question transformation productive",
    "question reformulation history science",
    "question reformulation systematic review",
    "wrong question science history",
    "question reformulation epistemology",
    "tractable question research",
]

MAX_PER_QUERY = 100
_DELAY_NO_KEY = 1.5   # sem S2_API_KEY
_DELAY_WITH_KEY = 0.4


def _headers() -> dict:
    key = os.getenv("S2_API_KEY", "")
    if key:
        return {"x-api-key": key}
    return {}


def _delay() -> float:
    return _DELAY_WITH_KEY if os.getenv("S2_API_KEY") else _DELAY_NO_KEY


def fetch_s2_candidates(output_path: Path, max_per_query: int = MAX_PER_QUERY) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        candidates = [json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cached_queries = {c.get("query", "") for c in candidates}
        pending = [q for q in REFORMULATION_QUERIES if q not in cached_queries]
        if not pending:
            print(f"Candidatos S2 carregados do cache: {len(candidates)}")
            return candidates
        print(f"Cache parcial S2 ({len(candidates)} candidatos). {len(pending)} queries pendentes.")
    else:
        candidates = []
        pending = REFORMULATION_QUERIES

    seen_ids: set[str] = {c["id"] for c in candidates}

    for query in tqdm(pending, desc="Buscando Semantic Scholar"):
        params = {
            "query": query,
            "limit": min(max_per_query, 100),
            "fields": _FIELDS,
        }
        data = None
        for attempt in range(3):
            try:
                r = requests.get(_S2_SEARCH, params=params, headers=_headers(), timeout=30)
                if r.status_code == 429:
                    wait = 90 * (attempt + 1)
                    print(f"  Rate limit — aguardando {wait}s (tentativa {attempt+1}/3)")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                print(f"  Erro na query '{query[:40]}': {e}")
                time.sleep(_delay() * 3)
        if data is None:
            continue

        new_this_query: list[dict] = []
        for paper in data.get("data", []):
            pid = paper.get("paperId", "")
            abstract = (paper.get("abstract") or "").replace("\n", " ").strip()
            if not pid or not abstract or pid in seen_ids:
                continue
            seen_ids.add(pid)
            # fieldsOfStudy pode ser lista de strings ou lista de dicts dependendo da versao da API
            fields_raw = paper.get("fieldsOfStudy") or []
            categories = [
                f if isinstance(f, str) else f.get("category", "")
                for f in fields_raw
            ]
            entry = {
                "id": f"s2_{pid}",
                "title": paper.get("title", ""),
                "abstract": abstract,
                "year": paper.get("year") or 0,
                "categories": categories,
                "source": "semantic_scholar",
                "query": query,
            }
            new_this_query.append(entry)
            candidates.append(entry)

        with output_path.open("a", encoding="utf-8") as fh:
            for e in new_this_query:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")

        time.sleep(_delay())

    print(f"Candidatos S2 salvos: {len(candidates)}")
    return candidates


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    out = Path("data/pairs/s2_candidates.jsonl")
    candidates = fetch_s2_candidates(out)
    print(f"\nTotal: {len(candidates)} papers candidatos")
    for c in candidates[:5]:
        print(f"  [{c['year']}] {c['title'][:70]}")
        print(f"        {c['abstract'][:100]}...")
