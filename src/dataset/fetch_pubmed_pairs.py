"""
Busca artigos no PubMed/MEDLINE com marcadores de reformulação de questões.

API NCBI E-utilities — gratuita, sem key (3 req/s), com NCBI_API_KEY (10 req/s).
Domínios: medicina, biologia, saúde pública — especialmente revisões sistemáticas
com refinamento explícito de PICO (Population, Intervention, Comparison, Outcome).

Referência: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests
from tqdm import tqdm

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Queries PubMed — mix de frases exatas e termos livres para maximizar recall
PUBMED_QUERIES = [
    # Frases exatas com campo TIAB (mais precisas quando existem)
    '"reformulate the question"[TIAB]',
    '"reframe the research question"[TIAB]',
    '"the right question is not"[TIAB]',
    '"instead of asking" "we ask"[TIAB]',
    '"research question" "reformulation"[TIAB] systematic review',
    '"PICO" "reformulation" systematic review question',
    '"question shifted from" OR "question changed from"[TIAB]',
    '"the question is no longer"[TIAB]',
    '"more tractable" question research methodology',
    '"paradigm shift" question replaced research',
    # Termos livres — maior recall, menor precisão (compensado pelo Claude extractor)
    "research question refinement systematic review methodology",
    "conceptual clarification question reformulation epistemology",
    "wrong question right question scientific progress",
    "question reformulation productivity epistemics",
    "reframing scientific question tractable operationalization",
]

MAX_PER_QUERY = 80
_BATCH_SIZE = 20  # efetch por vez
_DELAY = 0.35  # <3 req/s sem key; use NCBI_API_KEY para 10 req/s


def _api_key_param() -> dict:
    key = os.getenv("NCBI_API_KEY", "")
    if key:
        return {"api_key": key}
    return {}


def _search_ids(query: str, retmax: int) -> list[str]:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        **_api_key_param(),
    }
    r = requests.get(_ESEARCH, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])


def _fetch_abstracts(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        **_api_key_param(),
    }
    r = requests.get(_EFETCH, params=params, timeout=30)
    r.raise_for_status()
    xml = r.text

    import re

    articles = []
    # Parse simples via regex — evita dependência de lxml/xml.etree
    for block in re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL):
        pmid_m = re.search(r"<PMID[^>]*>(\d+)</PMID>", block)
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", block, re.DOTALL)
        abstract_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", block, re.DOTALL)
        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", block, re.DOTALL)

        pmid = pmid_m.group(1) if pmid_m else ""
        title = re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else ""
        abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1)) if abstract_m else ""
        year = int(year_m.group(1)) if year_m else 0

        if pmid and abstract:
            articles.append(
                {
                    "id": f"pubmed_{pmid}",
                    "title": title.replace("\n", " ").strip(),
                    "abstract": abstract.replace("\n", " ").strip(),
                    "year": year,
                    "categories": ["biomedical"],
                    "source": "pubmed",
                }
            )
    return articles


def fetch_pubmed_candidates(output_path: Path, max_per_query: int = MAX_PER_QUERY) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        candidates = [
            json.loads(l) for l in output_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        cached_queries = {c.get("query", "") for c in candidates}
        pending = [q for q in PUBMED_QUERIES if q not in cached_queries]
        if not pending:
            print(f"Candidatos PubMed carregados do cache: {len(candidates)}")
            return candidates
        print(
            f"Cache parcial PubMed ({len(candidates)} candidatos). {len(pending)} queries pendentes."
        )
    else:
        candidates = []
        pending = PUBMED_QUERIES

    seen_ids: set[str] = {c["id"] for c in candidates}

    for query in tqdm(pending, desc="Buscando PubMed"):
        try:
            pmids = _search_ids(query, max_per_query)
        except Exception as e:
            print(f"  Erro na busca '{query[:50]}': {e}")
            time.sleep(_DELAY * 3)
            continue

        time.sleep(_DELAY)

        # Processa em batches para não estourar a URL
        for i in range(0, len(pmids), _BATCH_SIZE):
            batch = [p for p in pmids[i : i + _BATCH_SIZE] if f"pubmed_{p}" not in seen_ids]
            if not batch:
                continue
            try:
                articles = _fetch_abstracts(batch)
            except Exception as e:
                print(f"  Erro no efetch batch: {e}")
                time.sleep(_DELAY * 3)
                continue

            new_batch: list[dict] = []
            for art in articles:
                if art["id"] not in seen_ids:
                    seen_ids.add(art["id"])
                    art["query"] = query
                    candidates.append(art)
                    new_batch.append(art)

            # Grava imediatamente — preserva progresso se interrompido
            with output_path.open("a", encoding="utf-8") as fh:
                for e in new_batch:
                    fh.write(json.dumps(e, ensure_ascii=False) + "\n")

            time.sleep(_DELAY)

    print(f"Candidatos PubMed salvos: {len(candidates)}")
    return candidates


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    out = Path("data/pairs/pubmed_candidates.jsonl")
    candidates = fetch_pubmed_candidates(out)
    print(f"\nTotal: {len(candidates)} artigos candidatos")
    for c in candidates[:5]:
        print(f"  [{c['year']}] {c['title'][:70]}")
        print(f"        {c['abstract'][:100]}...")
