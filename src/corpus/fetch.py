"""
Fetch papers from arXiv for the RAG corpus.
Domains: philosophy of science, scientific methodology, epistemology.
"""

import json
import time
import arxiv
from pathlib import Path
from tqdm import tqdm

SEARCH_QUERIES = [
    # Filosofia e metodologia (relevante para questoes label=0 — baixo score esperado)
    "philosophy of science research methodology",
    "scientific paradigm epistemology",
    "systematic review research question formulation",
    # Ciencia empirica — cobre dominios das questoes label=1
    "CRISPR gene editing molecular mechanisms",
    "transformer neural network scaling language model",
    "hippocampus neurogenesis sleep deprivation",
    "antibiotic resistance plasmid horizontal gene transfer",
    "meta-analysis publication bias statistical methods",
    "gut microbiome immune response vaccination",
    "working memory cognitive training transfer",
    "reinforcement learning reward prediction dopamine",
    "protein folding computational prediction",
    "climate carbon cycle feedback atmospheric",
    "minimum wage employment labor economics",
    "urban heat island precipitation climate",
    "epigenetics inheritance disease transgenerational",
    "convolutional neural network visual representation",
    "social network information diffusion",
    "replication crisis scientific reproducibility",
]

MAX_RESULTS_PER_QUERY = 50


def fetch_corpus(output_dir: Path, max_per_query: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "papers.jsonl"

    if cache_file.exists():
        papers = [json.loads(l) for l in cache_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"Corpus carregado do cache: {len(papers)} papers")
        return papers

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)
    seen_ids: set[str] = set()
    papers: list[dict] = []

    for query in tqdm(SEARCH_QUERIES, desc="Buscando arXiv"):
        search = arxiv.Search(
            query=query,
            max_results=max_per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            for result in client.results(search):
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)
                papers.append({
                    "id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary.replace("\n", " "),
                    "year": result.published.year,
                    "categories": result.categories,
                })
        except Exception as e:
            print(f"Erro na query '{query}': {e}")
        time.sleep(1.0)

    with cache_file.open("w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Corpus salvo: {len(papers)} papers em {cache_file}")
    return papers


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import os
    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    papers = fetch_corpus(corpus_dir)
    print(f"Total: {len(papers)} papers")
