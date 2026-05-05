"""
Extrai candidatos a pares da Stanford Encyclopedia of Philosophy (SEP).

Entradas selecionadas por cobrirem mudanças de paradigma documentadas —
casos onde a questão original foi substituída por uma mais tratável.

SEP é publicamente acessível via HTML. Estratégia:
  1. Lista curada de entradas sobre paradigm shifts conhecidos
  2. Download do HTML de cada entrada
  3. Extração de parágrafos relevantes (introdução + seções de mudança)
  4. Filtragem por densidade de marcadores de reformulação

O texto extraído vai para extract_pairs.py (Claude faz a extração do par).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from tqdm import tqdm

SEP_BASE = "https://plato.stanford.edu/entries"

# Entradas SEP curadas: cobertura explícita de reformulações de questões
SEP_ENTRIES = [
    # Filosofia da ciencia — mudancas de questao e paradigmas
    ("scientific-progress", "Scientific Progress"),
    ("scientific-revolutions", "Scientific Revolutions"),
    ("incommensurability", "Incommensurability"),
    ("scientific-explanation", "Scientific Explanation"),
    ("scientific-realism", "Scientific Realism"),
    ("lakatos", "Imre Lakatos"),
    ("feyerabend", "Paul Feyerabend"),
    ("thomas-kuhn", "Thomas Kuhn"),
    # Epistemologia da ciencia
    ("epistemology", "Epistemology"),
    ("scientific-knowledge-social", "Social Dimensions of Scientific Knowledge"),
    ("confirmation", "Confirmation"),
    ("popper", "Karl Popper"),
    ("abduction", "Abduction"),
    # Reducao e explicacao
    ("reduction-biology", "Reduction in Biology"),
    ("molecular-biology", "Molecular Biology"),
    ("biology-philosophy", "Philosophy of Biology"),
    ("scientific-reduction", "Scientific Reduction"),
    # Casos com reformulacoes historicas documentadas
    ("consciousness", "Consciousness"),
    ("causation-probabilistic", "Probabilistic Causation"),
    ("measurement-science", "Measurement in Science"),
    ("dispositions", "Dispositions"),
    ("models-science", "Models in Science"),
    ("science-theory-observation", "Theory and Observation in Science"),
    ("scientific-objectivity", "Scientific Objectivity"),
    ("scientific-underdetermination", "Underdetermination of Scientific Theory"),
    # Logica e metodologia
    ("scientific-method", "Scientific Method"),
    ("epistemology-bayesian", "Bayesian Epistemology"),
    ("probability-interpret", "Interpretations of Probability"),
    ("induction-problem", "The Problem of Induction"),
    ("thought-experiment", "Thought Experiments"),
    # Filosofia da mente e ciencias cognitivas
    ("artificial-intelligence", "Artificial Intelligence"),
    ("functionalism", "Functionalism"),
    ("mental-causation", "Mental Causation"),
    ("connectionism", "Connectionism"),
    ("behaviorism", "Behaviorism"),
    ("physicalism", "Physicalism"),
    ("dualism", "Dualism"),
    # Filosofia da fisica (slugs verificados)
    ("qm-everett", "Many-Worlds — Everett"),
    ("qm-collapse", "Collapse Theories"),
    ("spacetime-holearg", "Spacetime and Hole Argument"),
    ("time", "Time"),
    ("determinism-causal", "Causal Determinism"),
    ("physics-holism", "Holism in Physics"),
    ("symmetry-breaking", "Symmetry Breaking"),
    ("qt-quantcomp", "Quantum Computing"),
    # Filosofia das ciencias especiais (slugs verificados)
    ("economics", "Philosophy of Economics"),
    ("cognitive-science", "Cognitive Science"),
    ("neuroscience", "Philosophy of Neuroscience"),
    ("freud", "Freud and Psychoanalysis"),
    # Biologia (slugs verificados)
    ("natural-selection", "Natural Selection"),
    ("evolution", "Evolution"),
    ("species", "Species"),
    ("teleology-biology", "Teleology in Biology"),
    ("genetics", "Genetics"),
    ("fitness", "Fitness"),
    # Metafisica relevante
    ("natural-kinds", "Natural Kinds"),
    ("laws-of-nature", "Laws of Nature"),
    ("causation-metaphysics", "Causation"),
]


def _fetch_sep_entry(slug: str) -> str | None:
    url = f"{SEP_BASE}/{slug}/"
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0 (research bot)"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        print(f"  Erro ao buscar {slug}: {e}")
        return None


def _extract_text_from_html(html: str, max_chars: int = 8000) -> str:
    # Remove scripts, styles, nav
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL)
    html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL)
    html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL)

    # Foca no corpo do artigo
    main_match = re.search(r'<div[^>]+id=["\']main-text["\'][^>]*>(.*?)</div>', html, re.DOTALL)
    if main_match:
        html = main_match.group(1)

    # Remove tags restantes
    text = re.sub(r"<[^>]+>", " ", html)
    # Normaliza espaços
    text = re.sub(r"\s+", " ", text).strip()

    # Extrai parágrafos com maior densidade de marcadores primeiro
    # para maximizar a chance de capturar reformulações que aparecem no meio do artigo
    words = text.split()
    if len(words) <= max_chars // 5:
        return text

    # Janelas de 400 palavras, pontuadas por densidade de marcadores
    window = 80
    scored: list[tuple[float, int]] = []
    for i in range(0, len(words) - window, window // 2):
        chunk = " ".join(words[i:i + window]).lower()
        score = sum(chunk.count(m) for m in _REFORMULATION_MARKERS)
        scored.append((score, i))

    scored.sort(reverse=True)
    # Pega introdução (primeiras 2000 chars) + top segmentos com sinal
    intro = " ".join(words[:400])
    top_chunks = []
    seen_starts: set[int] = set()
    for _, start in scored[:6]:
        if start not in seen_starts:
            seen_starts.add(start)
            top_chunks.append(" ".join(words[start:start + window]))

    combined = intro + " ... " + " ... ".join(top_chunks)
    return combined[:max_chars]


_REFORMULATION_MARKERS = [
    "the question is not", "rather than asking", "instead of asking",
    "replaced by", "superseded", "abandoned", "the right question",
    "reframe", "reformulat", "more tractable", "better question",
    "shift from", "no longer ask", "transformed into",
]


def _has_reformulation_signal(text: str) -> bool:
    text_lower = text.lower()
    return any(m in text_lower for m in _REFORMULATION_MARKERS)


def fetch_sep_candidates(output_path: Path) -> list[dict]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache = output_path.with_suffix(".jsonl")

    if cache.exists():
        candidates = [json.loads(l) for l in cache.read_text(encoding="utf-8").splitlines() if l.strip()]
        cached_slugs = {c.get("slug", "") for c in candidates}
        pending = [(slug, label) for slug, label in SEP_ENTRIES if slug not in cached_slugs]
        if not pending:
            print(f"Candidatos SEP carregados do cache: {len(candidates)}")
            return candidates
        print(f"Cache parcial SEP ({len(candidates)} entradas). {len(pending)} slugs novos/corrigidos — buscando incrementalmente.")
    else:
        candidates = []
        pending = SEP_ENTRIES

    with cache.open("a", encoding="utf-8") as f:
        for slug, label in tqdm(pending, desc="Buscando SEP"):
            html = _fetch_sep_entry(slug)
            if not html:
                continue

            text = _extract_text_from_html(html)
            if len(text) < 200:
                continue

            # Ignora paginas de "Document Not Found" do SEP (retornam 200 mas com conteudo vazio)
            if "document not found" in text.lower()[:200]:
                print(f"  Slug inexistente (SEP 200/not-found): {slug}")
                continue

            has_signal = _has_reformulation_signal(text)
            entry = {
                "id": f"sep_{slug}",
                "title": f"SEP: {label}",
                "abstract": text,
                "year": 0,
                "categories": ["philosophy_of_science"],
                "source": "sep",
                "slug": slug,
                "has_reformulation_signal": has_signal,
            }
            candidates.append(entry)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            time.sleep(1.5)

    n_signal = sum(1 for c in candidates if c.get("has_reformulation_signal"))
    print(f"Candidatos SEP salvos: {len(candidates)} ({n_signal} com sinal de reformulacao)")
    return candidates


if __name__ == "__main__":
    out = Path("data/pairs/sep_candidates.jsonl")
    candidates = fetch_sep_candidates(out)
    print(f"\nTotal: {len(candidates)} entradas SEP")
    for c in candidates[:5]:
        signal = "[sinal]" if c["has_reformulation_signal"] else "[sem sinal]"
        print(f"  {signal} {c['title']}")
        print(f"        {c['abstract'][:120]}...")
