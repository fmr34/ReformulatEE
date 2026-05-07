"""
Fase 2 — Calibracao dos betas do reward model.

Pipeline:
  1. Computa e cacheia componentes EE para todas as questoes dos pares Layer 2:
       resp(q)  = respondibilidade  (BM25 + cross-encoder, local)
       tract(q) = tratabilidade     (Claude Haiku, ~326 calls, com resume)
       dist(pair) = cosine_distance(q_good, q_bad)  (sentence-transformers, local)

  2. Grid search sobre:
       beta1, beta2, beta3  (simplex, passo 0.05)
       bell_center          in {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60}
       bell_width           in {0.10, 0.15, 0.20, 0.25, 0.30}
       epsilon              in {0.05, 0.10, 0.15, 0.20}

     Para cada combinacao: success_rate = frac(EE(q_good) > EE(q_bad) + eps)
     Objetivo: maximizar success_rate; desempate por margin media.

  3. Gate: success_rate >= 0.90

  4. Salva:
       data/calibration/scores_cache.jsonl   -- cache de componentes
       data/calibration/grid_results.jsonl   -- todos os pontos do grid
       data/calibration/best_config.json     -- configuracao otima
       .env                                  -- atualizado com novos betas

Uso:
  .venv\\Scripts\\python -m src.ee.calibrate_betas
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from dotenv import set_key
from tqdm import tqdm

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

LAYER2_PATH = Path("data/pairs/pairs_layer2.jsonl")
CACHE_PATH = Path("data/calibration/scores_cache.jsonl")
GRID_PATH = Path("data/calibration/grid_results.jsonl")
BEST_PATH = Path("data/calibration/best_config.json")
ENV_PATH = Path(".env")

SUCCESS_GATE = 0.90
API_DELAY = 0.3  # segundos entre chamadas tratabilidade

# ---------------------------------------------------------------------------
# Grid de hiperparametros
# ---------------------------------------------------------------------------

BETA_STEP = 0.05
BELL_CENTERS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
BELL_WIDTHS = [0.10, 0.15, 0.20, 0.25, 0.30]
EPSILONS = [0.05, 0.10, 0.15, 0.20]


def _simplex_grid(step: float = BETA_STEP) -> list[tuple[float, float, float]]:
    """Todos os (b1, b2, b3) com b1+b2+b3=1, bi >= step, passo=step."""
    pts = []
    n = round(1.0 / step)
    for i in range(1, n):
        for j in range(1, n - i):
            k = n - i - j
            if k >= 1:
                pts.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
    return pts


def pr(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Etapa 1 — Cache de componentes
# ---------------------------------------------------------------------------


def _load_cache() -> dict[str, dict]:
    """Carrega cache existente: {question -> {resp, tract, trajectory, confidence}}."""
    cache = {}
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["question"]] = rec
    return cache


def _save_record(rec: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def compute_and_cache_scores(pairs: list[dict]) -> dict[str, dict]:
    """
    Para cada questao unica, computa e cacheia resp + tract.
    Resume automaticamente a partir do cache existente.
    """
    from src.corpus.index import build_index
    from src.ee.respondibilidade import respondibilidade
    from src.ee.tratabilidade import tratabilidade

    cache = _load_cache()
    pr(f"\n  Cache existente: {len(cache)} questoes")

    # Coleta todas as questoes unicas
    questoes: set[str] = set()
    for p in pairs:
        questoes.add(p["q_bad"])
        questoes.add(p["q_good"])
    pendentes = [q for q in sorted(questoes) if q not in cache]

    if not pendentes:
        pr("  Todas as questoes ja estao em cache.")
        return cache

    pr(f"  Questoes pendentes: {len(pendentes)}")

    # Carrega corpus
    pr("  Carregando corpus BM25...")
    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    index = build_index(corpus_dir)

    # Computa componentes
    pr("  Computando resp + tract (Claude Haiku)...")
    erros = 0
    for q in tqdm(pendentes, desc="  Cache scores"):
        try:
            resp = respondibilidade(q, index, top_k=10)
            tract = tratabilidade(q)
            rec = {
                "question": q,
                "resp": round(resp, 6),
                "tract": round(tract["prob_tractable"] * tract["confidence"], 6),
                "trajectory": tract["trajectory"],
                "confidence": round(tract["confidence"], 4),
            }
            cache[q] = rec
            _save_record(rec)
        except Exception as e:
            erros += 1
            pr(f"\n  ERRO em '{q[:50]}': {e}")
        time.sleep(API_DELAY)

    pr(f"  Concluido: {len(pendentes) - erros} ok, {erros} erros")
    return cache


# ---------------------------------------------------------------------------
# Etapa 1b — Distancias semanticas (nao_trivialidade)
# ---------------------------------------------------------------------------


def compute_pair_distances(pairs: list[dict]) -> list[float]:
    """Cosine distance entre q_good e q_bad para cada par."""
    from sentence_transformers import SentenceTransformer

    emb = SentenceTransformer("all-MiniLM-L6-v2")

    q_bads = [p["q_bad"] for p in pairs]
    q_goods = [p["q_good"] for p in pairs]

    all_texts = q_bads + q_goods
    all_embs = emb.encode(
        all_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    emb_bad = all_embs[: len(pairs)]
    emb_good = all_embs[len(pairs) :]

    # cosine dist = 1 - dot (embeddings normalizados)
    sims = (emb_bad * emb_good).sum(axis=1)
    return (1.0 - sims).tolist()


# ---------------------------------------------------------------------------
# Etapa 2 — Grid search
# ---------------------------------------------------------------------------


def bell(dist: float, center: float, width: float) -> float:
    return float(np.exp(-((dist - center) ** 2) / (2 * width**2)))


def grid_search(
    pairs: list[dict],
    cache: dict[str, dict],
    distances: list[float],
) -> list[dict]:
    """
    Varre o grid e retorna lista de resultados ordenada por success_rate desc, margin desc.
    """
    beta_grid = _simplex_grid(BETA_STEP)
    pr(f"\n  Grid de betas: {len(beta_grid)} pontos")
    pr(f"  Bell centers : {BELL_CENTERS}")
    pr(f"  Bell widths  : {BELL_WIDTHS}")
    pr(f"  Epsilons     : {EPSILONS}")

    total_combos = len(beta_grid) * len(BELL_CENTERS) * len(BELL_WIDTHS) * len(EPSILONS)
    pr(f"  Total combos : {total_combos:,}\n")

    # Pre-extrai vetores para velocidade
    resp_bad = np.array([cache.get(p["q_bad"], {}).get("resp", 0.0) for p in pairs])
    resp_good = np.array([cache.get(p["q_good"], {}).get("resp", 0.0) for p in pairs])
    tract_bad = np.array([cache.get(p["q_bad"], {}).get("tract", 0.0) for p in pairs])
    tract_good = np.array([cache.get(p["q_good"], {}).get("tract", 0.0) for p in pairs])
    dists = np.array(distances)
    # nt_bad_dist = np.zeros(len(pairs))  # dist(q_bad, q_bad) = 0 sempre (reservado)

    results = []
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=total_combos, desc="  Grid search", ncols=70) as pbar:
        with GRID_PATH.open("w", encoding="utf-8") as fout:
            for center, width in product(BELL_CENTERS, BELL_WIDTHS):
                nt_bad = bell(0.0, center, width)  # constante por (center, width)
                nt_good = np.array([bell(d, center, width) for d in dists])

                for b1, b2, b3 in beta_grid:
                    ee_bad = b1 * resp_bad + b2 * tract_bad + b3 * nt_bad
                    ee_good = b1 * resp_good + b2 * tract_good + b3 * nt_good

                    for eps in EPSILONS:
                        margin = ee_good - ee_bad
                        success = (margin > eps).mean()
                        avg_margin = float(margin.mean())

                        rec = {
                            "beta1": b1,
                            "beta2": b2,
                            "beta3": b3,
                            "bell_center": center,
                            "bell_width": width,
                            "epsilon": eps,
                            "success_rate": round(float(success), 6),
                            "avg_margin": round(avg_margin, 6),
                        }
                        results.append(rec)
                        fout.write(json.dumps(rec) + "\n")
                        pbar.update(1)

    # Ordena por success_rate desc, margin desc
    results.sort(key=lambda r: (-r["success_rate"], -r["avg_margin"]))
    return results


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


def calibrar() -> None:
    pr("=" * 60)
    pr("  Fase 2 — Calibracao dos betas do reward model")
    pr("=" * 60)

    # Carrega pares
    pairs = [
        json.loads(l)
        for l in LAYER2_PATH.read_text(encoding="utf-8").splitlines()
        if l.strip() and json.loads(l).get("q_bad") and json.loads(l).get("q_good")
    ]
    pr(f"\n[1/4] Pares Layer 2: {len(pairs)}")

    # Etapa 1: cache de componentes
    pr("\n[2/4] Computando/carregando componentes EE...")
    cache = compute_and_cache_scores(pairs)

    n_cached = sum(1 for p in pairs if p["q_bad"] in cache and p["q_good"] in cache)
    pr(f"  Pares com cache completo: {n_cached}/{len(pairs)}")

    # Filtra pares sem cache
    pairs_ok = [p for p in pairs if p["q_bad"] in cache and p["q_good"] in cache]
    if len(pairs_ok) < len(pairs):
        pr(f"  Aviso: {len(pairs) - len(pairs_ok)} pares ignorados (sem cache)")

    # Distancias semanticas
    pr("\n[3/4] Computando distancias semanticas (nao_trivialidade)...")
    distances = compute_pair_distances(pairs_ok)
    pr(
        f"  dist media: {np.mean(distances):.3f}  "
        f"min: {np.min(distances):.3f}  max: {np.max(distances):.3f}"
    )

    # Grid search
    pr("\n[4/4] Grid search...")
    results = grid_search(pairs_ok, cache, distances)

    best = results[0]
    top5 = results[:5]

    pr(f"\n{'='*60}")
    pr("  MELHOR CONFIGURACAO:")
    pr(f"    beta1 (Respondibilidade) : {best['beta1']:.2f}")
    pr(f"    beta2 (Tratabilidade)    : {best['beta2']:.2f}")
    pr(f"    beta3 (Nao-trivialidade) : {best['beta3']:.2f}")
    pr(f"    bell_center              : {best['bell_center']}")
    pr(f"    bell_width               : {best['bell_width']}")
    pr(f"    epsilon                  : {best['epsilon']}")
    pr(f"    success_rate             : {best['success_rate']:.4f}  (gate >= {SUCCESS_GATE})")
    pr(f"    avg_margin               : {best['avg_margin']:.4f}")

    gate_ok = best["success_rate"] >= SUCCESS_GATE
    pr(f"\n  GATE: {'PASSOU' if gate_ok else 'FALHOU'}")

    pr("\n  Top 5 configuracoes:")
    pr(
        f"  {'b1':>5} {'b2':>5} {'b3':>5} {'center':>7} {'width':>6} {'eps':>5} {'succ':>7} {'margin':>8}"
    )
    for r in top5:
        pr(
            f"  {r['beta1']:>5.2f} {r['beta2']:>5.2f} {r['beta3']:>5.2f} "
            f"{r['bell_center']:>7.2f} {r['bell_width']:>6.2f} "
            f"{r['epsilon']:>5.2f} {r['success_rate']:>7.4f} {r['avg_margin']:>8.4f}"
        )

    # Salva configuracao otima
    BEST_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    pr(f"\n  Configuracao salva em: {BEST_PATH}")

    # Atualiza .env
    if ENV_PATH.exists():
        set_key(str(ENV_PATH), "BETA1", str(best["beta1"]))
        set_key(str(ENV_PATH), "BETA2", str(best["beta2"]))
        set_key(str(ENV_PATH), "BETA3", str(best["beta3"]))
        set_key(str(ENV_PATH), "EPSILON", str(best["epsilon"]))
        set_key(str(ENV_PATH), "BELL_CENTER", str(best["bell_center"]))
        set_key(str(ENV_PATH), "BELL_WIDTH", str(best["bell_width"]))
        pr("  .env atualizado com novos betas.")
    pr(f"{'='*60}")

    # Distribuicao de margens com a melhor config
    _print_margin_distribution(pairs_ok, cache, distances, best)


def _print_margin_distribution(
    pairs: list[dict],
    cache: dict[str, dict],
    distances: list[float],
    cfg: dict,
) -> None:
    b1, b2, b3 = cfg["beta1"], cfg["beta2"], cfg["beta3"]
    center, width, eps = cfg["bell_center"], cfg["bell_width"], cfg["epsilon"]

    margins = []
    falhas = []
    for p, dist in zip(pairs, distances):
        r_bad = cache[p["q_bad"]]
        r_good = cache[p["q_good"]]
        ee_bad = b1 * r_bad["resp"] + b2 * r_bad["tract"] + b3 * bell(0.0, center, width)
        ee_good = b1 * r_good["resp"] + b2 * r_good["tract"] + b3 * bell(dist, center, width)
        margin = ee_good - ee_bad
        margins.append(margin)
        if margin <= eps:
            falhas.append((p["q_bad"], p["q_good"], margin, ee_bad, ee_good))

    margins_arr = np.array(margins)
    pr("\n  Distribuicao de margens EE(q_good) - EE(q_bad):")
    pr(f"    Media  : {margins_arr.mean():+.4f}")
    pr(f"    Mediana: {np.median(margins_arr):+.4f}")
    pr(f"    P25    : {np.percentile(margins_arr, 25):+.4f}")
    pr(f"    P10    : {np.percentile(margins_arr, 10):+.4f}")
    pr(f"    Min    : {margins_arr.min():+.4f}")

    if falhas:
        pr(f"\n  Pares que nao passam no gate ({len(falhas)}):")
        for q_bad, q_good, margin, ee_bad, ee_good in falhas[:6]:
            pr(f"    margin={margin:+.3f} EEbad={ee_bad:.3f} EEgood={ee_good:.3f}")
            pr(f"      q_bad : {q_bad[:65]}")
            pr(f"      q_good: {q_good[:65]}")


if __name__ == "__main__":
    calibrar()
