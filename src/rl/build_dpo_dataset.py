"""
Fase 3 — Construcao do dataset DPO com curriculo de alpha annealing.

Formato DPO:
  {
    "prompt":   instrucao de reformulacao + q_bad,
    "chosen":   q_good (reformulacao genuina, label=1),
    "rejected": q_good_fake (adversarial probe, label=0),
    "tier":     1 | 2 | 3  (curriculo de dificuldade),
    "alpha":    0.3 | 0.6 | 0.9  (peso de EE vs proximidade),
    "source_id": ...,
  }

Curriculo de alpha annealing (score = alpha*EE + (1-alpha)*Prox):
  Tier 1 (alpha=0.3): q_good proximo de q_bad (dist <= 0.35)
    -> O modelo aprende reformulacoes minimas e seguras
  Tier 2 (alpha=0.6): distancia intermediaria (0.35 < dist <= 0.60)
    -> O modelo aprende a equilibrar inovacao e coerencia
  Tier 3 (alpha=0.9): q_good distante de q_bad (dist > 0.60)
    -> O modelo aprende reformulacoes audaciosas de alta EE

Pairing strategy:
  - Para cada q_bad com probes adversariais: 1 chosen x 3 rejected (1 por tipo)
  - Para q_bad sem probes: 1 chosen x 1 rejected cross-domain
  - Prioridade rejected: parafrase > especulativa > desconectada > cross-domain

Saida: data/rl/dpo_dataset.jsonl  (todos os tiers)
       data/rl/dpo_tier{1,2,3}.jsonl  (por tier, para treinamento em curriculo)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

LAYER2_PATH = Path("data/pairs/pairs_layer2.jsonl")
PROBES_PATH = Path("data/pairs/adversarial_probes.jsonl")
CROSS_PATH = Path("data/pairs/adversarial_probes_crossdomain.jsonl")
OUT_DIR = Path("data/rl")

SEED = 42
TIER_CUTS = (0.35, 0.60)  # limites de distancia para os 3 tiers
ALPHA_PER_TIER = {1: 0.3, 2: 0.6, 3: 0.9}

PROMPT_TEMPLATE = (
    "You are an expert in philosophy of science. "
    "Reformulate the following research question to make it more epistemically tractable: "
    "operationalizable, methodologically grounded, and answerable with existing tools.\n\n"
    "Original question: {q_bad}\n\n"
    "Reformulated question:"
)


def pr(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Carrega dados
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _compute_distances(pairs: list[dict]) -> dict[str, float]:
    """
    Retorna {source_id: cosine_distance(q_good, q_bad)}.
    """
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    q_bads = [p["q_bad"] for p in pairs]
    q_goods = [p["q_good"] for p in pairs]

    all_embs = emb.encode(
        q_bads + q_goods, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    emb_bad = all_embs[: len(pairs)]
    emb_good = all_embs[len(pairs) :]
    sims = (emb_bad * emb_good).sum(axis=1)
    dists = 1.0 - sims

    return {p["source_id"]: float(d) for p, d in zip(pairs, dists)}


def _assign_tier(dist: float) -> int:
    if dist <= TIER_CUTS[0]:
        return 1
    if dist <= TIER_CUTS[1]:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Construcao do dataset
# ---------------------------------------------------------------------------


def build_dataset(seed: int = SEED) -> list[dict]:
    random.seed(seed)

    layer2 = _load_jsonl(LAYER2_PATH)
    probes = _load_jsonl(PROBES_PATH)
    cross = _load_jsonl(CROSS_PATH)

    # Indexa probes por (q_bad, tipo)
    probes_by_qbad: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in probes:
        probes_by_qbad[p["q_bad"]][p["probe_type"]] = p

    # Indexa cross-domain por q_bad
    cross_by_qbad: dict[str, list[dict]] = defaultdict(list)
    for c in cross:
        cross_by_qbad[c["q_bad"]].append(c)

    pr(f"  Layer2: {len(layer2)} pares")
    pr(f"  Probes adversariais: {len(probes)}")
    pr(f"  Probes cross-domain: {len(cross)}")

    # Computa distancias semanticas para curriculo
    pr("  Computando distancias semanticas...")
    dists = _compute_distances(layer2)

    samples: list[dict] = []

    for pair in layer2:
        sid = pair["source_id"]
        q_bad = pair["q_bad"]
        q_good = pair["q_good"]
        dist = dists.get(sid, 0.5)
        tier = _assign_tier(dist)
        alpha = ALPHA_PER_TIER[tier]
        prompt = PROMPT_TEMPLATE.format(q_bad=q_bad)

        # Candidatos rejected (prioridade: parafrase, especulativa, desconectada, cross)
        rejected_pool: list[tuple[str, str]] = []  # (q_fake, probe_type)

        tipo_pref = ["parafrase", "especulativa", "desconectada"]
        for tipo in tipo_pref:
            if tipo in probes_by_qbad[q_bad]:
                fake = probes_by_qbad[q_bad][tipo]["q_good_fake"]
                rejected_pool.append((fake, tipo))

        if cross_by_qbad[q_bad]:
            cx = random.choice(cross_by_qbad[q_bad])
            rejected_pool.append((cx["q_good_fake"], "desconectada_cross"))

        if not rejected_pool:
            pr(f"  Sem rejected para: {q_bad[:50]} — ignorado")
            continue

        # Gera 1 amostra por rejected disponivel (max 3)
        for fake_q, probe_type in rejected_pool[:3]:
            samples.append(
                {
                    "prompt": prompt,
                    "chosen": q_good,
                    "rejected": fake_q,
                    "tier": tier,
                    "alpha": alpha,
                    "dist_good": round(dist, 4),
                    "probe_type": probe_type,
                    "source_id": sid,
                    "domain": pair.get("domain", ""),
                    "source": pair.get("source", ""),
                }
            )

    return samples


# ---------------------------------------------------------------------------
# Salva e exibe resumo
# ---------------------------------------------------------------------------


def salvar(samples: list[dict]) -> None:
    from collections import Counter

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Arquivo completo
    full_path = OUT_DIR / "dpo_dataset.jsonl"
    with full_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    pr(f"\n  Salvo: {full_path}  ({len(samples)} amostras)")

    # Por tier
    for tier in [1, 2, 3]:
        tier_samples = [s for s in samples if s["tier"] == tier]
        path = OUT_DIR / f"dpo_tier{tier}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for s in tier_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        pr(f"  Salvo: {path}  ({len(tier_samples)} amostras, alpha={ALPHA_PER_TIER[tier]})")

    # Resumo
    tier_cnt = Counter(s["tier"] for s in samples)
    probe_cnt = Counter(s["probe_type"] for s in samples)
    source_cnt = Counter(s["source"] for s in samples)

    pr("\n  Distribuicao por tier:")
    for t, n in sorted(tier_cnt.items()):
        pr(f"    Tier {t} (alpha={ALPHA_PER_TIER[t]}): {n:>4} amostras")

    pr("\n  Distribuicao por tipo de rejected:")
    for pt, n in probe_cnt.most_common():
        pr(f"    {pt:<25}: {n:>4}")

    pr("\n  Distribuicao por fonte:")
    for s, n in source_cnt.most_common():
        pr(f"    {s:<25}: {n:>4}")

    dists_arr = np.array([s["dist_good"] for s in samples])
    pr("\n  Distancias semanticas q_good-q_bad:")
    pr(
        f"    media={dists_arr.mean():.3f}  "
        f"p25={np.percentile(dists_arr,25):.3f}  "
        f"p75={np.percentile(dists_arr,75):.3f}  "
        f"max={dists_arr.max():.3f}"
    )

    # Amostra por tier
    pr("\n  Amostra (1 por tier):")
    for tier in [1, 2, 3]:
        ex = next(s for s in samples if s["tier"] == tier)
        pr(f"\n  [TIER {tier} | alpha={ALPHA_PER_TIER[tier]} | dist={ex['dist_good']}]")
        pr(f"    prompt  : {ex['prompt'].split(chr(10))[-2][:70]}")
        pr(f"    chosen  : {ex['chosen'][:70]}")
        pr(f"    rejected: {ex['rejected'][:70]}  [{ex['probe_type']}]")


if __name__ == "__main__":
    pr("=" * 60)
    pr("  Fase 3 — Build DPO Dataset com curriculo alpha-annealing")
    pr("=" * 60)
    samples = build_dataset()
    pr(f"\n  Total amostras geradas: {len(samples)}")
    salvar(samples)
    pr("\nConcluido.")
