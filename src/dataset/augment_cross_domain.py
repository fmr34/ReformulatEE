"""
Augmentação cross-domain para a classe 'desconectada'.

Lógica:
  Para cada par (q_bad_A, q_good_A) de domínio A,
  emparelha com q_good_B de domínio B (B semanticamente distinto de A)
  → novo exemplo negativo: q_bad_A não é respondido por q_good_B.

Critério de distinção de domínio:
  - Overlap de palavras entre nomes de domínio < DOMAIN_SIM_THRESH
  - Evita pares dentro do mesmo cluster (ex: "molecular biology" vs
    "developmental biology" têm overlap alto → descartados)

Diversidade:
  - Máximo MAX_PER_QBAD amostras por q_bad (evita dominar o dataset)
  - Máximo MAX_PER_PAIR por combinação de domínios

Saída: data/pairs/adversarial_probes_crossdomain.jsonl
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

LAYER2_PATH  = Path("data/pairs/pairs_layer2.jsonl")
OUTPUT_PATH  = Path("data/pairs/adversarial_probes_crossdomain.jsonl")

N_TARGET         = 200   # exemplos a gerar
SEED             = 123
DOMAIN_SIM_THRESH = 0.20  # overlap Jaccard máximo entre nomes de domínio
MAX_PER_QBAD     = 3      # máx amostras por q_bad (garante diversidade)
MAX_PER_COMBO    = 2      # máx amostras por par de domínios (A, B)


# ---------------------------------------------------------------------------
# Similaridade entre nomes de domínio
# ---------------------------------------------------------------------------

def _dom_words(domain: str) -> set[str]:
    stop = {"of", "the", "a", "and", "in", "/", "or"}
    return {w.lower() for w in re.split(r"[\s/,]+", domain) if w.lower() not in stop and len(w) > 2}


def domain_jaccard(d1: str, d2: str) -> float:
    w1, w2 = _dom_words(d1), _dom_words(d2)
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------

def gerar_augmentacao(
    layer2_path: Path = LAYER2_PATH,
    output_path: Path = OUTPUT_PATH,
    n_target: int = N_TARGET,
    seed: int = SEED,
) -> list[dict]:
    random.seed(seed)

    # Carrega pares com domínio
    pares = [
        json.loads(l)
        for l in layer2_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    pares = [p for p in pares if p.get("q_bad") and p.get("q_good") and p.get("domain", "").strip()]

    # Agrupa por domínio
    por_dominio: dict[str, list[dict]] = defaultdict(list)
    for p in pares:
        por_dominio[p["domain"].strip()].append(p)

    dominios = list(por_dominio.keys())
    print(f"Pares com domínio   : {len(pares)}")
    print(f"Domínios distintos  : {len(dominios)}")

    # Pré-computa pares de domínios suficientemente diferentes
    doms_distintos: list[tuple[str, str]] = []
    for i, d1 in enumerate(dominios):
        for d2 in dominios[i + 1:]:
            if domain_jaccard(d1, d2) < DOMAIN_SIM_THRESH:
                doms_distintos.append((d1, d2))

    random.shuffle(doms_distintos)
    print(f"Pares de domínio elegíveis (Jaccard < {DOMAIN_SIM_THRESH}): {len(doms_distintos)}")

    # Gera candidatos cross-domain
    novos: list[dict] = []
    usos_qbad: dict[str, int] = defaultdict(int)
    usos_combo: dict[tuple[str, str], int] = defaultdict(int)

    # Itera em ordem aleatória para distribuir bem
    candidatos: list[tuple[dict, dict]] = []
    for d1, d2 in doms_distintos:
        for par_a in por_dominio[d1]:
            for par_b in por_dominio[d2]:
                # Nunca emparelha par com ele mesmo
                if par_a["source_id"] != par_b["source_id"]:
                    candidatos.append((par_a, par_b))
                    candidatos.append((par_b, par_a))  # direção inversa também

    random.shuffle(candidatos)

    for par_a, par_b in candidatos:
        if len(novos) >= n_target:
            break

        qbad_id = par_a["source_id"]
        combo   = (par_a["domain"].strip()[:20], par_b["domain"].strip()[:20])

        if usos_qbad[qbad_id] >= MAX_PER_QBAD:
            continue
        if usos_combo[combo] >= MAX_PER_COMBO:
            continue

        novo = {
            "q_bad":       par_a["q_bad"],
            "q_good_fake": par_b["q_good"],        # q_good real de outro domínio
            "q_good_real": par_a["q_good"],         # referência (não usado no treino)
            "probe_type":  "desconectada_cross",    # subtipo de augmentação
            "why_fake":    f"Cross-domain: q_bad from '{par_a['domain'][:40]}', "
                           f"q_cand from '{par_b['domain'][:40]}' — answers a different problem.",
            "domain":      par_a["domain"],
            "source":      par_a.get("source", ""),
            "source_id":   f"cross_{par_a['source_id']}_{par_b['source_id']}",
            "year":        par_a.get("year", 0),
            "label":       0,
        }
        novos.append(novo)
        usos_qbad[qbad_id] += 1
        usos_combo[combo] += 1

    # Salva
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in novos:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return novos


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------

def imprimir_resumo(novos: list[dict]) -> None:
    from collections import Counter
    dominios_bad  = Counter(n["domain"][:40] for n in novos)
    print(f"\n{'='*55}")
    print(f"  Novos exemplos desconectada_cross: {len(novos)}")
    print(f"\n  Top 10 domínios de q_bad:")
    for d, n in dominios_bad.most_common(10):
        print(f"    {d:<45} {n:>3}")

    print(f"\n  Amostra (3 exemplos):")
    for item in novos[:3]:
        print(f"\n  q_bad  : {item['q_bad'][:75]}")
        print(f"  q_cand : {item['q_good_fake'][:75]}")
        print(f"  why    : {item['why_fake'][:90]}")
    print(f"{'='*55}")


if __name__ == "__main__":
    print("=== Augmentação Cross-Domain ===\n")
    novos = gerar_augmentacao()
    imprimir_resumo(novos)
    print(f"\nSalvos em: {OUTPUT_PATH}")
