"""
Pipeline de avaliacao da Fase 0.

Gate: AUC > 0.70 sobre as 100 questoes curadas.

Nota sobre nao_trivialidade: eh uma metrica relacional (Q vs Q0) e nao faz
sentido em avaliacao standalone de questoes individuais. O score aqui usa
apenas respondibilidade + tratabilidade, que sao propriedades intrinsecas de Q.
nao_trivialidade entra apenas no loop RLHF quando temos pares (Q0, Q*).

Uso:
  python -m src.eval.evaluate [--no-tratabilidade] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

load_dotenv(override=True)

from src.corpus.fetch import fetch_corpus
from src.corpus.index import CorpusIndex
from src.corpus.index import build_index
from src.ee.respondibilidade import respondibilidade
from src.ee.tratabilidade import tratabilidade
from src.eval.curated import CuratedQuestion
from src.eval.curated import get_curated

# Pesos normalizados para avaliacao standalone (sem nao_trivialidade)
# B1 + B2 = 1.0
_B1 = 0.5  # respondibilidade
_B2 = 0.5  # tratabilidade


def score_question(
    q: CuratedQuestion,
    index: CorpusIndex,
    use_tratabilidade: bool = True,
) -> dict:
    resp = respondibilidade(q.text, index, top_k=10)

    if use_tratabilidade:
        tract_out = tratabilidade(q.text)
        tract = tract_out["prob_tractable"] * tract_out["confidence"]
        trajectory = tract_out.get("trajectory", "")
    else:
        tract = 0.5  # valor neutro quando API indisponivel
        trajectory = "unknown"

    # Score standalone: apenas respondibilidade + tratabilidade
    # nao_trivialidade entra apenas no loop RLHF com pares (Q0, Q*)
    ee = _B1 * resp + _B2 * tract

    return {
        "text": q.text,
        "label": q.label,
        "domain": q.domain,
        "respondibilidade": round(resp, 4),
        "tratabilidade": round(tract, 4),
        "trajectory": trajectory,
        "ee": round(ee, 4),
    }


def evaluate(
    corpus_dir: Path,
    use_tratabilidade: bool = True,
    output_path: Path | None = None,
    delay_between_calls: float = 0.3,
) -> dict:
    questions = get_curated()
    fetch_corpus(corpus_dir)
    index = build_index(corpus_dir)

    results = []
    for q in tqdm(questions, desc="Avaliando questoes"):
        r = score_question(q, index, use_tratabilidade=use_tratabilidade)
        results.append(r)
        if use_tratabilidade:
            time.sleep(delay_between_calls)

    labels = np.array([r["label"] for r in results])
    scores = np.array([r["ee"] for r in results])

    auc = roc_auc_score(labels, scores)
    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)
    report = classification_report(
        labels, preds, target_names=["baixa_EE", "alta_EE"], output_dict=True
    )

    summary = {
        "auc": round(auc, 4),
        "gate_passed": auc >= 0.70,
        "threshold_used": round(threshold, 4),
        "n_questions": len(questions),
        "classification_report": report,
        "per_question": results,
    }

    print(f"\n{'='*50}")
    print(f"  AUC = {auc:.4f}  |  Gate (>= 0.70): {'PASSOU' if auc >= 0.70 else 'FALHOU'}")
    print(f"{'='*50}")
    print(f"  Threshold (mediana): {threshold:.4f}")
    print(f"  Precision alta_EE:   {report['alta_EE']['precision']:.3f}")
    print(f"  Recall    alta_EE:   {report['alta_EE']['recall']:.3f}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n  Resultados salvos em: {output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fase 0 — avaliacao de EE(Q)")
    parser.add_argument(
        "--no-tratabilidade",
        action="store_true",
        help="Desabilita chamadas API (usa valor neutro 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/fase0_eval.json",
        help="Caminho para salvar resultados JSON",
    )
    args = parser.parse_args()

    corpus_dir = Path(os.getenv("CORPUS_DIR", "data/corpus"))
    output_path = Path(args.output)

    evaluate(
        corpus_dir=corpus_dir,
        use_tratabilidade=not args.no_tratabilidade,
        output_path=output_path,
    )
