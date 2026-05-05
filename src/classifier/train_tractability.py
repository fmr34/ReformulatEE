"""
Treina o classificador local de Tratabilidade.

Estratégia:
  1. Carrega as 100 questões curated (50 alta EE / 50 baixa EE)
  2. [--api] Obtém scores reais via Claude API (usa cache SQLite automaticamente)
  3. [padrão] Usa labels binárias como proxy: 0 → 0.05, 1 → 0.80
  4. Treina Ridge regression sobre embeddings all-MiniLM-L6-v2
  5. Salva em data/models/tractability/classifier.pkl

Modos:
  python -m src.classifier.train_tractability          # labels binárias (rápido, ~30s)
  python -m src.classifier.train_tractability --api    # scores reais via API (~5 min, mais preciso)
  python -m src.classifier.train_tractability --eval   # avalia modelo já treinado
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Força UTF-8 no stdout do Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def _get_scores_from_cache(questions: list[str]) -> dict[str, float]:
    """Puxa scores já calculados do cache SQLite (gratuito)."""
    try:
        from src.db.historico import get_trat_cache
        result = {}
        for q in questions:
            cached = get_trat_cache(q)
            if cached:
                result[q] = cached["prob_tractable"] * cached["confidence"]
        return result
    except Exception:
        return {}


def _get_scores_from_api(questions: list[str]) -> dict[str, float]:
    """Gera scores via Claude API — usa cache automaticamente."""
    from src.ee.tratabilidade import tratabilidade
    result = {}
    for i, q in enumerate(questions, 1):
        print(f"  [{i:3d}/{len(questions)}] {q[:70]}")
        r = tratabilidade(q)
        result[q] = r["prob_tractable"] * r["confidence"]
    return result


def _evaluate(questions: list[str], targets: list[float]) -> None:
    """Avalia o modelo treinado contra os targets."""
    from src.classifier.tractability_local import predict_local
    import numpy as np

    preds  = [predict_local(q)["prob_tractable"] for q in questions]
    errors = [abs(p - t) for p, t in zip(preds, targets)]
    mae    = np.mean(errors)

    # Acurácia binária (threshold 0.4)
    correct = sum(
        (p >= 0.4) == (t >= 0.4)
        for p, t in zip(preds, targets)
    )
    acc = correct / len(questions)

    print(f"\n  Avaliação sobre {len(questions)} exemplos:")
    print(f"  MAE       : {mae:.4f}")
    print(f"  Acurácia  : {acc:.1%}  (threshold 0.40)")
    print()

    # Casos com maior erro
    top_erros = sorted(zip(errors, questions, targets, preds), reverse=True)[:5]
    print("  Top-5 maiores erros:")
    for err, q, t, p in top_erros:
        print(f"  [{err:.3f}] target={t:.2f} pred={p:.2f} | {q[:65]}")


def main(use_api: bool = False, only_eval: bool = False) -> None:
    from src.eval.curated import get_curated
    from src.classifier.tractability_local import is_trained

    print("=" * 65)
    print("  Treino — Classificador Local de Tratabilidade")
    print("=" * 65)

    curated   = get_curated()
    questions = [q.text  for q in curated]
    labels    = [q.label for q in curated]

    print(f"\n  {len(curated)} questões carregadas  "
          f"({sum(labels)} alta EE / {len(labels)-sum(labels)} baixa EE)")

    # ── Modo avaliação ──────────────────────────────────────────────
    if only_eval:
        if not is_trained():
            print("  ⚠ Modelo não treinado. Execute sem --eval primeiro.")
            return
        targets = [0.05 if l == 0 else 0.80 for l in labels]
        _evaluate(questions, targets)
        return

    # ── Determina scores alvo ───────────────────────────────────────
    if use_api:
        print("\n  Consultando cache SQLite...")
        cached = _get_scores_from_cache(questions)
        print(f"  {len(cached)}/{len(questions)} já em cache.")

        pending = [q for q in questions if q not in cached]
        if pending:
            print(f"  Chamando API para {len(pending)} questões pendentes...")
            api_scores = _get_scores_from_api(pending)
            cached.update(api_scores)

        target_scores = [
            cached.get(q, 0.05 if labels[i] == 0 else 0.80)
            for i, q in enumerate(questions)
        ]
        print(f"  {len(cached)}/{len(questions)} scores obtidos.")
    else:
        target_scores = [0.05 if l == 0 else 0.80 for l in labels]
        print("\n  Modo: labels binarias  (0=0.05, 1=0.80)")
        print("  Dica: adicione --api para scores reais (mais precisos).")

    # ── Treina ──────────────────────────────────────────────────────
    print()
    from src.classifier.tractability_local import train_and_save
    metrics = train_and_save(questions, target_scores, alpha=50.0)

    # ── Resultados ──────────────────────────────────────────────────
    print("\n  ─── Métricas de treino ────────────────────────────────")
    print(f"  Exemplos     : {metrics['n_train']}")
    print(f"  R²           : {metrics['r2']:.4f}")
    print(f"  RMSE         : {metrics['rmse']:.4f}")
    if metrics['cv_rmse_mean'] is not None:
        print(f"  RMSE CV 5-fold: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
    print(f"  Modelo salvo : {metrics['model_path']}")
    print("  ───────────────────────────────────────────────────────")

    # ── Teste rápido ────────────────────────────────────────────────
    print("\n  Teste rápido (4 questões):")
    from src.classifier.tractability_local import predict_local
    test_cases = [
        ("What are the molecular mechanisms of CRISPR-Cas9 DNA repair?",  "→ ALTO"),
        ("What is the essence of life?",                                   "→ BAIXO"),
        ("How does transformer architecture scaling affect performance?",  "→ ALTO"),
        ("What is the true nature of consciousness?",                      "→ BAIXO"),
    ]
    for q, expected in test_cases:
        r     = predict_local(q)
        score = r["prob_tractable"]
        label = "✅" if (score >= 0.4) == (expected == "→ ALTO") else "⚠️"
        print(f"  {label} [{score:.3f}] {expected} | {q[:60]}")

    # Avaliação completa
    _evaluate(questions, target_scores)

    print("  ✅ tratabilidade.py usará o modelo local automaticamente.")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        use_api   = "--api"  in args,
        only_eval = "--eval" in args,
    )
