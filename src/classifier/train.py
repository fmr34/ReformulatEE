"""
Patch A — Treinamento do paradigm_classifier

Fluxo:
  1. Carrega pairs_layer2.jsonl  -> pares positivos (label=1)
  2. Carrega adversarial_probes  -> pares negativos (label=0)
  3. Extrai features estruturais + embedding difference (384 dims)
  4. Pipeline: ColumnTransformer(struct->StandardScaler,
                                 emb_diff->StandardScaler+PCA(50))
             + GradientBoostingClassifier
  5. StratifiedKFold(5): accuracy + Cohen kappa
  6. Gate: accuracy > 0.85 e kappa > 0.70
  7. Treina modelo final + salva em data/models/

Uso:
  .venv\\Scripts\\python -m src.classifier.train
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.classifier.paradigm_classifier import N_STRUCTURAL_TOTAL
from src.classifier.paradigm_classifier import ParadigmClassifier

# ---------------------------------------------------------------------------
# Caminhos e parametros
# ---------------------------------------------------------------------------

LAYER2_PATH = Path("data/pairs/pairs_layer2.jsonl")
PROBES_PATH = Path("data/pairs/adversarial_probes.jsonl")
CROSSDOMAIN_PATH = Path("data/pairs/adversarial_probes_crossdomain.jsonl")
MODEL_PATH = Path("data/models/paradigm_classifier.pkl")

ACCURACY_GATE = 0.85
KAPPA_GATE = 0.70
PCA_DIMS = 40  # dimensoes para compressao de cada bloco de embedding

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def pr(text: str) -> None:
    """Print seguro para terminais Windows (cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def carregar_pares() -> tuple[list[tuple[str, str]], list[int], list[str]]:
    pares: list[tuple[str, str]] = []
    labels: list[int] = []
    probe_types: list[str] = []

    for linha in LAYER2_PATH.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        rec = json.loads(linha)
        if rec.get("q_bad") and rec.get("q_good"):
            pares.append((rec["q_bad"], rec["q_good"]))
            labels.append(1)
            probe_types.append("")

    for path in (PROBES_PATH, CROSSDOMAIN_PATH):
        if not path.exists():
            continue
        for linha in path.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            rec = json.loads(linha)
            if rec.get("q_bad") and rec.get("q_good_fake"):
                pares.append((rec["q_bad"], rec["q_good_fake"]))
                labels.append(0)
                probe_types.append(rec.get("probe_type", ""))

    return pares, labels, probe_types


# ---------------------------------------------------------------------------
# Construcao do pipeline sklearn
# ---------------------------------------------------------------------------


def build_pipeline(n_total_features: int, pca_dims: int) -> Pipeline:
    """
    Layout de colunas esperado (N_FEATURES_RAW = 1185):
      [0:32]     estruturais + cross_enc ao final -> StandardScaler
      [32:416]   emb_bad  -> StandardScaler + PCA(pca_dims)
      [416:800]  emb_cand -> StandardScaler + PCA(pca_dims)
      [800:1184] emb_diff -> StandardScaler + PCA(pca_dims)
      [1184]     cross_encoder score -> incluido no bloco struct (StandardScaler)

    Features apos PCA: N_STRUCTURAL_TOTAL + 1 (cross_enc) + pca_dims*3
    """
    from src.classifier.paradigm_classifier import EMB_DIM
    from src.classifier.paradigm_classifier import N_STRUCTURAL_TOTAL

    s = N_STRUCTURAL_TOTAL
    emb_size = EMB_DIM
    ce_idx = s + 3 * emb_size  # indice 1184

    # Estruturais + cross_encoder no mesmo bloco (ambos passam por StandardScaler)
    struct_idx = list(range(s)) + [ce_idx]
    emb_bad_idx = list(range(s, s + emb_size))
    emb_cand_idx = list(range(s + emb_size, s + 2 * emb_size))
    emb_diff_idx = list(range(s + 2 * emb_size, s + 3 * emb_size))

    preprocessor = ColumnTransformer(
        [
            ("struct", StandardScaler(), struct_idx),
            (
                "emb_bad",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(pca_dims, random_state=42))]),
                emb_bad_idx,
            ),
            (
                "emb_cand",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(pca_dims, random_state=42))]),
                emb_cand_idx,
            ),
            (
                "emb_diff",
                Pipeline([("sc", StandardScaler()), ("pca", PCA(pca_dims, random_state=42))]),
                emb_diff_idx,
            ),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=400,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_leaf=5,
                    l2_regularization=0.1,
                    class_weight="balanced",  # corrige desbalanceamento 163 pos / 410 neg
                    random_state=42,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Treino principal
# ---------------------------------------------------------------------------


def treinar() -> None:
    pr("=" * 55)
    pr("  Patch A - paradigm_classifier")
    pr("=" * 55)

    # ------------------------------------------------------------------
    # 1. Dados
    # ------------------------------------------------------------------
    pr("\n[1/5] Carregando dados...")
    pares, labels, probe_types = carregar_pares()
    y = np.array(labels)

    contagem = Counter(labels)
    pr(f"  Positivos (label=1): {contagem[1]}")
    pr(f"  Negativos (label=0): {contagem[0]}")
    pr(f"  Total:               {len(pares)}")

    # ------------------------------------------------------------------
    # 2. Features
    # ------------------------------------------------------------------
    pr("\n[2/5] Extraindo features + embeddings + cross-encoder...")
    pr("  (carregando all-MiniLM-L6-v2 + cross-encoder/ms-marco-MiniLM-L-6-v2...)")

    clf_obj = ParadigmClassifier()
    t0 = time.time()
    X = clf_obj.build_features(pares)
    elapsed = time.time() - t0

    from src.classifier.paradigm_classifier import N_CROSS_ENC
    from src.classifier.paradigm_classifier import N_FEATURES_RAW

    pr(f"  Shape de X : {X.shape}  (esperado {N_FEATURES_RAW})")
    pr(f"  Estruturais: {N_STRUCTURAL_TOTAL} | Emb 3x384 | Cross-enc: {N_CROSS_ENC}")
    pr(f"  PCA: 3x 384 -> {PCA_DIMS} | Features pos-PCA: {N_STRUCTURAL_TOTAL + 1 + PCA_DIMS * 3}")
    pr(f"  Tempo      : {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 3. Pipeline
    # ------------------------------------------------------------------
    pr("\n[3/5] Configurando pipeline ColumnTransformer + GradientBoosting...")
    pipeline = build_pipeline(X.shape[1], PCA_DIMS)

    # ------------------------------------------------------------------
    # 4. Validacao cruzada
    # ------------------------------------------------------------------
    pr("\n[4/5] Validacao cruzada StratifiedKFold(5)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=1)

    acc_global = float((y_pred == y).mean())
    kappa_global = float(cohen_kappa_score(y, y_pred))

    pr(f"\n  {'Metrica':<25} {'Global':>8}")
    pr(f"  {'-'*35}")
    pr(f"  {'Accuracy':<25} {acc_global:>8.4f}")
    pr(f"  {'Cohen kappa':<25} {kappa_global:>8.4f}")

    # Por tipo de probe
    pr("\n  Accuracy por tipo de probe (negativos):")
    for tipo in ["parafrase", "desconectada", "desconectada_cross", "especulativa"]:
        idxs = [i for i, pt in enumerate(probe_types) if pt == tipo]
        if not idxs:
            continue
        acc_t = float((y_pred[idxs] == y[idxs]).mean())
        pr(f"    {tipo:<15}: {acc_t:.4f}  ({len(idxs)} amostras)")

    # Subconjunto adversarial (probes + positivos)
    idx_adv = [i for i, pt in enumerate(probe_types) if pt != ""]
    idx_pos = [i for i, l in enumerate(labels) if l == 1]
    idx_eval = sorted(set(idx_adv + idx_pos))
    y_eval = y[idx_eval]
    yp_eval = y_pred[idx_eval]
    acc_adv = float((yp_eval == y_eval).mean())
    kappa_adv = float(cohen_kappa_score(y_eval, yp_eval))

    pr(f"\n  Subconjunto adversarial ({len(idx_eval)} amostras):")
    pr(f"    Accuracy : {acc_adv:.4f}  (gate >{ACCURACY_GATE})")
    pr(f"    Kappa    : {kappa_adv:.4f}  (gate >{KAPPA_GATE})")

    pr("\n  Relatorio completo:")
    report = classification_report(
        y,
        y_pred,
        target_names=["intratavel/fake", "melhoria_genuina"],
        digits=4,
    )
    pr(report)

    # ------------------------------------------------------------------
    # 5. Gate e salvamento
    # ------------------------------------------------------------------
    gate_ok = acc_adv > ACCURACY_GATE and kappa_adv > KAPPA_GATE

    pr("=" * 55)
    if gate_ok:
        pr(
            f"  GATE PASSOU  (acc={acc_adv:.3f} > {ACCURACY_GATE}, "
            f"kappa={kappa_adv:.3f} > {KAPPA_GATE})"
        )
    else:
        pr(f"  GATE FALHOU  (acc={acc_adv:.3f}, kappa={kappa_adv:.3f})")
        pr("     -> modelo NAO salvo. Revisar features ou dataset.")
        return

    pr("\n  Treinando modelo final em todos os dados...")
    pipeline.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    pr(f"  Modelo salvo em: {MODEL_PATH}")
    pr("=" * 55)

    # Metadados
    meta = {
        "acc_global": round(acc_global, 6),
        "kappa_global": round(kappa_global, 6),
        "acc_adversarial": round(acc_adv, 6),
        "kappa_adversarial": round(kappa_adv, 6),
        "n_train": len(pares),
        "n_positivos": int(contagem[1]),
        "n_negativos": int(contagem[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_pos_pca": int(N_STRUCTURAL_TOTAL + 1 + PCA_DIMS * 3),
        "gate_passou": gate_ok,
        "model_path": str(MODEL_PATH),
    }
    meta_path = MODEL_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pr(f"  Metadados em  : {meta_path}")


if __name__ == "__main__":
    treinar()
