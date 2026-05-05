"""
Classificador local de Tratabilidade.

Arquitetura: sentence-transformers/all-MiniLM-L6-v2 → Ridge regression (numpy puro).
Substitui chamadas à Claude API para pontuação de tratabilidade:
  - Latência: ~5ms vs ~600ms da API
  - Custo: R$0,00 por chamada
  - Performance esperada: ~85% de concordância com a API

Para treinar:
  python -m src.classifier.train_tractability          # usa labels binárias (rápido)
  python -m src.classifier.train_tractability --api    # gera scores reais via API (mais preciso)
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

_MODEL_PATH = Path(os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'models', 'tractability', 'classifier.pkl')
))

# Lazy-load do modelo em memória
_classifier = None


class _RidgeClassifier:
    """Regressão Ridge com padronização de features — numpy puro, sem sklearn."""

    def __init__(self,
                 weights:   np.ndarray,
                 bias:      float,
                 feat_mean: np.ndarray,
                 feat_std:  np.ndarray):
        self.weights   = weights
        self.bias      = bias
        self.feat_mean = feat_mean
        self.feat_std  = feat_std

    def predict_single(self, embedding: np.ndarray) -> float:
        x = (embedding - self.feat_mean) / (self.feat_std + 1e-8)
        score = float(x @ self.weights + self.bias)
        return max(0.0, min(1.0, score))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> '_RidgeClassifier':
        with open(path, 'rb') as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def is_trained() -> bool:
    """Retorna True se o modelo local já foi treinado e está disponível."""
    return _MODEL_PATH.exists()


def _get_classifier() -> _RidgeClassifier | None:
    global _classifier
    if _classifier is None and _MODEL_PATH.exists():
        _classifier = _RidgeClassifier.load(_MODEL_PATH)
    return _classifier


def predict_local(query: str) -> dict:
    """
    Prediz tratabilidade de uma questão de pesquisa localmente.
    Retorna dict compatível com tratabilidade() (API Claude).

    Raises RuntimeError se o modelo não estiver treinado.
    """
    clf = _get_classifier()
    if clf is None:
        raise RuntimeError(
            "Modelo local de tratabilidade não encontrado. "
            "Execute: python -m src.classifier.train_tractability"
        )

    from src.ee.nao_trivialidade import embed
    emb = embed(query)
    score = clf.predict_single(emb)

    if score >= 0.65:
        trajectory = "rising"
    elif score >= 0.35:
        trajectory = "plateau"
    elif score >= 0.15:
        trajectory = "declining"
    else:
        trajectory = "absent"

    return {
        "prob_tractable": score,
        "trajectory":     trajectory,
        "confidence":     1.0,   # score já inclui calibração; não divide por confidence
        "nearest_resolved_question": "",
    }


def train_and_save(questions: list[str],
                   scores:    list[float],
                   alpha:     float = 1.0) -> dict:
    """
    Treina o classificador Ridge e persiste o modelo.

    Args:
        questions: lista de perguntas de pesquisa
        scores:    scores alvo de tratabilidade (prob_tractable * confidence)
        alpha:     regularização L2

    Returns:
        dict com métricas (r2, rmse, cv_rmse_mean, cv_rmse_std, model_path)
    """
    from src.ee.nao_trivialidade import embed

    print(f"  Gerando embeddings para {len(questions)} questões...")
    X = np.array([embed(q) for q in questions], dtype=np.float64)
    y = np.array(scores, dtype=np.float64)

    # Padronização feature-wise
    feat_mean = X.mean(axis=0)
    feat_std  = X.std(axis=0)
    X_norm = (X - feat_mean) / (feat_std + 1e-8)

    # Ridge: w = (X'X + αI)^{-1} X'y
    n_feat = X_norm.shape[1]
    A = X_norm.T @ X_norm + alpha * np.eye(n_feat)
    b = X_norm.T @ y
    weights = np.linalg.solve(A, b)
    bias    = float(y.mean() - (X_norm @ weights).mean())

    # Métricas de treino
    y_pred = np.clip(X_norm @ weights + bias, 0.0, 1.0)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2   = 1.0 - ss_res / (ss_tot + 1e-10)
    rmse = float(np.sqrt(ss_res / len(y)))

    # Cross-validation 5-fold
    n = len(y)
    fold_size = max(n // 5, 1)
    cv_rmse = []
    for fold in range(5):
        val_idx   = list(range(fold * fold_size, min((fold + 1) * fold_size, n)))
        train_idx = [i for i in range(n) if i not in val_idx]
        if not train_idx or not val_idx:
            continue
        Xtr, ytr   = X_norm[train_idx], y[train_idx]
        Xval, yval = X_norm[val_idx],   y[val_idx]
        A_cv = Xtr.T @ Xtr + alpha * np.eye(n_feat)
        w_cv = np.linalg.solve(A_cv, Xtr.T @ ytr)
        b_cv = float(ytr.mean() - (Xtr @ w_cv).mean())
        pred = np.clip(Xval @ w_cv + b_cv, 0.0, 1.0)
        cv_rmse.append(float(np.sqrt(np.mean((yval - pred) ** 2))))

    # Salva
    clf = _RidgeClassifier(
        weights   = weights.astype(np.float32),
        bias      = bias,
        feat_mean = feat_mean.astype(np.float32),
        feat_std  = feat_std.astype(np.float32),
    )
    clf.save(_MODEL_PATH)

    # Invalida cache em memória
    global _classifier
    _classifier = None

    return {
        "n_train":      len(y),
        "r2":           round(r2, 4),
        "rmse":         round(rmse, 4),
        "cv_rmse_mean": round(float(np.mean(cv_rmse)), 4) if cv_rmse else None,
        "cv_rmse_std":  round(float(np.std(cv_rmse)),  4) if cv_rmse else None,
        "model_path":   str(_MODEL_PATH),
    }
