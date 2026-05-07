"""
Patch A — paradigm_classifier

Classifica pares (q_bad, q_candidato) como melhoria genuina (label=1)
ou probe adversarial / pergunta inocua (label=0).

Dataset de treino:
  - pairs_layer2.jsonl        : (q_bad, q_good)      -> 163 positivos (label=1)
  - adversarial_probes        : (q_bad, q_good_fake)  -> 210 negativos (label=0)
  - adversarial_probes_cross  : cross-domain          -> 200 negativos (label=0)
  Total: 573 pares

Layout de X por par (1185 colunas):
  [0:32]     estruturais (32): len, spec, op, meas, hedg, nomin, wh, meth (x bad+cand+delta) + sim + jaccard
  [32:416]   emb_bad  (384): embedding sentence-transformer do q_bad
  [416:800]  emb_cand (384): embedding sentence-transformer do q_cand
  [800:1184] emb_diff (384): emb_cand - emb_bad
  [1184]     cross_enc (1): score do cross-encoder ms-marco (relevancia q_bad -> q_cand)

Pipeline de treino:
  ColumnTransformer:
    struct + cross_enc -> StandardScaler
    emb_bad, emb_cand, emb_diff -> StandardScaler + PCA(40)
  HistGradientBoostingClassifier(class_weight='balanced')

Gate: accuracy > 0.85 e kappa > 0.70
Saida: data/models/paradigm_classifier.pkl
"""

from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Wordlists para features estruturais
# ---------------------------------------------------------------------------

_SPECULATIVE = {
    "consciousness",
    "awareness",
    "qualia",
    "phenomenal",
    "subjective",
    "essence",
    "nature",
    "meaning",
    "reality",
    "existence",
    "being",
    "ontological",
    "ontology",
    "metaphysical",
    "metaphysics",
    "noumenal",
    "teleological",
    "teleology",
    "intrinsic",
    "ineffable",
    "transcendent",
    "transcendence",
    "ultimate",
    "fundamental",
    "emergent",
    "emergence",
    "substrate",
    "irreducible",
    "irreducibility",
    "holistic",
    "holism",
    "vitalism",
    "panpsychism",
    "epiphenomenal",
    "epiphenomenalism",
    "supervenience",
    "grounding",
    "instantiation",
    "potentiality",
    "actuality",
    "telos",
    "logos",
    "ontogenetic",
    "morphogenetic",
}

_OPERATIONAL = {
    "measure",
    "measures",
    "measured",
    "measuring",
    "measurement",
    "test",
    "tests",
    "tested",
    "testing",
    "experiment",
    "experimental",
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "correlate",
    "correlation",
    "predict",
    "prediction",
    "predictive",
    "quantify",
    "quantification",
    "identify",
    "isolate",
    "control",
    "replicate",
    "simulate",
    "simulation",
    "model",
    "models",
    "optimize",
    "detect",
    "estimate",
    "calculate",
    "compute",
    "observe",
    "observation",
    "classify",
    "validate",
    "validation",
    "calibrate",
    "calibration",
    "statistically",
    "empirically",
    "operationalize",
    "operationalized",
    "protocol",
    "methodology",
    "randomized",
    "controlled",
    "blinded",
    "sequence",
    "genome",
    "gene",
    "protein",
    "pathway",
    "mechanism",
    "circuit",
    "neural",
    "behavioral",
    "cognitive",
    "physiological",
    "specific",
    "particular",
    "defined",
    "characterized",
}

_MEASUREMENT = {
    "rate",
    "frequency",
    "level",
    "concentration",
    "correlation",
    "coefficient",
    "proportion",
    "ratio",
    "percentage",
    "threshold",
    "range",
    "scale",
    "index",
    "score",
    "metric",
    "magnitude",
    "intensity",
    "duration",
    "latency",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "variance",
    "deviation",
    "gradient",
    "density",
    "flux",
    "potential",
    "resistance",
    "temperature",
    "velocity",
    "mass",
    "volume",
    "charge",
}

_HEDGING = {
    "might",
    "could",
    "possibly",
    "perhaps",
    "presumably",
    "arguably",
    "allegedly",
    "seemingly",
    "apparently",
    "conceivably",
    "hypothetically",
    "theoretically",
    "speculatively",
    "putatively",
}

_WH_WORDS = {
    "what": 0,
    "how": 1,
    "why": 2,
    "which": 3,
    "whether": 4,
    "when": 5,
    "where": 6,
    "who": 7,
    "whom": 8,
    "whose": 9,
}

_NOMINALIZATIONS = re.compile(
    r"\b\w+(?:tion|ity|ness|ism|ence|ance|ment|hood|ship|ics)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Constantes de layout
# ---------------------------------------------------------------------------

N_STRUCT = 10  # features estruturais por questao
N_STRUCTURAL_TOTAL = N_STRUCT * 3 + 2  # 32: bad + cand + delta + sim + jaccard
EMB_DIM = 384  # all-MiniLM-L6-v2
N_CROSS_ENC = 1  # score do cross-encoder
# Total colunas brutas: 32 + 384*3 + 1 = 1185
N_FEATURES_RAW = N_STRUCTURAL_TOTAL + EMB_DIM * 3 + N_CROSS_ENC


# ---------------------------------------------------------------------------
# Extratores de features
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _structural_features(q: str) -> list[float]:
    words = _words(q)
    wset = set(words)
    n = max(len(words), 1)
    wh = _WH_WORDS.get(words[0] if words else "", -1)
    return [
        len(words),
        sum(1 for w in words if w in _SPECULATIVE) / n,
        sum(1 for w in words if w in _OPERATIONAL) / n,
        sum(1 for w in words if w in _MEASUREMENT) / n,
        sum(1 for w in words if w in _HEDGING) / n,
        len(_NOMINALIZATIONS.findall(q)) / n,
        float(wh),
        float(any(w in wset for w in {"protocol", "methodology", "randomized", "blinded"})),
        float(sum(1 for w in words if w in _SPECULATIVE)),
        float(sum(1 for w in words if w in _OPERATIONAL)),
    ]


def _jaccard(q1: str, q2: str) -> float:
    stop = {
        "is",
        "are",
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "do",
        "does",
        "did",
        "can",
        "could",
        "how",
        "what",
        "why",
        "which",
        "when",
        "where",
        "who",
        "that",
        "this",
        "these",
        "those",
        "and",
        "or",
        "but",
        "for",
        "with",
        "between",
        "among",
    }
    w1 = {w for w in _words(q1) if len(w) > 3 and w not in stop}
    w2 = {w for w in _words(q2) if len(w) > 3 and w not in stop}
    if not w1 and not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def build_structural_matrix(
    q_bads: list[str],
    q_cands: list[str],
    emb_bad: np.ndarray,
    emb_cand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (X_struct, emb_bad, emb_cand, emb_diff).
    X_struct shape: (n, N_STRUCTURAL_TOTAL)
    """
    sims = (emb_bad * emb_cand).sum(axis=1)  # cosine (embeddings normalizados)
    rows = []
    for q_bad, q_cand, sim in zip(q_bads, q_cands, sims):
        fb = _structural_features(q_bad)
        fc = _structural_features(q_cand)
        delta = [c - b for b, c in zip(fb, fc)]
        rows.append(fb + fc + delta + [float(sim), _jaccard(q_bad, q_cand)])
    return (
        np.array(rows, dtype=float),
        emb_bad,
        emb_cand,
        emb_cand - emb_bad,
    )


# ---------------------------------------------------------------------------
# Wrapper do modelo treinado
# ---------------------------------------------------------------------------


class ParadigmClassifier:
    """Classificador de pares (q_bad, q_candidato)."""

    CROSS_ENC_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path or "data/models/paradigm_classifier.pkl")
        self._pipeline = None
        self._embedder = None
        self._cross_encoder = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.EMBED_MODEL)
        return self._embedder

    def _load_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.CROSS_ENC_MODEL)
        return self._cross_encoder

    def load(self) -> "ParadigmClassifier":
        self._pipeline = joblib.load(self.model_path)
        return self

    # ------------------------------------------------------------------
    # Computacao de features
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return self._load_embedder().encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )

    def _cross_enc_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Retorna array (n, 1) com scores do cross-encoder."""
        scores = self._load_cross_encoder().predict(pairs)
        return scores.reshape(-1, 1).astype(float)

    def build_features(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """
        Extrai X de shape (n, N_FEATURES_RAW = 1185).

        Layout:
          [0:32]     estruturais
          [32:416]   emb_bad
          [416:800]  emb_cand
          [800:1184] emb_diff
          [1184]     cross_encoder score
        """
        q_bads = [p[0] for p in pairs]
        q_cands = [p[1] for p in pairs]

        all_embs = self._embed_batch(q_bads + q_cands)
        emb_bad = all_embs[: len(q_bads)]
        emb_cand = all_embs[len(q_bads) :]

        X_struct, X_eb, X_ec, X_ed = build_structural_matrix(q_bads, q_cands, emb_bad, emb_cand)
        X_ce = self._cross_enc_batch(pairs)  # (n, 1)

        return np.hstack([X_struct, X_eb, X_ec, X_ed, X_ce])

    # ------------------------------------------------------------------
    # Predicao (requer load() antes)
    # ------------------------------------------------------------------

    def predict(self, q_bad: str, q_cand: str) -> int:
        X = self.build_features([(q_bad, q_cand)])
        return int(self._pipeline.predict(X)[0])

    def predict_proba(self, q_bad: str, q_cand: str) -> float:
        """Retorna P(label=1) — probabilidade de melhoria genuina."""
        X = self.build_features([(q_bad, q_cand)])
        return float(self._pipeline.predict_proba(X)[0, 1])
