"""
Traducao local usando Helsinki-NLP MarianMT — substitui chamadas Claude API.

Modelos (~300 MB cada, baixados automaticamente na primeira chamada):
  pt→en : Helsinki-NLP/opus-mt-tc-big-pt-en
  en→pt : Helsinki-NLP/opus-mt-en-ROMANCE  (prefixo >>pt<< obrigatorio)

Uso:
  from src.ee.translate_local import translate
  en = translate("O que causa o envelhecimento?", "pt_to_en")
  pt = translate("What causes aging?", "en_to_pt")

Requisitos:
  pip install transformers sentencepiece sacremoses
"""

from __future__ import annotations

import os
from pathlib import Path

# Diretório de cache local para os modelos (evita re-download)
_MODEL_CACHE = Path(os.getenv("HF_HOME", "data/models/hf_cache"))

# Nomes dos modelos Helsinki-NLP
_PT_EN_MODEL = "Helsinki-NLP/opus-mt-tc-big-pt-en"
_EN_PT_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"  # usa prefixo >>pt<<

# Pipelines lazy-loaded
_pipe_pt_en = None
_pipe_en_pt = None


def _load_pipeline(direction: str):
    """Carrega pipeline MarianMT com cache local."""
    from transformers import pipeline

    _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    cache_dir = str(_MODEL_CACHE)

    if direction == "pt_to_en":
        global _pipe_pt_en
        if _pipe_pt_en is None:
            print(f"  [translate_local] Carregando {_PT_EN_MODEL}...")
            _pipe_pt_en = pipeline(
                "translation",
                model=_PT_EN_MODEL,
                cache_dir=cache_dir,
                device=-1,  # CPU
            )
        return _pipe_pt_en
    else:  # en_to_pt
        global _pipe_en_pt
        if _pipe_en_pt is None:
            print(f"  [translate_local] Carregando {_EN_PT_MODEL}...")
            _pipe_en_pt = pipeline(
                "translation",
                model=_EN_PT_MODEL,
                cache_dir=cache_dir,
                device=-1,  # CPU
            )
        return _pipe_en_pt


def translate(text: str, direction: str) -> str:
    """
    Traduz texto localmente via MarianMT (sem API).

    Args:
        text:      Texto a traduzir
        direction: 'pt_to_en' | 'en_to_pt'

    Returns:
        Texto traduzido

    Raises:
        ImportError: se transformers/sentencepiece nao estiverem instalados
        ValueError:  se direction for invalido
    """
    if direction not in ("pt_to_en", "en_to_pt"):
        raise ValueError(f"direction invalido: {direction!r}. Use 'pt_to_en' ou 'en_to_pt'.")

    pipe = _load_pipeline(direction)

    if direction == "en_to_pt":
        # opus-mt-en-ROMANCE requer prefixo de lingua alvo
        input_text = f">>pt<< {text}"
    else:
        input_text = text

    result = pipe(input_text, max_length=512)
    return result[0]["translation_text"].strip()


def is_available() -> bool:
    """Retorna True se transformers e sentencepiece estiverem instalados."""
    try:
        import sentencepiece  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False
