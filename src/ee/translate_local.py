"""
Traducao local usando Helsinki-NLP MarianMT — substitui chamadas Claude API.

Modelos (~300 MB cada, baixados automaticamente na primeira chamada):
  pt→en : Helsinki-NLP/opus-mt-ROMANCE-en  (prefixo >>pt<< opcional)
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
_PT_EN_MODEL = "Helsinki-NLP/opus-mt-ROMANCE-en"
_EN_PT_MODEL = "Helsinki-NLP/opus-mt-en-ROMANCE"  # usa prefixo >>pt<<

# Modelos lazy-loaded (tokenizer + model)
_models: dict = {}


def _load_model(model_name: str):
    """Carrega MarianMTModel + tokenizer com cache local."""
    if model_name in _models:
        return _models[model_name]

    from transformers import MarianMTModel
    from transformers import MarianTokenizer

    _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    cache_dir = str(_MODEL_CACHE)

    print(f"  [translate_local] Carregando {model_name}...")
    tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
    _models[model_name] = (tokenizer, model)
    return tokenizer, model


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

    if direction == "en_to_pt":
        model_name = _EN_PT_MODEL
        input_text = f">>pt<< {text}"  # opus-mt-en-ROMANCE requer prefixo
    else:
        model_name = _PT_EN_MODEL
        input_text = text

    tokenizer, model = _load_model(model_name)
    inputs = tokenizer(
        [input_text], return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def is_available() -> bool:
    """Retorna True se transformers e sentencepiece estiverem instalados."""
    try:
        import sentencepiece  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False
