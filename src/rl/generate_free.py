"""
Backends de geração gratuitos para substituir a Claude API.

Hierarquia de backends (tentados em ordem):
  1. hf_inference  — HuggingFace Inference API (gratuita, ideal para HF Spaces)
  2. claude        — Claude API (fallback, requer ANTHROPIC_API_KEY)

Backend GGUF local (llama-cpp-python) é documentado como opcional:
  → Requer Windows Long Path habilitado (admin) + pip install llama-cpp-python
  → Configurar: INFERENCE_BACKEND=gguf no .env
  → Ver README para instruções de instalação no Intel Arc (SYCL)

Configuração via .env:
  INFERENCE_BACKEND   = hf_inference | gguf | claude   (default: auto)
  HF_MODEL            = Qwen/Qwen2.5-1.5B-Instruct     (modelo para HF Inference)
  HF_TOKEN            = hf_...                          (opcional, aumenta rate limit)
  GGUF_MODEL_REPO     = Qwen/Qwen2.5-1.5B-Instruct-GGUF
  GGUF_MODEL_FILE     = qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SYSTEM = (
    "You are an expert in philosophy of science. "
    "Your task is to reformulate research questions to make them more epistemically tractable: "
    "operationalizable, methodologically grounded, and answerable with existing tools. "
    "Respond with ONLY the reformulated question — no explanation, no preamble."
)

# Modelo padrão para HF Inference API
_HF_MODEL   = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
_HF_TOKEN   = os.getenv("HF_TOKEN")

# Modelo padrão para GGUF local
_GGUF_REPO  = os.getenv("GGUF_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
_GGUF_FILE  = os.getenv("GGUF_MODEL_FILE",  "qwen2.5-1.5b-instruct-q4_k_m.gguf")
_GGUF_PATH  = Path(os.getenv("GGUF_LOCAL_PATH",
                              "data/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"))

_gguf_model = None   # lazy-load


# ---------------------------------------------------------------------------
# HF Inference API (primário — ideal para HF Spaces e zero-cost)
# ---------------------------------------------------------------------------

def _hf_single_call(q_bad: str) -> str:
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=_HF_MODEL, token=_HF_TOKEN)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": f"Original question: {q_bad}"},
        ],
        max_tokens=100,
        temperature=1.1,
    )
    return resp.choices[0].message.content.strip().split("\n")[0].strip()


def generate_hf_inference(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos via HuggingFace Inference API em paralelo.
    Gratuita para modelos públicos; não requer download local.
    """
    candidates = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_hf_single_call, q_bad) for _ in range(n)]
        for f in as_completed(futures):
            try:
                text = f.result()
                if text:
                    candidates.append(text)
            except Exception as e:
                print(f"  [hf_inference] aviso: {e}")
    return candidates


# ---------------------------------------------------------------------------
# GGUF local (opcional — requer llama-cpp-python instalado)
# ---------------------------------------------------------------------------

def _gguf_available() -> bool:
    """True se llama-cpp-python está instalado e o modelo existe localmente."""
    try:
        import llama_cpp  # noqa: F401
        return _GGUF_PATH.exists()
    except ImportError:
        return False


def _load_gguf():
    global _gguf_model
    if _gguf_model is not None:
        return _gguf_model
    from llama_cpp import Llama
    n_threads = min(os.cpu_count() or 4, 8)
    print(f"  Carregando modelo GGUF: {_GGUF_PATH} ({n_threads} threads)...")
    _gguf_model = Llama(
        model_path=str(_GGUF_PATH),
        n_ctx=1024,
        n_threads=n_threads,
        verbose=False,
    )
    return _gguf_model


def download_gguf_model() -> Path:
    """Baixa o modelo GGUF do HuggingFace Hub se não existir localmente."""
    if _GGUF_PATH.exists():
        print(f"  Modelo já existe: {_GGUF_PATH}")
        return _GGUF_PATH
    from huggingface_hub import hf_hub_download
    _GGUF_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Baixando {_GGUF_FILE} de {_GGUF_REPO}...")
    path = hf_hub_download(
        repo_id=_GGUF_REPO,
        filename=_GGUF_FILE,
        local_dir=str(_GGUF_PATH.parent),
    )
    print(f"  Modelo salvo em: {path}")
    return Path(path)


def generate_gguf(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos usando modelo GGUF local (sequencial — thread-unsafe).
    Requer llama-cpp-python e modelo baixado via download_gguf_model().
    """
    model = _load_gguf()
    candidates = []
    for i in range(n):
        try:
            out = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": f"Original question: {q_bad}"},
                ],
                max_tokens=100,
                temperature=1.1,
                seed=i * 42,
            )
            text = out["choices"][0]["message"]["content"].strip().split("\n")[0].strip()
            if text:
                candidates.append(text)
        except Exception as e:
            print(f"  [gguf] aviso chamada {i}: {e}")
    return candidates


# ---------------------------------------------------------------------------
# Interface unificada com fallback automático
# ---------------------------------------------------------------------------

def _detect_backend() -> str:
    """Auto-detecta o melhor backend disponível."""
    explicit = os.getenv("INFERENCE_BACKEND", "").lower()
    if explicit in ("hf_inference", "gguf", "claude", "local"):
        return explicit

    # HF Spaces detectado
    if os.getenv("SPACE_ID"):
        return "hf_inference"

    # Modelo GGUF local disponível
    if _gguf_available():
        return "gguf"

    # HF Inference API (padrão zero-cost)
    return "hf_inference"


def generate(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos usando o melhor backend disponível.
    Fallback automático: GGUF → HF Inference → Claude.
    """
    backend = _detect_backend()

    if backend == "gguf":
        try:
            return generate_gguf(q_bad, n)
        except Exception as e:
            print(f"  [gguf] falhou ({e}), usando HF Inference...")
            backend = "hf_inference"

    if backend == "hf_inference":
        try:
            return generate_hf_inference(q_bad, n)
        except Exception as e:
            print(f"  [hf_inference] falhou ({e}), usando Claude...")
            backend = "claude"

    # Fallback final: Claude API
    if os.getenv("ANTHROPIC_API_KEY"):
        from src.rl.inference import _generate_claude
        return _generate_claude(q_bad, n)

    return []
