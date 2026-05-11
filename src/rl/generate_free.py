"""
Backends de geração gratuitos para substituir a Claude API.

Hierarquia de backends (tentados em ordem):
  1. ollama       — Ollama local (ideal para uso local com modelo fine-tuned)
  2. hf_inference — HuggingFace Inference API (gratuita, ideal para HF Spaces)
  3. gguf         — llama-cpp-python local (opcional)
  4. claude       — Claude API (fallback final, requer ANTHROPIC_API_KEY)

Configuração via .env:
  INFERENCE_BACKEND   = ollama | hf_inference | gguf | claude   (default: auto)
  OLLAMA_MODEL        = reformulatee        (nome do modelo no Ollama)
  OLLAMA_BASE_URL     = http://localhost:11434  (padrão)
  HF_MODEL            = Qwen/Qwen2.5-1.5B-Instruct
  HF_TOKEN            = hf_...
  GGUF_MODEL_REPO     = Qwen/Qwen2.5-1.5B-Instruct-GGUF
  GGUF_MODEL_FILE     = qwen2.5-1.5b-instruct-q4_k_m.gguf
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

_SYSTEM = (
    "You are a research question reformulator. "
    "Rules: (1) output exactly ONE sentence; (2) it must end with '?'; "
    "(3) no preamble, no labels, no numbering, no explanation — only the question itself. "
    "Bad output: 'Here is the reformulation: ...' "
    "Good output: 'How does X affect Y under condition Z?'"
)

# Modelo padrão para HF Inference API
_HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")  # base model (serverless)
_HF_TOKEN = os.getenv("HF_TOKEN")

# Ollama
_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "reformulatee")
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Modelo padrão para GGUF local
_GGUF_REPO = os.getenv("GGUF_MODEL_REPO", "Qwen/Qwen2.5-1.5B-Instruct-GGUF")
_GGUF_FILE = os.getenv("GGUF_MODEL_FILE", "qwen2.5-1.5b-instruct-q4_k_m.gguf")
_GGUF_PATH = Path(
    os.getenv("GGUF_LOCAL_PATH", "data/models/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf")
)

_gguf_model = None  # lazy-load

# Prefixos que o modelo às vezes adiciona antes da pergunta
_PREFIXES = (
    "rephrased question:",
    "reformulated question:",
    "reformulation:",
    "question:",
    "rephrased:",
    "reformulated:",
    "new question:",
    "answer:",
    "answered question:",
    "rewritten answer based on the given prompt:",
    "rewritten question:",
)

# Palavras que indicam que uma frase é uma pergunta
_QUESTION_STARTERS = (
    "what",
    "how",
    "why",
    "when",
    "where",
    "who",
    "which",
    "does",
    "do",
    "is",
    "are",
    "can",
    "could",
    "would",
    "will",
    "has",
    "have",
    "did",
    "should",
    "might",
    "may",
    "was",
    "were",
    "in what",
    "to what",
    "under what",
    "at what",
    "for what",
)


def _clean_output(raw: str) -> str:
    """
    Extrai apenas a pergunta reformulada de uma resposta bruta do modelo.

    Estratégia:
      1. Remove prefixos conhecidos ("Rephrased question:", etc.)
      2. Procura a primeira frase que termina em '?' — descarta o resto
      3. Se o texto não contiver '?', retorna string vazia (candidato descartado)
    """
    import re

    text = raw.strip()

    # 1. Remove prefixo comum (pode aparecer antes da pergunta real)
    lower = text.lower()
    for prefix in _PREFIXES:
        if lower.startswith(prefix):
            text = text[len(prefix) :].strip()
            lower = text.lower()
            break

    # 2. Tenta encontrar uma frase que termina em '?'
    # Divide em sentenças usando pontuação como separador
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        sent = sent.strip()
        if sent.endswith("?") and len(sent) >= 15:
            return sent

    # 3. Extrai tudo até o primeiro '?' (cobre casos sem espaço após ponto)
    idx = text.find("?")
    if idx != -1:
        candidate = text[: idx + 1].strip()
        if len(candidate) >= 15:
            return candidate

    # 4. Sem '?' encontrado → descarta (não é uma pergunta)
    return ""


# ---------------------------------------------------------------------------
# Ollama (primário local — usa modelo fine-tuned via API OpenAI-compatible)
# ---------------------------------------------------------------------------


def _ollama_available() -> bool:
    """True se Ollama está rodando e o modelo está carregado."""
    import urllib.request

    try:
        url = f"{_OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=2) as resp:
            import json

            data = json.loads(resp.read())
            model = os.getenv("OLLAMA_MODEL", _OLLAMA_MODEL)
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            return model in models
    except Exception:
        return False


def _ollama_single_call(q_bad: str, seed: int = 0) -> str:
    import urllib.request
    import json

    model = os.getenv("OLLAMA_MODEL", _OLLAMA_MODEL)
    base_url = os.getenv("OLLAMA_BASE_URL", _OLLAMA_BASE_URL)
    print(f"  [ollama] modelo={model}")

    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"Reformulate into one tractable research question: {q_bad}",
                },
                {"role": "assistant", "content": ""},
            ],
            "stream": False,
            "options": {
                "temperature": 1.0,
                "num_predict": 80,
                "seed": seed,
                "stop": ["\n", "\n\n"],
            },
        }
    ).encode()

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except TimeoutError:
        print(f"  [ollama] timeout após 30s — modelo={os.getenv('OLLAMA_MODEL', _OLLAMA_MODEL)}")
        return ""
    return _clean_output(data["message"]["content"])


def generate_ollama(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos via Ollama local em paralelo.
    Requer: ollama rodando + modelo registrado (ollama create reformulatee -f Modelfile).
    """
    candidates = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_ollama_single_call, q_bad, i) for i in range(n)]
        for f in as_completed(futures):
            try:
                text = f.result()
                if text:
                    candidates.append(text)
            except Exception as e:
                print(f"  [ollama] ERRO: {e}")
    return candidates


# ---------------------------------------------------------------------------
# HF Inference API (primário remoto — ideal para HF Spaces e zero-cost)
# ---------------------------------------------------------------------------


def _hf_single_call(q_bad: str) -> str:
    from huggingface_hub import InferenceClient

    model = os.getenv("HF_MODEL", _HF_MODEL)
    token = os.getenv("HF_TOKEN", _HF_TOKEN)
    print(f"  [hf_inference] modelo={model} token={'***' if token else 'None'}")
    client = InferenceClient(model=model, token=token)
    resp = client.chat_completion(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Original question: {q_bad}"},
        ],
        max_tokens=100,
        temperature=1.1,
    )
    return _clean_output(resp.choices[0].message.content)


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
                print(f"  [hf_inference] ERRO: {e}")
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
                    {"role": "user", "content": f"Original question: {q_bad}"},
                ],
                max_tokens=100,
                temperature=1.1,
                seed=i * 42,
            )
            text = _clean_output(out["choices"][0]["message"]["content"])
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
    if explicit in ("ollama", "hf_inference", "gguf", "claude", "local"):
        return explicit

    # HF Spaces detectado — sempre usa HF Inference API
    if os.getenv("SPACE_ID"):
        return "hf_inference"

    # Ollama rodando localmente com modelo carregado
    if _ollama_available():
        return "ollama"

    # Modelo GGUF local disponível
    if _gguf_available():
        return "gguf"

    # HF Inference API (padrão zero-cost)
    return "hf_inference"


def generate(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos usando o melhor backend disponível.
    Fallback automático: Ollama → GGUF → HF Inference → Claude.
    """
    backend = _detect_backend()

    if backend == "ollama":
        try:
            result = generate_ollama(q_bad, n)
            if result:
                return result
            print("  [ollama] sem candidatos, tentando HF Inference...")
            backend = "hf_inference"
        except Exception as e:
            print(f"  [ollama] falhou ({e}), tentando HF Inference...")
            backend = "hf_inference"

    if backend == "gguf":
        try:
            return generate_gguf(q_bad, n)
        except Exception as e:
            print(f"  [gguf] falhou ({e}), usando HF Inference...")
            backend = "hf_inference"

    if backend == "hf_inference":
        try:
            result = generate_hf_inference(q_bad, n)
            if result:
                return result
            print("  [hf_inference] sem candidatos, usando Claude como fallback...")
            backend = "claude"
        except Exception as e:
            print(f"  [hf_inference] falhou ({e}), usando Claude...")
            backend = "claude"

    # Fallback final: Claude API
    if os.getenv("ANTHROPIC_API_KEY"):
        from src.rl.inference import _generate_claude

        return _generate_claude(q_bad, n)

    return []
