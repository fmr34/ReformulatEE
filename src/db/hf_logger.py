"""
Logger assíncrono para HuggingFace Dataset.

Persiste perguntas e feedbacks em <HF_LOG_REPO>/logs.jsonl (dataset privado).
Usa fila em memória + thread de background para não bloquear a UI do Space.

Fluxo:
  salvar()             → log_async()   — enfileira, flush a cada 10 itens ou 2 min
  registrar_feedback() → log_urgent()  — enfileira + flush imediato
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.request

_REPO_ID = os.getenv("HF_LOG_REPO", "fmr34/reformulatee-logs")
_FILE = "logs.jsonl"
_FLUSH_BATCH = 10
_FLUSH_INTERVAL = 120  # segundos

_q: queue.Queue[dict] = queue.Queue()
_lock = threading.Lock()
_started = False


# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------


def _token() -> str | None:
    return os.getenv("HF_TOKEN")


def _get_api():
    tok = _token()
    if not tok:
        return None
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        api.create_repo(_REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        return api
    except Exception as e:
        print(f"[hf_logger] init error: {e}")
        return None


def _download_lines() -> list[str]:
    """Baixa o conteúdo atual de logs.jsonl via HTTP (evita cache local)."""
    tok = _token()
    if not tok:
        return []
    url = f"https://huggingface.co/datasets/{_REPO_ID}/resolve/main/{_FILE}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8").splitlines(keepends=True)
    except Exception:
        return []


def _push(api, records: list[dict]) -> bool:
    """Baixa arquivo atual, adiciona novos registros e faz upload."""
    with _lock:
        try:
            existing = _download_lines()
            new_lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
            content = ("".join(existing + new_lines)).encode("utf-8")
            api.upload_file(
                path_or_fileobj=content,
                path_in_repo=_FILE,
                repo_id=_REPO_ID,
                repo_type="dataset",
                commit_message=f"log: +{len(records)}",
            )
            return True
        except Exception as e:
            print(f"[hf_logger] push error: {e}")
            return False


def _worker():
    pending: list[dict] = []
    last_flush = time.monotonic()
    api = None

    while True:
        try:
            record = _q.get(timeout=5)
            if record.get("__flush__"):
                if pending:
                    if api is None:
                        api = _get_api()
                    if api and _push(api, pending):
                        pending = []
                        last_flush = time.monotonic()
            else:
                pending.append(record)
            _q.task_done()
        except queue.Empty:
            pass

        now = time.monotonic()
        if pending and (len(pending) >= _FLUSH_BATCH or now - last_flush >= _FLUSH_INTERVAL):
            if api is None:
                api = _get_api()
            if api and _push(api, pending):
                pending = []
                last_flush = now


def _ensure_started():
    global _started
    if not _started and _token():
        threading.Thread(target=_worker, daemon=True).start()
        _started = True


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------


def log_async(record: dict) -> None:
    """Enfileira registro para upload em batch (não bloqueia)."""
    if not _token():
        return
    _ensure_started()
    _q.put(record)


def log_urgent(record: dict) -> None:
    """Enfileira registro e dispara flush imediato (para feedback)."""
    if not _token():
        return
    _ensure_started()
    _q.put(record)
    _q.put({"__flush__": True})


def ultimas_hf(n: int = 8) -> list[list[str]]:
    """Retorna as últimas n perguntas do HF Dataset (fallback cross-session)."""
    lines = _download_lines()
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("type") == "record":
                records.append(row)
        except json.JSONDecodeError:
            continue
    records.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return [[r.get("pergunta_orig", ""), r.get("idioma", "Português")] for r in records[:n]]
