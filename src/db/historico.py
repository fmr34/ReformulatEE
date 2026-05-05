"""
Banco de dados SQLite para histórico de reformulações.

Tabelas:
  historico          — cada pergunta processada + resultado + feedback
  cache_tratabilidade — cache persistente das chamadas à Claude API
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone

_DB = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'historico.db')
)


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB), exist_ok=True)
    return sqlite3.connect(_DB)


def init_db() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS historico (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            TEXT    NOT NULL,
                idioma        TEXT    NOT NULL,
                pergunta_orig TEXT    NOT NULL,
                pergunta_en   TEXT,
                candidatos    TEXT    NOT NULL,
                melhor        TEXT    NOT NULL,
                melhor_pt     TEXT,
                ee_antes      REAL    NOT NULL,
                ee_depois     REAL    NOT NULL,
                stage1_pass   INTEGER NOT NULL,
                feedback      INTEGER DEFAULT NULL   -- 1=positivo, -1=negativo
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS cache_tratabilidade (
                hash          TEXT PRIMARY KEY,
                query         TEXT NOT NULL,
                resultado     TEXT NOT NULL,          -- JSON
                ts            TEXT NOT NULL
            )
        """)


# ---------------------------------------------------------------------------
# Histórico
# ---------------------------------------------------------------------------

def salvar(
    idioma: str,
    pergunta_orig: str,
    pergunta_en: str | None,
    candidatos: list[dict],
    melhor: str,
    melhor_pt: str | None,
    ee_antes: float,
    ee_depois: float,
    stage1_pass: bool,
) -> int | None:
    """Persiste um resultado. Retorna o ID do registro inserido."""
    try:
        with _conn() as c:
            cur = c.execute("""
                INSERT INTO historico
                  (ts, idioma, pergunta_orig, pergunta_en, candidatos,
                   melhor, melhor_pt, ee_antes, ee_depois, stage1_pass)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                idioma,
                pergunta_orig,
                pergunta_en,
                json.dumps(candidatos, ensure_ascii=False),
                melhor,
                melhor_pt,
                float(ee_antes),
                float(ee_depois),
                int(stage1_pass),
            ))
            return cur.lastrowid
    except Exception as e:
        print(f"[historico] Erro ao salvar: {e}")
        return None


def registrar_feedback(record_id: int, valor: int) -> None:
    """Registra feedback do usuário: valor = 1 (👍) ou -1 (👎)."""
    try:
        with _conn() as c:
            c.execute(
                "UPDATE historico SET feedback = ? WHERE id = ?",
                (valor, record_id)
            )
    except Exception as e:
        print(f"[historico] Erro ao salvar feedback: {e}")


def ultimas(n: int = 8) -> list[list[str]]:
    """Retorna as últimas n perguntas como [[pergunta_orig, idioma], ...]."""
    try:
        with _conn() as c:
            rows = c.execute("""
                SELECT pergunta_orig, idioma
                FROM historico
                ORDER BY id DESC
                LIMIT ?
            """, (n,)).fetchall()
        return [list(r) for r in rows]
    except Exception:
        return []


def todas() -> list[dict]:
    """Retorna todos os registros completos para análise/exportação."""
    try:
        with _conn() as c:
            rows = c.execute("""
                SELECT id, ts, idioma, pergunta_orig, pergunta_en,
                       candidatos, melhor, melhor_pt,
                       ee_antes, ee_depois, stage1_pass, feedback
                FROM historico
                ORDER BY id DESC
            """).fetchall()
        cols = ['id', 'ts', 'idioma', 'pergunta_orig', 'pergunta_en',
                'candidatos', 'melhor', 'melhor_pt',
                'ee_antes', 'ee_depois', 'stage1_pass', 'feedback']
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            d['candidatos'] = json.loads(d['candidatos'])
            result.append(d)
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Cache de Tratabilidade (cross-session)
# ---------------------------------------------------------------------------

def _hash_query(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def get_trat_cache(query: str) -> dict | None:
    """Retorna resultado cached de tratabilidade, ou None se não encontrado."""
    try:
        h = _hash_query(query)
        with _conn() as c:
            row = c.execute(
                "SELECT resultado FROM cache_tratabilidade WHERE hash = ?", (h,)
            ).fetchone()
        if row:
            return json.loads(row[0])
        return None
    except Exception:
        return None


def set_trat_cache(query: str, resultado: dict) -> None:
    """Persiste resultado de tratabilidade no cache."""
    try:
        h = _hash_query(query)
        with _conn() as c:
            c.execute("""
                INSERT OR REPLACE INTO cache_tratabilidade (hash, query, resultado, ts)
                VALUES (?, ?, ?, ?)
            """, (
                h, query,
                json.dumps(resultado, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ))
    except Exception as e:
        print(f"[historico] Erro ao salvar cache tratabilidade: {e}")
