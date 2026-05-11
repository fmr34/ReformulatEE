"""
Consolida todas as fontes de pares DPO em um único dataset de treino.

Fontes (em ordem de prioridade/qualidade):
  1. data/rl/dpo_tier2.jsonl   — pares validados com adversarial probes
  2. data/rl/dpo_tier3.jsonl   — pares adversariais cross-domain
  3. data/rl/dpo_tier1.jsonl   — pares base (curated)
  4. data/rl/batch_pairs.jsonl — pares gerados via Claude Batch API
  5. historico.db              — feedback positivo (👍) do usuário

Saída:
  data/rl/dpo_final.jsonl   — dataset unificado, deduplicado, pronto para Colab

Uso:
  python -m src.dataset.prepare_dpo
  python -m src.dataset.prepare_dpo --stats   # só mostra estatísticas
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Forca UTF-8 no stdout do Windows
import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

_RL_DIR = Path("data/rl")
_OUT = _RL_DIR / "dpo_final.jsonl"

_SYSTEM_PROMPT = (
    "You are an expert in philosophy of science. "
    "Reformulate the following research question to make it more epistemically tractable: "
    "operationalizable, methodologically grounded, and answerable with existing tools.\n\n"
)

_TIER_ORDER = ["dpo_tier3.jsonl", "dpo_tier2.jsonl", "dpo_tier1.jsonl", "dpo_dataset.jsonl"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prompt(q_bad: str) -> str:
    return f"{_SYSTEM_PROMPT}" f"Original question: {q_bad}\n\n" "Reformulated question:"


def _prompt_key(prompt: str) -> str:
    """Chave de deduplicação: só a pergunta original (após 'Original question:')."""
    marker = "Original question:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].split("\n")[0].strip().lower()
    return prompt.strip().lower()


# ---------------------------------------------------------------------------
# Carregadores por fonte
# ---------------------------------------------------------------------------


def _load_existing_jsonl(path: Path) -> list[dict]:
    """Carrega JSONL com formato DPO padrão (prompt/chosen/rejected)."""
    pairs = []
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "prompt" in row and "chosen" in row and "rejected" in row:
                    pairs.append(row)
            except json.JSONDecodeError:
                pass
    return pairs


def _load_batch_pairs(path: Path) -> list[dict]:
    """
    Carrega batch_pairs.jsonl (formato: q_bad / q_good / source / domain)
    e converte para formato DPO.
    """
    pairs = []
    if not path.exists():
        return pairs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                q_bad = row.get("q_bad", "").strip()
                q_good = row.get("q_good", "").strip()
                if not q_bad or not q_good or q_bad == q_good:
                    continue
                pairs.append(
                    {
                        "prompt": _make_prompt(q_bad),
                        "chosen": q_good,
                        "rejected": q_bad,
                        "source": row.get("source", "batch"),
                        "domain": row.get("domain", ""),
                    }
                )
            except json.JSONDecodeError:
                pass
    return pairs


def _load_hf_logs_pairs() -> list[dict]:
    """
    Lê logs.jsonl do HF Dataset repo e extrai pares com feedback=1 (👍).

    Formato dos registros:
      {"type": "record", "id": 123, ...campos..., "feedback": null}
      {"type": "feedback", "id": 123, "feedback": 1, "ts": "..."}

    Merge: aplica eventos de feedback sobre os registros correspondentes.
    """
    import os
    import urllib.request

    tok = os.getenv("HF_TOKEN")
    repo = os.getenv("HF_LOG_REPO", "fmr34/reformulatee-logs")
    if not tok:
        return []

    url = f"https://huggingface.co/datasets/{repo}/resolve/main/logs.jsonl"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    except Exception as e:
        print(f"  [hf_logs] Não foi possível baixar logs: {e}")
        return []

    records: dict[int, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue

        if row.get("type") == "record":
            rid = row.get("id")
            if rid is not None:
                records[rid] = row
        elif row.get("type") == "feedback":
            rid = row.get("id")
            if rid is not None and rid in records:
                records[rid]["feedback"] = row.get("feedback")

    pairs = []
    for r in records.values():
        if r.get("feedback") != 1:
            continue
        q_bad = (r.get("pergunta_en") or r.get("pergunta_orig", "")).strip()
        chosen = r.get("melhor", "").strip()
        cands = r.get("candidatos") or []

        if not q_bad or not chosen:
            continue

        if cands:
            worst = min(cands, key=lambda c: c.get("ee", 1.0))
            rejected = worst.get("text", "").strip()
        else:
            rejected = q_bad

        if not rejected or rejected == chosen:
            rejected = q_bad
        if chosen == rejected:
            continue

        pairs.append(
            {
                "prompt": _make_prompt(q_bad),
                "chosen": chosen,
                "rejected": rejected,
                "source": "hf_logs_feedback",
                "domain": "",
            }
        )
    return pairs


def _load_feedback_pairs() -> list[dict]:
    """
    Extrai pares de treino do histórico: registros com feedback=1 (👍).
    chosen  = melhor reformulação selecionada pelo sistema
    rejected = candidato com menor EE do mesmo registro
    """
    pairs = []
    try:
        from src.db.historico import todas

        registros = todas()
    except Exception as e:
        print(f"  [aviso] Não foi possível carregar historico.db: {e}")
        return pairs

    for r in registros:
        if r.get("feedback") != 1:
            continue
        q_bad = (r.get("pergunta_en") or r.get("pergunta_orig", "")).strip()
        chosen = r.get("melhor", "").strip()
        cands = r.get("candidatos") or []

        if not q_bad or not chosen:
            continue

        # Escolhe o candidato com menor EE como rejected
        if cands:
            worst = min(cands, key=lambda c: c.get("ee", 1.0))
            rejected = worst.get("text", "").strip()
        else:
            rejected = q_bad

        if not rejected or rejected == chosen:
            rejected = q_bad

        if chosen == rejected:
            continue

        pairs.append(
            {
                "prompt": _make_prompt(q_bad),
                "chosen": chosen,
                "rejected": rejected,
                "source": "feedback_positive",
                "domain": "",
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def prepare(stats_only: bool = False) -> None:
    print("=" * 65)
    print("  Preparação do Dataset DPO Final")
    print("=" * 65)

    all_pairs: list[dict] = []
    source_counts: dict[str, int] = {}

    # 1. Tiers (maior qualidade primeiro — tiers adversariais)
    for fname in _TIER_ORDER:
        path = _RL_DIR / fname
        loaded = _load_existing_jsonl(path)
        if loaded:
            tag = fname.replace(".jsonl", "")
            source_counts[tag] = len(loaded)
            all_pairs.extend(loaded)
            print(f"  {tag:<25} {len(loaded):>4} pares")

    # 2. Batch API pairs (curated questions → variações)
    batch_path = _RL_DIR / "batch_pairs.jsonl"
    batch = _load_batch_pairs(batch_path)
    if batch:
        source_counts["batch_pairs"] = len(batch)
        all_pairs.extend(batch)
        print(f"  {'batch_pairs':<25} {len(batch):>4} pares")
    else:
        print(f"  {'batch_pairs':<25}    0 pares  (execute expand_via_batch primeiro)")

    # 3. Batch domínios (novos domínios sub-representados)
    domains_path = _RL_DIR / "batch_domains.jsonl"
    domains = _load_batch_pairs(domains_path)
    if domains:
        source_counts["batch_domains"] = len(domains)
        all_pairs.extend(domains)
        print(f"  {'batch_domains':<25} {len(domains):>4} pares")
    else:
        print(f"  {'batch_domains':<25}    0 pares  (execute expand_domains primeiro)")

    # 4. Batch large (novos domínios + arquétipos de questão-ruim)
    large_path = _RL_DIR / "batch_large.jsonl"
    large = _load_batch_pairs(large_path)
    if large:
        source_counts["batch_large"] = len(large)
        all_pairs.extend(large)
        print(f"  {'batch_large':<25} {len(large):>4} pares")
    else:
        print(f"  {'batch_large':<25}    0 pares  (execute expand_large primeiro)")

    # 5. Feedback positivo do usuário (SQLite local)
    feedback = _load_feedback_pairs()
    if feedback:
        source_counts["feedback_positive"] = len(feedback)
        all_pairs.extend(feedback)
        print(f"  {'feedback_positive':<25} {len(feedback):>4} pares")

    # 6. Feedback do HF Dataset (logs do Space online)
    hf_feedback = _load_hf_logs_pairs()
    if hf_feedback:
        source_counts["hf_logs_feedback"] = len(hf_feedback)
        all_pairs.extend(hf_feedback)
        print(f"  {'hf_logs_feedback':<25} {len(hf_feedback):>4} pares")
    else:
        print(f"  {'hf_logs_feedback':<25}    0 pares  (sem HF_TOKEN ou repo vazio)")

    print(f"\n  Total bruto : {len(all_pairs)} pares")

    # Deduplicação por pergunta original
    seen: set[str] = set()
    deduped: list[dict] = []
    for pair in all_pairs:
        key = _prompt_key(pair["prompt"])
        if key not in seen:
            seen.add(key)
            deduped.append(pair)

    removed = len(all_pairs) - len(deduped)
    print(f"  Duplicatas  : {removed} removidas")
    print(f"  Total final : {len(deduped)} pares únicos")

    if stats_only:
        print("\n  (modo --stats: arquivo não salvo)")
        return

    # Salva
    _RL_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        for pair in deduped:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  ✅ Salvo em: {_OUT}")

    # Distribuição por domínio (top-5)
    domains: dict[str, int] = {}
    for p in deduped:
        d = p.get("domain") or "unknown"
        domains[d] = domains.get(d, 0) + 1
    top = sorted(domains.items(), key=lambda x: -x[1])[:6]
    print("\n  Top domínios:")
    for dom, cnt in top:
        print(f"    {dom or 'unknown':<30} {cnt:>4}")

    print()


if __name__ == "__main__":
    stats_only = "--stats" in sys.argv
    prepare(stats_only=stats_only)
