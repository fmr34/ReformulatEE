"""
Expansão de dataset via Claude Batch API (50% mais barato que chamadas individuais).

Gera pares (q_bad, q_good) para treino DPO a partir das questões curated:
  - Tipo A: q_good → versão vaga/intratável (q_bad gerada)
  - Tipo B: q_bad  → versão tratável (q_good gerada)

Fluxo assíncrono:
  1. python -m src.dataset.expand_via_batch --submit        # cria o batch, imprime ID
  2. (aguardar ~5-30 min)
  3. python -m src.dataset.expand_via_batch --retrieve ID   # baixa resultados e salva

Resultado salvo em: data/rl/batch_pairs.jsonl
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

load_dotenv(override=True)

import anthropic

_OUT_PATH = Path("data/rl/batch_pairs.jsonl")
_BATCH_ID_FILE = Path(".claude/batch_id.txt")

_SYSTEM_A = (
    "Make this research question vaguer, more philosophical, and harder to study empirically. "
    "The result should be ontologically broad, lacking a clear methodology, and resistant "
    "to empirical investigation. Respond with ONLY the vaguer question — no explanation."
)

_SYSTEM_B = (
    "Make this research question more specific, empirically testable, and methodologically tractable. "
    "Operationalize it: identify a measurable phenomenon, a methodology, and a context. "
    "Respond with ONLY the improved question — no explanation."
)


def _build_requests(curated) -> list[dict]:
    """Constrói lista de requisições para o Batch API."""
    requests = []

    for i, q in enumerate(curated):
        if q.label == 1:
            # Tipo A: boa → ruim
            requests.append(
                {
                    "custom_id": f"A_{i}",
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 120,
                        "system": _SYSTEM_A,
                        "messages": [{"role": "user", "content": q.text}],
                    },
                }
            )
        else:
            # Tipo B: ruim → boa
            requests.append(
                {
                    "custom_id": f"B_{i}",
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 120,
                        "system": _SYSTEM_B,
                        "messages": [{"role": "user", "content": q.text}],
                    },
                }
            )

    return requests


def cmd_submit() -> None:
    """Submete o batch e salva o ID."""
    from src.eval.curated import get_curated

    curated = get_curated()
    requests = _build_requests(curated)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"  Submetendo batch com {len(requests)} requisições...")
    batch = client.messages.batches.create(requests=requests)

    _BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_ID_FILE.write_text(batch.id)

    print("\n  ✅ Batch submetido!")
    print(f"  ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ID salvo em: {_BATCH_ID_FILE}")
    print("\n  Aguarde ~5-30 min e então execute:")
    print(f"  python -m src.dataset.expand_via_batch --retrieve {batch.id}")


def cmd_retrieve(batch_id: str) -> None:
    """Recupera resultados, processa e salva os pares."""
    from src.eval.curated import get_curated

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    curated = get_curated()
    curated_by_idx = {str(i): q for i, q in enumerate(curated)}

    # Verifica status
    batch = client.messages.batches.retrieve(batch_id)
    print(f"  Status: {batch.processing_status}")

    if batch.processing_status == "in_progress":
        print("  ⏳ Batch ainda em processamento. Tente novamente em alguns minutos.")
        return

    if batch.processing_status != "ended":
        print(f"  ⚠ Status inesperado: {batch.processing_status}")
        return

    # Processa resultados
    pairs = []
    errors = 0
    skipped = 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id  # "A_5" ou "B_12"
        tipo, idx_str = cid.split("_", 1)
        original = curated_by_idx.get(idx_str)

        if not original:
            skipped += 1
            continue

        if result.result.type != "succeeded":
            errors += 1
            continue

        generated = result.result.message.content[0].text.strip().split("\n")[0].strip()
        if not generated:
            skipped += 1
            continue

        if tipo == "A":
            # A: original é q_good, gerada é q_bad
            pairs.append(
                {
                    "q_bad": generated,
                    "q_good": original.text,
                    "source": "batch_A",
                    "domain": original.domain,
                }
            )
        else:
            # B: original é q_bad, gerada é q_good
            pairs.append(
                {
                    "q_bad": original.text,
                    "q_good": generated,
                    "source": "batch_B",
                    "domain": original.domain,
                }
            )

    # Salva
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n  ✅ Resultados processados:")
    print(f"  Pares gerados : {len(pairs)}")
    print(f"  Erros API     : {errors}")
    print(f"  Ignorados     : {skipped}")
    print(f"  Arquivo       : {_OUT_PATH}")
    print(
        f"\n  Dataset expandido. Total em batch_pairs.jsonl: "
        f"{sum(1 for _ in open(_OUT_PATH, encoding='utf-8'))} pares"
    )


def cmd_status(batch_id: str | None = None) -> None:
    """Consulta status do batch mais recente ou do ID fornecido."""
    if not batch_id and _BATCH_ID_FILE.exists():
        batch_id = _BATCH_ID_FILE.read_text().strip()
    if not batch_id:
        print("  Nenhum batch encontrado. Use --submit primeiro.")
        return

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    batch = client.messages.batches.retrieve(batch_id)

    print(f"  ID      : {batch.id}")
    print(f"  Status  : {batch.processing_status}")
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        print(f"  Sucesso : {rc.succeeded}")
        print(f"  Erros   : {rc.errored}")
        print(f"  Pendente: {rc.processing}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--submit" in args:
        cmd_submit()

    elif "--retrieve" in args:
        idx = args.index("--retrieve")
        bid = args[idx + 1] if idx + 1 < len(args) else None
        if not bid and _BATCH_ID_FILE.exists():
            bid = _BATCH_ID_FILE.read_text().strip()
        if not bid:
            print("Uso: --retrieve <batch_id>")
            sys.exit(1)
        cmd_retrieve(bid)

    elif "--status" in args:
        idx = args.index("--status")
        bid = args[idx + 1] if idx + 1 < len(args) else None
        cmd_status(bid)

    else:
        print(__doc__)
