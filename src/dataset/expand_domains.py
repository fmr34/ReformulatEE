"""
Expande o dataset DPO com pares em domínios sub-representados via Claude Batch API.

Estratégia: para cada domínio-alvo, gera N pares (q_bad, q_good) diretamente,
sem depender das questões curated existentes. Isso cobre lacunas de cobertura.

Domínios alvo: sociology, psychology, linguistics, ecology, epidemiology,
anthropology, geology, genetics, public health, climate science, ethics,
aesthetics, political science, history, cognitive science, economics (applied),
law, education, architecture/urban planning, environmental science.

Fluxo:
  1. python -m src.dataset.expand_domains --submit        # cria batch
  2. (aguardar ~5-30 min)
  3. python -m src.dataset.expand_domains --retrieve ID   # salva pares

Saída: data/rl/batch_domains.jsonl  (formato idêntico a batch_pairs.jsonl)
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..')))

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv(override=True)

import anthropic

_OUT_PATH      = Path("data/rl/batch_domains.jsonl")
_BATCH_ID_FILE = Path(".claude/batch_domains_id.txt")

# ---------------------------------------------------------------------------
# Domínios alvo com peso (pares a gerar por domínio)
# ---------------------------------------------------------------------------
_DOMAINS = [
    # (domínio, n_pares, instrução de contexto)
    ("sociology",                  8,  "social structures, institutions, inequality, group behavior"),
    ("psychology",                 8,  "cognition, behavior, mental processes, individual differences"),
    ("linguistics",                8,  "language structure, acquisition, processing, evolution"),
    ("ecology",                    8,  "species interactions, ecosystems, biodiversity, conservation"),
    ("epidemiology",               8,  "disease distribution, risk factors, population health"),
    ("anthropology",               8,  "human cultures, evolution, archaeology, kinship systems"),
    ("geology",                    6,  "rock formation, plate tectonics, earth history, mineralogy"),
    ("genetics",                   8,  "gene expression, heredity, genomics, mutations"),
    ("public health",              8,  "health interventions, policy, disease prevention, equity"),
    ("climate science",            8,  "climate dynamics, feedback loops, attribution, modeling"),
    ("ethics",                     6,  "moral philosophy, normative theory, applied ethics"),
    ("aesthetics",                 6,  "art perception, beauty, aesthetic experience, criticism"),
    ("political science",          8,  "governance, institutions, voting behavior, international relations"),
    ("history",                    6,  "historical causation, sources, periodization, methodology"),
    ("cognitive science",          8,  "mental representations, reasoning, decision-making, embodied cognition"),
    ("law",                        6,  "legal reasoning, jurisprudence, rights, enforcement"),
    ("education",                  6,  "learning theory, pedagogy, curriculum, educational equity"),
    ("environmental science",      6,  "pollution, resource management, sustainability, human impact"),
    ("microbiology",               6,  "microbial ecology, pathogenesis, symbiosis, evolution"),
    ("urban planning",             6,  "city design, land use, mobility, housing, social equity"),
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_GENERATE = (
    "You are a research methodology expert. "
    "Your task: given a scientific domain and context keywords, generate ONE pair of research questions:\n"
    "  1. q_bad  — vague, overly philosophical, ontologically broad, and hard to study empirically\n"
    "  2. q_good — operationalizable, specific, methodologically tractable, answerable with existing tools\n\n"
    "The q_good should be a concrete improvement of q_bad — same topic, different tractability.\n\n"
    "Respond in valid JSON only, no explanation:\n"
    '{\"q_bad\": \"...\", \"q_good\": \"...\"}'
)


# ---------------------------------------------------------------------------
# Build requests
# ---------------------------------------------------------------------------

def _build_requests() -> list[dict]:
    requests = []
    for domain, n_pairs, context in _DOMAINS:
        for i in range(n_pairs):
            requests.append({
                "custom_id": f"{domain.replace(' ', '_')}_{i}",
                "params": {
                    "model": "claude-haiku-4-5",
                    "max_tokens": 200,
                    "system": _SYSTEM_GENERATE,
                    "messages": [{
                        "role": "user",
                        "content": (
                            f"Domain: {domain}\n"
                            f"Context: {context}\n"
                            f"Generate pair #{i + 1} (make it distinct from obvious examples)."
                        ),
                    }],
                },
            })
    return requests


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit() -> None:
    requests = _build_requests()
    total = len(requests)

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print(f"  Submetendo batch com {total} requisições ({len(_DOMAINS)} domínios)...")

    batch = client.messages.batches.create(requests=requests)

    _BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_ID_FILE.write_text(batch.id)

    print(f"\n  ✅ Batch submetido!")
    print(f"  ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  ID salvo em: {_BATCH_ID_FILE}")
    print(f"\n  Aguarde ~5-30 min e execute:")
    print(f"  python -m src.dataset.expand_domains --retrieve {batch.id}")


def cmd_retrieve(batch_id: str) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch = client.messages.batches.retrieve(batch_id)
    print(f"  Status: {batch.processing_status}")

    if batch.processing_status == "in_progress":
        print("  ⏳ Ainda processando. Tente novamente em alguns minutos.")
        return

    if batch.processing_status != "ended":
        print(f"  ⚠ Status inesperado: {batch.processing_status}")
        return

    pairs   = []
    errors  = 0
    skipped = 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id   # "sociology_0", "genetics_3", etc.

        # Extrai domínio a partir do custom_id
        parts  = cid.rsplit("_", 1)
        domain = parts[0].replace("_", " ") if len(parts) == 2 else cid

        if result.result.type != "succeeded":
            errors += 1
            continue

        raw = result.result.message.content[0].text.strip()

        # Remove possível markdown ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
            q_bad  = parsed.get("q_bad",  "").strip()
            q_good = parsed.get("q_good", "").strip()
        except json.JSONDecodeError:
            skipped += 1
            continue

        if not q_bad or not q_good or q_bad == q_good:
            skipped += 1
            continue

        pairs.append({
            "q_bad":   q_bad,
            "q_good":  q_good,
            "source":  "batch_domains",
            "domain":  domain,
        })

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n  ✅ Resultados:")
    print(f"  Pares gerados : {len(pairs)}")
    print(f"  Erros API     : {errors}")
    print(f"  Ignorados     : {skipped}")
    print(f"  Arquivo       : {_OUT_PATH}")

    # Resumo por domínio
    from collections import Counter
    domain_counts = Counter(p["domain"] for p in pairs)
    print("\n  Pares por domínio:")
    for dom, cnt in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {dom:<35} {cnt:>3}")


def cmd_status(batch_id: str | None = None) -> None:
    if not batch_id and _BATCH_ID_FILE.exists():
        batch_id = _BATCH_ID_FILE.read_text().strip()
    if not batch_id:
        print("  Nenhum batch encontrado. Use --submit primeiro.")
        return
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    batch  = client.messages.batches.retrieve(batch_id)
    print(f"  ID      : {batch.id}")
    print(f"  Status  : {batch.processing_status}")
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        print(f"  Sucesso : {rc.succeeded}")
        print(f"  Erros   : {rc.errored}")
        print(f"  Pendente: {rc.processing}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    n_total = sum(n for _, n, _ in _DOMAINS)
    print(f"  expand_domains — {n_total} pares em {len(_DOMAINS)} domínios")
    print(f"  Custo estimado: ~${n_total * 0.0003:.2f} (Haiku Batch API)")
    print()

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
