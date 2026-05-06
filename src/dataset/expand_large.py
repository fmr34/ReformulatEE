"""
Expansão grande do dataset DPO — ~900 pares via Claude Batch API.

Estratégia dupla:
  1. Novos domínios (30) não cobertos pelos batches anteriores
  2. 4 arquétipos de questão-ruim para os domínios existentes:
       - ontológica   ("What is the nature of X?")
       - excessivamente ampla ("How does X affect everything?")
       - metodologicamente ingênua ("Does X cause Y?")
       - especulativa ("Could X eventually explain Z?")

  Isso dá sinal de treino mais rico: o modelo aprende a reconhecer
  diferentes falhas epistêmicas, não só a "filosofia excessiva".

Fluxo:
  1. python -m src.dataset.expand_large --submit
  2. (aguardar ~10-30 min)
  3. python -m src.dataset.expand_large --retrieve

Saída: data/rl/batch_large.jsonl
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv(override=True)

import anthropic

_OUT_PATH = Path("data/rl/batch_large.jsonl")
_BATCH_ID_FILE = Path(".claude/batch_large_id.txt")

# ---------------------------------------------------------------------------
# Domínios novos (não cobertos pelos batches anteriores)
# ---------------------------------------------------------------------------

_NEW_DOMAINS = [
    ("bioinformatics", "sequence analysis, protein structure prediction, genomics pipelines"),
    ("materials science", "crystal structure, mechanical properties, thin films, alloys"),
    ("nuclear physics", "decay modes, fission, fusion reactions, nuclear structure"),
    ("archaeology", "artifact dating, site stratigraphy, material culture, taphonomy"),
    ("musicology", "harmonic analysis, music cognition, historical performance practice"),
    (
        "sports science",
        "athletic performance, biomechanics, exercise physiology, injury prevention",
    ),
    (
        "nutrition science",
        "dietary patterns, nutrient metabolism, bioavailability, intervention trials",
    ),
    ("pharmacology", "drug mechanisms, receptor binding, pharmacokinetics, dose-response"),
    ("veterinary science", "animal disease, zoonosis, livestock health, companion animal medicine"),
    (
        "forensic science",
        "evidence analysis, DNA profiling, toxicology, crime scene reconstruction",
    ),
    ("demography", "population dynamics, fertility rates, migration, age structure"),
    (
        "development economics",
        "poverty traps, microfinance, institutional capacity, growth determinants",
    ),
    ("chemical engineering", "reaction kinetics, process design, separation, catalysis"),
    (
        "biomedical engineering",
        "implant design, biosensors, drug delivery systems, tissue scaffolds",
    ),
    ("agricultural science", "crop yield, soil fertility, pest resistance, irrigation efficiency"),
    ("oceanography", "ocean circulation, marine chemistry, deep-sea ecosystems, salinity"),
    (
        "atmospheric science",
        "aerosol dynamics, cloud microphysics, boundary layer, radiative forcing",
    ),
    ("paleontology", "fossil morphology, extinction events, phylogenetic reconstruction"),
    ("nanotechnology", "nanomaterial synthesis, surface area, quantum effects, self-assembly"),
    ("information science", "knowledge organization, retrieval models, metadata, semantic web"),
    ("game theory", "equilibria, mechanism design, repeated games, coalition formation"),
    ("space medicine", "microgravity physiology, radiation exposure, long-duration spaceflight"),
    ("philosophy of mind", "consciousness, qualia, intentionality, mental causation"),
    ("philosophy of mathematics", "mathematical objects, proof, formalism, intuitionism"),
    ("comparative literature", "translation theory, intertextuality, genre, world literature"),
    ("religious studies", "ritual practice, theology, sacred texts, religious experience"),
    ("food science", "food processing, texture, shelf life, sensory evaluation"),
    ("operations research", "linear programming, queuing theory, inventory models, scheduling"),
    ("actuarial science", "mortality tables, risk modeling, insurance pricing, solvency"),
    (
        "developmental psychology",
        "cognitive milestones, attachment theory, language acquisition, adolescence",
    ),
]

# ---------------------------------------------------------------------------
# Arquétipos de questão-ruim para domínios já cobertos
# ---------------------------------------------------------------------------

_ARCHETYPES = [
    (
        "ontological",
        "The q_bad must be an ontological question: 'What is the nature/essence/being of X?'. "
        "It should sound deep but be unanswerable empirically.",
    ),
    (
        "too_broad",
        "The q_bad must be excessively broad: 'How does X affect everything in Y?'. "
        "It lacks scope, population, timeframe, or outcome specification.",
    ),
    (
        "naive_causation",
        "The q_bad must conflate correlation with causation: 'Does X cause Y?' "
        "without specifying mechanism, controls, population, or study design.",
    ),
    (
        "speculative",
        "The q_bad must be speculative and unfalsifiable: 'Could X eventually explain Z?' or "
        "'Will X ever achieve Y?'. No testable hypothesis, no operationalization.",
    ),
]

_ARCHETYPE_DOMAINS = [
    ("sociology", "social structures, institutions, inequality"),
    ("psychology", "cognition, behavior, mental processes"),
    ("ecology", "species interactions, ecosystems, biodiversity"),
    ("epidemiology", "disease distribution, risk factors, population health"),
    ("economics", "markets, incentives, welfare, growth"),
    ("neuroscience", "neural circuits, brain regions, behavior"),
    ("physics", "forces, fields, particles, thermodynamics"),
    ("molecular biology", "gene expression, protein function, cell signaling"),
    ("climate science", "climate dynamics, feedback loops, attribution"),
    ("linguistics", "language structure, acquisition, pragmatics"),
    ("genetics", "heredity, genomics, gene regulation"),
    ("political science", "governance, institutions, power, policy"),
    ("cognitive science", "reasoning, decision-making, representations"),
    ("anthropology", "culture, kinship, ritual, material culture"),
    ("education", "learning, pedagogy, assessment, equity"),
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_NEW = (
    "You are a research methodology expert. "
    "Given a scientific domain and context, generate ONE research question pair:\n"
    "  q_bad  — vague, overly philosophical, not empirically tractable\n"
    "  q_good — specific, operationalizable, methodologically sound\n\n"
    "Same topic, radically different tractability. "
    "Respond in valid JSON only:\n"
    '{"q_bad": "...", "q_good": "..."}'
)

_SYSTEM_ARCHETYPE = (
    "You are a research methodology expert. "
    "Generate ONE pair of research questions for the given domain.\n\n"
    "The q_bad must follow the SPECIFIC archetype described. "
    "The q_good must be a concrete, empirically tractable improvement.\n\n"
    "Respond in valid JSON only:\n"
    '{"q_bad": "...", "q_good": "..."}'
)

# ---------------------------------------------------------------------------
# Build requests
# ---------------------------------------------------------------------------


def _build_requests() -> list[dict]:
    requests = []

    # Part 1: New domains — 12 pairs each
    for domain, context in _NEW_DOMAINS:
        for i in range(12):
            requests.append(
                {
                    "custom_id": f"new__{domain.replace(' ', '_')}__{i}",
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 220,
                        "system": _SYSTEM_NEW,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Domain: {domain}\n"
                                    f"Context: {context}\n"
                                    f"Pair #{i + 1}: vary the research area within this domain."
                                ),
                            }
                        ],
                    },
                }
            )

    # Part 2: Archetype variants for existing domains — 4 archetypes × 15 domains = 60 pairs
    for archetype_name, archetype_instr in _ARCHETYPES:
        for domain, context in _ARCHETYPE_DOMAINS:
            requests.append(
                {
                    "custom_id": f"arch__{archetype_name}__{domain.replace(' ', '_')}",
                    "params": {
                        "model": "claude-haiku-4-5",
                        "max_tokens": 220,
                        "system": _SYSTEM_ARCHETYPE,
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    f"Domain: {domain}\n"
                                    f"Context: {context}\n"
                                    f"Archetype for q_bad: {archetype_instr}"
                                ),
                            }
                        ],
                    },
                }
            )

    return requests


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_submit() -> None:
    requests = _build_requests()
    n_new = len(_NEW_DOMAINS) * 12
    n_arch = len(_ARCHETYPES) * len(_ARCHETYPE_DOMAINS)
    total = len(requests)
    cost = total * 0.00006  # estimativa conservadora por request

    print("  Breakdown:")
    print(f"    Novos domínios ({len(_NEW_DOMAINS)} × 12): {n_new} pares")
    print(f"    Arquétipos    ({len(_ARCHETYPES)} × {len(_ARCHETYPE_DOMAINS)}):  {n_arch} pares")
    print(f"    Total         : {total} requisições")
    print(f"    Custo estimado: ~${cost:.3f}")
    print()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    print("  Submetendo batch...")
    batch = client.messages.batches.create(requests=requests)

    _BATCH_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BATCH_ID_FILE.write_text(batch.id)

    print("\n  ✅ Batch submetido!")
    print(f"  ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print("\n  Aguarde ~10-30 min e execute:")
    print(f"  python -m src.dataset.expand_large --retrieve {batch.id}")


def cmd_retrieve(batch_id: str) -> None:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    batch = client.messages.batches.retrieve(batch_id)
    print(f"  Status: {batch.processing_status}")

    if batch.processing_status == "in_progress":
        print("  ⏳ Ainda processando.")
        return
    if batch.processing_status != "ended":
        print(f"  ⚠ Status inesperado: {batch.processing_status}")
        return

    pairs, errors, skipped = [], 0, 0

    for result in client.messages.batches.results(batch_id):
        cid = result.custom_id
        parts = cid.split("__")

        if parts[0] == "new":
            domain = parts[1].replace("_", " ") if len(parts) >= 2 else "unknown"
        elif parts[0] == "arch":
            domain = parts[2].replace("_", " ") if len(parts) >= 3 else "unknown"
            # arquétipo como metadata extra
        else:
            domain = cid

        if result.result.type != "succeeded":
            errors += 1
            continue

        raw = result.result.message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            parsed = json.loads(raw)
            q_bad = parsed.get("q_bad", "").strip()
            q_good = parsed.get("q_good", "").strip()
        except json.JSONDecodeError:
            skipped += 1
            continue

        if not q_bad or not q_good or q_bad == q_good:
            skipped += 1
            continue

        pair = {
            "q_bad": q_bad,
            "q_good": q_good,
            "source": "batch_large",
            "domain": domain,
        }
        if parts[0] == "arch":
            pair["archetype"] = parts[1] if len(parts) >= 2 else ""

        pairs.append(pair)

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "a", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n  ✅ Resultados:")
    print(f"  Pares gerados : {len(pairs)}")
    print(f"  Erros API     : {errors}")
    print(f"  Ignorados     : {skipped}")
    print(f"  Arquivo       : {_OUT_PATH}")

    from collections import Counter

    by_source = Counter(
        ("arquétipo: " + p.get("archetype", "?")) if p.get("archetype") else p["domain"]
        for p in pairs
    )
    print("\n  Top domínios/arquétipos:")
    for k, cnt in sorted(by_source.items(), key=lambda x: -x[1])[:15]:
        print(f"    {k:<45} {cnt:>3}")


def cmd_status(batch_id: str | None = None) -> None:
    if not batch_id and _BATCH_ID_FILE.exists():
        batch_id = _BATCH_ID_FILE.read_text().strip()
    if not batch_id:
        print("  Nenhum batch encontrado.")
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

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
