"""
Layer 4 — Geração de adversarial probes (Patch B).

Para cada par (q_bad, q_good) selecionado da Layer 2, gera três variantes
de Q_bem_falso — reformulações que parecem boas mas não são:

  Tipo 1 — parafrase:     paráfrase sofisticada de q_bad sem ganho epistêmico
  Tipo 2 — desconectada:  tratável aparentemente, mas não responde q_bad
  Tipo 3 — especulativa:  troca termos especulativos por outros igualmente especulativos

O reward model deve atribuir EE baixa a todos os probes (label=0).
Se não atribuir, é evidência de circularidade no dataset → re-treino.

Saída: data/pairs/adversarial_probes.jsonl
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

SEED = 42
N_PARES = 70  # pares selecionados da Layer 2
TIPOS = ("parafrase", "desconectada", "especulativa")
DELAY = 0.4
OUTPUT_PATH = Path("data/pairs/adversarial_probes.jsonl")
LAYER2_PATH = Path("data/pairs/pairs_layer2.jsonl")

# ---------------------------------------------------------------------------
# Prompts por tipo
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert in philosophy of science and epistemic analysis.
Your task: generate a FAKE reformulation of a research question.
The fake reformulation must LOOK like a genuine improvement but MUST NOT be one.
Respond ONLY with valid JSON. No markdown, no explanation outside JSON.
"""

_PROMPTS: dict[str, str] = {
    "parafrase": """\
Generate a fake reformulation of type SOPHISTICATED PARAPHRASE.

Rules:
- Use more sophisticated academic vocabulary than q_bad
- Sound like a genuine improvement to a non-expert
- Preserve the SAME fundamental unoperationalizable concepts as q_bad
- Do NOT introduce any new methodology, measurement tools, or empirical access
- A philosopher of science would immediately recognize it as the same question in fancier dress

Original question (q_bad): {q_bad}
Real good reformulation (q_good, for contrast — do NOT copy this): {q_good}
Domain: {domain}

JSON output:
{{
  "q_fake": "<the sophisticated paraphrase — must end with ?>",
  "why_fake": "<one sentence: which unoperationalizable concept was preserved>"
}}
""",
    "desconectada": """\
Generate a fake reformulation of type TRACTABLE BUT DISCONNECTED.

Rules:
- Must be a GENUINELY tractable question (clear methodology, measurable outcomes)
- Must sound thematically related to q_bad
- BUT: answering it would NOT meaningfully advance answering q_bad
- The inferential connection q_fake → q_bad must be absent or very weak
- A domain expert would say "interesting question, but answers a different problem"

Original question (q_bad): {q_bad}
Real good reformulation (q_good, for contrast — do NOT copy this): {q_good}
Domain: {domain}

JSON output:
{{
  "q_fake": "<the tractable but disconnected question — must end with ?>",
  "why_fake": "<one sentence: why answering this does NOT help answer q_bad>"
}}
""",
    "especulativa": """\
Generate a fake reformulation of type SPECULATIVE-TO-SPECULATIVE.

Rules:
- Replace the vague/speculative terms in q_bad with DIFFERENT vague/speculative terms
- Use modern or technical-sounding language (e.g. "emergent", "ontological", "substrate")
- The result must remain equally unanswerable and equally lacking in methodology
- It should sound more sophisticated but be just as intractable as q_bad
- Do NOT introduce any operational definition or measurable proxy

Original question (q_bad): {q_bad}
Real good reformulation (q_good, for contrast — do NOT copy this): {q_good}
Domain: {domain}

JSON output:
{{
  "q_fake": "<the speculative-to-speculative reformulation — must end with ?>",
  "why_fake": "<one sentence: which new speculative terms replaced the old ones>"
}}
""",
}

# ---------------------------------------------------------------------------
# Geração
# ---------------------------------------------------------------------------


def _gerar_probe(
    client: anthropic.Anthropic,
    par: dict,
    tipo: str,
) -> dict | None:
    prompt = _PROMPTS[tipo].format(
        q_bad=par["q_bad"],
        q_good=par["q_good"],
        domain=par.get("domain", "unknown"),
    )
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = re.sub(r"```[a-z]*\n?", "", msg.content[0].text).strip()
        data = json.loads(raw)
    except Exception:
        return None

    q_fake = data.get("q_fake", "").strip()
    why = data.get("why_fake", "").strip()

    # Validações básicas
    if not q_fake or not q_fake.endswith("?"):
        return None
    if len(q_fake) < 20:
        return None
    # Não deve ser cópia de q_bad ou q_good
    if q_fake.lower() == par["q_bad"].lower():
        return None
    if q_fake.lower() == par["q_good"].lower():
        return None

    return {
        "q_bad": par["q_bad"],
        "q_good_fake": q_fake,
        "q_good_real": par["q_good"],  # referência — não usado no treino
        "probe_type": tipo,
        "why_fake": why,
        "domain": par.get("domain", ""),
        "source": par.get("source", ""),
        "source_id": par.get("source_id", ""),
        "year": par.get("year", 0),
        "label": 0,  # EE deve ser BAIXA para este probe
    }


def gerar_probes(
    layer2_path: Path,
    output_path: Path,
    n_pares: int = N_PARES,
    seed: int = SEED,
    delay: float = DELAY,
) -> list[dict]:
    pares = [
        json.loads(l) for l in layer2_path.read_text(encoding="utf-8").splitlines() if l.strip()
    ]

    # Retoma: carrega probes já gerados
    done_keys: set[tuple] = set()
    probes: list[dict] = []
    if output_path.exists():
        for l in output_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                p = json.loads(l)
                done_keys.add((p.get("source_id", ""), p.get("probe_type", "")))
                probes.append(p)
        print(f"Retomando: {len(probes)} probes já gerados.")

    # Seleciona amostra balanceada
    random.seed(seed)
    candidatos = [p for p in pares if p.get("q_bad") and p.get("q_good")]
    amostra = random.sample(candidatos, min(n_pares, len(candidatos)))

    # Tarefas pendentes
    tarefas = [
        (par, tipo)
        for par in amostra
        for tipo in TIPOS
        if (par.get("source_id", ""), tipo) not in done_keys
    ]

    print("\n=== Layer 4 — Adversarial Probes ===")
    print(f"Pares selecionados : {len(amostra)}")
    print(f"Tipos por par      : {len(TIPOS)}")
    print(f"Total esperado     : {len(amostra) * len(TIPOS)}")
    print(f"Já gerados         : {len(probes)}")
    print(f"Pendentes          : {len(tarefas)}\n")

    client = anthropic.Anthropic()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as f:
        for par, tipo in tqdm(tarefas, desc="Gerando probes"):
            probe = _gerar_probe(client, par, tipo)
            if probe:
                f.write(json.dumps(probe, ensure_ascii=False) + "\n")
                f.flush()
                probes.append(probe)
            time.sleep(delay)

    return probes


def imprimir_resumo(probes: list[dict]) -> None:
    from collections import Counter

    tipos = Counter(p["probe_type"] for p in probes)
    fontes = Counter(p["source"] for p in probes)

    print(f"\n{'='*50}")
    print(f"  Total probes gerados: {len(probes)}")
    print("\n  Por tipo:")
    for t, n in tipos.most_common():
        print(f"    {t:<15} {n}")
    print("\n  Por fonte:")
    for s, n in fontes.most_common():
        print(f"    {s:<20} {n}")
    print(f"{'='*50}")

    print("\n=== Amostra (1 de cada tipo) ===")
    vistos: set[str] = set()
    for p in probes:
        if p["probe_type"] not in vistos:
            vistos.add(p["probe_type"])
            print(f"\n  [{p['probe_type'].upper()}]")
            print(f"  q_bad      : {p['q_bad'][:80]}")
            print(f"  q_good_fake: {p['q_good_fake'][:80]}")
            print(f"  Por que falso: {p['why_fake'][:100]}")


if __name__ == "__main__":
    probes = gerar_probes(
        layer2_path=LAYER2_PATH,
        output_path=OUTPUT_PATH,
    )
    imprimir_resumo(probes)
    print(f"\nSalvos em: {OUTPUT_PATH}")
