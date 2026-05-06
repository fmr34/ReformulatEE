"""
Fase 3 — Inferencia guiada por EE (best-of-N sampling).

Estrategia:
  1. Policy model gera N candidatos para q_bad
  2. Cada candidato passa pelo Filtro Stage 1: EE(q_cand) > EE(q_bad) + epsilon
  3. Entre os aprovados, seleciona o de maior score(alpha)
  4. Fallback: se nenhum aprovado, retorna o de maior EE entre todos

Policy backends suportados:
  - hf_inference : HuggingFace Inference API (gratuita, zero-cost) — PADRAO
  - gguf         : modelo GGUF local via llama-cpp-python (opcional)
  - claude       : Claude API (requer ANTHROPIC_API_KEY)
  - local        : modelo PEFT local (DPO checkpoint) com transformers + PEFT

Traducao pt-br:
  - Padrao: Helsinki-NLP MarianMT local (zero-cost, ~300 MB por modelo)
  - Fallback: Claude API (se ANTHROPIC_API_KEY definida)

Otimizacoes (Onda 1):
  - Geracao paralela: 8 chamadas simultaneas via ThreadPoolExecutor
  - Scoring paralelo: 8 scores EE simultaneos via ThreadPoolExecutor
  - Prompt caching: cache_control ephemeral (Claude API)
  - Cache de tratabilidade: in-memory + SQLite

Configuracao via .env:
  INFERENCE_BACKEND        = hf_inference | gguf | claude | local  (default: auto)
  INFERENCE_MODEL_DIR      = data/models/dpo_policy/tier3/final
  DPO_MODEL                = gpt2             (base model para backend local)
  INFERENCE_N              = 8                (candidatos por query)
  INFERENCE_ALPHA          = 0.5              (peso EE vs proximidade no ranking)
  INFERENCE_MAX_NEW_TOKENS = 80
  INFERENCE_TEMPERATURE    = 1.1
  INFERENCE_TOP_P          = 0.95
  TRANSLATE_BACKEND        = local | claude   (default: local)

Uso (ingles):
  .venv\\Scripts\\python -m src.rl.inference "What is the essence of life?"
  .venv\\Scripts\\python -m src.rl.inference --batch caminho/para/perguntas.txt
  .venv\\Scripts\\python -m src.rl.inference --demo

Uso (portugues — traducao automatica):
  .venv\\Scripts\\python -m src.rl.inference --pt "O que e a consciencia?"
  .venv\\Scripts\\python -m src.rl.inference --pt --batch caminho/para/perguntas_pt.txt
  .venv\\Scripts\\python -m src.rl.inference --pt --demo
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------

BACKEND = os.getenv("INFERENCE_BACKEND", "auto")
TRANSLATE_BACKEND = os.getenv("TRANSLATE_BACKEND", "local")
MODEL_DIR = Path(os.getenv("INFERENCE_MODEL_DIR", "data/models/dpo_policy/tier3/final"))
BASE_MODEL = os.getenv("DPO_MODEL", "gpt2")
N_CANDIDATES = int(os.getenv("INFERENCE_N", "8"))
ALPHA = float(os.getenv("INFERENCE_ALPHA", "0.5"))
MAX_NEW_TOKENS = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", "80"))
TEMPERATURE = float(os.getenv("INFERENCE_TEMPERATURE", "1.1"))
TOP_P = float(os.getenv("INFERENCE_TOP_P", "0.95"))
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "data/corpus"))

PROMPT_TEMPLATE = (
    "You are an expert in philosophy of science. "
    "Reformulate the following research question to make it more epistemically tractable: "
    "operationalizable, methodologically grounded, and answerable with existing tools.\n\n"
    "Original question: {q_bad}\n\n"
    "Reformulated question:"
)

# System prompts extraídos como constantes para reutilização com cache_control
_GENERATION_SYSTEM = (
    "You are an expert in philosophy of science. "
    "Your task is to reformulate research questions to make them more epistemically tractable: "
    "operationalizable, methodologically grounded, and answerable with existing tools. "
    "Respond with ONLY the reformulated question — no explanation, no preamble."
)

_TRANSLATE_SYSTEMS = {
    "pt_to_en": (
        "Translate the following research question from Portuguese to English. "
        "Preserve the exact meaning and academic tone. "
        "Respond with ONLY the translated question, nothing else."
    ),
    "en_to_pt": (
        "Translate the following research question from English to Portuguese (Brazilian). "
        "Preserve the exact meaning and academic tone. "
        "Respond with ONLY the translated question, nothing else."
    ),
}

# Cliente Anthropic compartilhado (thread-safe)
_claude_client = None


def _get_claude_client():
    global _claude_client
    if _claude_client is None:
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY nao definido no .env. "
                "Configure-o ou use INFERENCE_BACKEND=local."
            )
        _claude_client = anthropic.Anthropic(api_key=api_key)
    return _claude_client


def pr(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# Resultado de inferencia
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    q_bad: str
    best: str
    ee_bad: float
    ee_best: float
    score_best: float
    stage1_pass: bool
    candidates: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Input  : {self.q_bad[:80]}",
            f"  Output : {self.best[:80]}",
            f"  EE     : {self.ee_bad:.3f} -> {self.ee_best:.3f}  "
            f"({'PASS' if self.stage1_pass else 'FALLBACK'})",
            f"  Score  : {self.score_best:.3f}  (alpha={ALPHA})",
            f"  N cand : {len(self.candidates)}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Corpus index (lazy)
# ---------------------------------------------------------------------------

_corpus_index = None


def _get_index():
    global _corpus_index
    if _corpus_index is None:
        from src.corpus.index import build_index

        index_pkl = CORPUS_DIR / "bm25_index.pkl"
        if not index_pkl.exists():
            pr("  [aviso] Corpus BM25 nao encontrado — respondibilidade sera 0.")
            _corpus_index = _NullIndex()
        else:
            _corpus_index = build_index(CORPUS_DIR)
    return _corpus_index


class _NullIndex:
    """Fallback quando corpus nao esta disponivel."""

    def search(self, *args, **kwargs):
        return []


# ---------------------------------------------------------------------------
# EE scoring
# ---------------------------------------------------------------------------


def _score_candidate(q_cand: str, q_bad: str) -> dict:
    """Retorna dict com ee, score, prox para um candidato."""
    from src.ee.reward import compute_ee
    from src.ee.reward import compute_score

    index = _get_index()
    try:
        result = compute_ee(q_cand, q_bad, index)
        score = compute_score(result, alpha=ALPHA)
        return {
            "text": q_cand,
            "ee": result.ee,
            "score": score,
            "prox": result.prox,
            "resp": result.respondibilidade,
            "tract": result.tratabilidade,
            "nt": result.nao_trivialidade,
        }
    except Exception as exc:
        return {
            "text": q_cand,
            "ee": 0.0,
            "score": 0.0,
            "prox": 0.0,
            "resp": 0.0,
            "tract": 0.0,
            "nt": 0.0,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Backend LOCAL (PEFT + transformers)
# ---------------------------------------------------------------------------

_local_pipeline = None


def _load_local_pipeline():
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline

    import torch
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    from transformers import pipeline

    pr(f"  Carregando modelo local: {MODEL_DIR}")

    # Tenta carregar PEFT adapter; fallback para base model
    if (MODEL_DIR / "adapter_config.json").exists():
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, str(MODEL_DIR))
        model = model.merge_and_unload()
        pr("  PEFT adapter carregado e mesclado.")
    elif MODEL_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            device_map="auto" if torch.cuda.is_available() else None,
        )
        pr("  Modelo completo carregado.")
    else:
        pr(f"  [aviso] MODEL_DIR nao encontrado: {MODEL_DIR}")
        pr(f"  Usando modelo base: {BASE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = 0 if torch.cuda.is_available() else -1
    _local_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return _local_pipeline


def _generate_local(q_bad: str, n: int) -> list[str]:
    """Gera n candidatos usando modelo local."""
    from transformers import GenerationConfig

    pipe = _load_local_pipeline()
    prompt = PROMPT_TEMPLATE.format(q_bad=q_bad)

    gen_config = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        num_return_sequences=n,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    outputs = pipe(prompt, generation_config=gen_config)

    candidates = []
    for out in outputs:
        text = out["generated_text"]
        # Remove o prompt, ficando apenas a reformulacao
        if "Reformulated question:" in text:
            text = text.split("Reformulated question:")[-1]
        text = text.strip().split("\n")[0].strip()
        if text:
            candidates.append(text)

    return candidates


# ---------------------------------------------------------------------------
# Backend CLAUDE (Anthropic API)
# ---------------------------------------------------------------------------


def _generate_claude(q_bad: str, n: int) -> list[str]:
    """
    Gera n candidatos usando Claude API em paralelo (ThreadPoolExecutor).
    Usa prompt caching no system prompt para reduzir custo.
    """
    client = _get_claude_client()
    system = [
        {
            "type": "text",
            "text": _GENERATION_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    user_msg = [{"role": "user", "content": f"Original question: {q_bad}"}]

    def _single_call(_):
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            temperature=1.0,
            system=system,
            messages=user_msg,
        )
        return msg.content[0].text.strip().split("\n")[0].strip()

    candidates = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [ex.submit(_single_call, i) for i in range(n)]
        for f in as_completed(futures):
            try:
                text = f.result()
                if text:
                    candidates.append(text)
            except Exception as exc:
                pr(f"  [aviso] Erro na API Claude: {exc}")

    return candidates


# ---------------------------------------------------------------------------
# Interface principal
# ---------------------------------------------------------------------------


def _score_all(candidates_text: list[str], q_bad: str) -> list[dict]:
    """
    Pontua todos os candidatos em paralelo via ThreadPoolExecutor.
    Cada score chama tratabilidade() (API Claude), que usa cache interno.
    """
    with ThreadPoolExecutor(max_workers=len(candidates_text)) as ex:
        futures = {ex.submit(_score_candidate, c, q_bad): c for c in candidates_text}
        results = []
        for f in as_completed(futures):
            try:
                results.append(f.result())
            except Exception as exc:
                q = futures[f]
                pr(f"  [aviso] Erro ao pontuar candidato: {exc}")
                results.append(
                    {
                        "text": q,
                        "ee": 0.0,
                        "score": 0.0,
                        "prox": 0.0,
                        "resp": 0.0,
                        "tract": 0.0,
                        "nt": 0.0,
                    }
                )
    return results


def gerar_candidatos(q_bad: str, n: int = N_CANDIDATES) -> list[str]:
    """
    Gera n reformulacoes candidatas para q_bad.

    Hierarquia de backends:
      auto/hf_inference/gguf → generate_free.generate() (zero-cost)
      claude                 → _generate_claude() (requer API key)
      local                  → _generate_local() (modelo PEFT local)
    """
    if BACKEND == "claude":
        return _generate_claude(q_bad, n)
    elif BACKEND == "local":
        return _generate_local(q_bad, n)
    else:
        # auto, hf_inference, gguf — delega ao módulo zero-cost
        from src.rl.generate_free import generate as _generate_free

        return _generate_free(q_bad, n)


def reformular(q_bad: str, n: int = N_CANDIDATES) -> InferenceResult:
    """
    Pipeline completo: gera N candidatos, pontua, filtra e retorna o melhor.

    Args:
        q_bad: Pergunta de pesquisa original (pouco tratavel)
        n:     Numero de candidatos a gerar

    Returns:
        InferenceResult com o melhor candidato e metricas
    """
    from src.ee.reward import compute_ee

    # Score da pergunta original
    index = _get_index()
    r_bad = compute_ee(q_bad, q_bad, index)
    ee_bad = r_bad.ee

    # Gera candidatos
    candidates_text = gerar_candidatos(q_bad, n)
    if not candidates_text:
        # Fallback: devolve a pergunta original
        return InferenceResult(
            q_bad=q_bad,
            best=q_bad,
            ee_bad=ee_bad,
            ee_best=ee_bad,
            score_best=0.0,
            stage1_pass=False,
            candidates=[],
        )

    # Pontua todos os candidatos em paralelo
    scored = _score_all(candidates_text, q_bad)

    # Filtro Stage 1: EE(cand) > EE(q_bad) + epsilon
    from src.ee.reward import _EPSILON

    approved = [s for s in scored if s["ee"] > ee_bad + _EPSILON]

    if approved:
        best = max(approved, key=lambda s: s["score"])
        stage1_pass = True
    else:
        # Fallback: melhor EE entre todos
        best = max(scored, key=lambda s: s["ee"])
        stage1_pass = False

    return InferenceResult(
        q_bad=q_bad,
        best=best["text"],
        ee_bad=ee_bad,
        ee_best=best["ee"],
        score_best=best["score"],
        stage1_pass=stage1_pass,
        candidates=scored,
    )


# ---------------------------------------------------------------------------
# Traducao automatica pt-br (MarianMT local por padrao; Claude como fallback)
# ---------------------------------------------------------------------------


def _translate_claude(text: str, direction: str) -> str:
    """Traduz via Claude API com prompt caching. direction: 'pt_to_en' | 'en_to_pt'."""
    client = _get_claude_client()
    system = [
        {
            "type": "text",
            "text": _TRANSLATE_SYSTEMS[direction],
            "cache_control": {"type": "ephemeral"},
        }
    ]
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    return msg.content[0].text.strip()


def _translate(text: str, direction: str) -> str:
    """
    Traduz texto pt-br <-> en.
    Usa MarianMT local por padrao (zero-cost); fallback para Claude API.

    direction: 'pt_to_en' | 'en_to_pt'
    """
    if TRANSLATE_BACKEND == "claude":
        return _translate_claude(text, direction)

    # Backend local (padrao)
    try:
        from src.ee.translate_local import is_available
        from src.ee.translate_local import translate as _translate_local

        if is_available():
            return _translate_local(text, direction)
        # transformers nao instalado — tenta Claude
        if os.getenv("ANTHROPIC_API_KEY"):
            pr("  [translate] transformers nao disponivel, usando Claude API...")
            return _translate_claude(text, direction)
        raise RuntimeError(
            "Nenhum backend de traducao disponivel. "
            "Instale transformers+sentencepiece ou defina ANTHROPIC_API_KEY."
        )
    except ImportError:
        if os.getenv("ANTHROPIC_API_KEY"):
            pr("  [translate] translate_local nao encontrado, usando Claude API...")
            return _translate_claude(text, direction)
        raise


@dataclass
class PtBrResult:
    q_bad_pt: str  # pergunta original em pt-br
    q_bad_en: str  # traducao para ingles
    best_en: str  # melhor reformulacao em ingles
    best_pt: str  # melhor reformulacao em pt-br
    ee_bad: float
    ee_best: float
    score_best: float
    stage1_pass: bool
    candidates: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Entrada  : {self.q_bad_pt[:80]}",
            f"  (ingles) : {self.q_bad_en[:80]}",
            f"  Resultado: {self.best_pt[:80]}",
            f"  (ingles) : {self.best_en[:80]}",
            f"  EE       : {self.ee_bad:.3f} -> {self.ee_best:.3f}  "
            f"({'PASS' if self.stage1_pass else 'FALLBACK'})",
            f"  Score    : {self.score_best:.3f}  (alpha={ALPHA})",
            f"  N cand   : {len(self.candidates)}",
        ]
        return "\n".join(lines)


def reformular_ptbr(q_bad_pt: str, n: int = N_CANDIDATES) -> PtBrResult:
    """
    Pipeline completo com suporte a portugues:
      1. Traduz q_bad_pt (pt-br) -> ingles
      2. Roda reformular() em ingles
      3. Traduz o melhor resultado de volta para pt-br

    Args:
        q_bad_pt: Pergunta em portugues (pt-br)
        n:        Numero de candidatos a gerar

    Returns:
        PtBrResult com entrada e saida em pt-br e ingles
    """
    pr("  [1/3] Traduzindo entrada (pt -> en)...")
    q_bad_en = _translate(q_bad_pt, "pt_to_en")
    pr(f"  -> {q_bad_en[:80]}")

    pr("  [2/3] Reformulando (pipeline EE)...")
    result = reformular(q_bad_en, n)

    pr("  [3/3] Traduzindo resultado (en -> pt)...")
    best_pt = _translate(result.best, "en_to_pt")
    pr(f"  -> {best_pt[:80]}")

    return PtBrResult(
        q_bad_pt=q_bad_pt,
        q_bad_en=q_bad_en,
        best_en=result.best,
        best_pt=best_pt,
        ee_bad=result.ee_bad,
        ee_best=result.ee_best,
        score_best=result.score_best,
        stage1_pass=result.stage1_pass,
        candidates=result.candidates,
    )


# ---------------------------------------------------------------------------
# Demo interativa
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "What is the meaning of life?",
    "What is consciousness?",
    "Does free will exist?",
    "What is the nature of time?",
    "Is there a theory of everything in physics?",
]

DEMO_QUESTIONS_PT = [
    "O que e a consciencia?",
    "O livre-arbitrio existe?",
    "Qual e a natureza do tempo?",
    "O que causa o envelhecimento biologico?",
    "Como surgiu a vida na Terra?",
]


def run_demo() -> None:
    pr("=" * 65)
    pr("  Fase 3 — Demo Inferencia DPO (best-of-N + EE scoring)")
    pr("=" * 65)
    from src.rl.generate_free import _detect_backend as _det

    _resolved = _det() if BACKEND not in ("claude", "local") else BACKEND
    _blabel = f"{BACKEND} → {_resolved}" if BACKEND not in ("claude", "local") else BACKEND
    pr(f"\n  Backend  : {_blabel}")
    pr(f"  N candid.: {N_CANDIDATES}")
    pr(f"  Alpha    : {ALPHA}")
    pr(f"  Temp     : {TEMPERATURE}")
    pr(f"  Model dir: {MODEL_DIR}")

    pr("\n  Carregando pipeline de scoring...")
    _get_index()  # pre-carrega o corpus index

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        pr(f"\n[{i}/{len(DEMO_QUESTIONS)}] Reformulando...")
        result = reformular(q)
        pr(result.summary())

        # Top-3 candidatos
        sorted_cands = sorted(result.candidates, key=lambda s: s["score"], reverse=True)
        pr("\n  Top-3 candidatos:")
        for j, c in enumerate(sorted_cands[:3], 1):
            pr(f"    {j}. EE={c['ee']:.3f} | Score={c['score']:.3f} | {c['text'][:70]}")

    pr(f"\n{'='*65}")
    pr("  Demo concluida.")
    pr(f"{'='*65}")


def run_demo_pt() -> None:
    pr("=" * 65)
    pr("  Fase 3 — Demo pt-br (traducao automatica + EE scoring)")
    pr("=" * 65)
    from src.rl.generate_free import _detect_backend as _det_pt

    _resolved_pt = _det_pt() if BACKEND not in ("claude", "local") else BACKEND
    _blabel_pt = f"{BACKEND} → {_resolved_pt}" if BACKEND not in ("claude", "local") else BACKEND
    pr(f"\n  Backend  : {_blabel_pt}")
    pr(f"  N candid.: {N_CANDIDATES}")
    translate_info = "Claude Haiku" if TRANSLATE_BACKEND == "claude" else "MarianMT local"
    pr(f"  Traducao : {translate_info} (pt <-> en)")

    _get_index()

    for i, q in enumerate(DEMO_QUESTIONS_PT, 1):
        pr(f"\n[{i}/{len(DEMO_QUESTIONS_PT)}] ----------------------------------------")
        result = reformular_ptbr(q)
        pr("\n" + result.summary())

    pr(f"\n{'='*65}")
    pr("  Demo pt-br concluida.")
    pr(f"{'='*65}")


def run_batch(path: str, ptbr: bool = False) -> None:
    """Processa um arquivo .txt com uma pergunta por linha."""
    questions = [
        l.strip()
        for l in Path(path).read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")
    ]
    pr(f"  Processando {len(questions)} perguntas de {path}...")

    results = []
    for i, q in enumerate(questions, 1):
        pr(f"\n[{i}/{len(questions)}]")
        if ptbr:
            r = reformular_ptbr(q)
            pr(r.summary())
            results.append(
                {
                    "q_bad_pt": r.q_bad_pt,
                    "q_bad_en": r.q_bad_en,
                    "best_en": r.best_en,
                    "best_pt": r.best_pt,
                    "ee_bad": round(r.ee_bad, 4),
                    "ee_best": round(r.ee_best, 4),
                    "score_best": round(r.score_best, 4),
                    "stage1_pass": r.stage1_pass,
                }
            )
        else:
            r = reformular(q)
            pr(r.summary())
            results.append(
                {
                    "q_bad": r.q_bad,
                    "best": r.best,
                    "ee_bad": round(r.ee_bad, 4),
                    "ee_best": round(r.ee_best, 4),
                    "score_best": round(r.score_best, 4),
                    "stage1_pass": r.stage1_pass,
                }
            )

    out_path = Path(path).with_suffix(".results.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pr(f"\n  Resultados salvos em: {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    ptbr = "--pt" in args
    if ptbr:
        args = [a for a in args if a != "--pt"]

    if "--demo" in args:
        if ptbr:
            run_demo_pt()
        else:
            run_demo()

    elif "--batch" in args:
        idx = args.index("--batch")
        if idx + 1 >= len(args):
            pr("Uso: python -m src.rl.inference [--pt] --batch caminho/para/arquivo.txt")
            sys.exit(1)
        run_batch(args[idx + 1], ptbr=ptbr)

    elif args and not args[0].startswith("--"):
        q = " ".join(args)
        pr(f"\n  Entrada: {q}")
        if ptbr:
            result = reformular_ptbr(q)
            pr("\n" + result.summary())
            pr("\n  Top candidatos (decrescente por score):")
            for j, c in enumerate(
                sorted(result.candidates, key=lambda s: s["score"], reverse=True), 1
            ):
                status = "PASS" if c["ee"] > result.ee_bad + 0.05 else "FAIL"
                pr(f"  {j:2}. [{status}] EE={c['ee']:.3f} Sc={c['score']:.3f} | {c['text'][:75]}")
        else:
            result = reformular(q)
            pr(result.summary())
            pr("\n  Todos os candidatos (decrescente por score):")
            for j, c in enumerate(
                sorted(result.candidates, key=lambda s: s["score"], reverse=True), 1
            ):
                status = "PASS" if c["ee"] > result.ee_bad + 0.05 else "FAIL"
                pr(f"  {j:2}. [{status}] EE={c['ee']:.3f} Sc={c['score']:.3f} | {c['text'][:75]}")
    else:
        pr(__doc__)
        sys.exit(0)
