---
title: ReformulatEE
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Turn vague research questions into testable hypotheses
---

# 🔬 ReformulatEE — Epistemic Effectiveness Reformulation

[![Live Demo](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/fmr34/reformulatee)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/fmr34/ReformulatEE/actions/workflows/test.yml/badge.svg)](https://github.com/fmr34/ReformulatEE/actions)

ReformulatEE transforms vague research questions into concrete, testable hypotheses by maximizing their **Epistemic Effectiveness (EE)** — a composite score measuring how operationalizable, tractable, and non-trivial a question is.

> **[Try it live →](https://huggingface.co/spaces/fmr34/reformulatee)**

---

## How It Works

```
Input question  →  Generate 8 candidates  →  Score each (EE)  →  Filter  →  Best output
```

**EE(Q) = 0.05 · Respondibilidade + 0.05 · Tratabilidade + 0.90 · Não-trivialidade**

| Component | Description | Method |
|-----------|-------------|--------|
| **Respondibilidade** | Is there an active research corpus? | BM25 + semantic search over 919 papers |
| **Tratabilidade** | Can it be answered with existing tools? | Local Ridge classifier on sentence embeddings |
| **Não-trivialidade** | Is it meaningfully different from the original? | Semantic dissimilarity probe |

**Example:**

| Input | Output | EE |
|-------|--------|----|
| "What is consciousness?" | "What measurable neural correlates distinguish conscious from unconscious processing?" | 0.137 → **0.926** |
| "O que causa o envelhecimento?" | "Como a taxa de dano ao DNA mitocondrial correlaciona com biomarcadores de senescência?" | 0.153 → **0.891** |

---

## Quick Start

There are two ways to use ReformulatEE:

| | Public Demo (HF Space) | Local (Ollama fine-tuned) |
|---|---|---|
| **Setup** | None — open in browser | Clone repo + Python env + Ollama |
| **Generation** | Claude Haiku API | Fine-tuned GGUF model via Ollama |
| **Translation** | MarianMT (local, free) | MarianMT (local, free) |
| **EE Scoring** | Full | Full |
| **Cost** | Free to use | $0 (fully local) |
| **Link** | [HF Space](https://huggingface.co/spaces/fmr34/reformulatee) | See below |

### Option 1 — Public Demo

**[→ Try it live on HuggingFace Spaces](https://huggingface.co/spaces/fmr34/reformulatee)**

No installation required. Questions are logged anonymously for research purposes.

### Option 2 — Local with Fine-tuned Model (zero cost)

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e .
cp .env.example .env   # add your API keys
```

With [Ollama](https://ollama.com) and the fine-tuned model:
```bash
# Download the Modelfile and GGUF from HF Hub, then:
ollama create reformulatee -f Modelfile
# Set INFERENCE_BACKEND=ollama and OLLAMA_MODEL=reformulatee in .env
python app.py          # opens http://localhost:7860
```

The local setup uses Ollama for generation, MarianMT for translation, and the Ridge classifier for tractability — **zero API cost per request**.

### CLI

```bash
# English
python -m src.rl.inference "Does free will exist?"

# Portuguese
python -m src.rl.inference --pt "O que é a consciência?"

# Batch
python -m src.rl.inference --pt --batch questions.txt
```

---

## Configuration

Copy `.env.example` to `.env` and set:

```bash
# Required for tractability scoring (fallback) and generation on HF Space
ANTHROPIC_API_KEY=sk-ant-...

# Generation backend:
# ollama       = local fine-tuned model via Ollama (recommended for local, zero cost)
# claude       = Claude Haiku API (used automatically on HF Space)
# hf_inference = HuggingFace Inference API (public models, free)
INFERENCE_BACKEND=ollama
OLLAMA_MODEL=reformulatee
OLLAMA_BASE_URL=http://localhost:11434

# HuggingFace token — for cross-session history logging (optional locally)
HF_TOKEN=hf_...

# Optional
CORPUS_DIR=data/corpus   # path to paper corpus
```

---

## Architecture

```
Gradio Web Interface (rate-limited: 10 req/min/session)
        │
   Translation (MarianMT pt ↔ en — local, free)
        │
   Generation — 8 parallel candidates
   ├─ ollama        Fine-tuned GGUF model (local, zero cost) ← recommended locally
   ├─ claude        Claude Haiku (used on HF Space)
   └─ hf_inference  HF Inference API (public models, free)
        │
   EE Scoring
   ├─ Respondibilidade  BM25 + cosine re-ranking over 919 papers
   ├─ Tratabilidade     Ridge classifier on MiniLM embeddings (local)
   └─ Não-trivialidade  Semantic dissimilarity probe
        │
   Stage 1 Filter  EE(candidate) > EE(original) + ε
        │
   Best candidate → SQLite (local) + HF Dataset (cross-session log)
        │
   User feedback (👍/👎) → DPO training pipeline
```

---

## Models

| Model | HuggingFace | Description |
|-------|-------------|-------------|
| Generator | [fmr34/reformulatee-reformulator-merged](https://huggingface.co/fmr34/reformulatee-reformulator-merged) | Qwen2.5-1.5B fine-tuned with DPO |
| LoRA adapter | [fmr34/reformulatee-reformulator](https://huggingface.co/fmr34/reformulatee-reformulator) | Adapter only (17 MB) |
| Translation pt→en | Helsinki-NLP/opus-mt-ROMANCE-en | MarianMT local inference |
| Translation en→pt | Helsinki-NLP/opus-mt-en-ROMANCE | MarianMT local inference |
| Embeddings | all-MiniLM-L6-v2 | Sentence embeddings for scoring |

---

## Installation

### Requirements

- Python 3.9+
- ~1 GB disk (models downloaded on first run)

### Dependencies

```bash
pip install -e .                     # core
pip install -e ".[training]"         # + DPO fine-tuning (trl, peft)
pip install -e ".[dev]"              # + development tools
```

---

## Fine-tuning

The generator was fine-tuned using **DPO (Direct Preference Optimization)** on curated chosen/rejected pairs from multiple sources: adversarial probes, cross-domain pairs, Batch API expansion, and user feedback (👍).

To replicate or extend the dataset:
1. `python -m src.dataset.prepare_dpo` — consolidate all sources into `data/rl/dpo_final.jsonl`
2. Open `notebooks/dpo_finetune_colab.ipynb` in Google Colab (free T4 GPU)
3. Upload `data/rl/dpo_final.jsonl`, run training (~45 min)
4. Publish to HF Hub and update `OLLAMA_MODEL` / `HF_MODEL` in `.env`

**DPO data sources (in priority order):**
- `dpo_tier3.jsonl` — adversarial cross-domain pairs
- `dpo_tier2.jsonl` — adversarial validated pairs
- `dpo_tier1.jsonl` — curated base pairs
- `batch_pairs.jsonl` / `batch_domains.jsonl` / `batch_large.jsonl` — API-expanded pairs
- `historico.db` + HF Dataset — user feedback (👍 = chosen, worst candidate = rejected)

---

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — component breakdown and data flow
- [Deployment](docs/DEPLOYMENT.md) — HF Spaces, Docker, local setup
- [Contributing](docs/CONTRIBUTING.md) — contribution guidelines

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{reformulatee_2025,
  title   = {ReformulatEE: Epistemic Effectiveness Reformulation},
  author  = {fmr34},
  year    = {2025},
  url     = {https://github.com/fmr34/ReformulatEE},
  license = {Apache-2.0}
}
```
