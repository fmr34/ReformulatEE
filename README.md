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
| **Respondibilidade** | Is there an active research corpus? | BM25 + semantic search over 30K+ papers |
| **Tratabilidade** | Can it be answered with existing tools? | Local Ridge classifier on sentence embeddings |
| **Não-trivialidade** | Is it meaningfully different from the original? | Semantic dissimilarity probe |

**Example:**

| Input | Output | EE |
|-------|--------|----|
| "What is consciousness?" | "What measurable neural correlates distinguish conscious from unconscious processing?" | 0.137 → **0.926** |
| "O que causa o envelhecimento?" | "Como a taxa de dano ao DNA mitocondrial correlaciona com biomarcadores de senescência?" | 0.153 → **0.891** |

---

## Quick Start

### Web Interface

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e .
cp .env.example .env   # add your API keys
python app.py          # opens http://localhost:7860
```

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
# Required for tractability scoring
ANTHROPIC_API_KEY=sk-ant-...

# Generation backend
INFERENCE_BACKEND=hf_inference   # hf_inference | claude | gguf | local
HF_TOKEN=hf_...                  # HuggingFace token (Inference API)
HF_MODEL=fmr34/reformulatee-reformulator-merged   # fine-tuned model

# Optional
CORPUS_DIR=data/corpus           # path to paper corpus
```

---

## Architecture

```
Gradio Web Interface
        │
   Translation (MarianMT pt ↔ en)
        │
   Generation — 8 parallel candidates
   ├─ hf_inference  HF Inference API (default, free)
   ├─ claude        Claude Haiku (fallback)
   └─ gguf          Local quantized model
        │
   EE Scoring
   ├─ Respondibilidade  BM25 + cosine re-ranking
   ├─ Tratabilidade     Ridge(α=50) on MiniLM embeddings
   └─ Não-trivialidade  Semantic dissimilarity
        │
   Stage 1 Filter  EE(candidate) > EE(original) + ε
        │
   Best candidate → Database (SQLite) → User feedback
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

The generator was fine-tuned using **DPO (Direct Preference Optimization)** on ~700 chosen/rejected pairs of research question reformulations.

To replicate:
1. `python -m src.dataset.prepare_dpo` — consolidate dataset
2. Open `notebooks/dpo_finetune_colab.ipynb` in Google Colab (free T4 GPU)
3. Upload `data/rl/dpo_final.jsonl`, run training (~45 min)
4. Publish to HF Hub and update `HF_MODEL` in `.env`

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
