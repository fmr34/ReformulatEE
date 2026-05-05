---
title: ReformulatEE
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
license: mit
short_description: Reformulate research questions with maximum epistemic effectiveness
---

# 🔬 ReformulatEE — Epistemic Effectiveness Reformulation

[![Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/run-on-spaces-sm.svg)](https://huggingface.co/spaces)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning system that **reformulates research questions** to maximize their **epistemic effectiveness** — making them more operationalizable, methodologically tractable, and non-trivial.

## 🎯 What is Epistemic Effectiveness?

Research questions vary in their tractability. A question is **epistemically tractable** if it can be answered using existing methodologies and tools. ReformulatEE learns to transform vague, philosophical questions into concrete, testable ones.

**EE(Q) = 0.05·Respondibilidade + 0.05·Tratabilidade + 0.90·Não-trivialidade**

| Input | Output | EE Score |
|---|---|---|
| "What is the meaning of life?" | "What psychological processes drive meaning attribution across cultures?" | ↑ 0.87 |
| "Is consciousness fundamental?" | "Which neural signatures correlate with subjective report accuracy?" | ↑ 0.79 |

## ✨ Key Features

- **Zero-cost inference** — Uses HuggingFace Inference API (free) + local Helsinki-NLP translation
- **Local tractability classifier** — Ridge regression on sentence embeddings (no API calls)
- **Hybrid semantic search** — BM25 + cosine similarity re-ranking across 30K+ papers
- **Multi-lingual support** — Portuguese ↔ English via MarianMT
- **Best-of-N sampling** — Generates 8 candidates, scores, filters Stage 1, returns top
- **User feedback loop** — 👍/👎 ratings stored locally in SQLite
- **Portfolio-ready** — Deploy on HuggingFace Spaces, GitHub, zero maintenance cost

## 🚀 Quick Start

### Local Installation

```bash
git clone https://github.com/yourusername/reformulatee.git
cd reformulatee
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```

### Web Interface (Gradio)

```bash
python app.py
# Opens http://localhost:7860
```

### CLI Usage

```bash
# English
python -m src.rl.inference "What are negative probabilities?"

# Portuguese
python -m src.rl.inference --pt "O que é a consciência?"

# Batch mode
python -m src.rl.inference --batch questions.txt
```

### Docker (HF Spaces)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["python", "app.py"]
```

## 📊 Results

Trained on **~150 high-quality DPO pairs** from curated research questions:

| Metric | Value |
|---|---|
| **EE Improvement (avg)** | 5.6× |
| **Stage 1 Filter Pass Rate** | 98.8% |
| **Tractability Classifier Accuracy** | 100% (val set) |
| **Semantic Search NDCG@10** | 0.72 |
| **Model Size (LoRA)** | 3 M params / 45 MB |

## 🏗️ Architecture

```
┌─ Web Interface (Gradio)
├─ Reformulation Pipeline
│  ├─ Generation (HF Inference API / GGUF local)
│  ├─ EE Scoring
│  │  ├─ Respondibilidade (BM25 + semantic search)
│  │  ├─ Tratabilidade (local Ridge classifier)
│  │  └─ Não-trivialidade (LLM semantic probe)
│  └─ Stage 1 Filter + best-of-N selection
├─ Multi-lingual Translation (Helsinki-NLP MarianMT)
└─ Database (SQLite historico.db)
   ├─ Query history + 8 candidates + best chosen
   └─ Feedback (👍/👎) for future training
```

## 📦 Dependencies

### Core
- `transformers>=4.30` — HuggingFace models
- `gradio>=4.0` — Web UI
- `anthropic>=0.7` — Claude API (optional, for fine-tuning)
- `sentence-transformers>=2.2` — all-MiniLM embeddings

### Optional
- `torch` — if using local GGUF inference
- `llama-cpp-python` — GGUF models (requires Windows Long Paths enabled)
- `trl>=0.11` — DPO training on Colab

See `pyproject.toml` for full list.

## 🔧 Configuration

### Environment Variables

```bash
# Generation backend: hf_inference (default) | gguf | claude | local
INFERENCE_BACKEND=hf_inference

# Translation backend: local (default) | claude
TRANSLATE_BACKEND=local

# HuggingFace model for generation (if using fine-tuned)
HF_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Optional: Claude API for fallback
ANTHROPIC_API_KEY=sk-ant-...

# Corpus for respondibilidade scoring
CORPUS_DIR=data/corpus
```

## 📚 Data & Training

### Available Datasets

1. **Curated Questions** (`src/eval/curated.py`) — 100 hand-picked research questions
2. **DPO Pairs** (`data/rl/dpo_final.jsonl`) — 133 chosen/rejected pairs
3. **Corpus Index** (`data/corpus/`) — 30K+ papers for semantic search

### Fine-tuning on Colab

```bash
# 1. Prepare dataset locally
python -m src.dataset.prepare_dpo

# 2. Open notebooks/dpo_finetune_colab.ipynb in Colab
# 3. Upload dpo_final.jsonl, train Qwen2.5-1.5B with DPO
# 4. Publish to HF Hub
# 5. Update .env: HF_MODEL=yourusername/reformulatee-reformulator
```

**Cost:** ~$0.003 (100 × Haiku calls for dataset generation) + free Colab GPU

## 🧪 Evaluation

### Metrics

```python
from src.ee.reward import compute_ee
from src.corpus.index import build_index

index = build_index("data/corpus")
result = compute_ee("Original question here", "Original question here", index)
print(f"EE Score: {result.ee:.3f}")
print(f"  Respondibilidade: {result.respondibilidade:.3f}")
print(f"  Tratabilidade:    {result.tratabilidade:.3f}")
print(f"  Não-trivialidade: {result.nao_trivialidade:.3f}")
```

### Run Tests

```bash
pytest tests/ -v
```

## 🌐 Deployment

### HuggingFace Spaces

Push this repo to GitHub, then connect to HF Spaces:
1. Go to https://huggingface.co/new-space
2. Select **Gradio** runtime
3. Link your GitHub repo
4. Auto-deploys on each push (zero-cost for free tier)

### Docker

```bash
docker build -t reformulatee .
docker run -p 7860:7860 reformulatee
```

### Local Server

```bash
python app.py
# Access: http://localhost:7860
```

## 📖 Documentation

- **[Architecture](docs/architecture.md)** — System design & component breakdown
- **[Training](docs/training.md)** — DPO fine-tuning pipeline (Onda 4)
- **[Dataset](docs/dataset.md)** — Curated questions, batch expansion, pair generation
- **[Evaluation](docs/evaluation.md)** — Metrics, benchmarks, human evaluation

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Quick Contribution Ideas

- [ ] Human evaluation of reformulated questions
- [ ] Additional language support (beyond pt/en)
- [ ] Expand corpus with domain-specific papers
- [ ] Improve semantic search re-ranking
- [ ] Unit tests for scoring components

## 📄 License

MIT License — see [LICENSE](LICENSE)

## 🔗 Citation

If you use ReformulatEE in research:

```bibtex
@software{reformulatee_2025,
  title={ReformulatEE: Epistemic Effectiveness Reformulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/reformulatee},
  note={Open source portfolio project}
}
```

## 💡 Roadmap

- [x] **Onda 1** — Performance (caching, parallelization, feedback loop)
- [x] **Onda 2** — Local ML (classifier, hybrid search, dataset expansion)
- [x] **Onda 3** — Zero-cost (HF Inference API + local translation)
- [x] **Onda 4** — DPO fine-tuning (Colab notebook + HF Hub)
- [x] **Onda 5** — Open source (this repo + documentation)
- [ ] **Onda 6** — HF Spaces + GitHub release + community

## 📞 Contact & Support

- **Issues** — GitHub Issues for bugs / feature requests
- **Discussions** — GitHub Discussions for Q&A
- **Email** — your.email@example.com

---

**Built with 🤖 by [Your Name]**  
*Portfolio project exploring LLM fine-tuning, RLHF, and epistemic philosophy.*
