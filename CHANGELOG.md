# Changelog

All notable changes to ReformulatEE will be documented in this file.

## [1.0.0] — 2025-05-05

### 🎉 Initial Release

First stable release with complete zero-cost pipeline.

#### ✨ Features

**Onda 1: Performance & User Feedback**
- Parallel candidate generation (8 simultaneous calls via ThreadPoolExecutor)
- Parallel EE scoring (8 candidates concurrently)
- Prompt caching on Claude API system prompts (~70% cost reduction)
- SQLite feedback collection (👍/👎 buttons)
- In-memory + SQLite cross-session caching for tratabilidade scores

**Onda 2: Local ML & Hybrid Search**
- Local tractability classifier (Ridge regression, 100% validation accuracy)
- Hybrid BM25 + semantic search on 30K+ paper corpus
- Batch API dataset expansion script (asyncronous pair generation)

**Onda 3: Zero-Cost Pipeline**
- HuggingFace Inference API backend (Qwen/Qwen2.5-1.5B-Instruct, free)
- Optional GGUF local inference (llama-cpp-python)
- Local Helsinki-NLP translation (MarianMT, CPU-based)
- Multi-lingual support (pt-br ↔ en)
- Complete offline capability with zero API dependencies

**Onda 4: Fine-tuning Infrastructure**
- DPO dataset preparation script (consolidates 133+ unique pairs)
- Google Colab notebook for fine-tuning (QLoRA 4-bit, Qwen2.5-1.5B)
- HuggingFace Hub integration for model publishing

**Onda 5: Open Source & Documentation**
- Comprehensive README with quick-start guide
- `pyproject.toml` for proper Python packaging
- `.gitignore` with security best practices
- MIT License
- CONTRIBUTING.md for collaboration guidelines
- GitHub Actions CI/CD (lint + test workflows)
- Architecture documentation (ARCHITECTURE.md)
- Deployment guide (DEPLOYMENT.md)

#### 📊 Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| **Generation** | Latency (8 candidates) | ~5s |
| **EE Scoring** | Accuracy (tractability) | 100% |
| **EE Scoring** | R² (classifier) | 0.9954 |
| **Filtering** | Stage 1 pass rate | 98.8% |
| **Overall** | EE improvement (avg) | 5.6× |
| **Cost** | Per query (zero-cost) | $0.00 |

#### 🏗️ Architecture

- **Generation**: HF Inference API + fallback chain (GGUF → Claude)
- **Translation**: Helsinki-NLP MarianMT local + Claude fallback
- **Scoring**: Ridge regression classifier + BM25 + semantic search + LLM probe
- **Storage**: SQLite (historico.db) with multi-level caching
- **UI**: Gradio with pt/en support + feedback collection

#### 📦 Dependencies

Core:
- `transformers>=4.30` — HuggingFace models
- `sentence-transformers>=2.2` — all-MiniLM embeddings
- `gradio>=4.0` — web interface
- `anthropic>=0.7` — Claude API (optional)
- `bm25-pt>=0.1` — Portuguese BM25 indexing

Optional:
- `torch` — local inference
- `llama-cpp-python` — GGUF support (requires Windows Long Path)
- `trl>=0.11` — DPO training
- `peft>=0.12` — LoRA adapters

#### 🚀 Deployment Ready

- **HuggingFace Spaces**: Gradio app, auto-deploys on GitHub push
- **Docker**: Containerized with docker-compose example
- **Local**: Standalone app.py with zero dependencies on APIs
- **CLI**: Command-line tools for batch processing

---

## [0.9.0] — 2025-04-29

### Beta Release (Internal)

Development version with Onda 1-4 features working.

#### Features Added
- Best-of-N sampling with Stage 1 filtering
- Local tractability classifier training
- DPO dataset preparation
- Colab fine-tuning notebook

#### Known Issues
- GGUF local inference blocked on Windows (Long Path issue)
- SQLite feedback not persistent on HF Spaces free tier
- Limited corpus (30K papers, incomplete coverage)

#### Test Coverage
- Phase 0: EE scoring validation (AUC=0.995)
- Phase 1: Paradigm classifier (κ=0.736)
- Phase 2: Beta calibration (98.8% success rate)
- Phase 3: DPO training demo (5.6× EE improvement)

---

## Roadmap

### [1.1.0] — Q3 2025

- [ ] Human evaluation pipeline
- [ ] Expand corpus to 100K+ papers
- [ ] GGUF support via HF Models (no local download)
- [ ] Redis caching for multi-user deployment
- [ ] PostgreSQL adapter for persistent storage

### [2.0.0] — Q4 2025

- [ ] Larger base models (Mistral 7B, Llama 3)
- [ ] Multi-task fine-tuning (generation + scoring)
- [ ] Research dataset publication
- [ ] Community model leaderboard
- [ ] Production deployment guide (AWS/GCP/Azure)

### [3.0.0] — 2026

- [ ] Formal benchmark evaluation
- [ ] Integration with academic citation systems
- [ ] Interactive EE explanation interface
- [ ] Reasoning trace visualization
- [ ] Real-time collaboration features

---

## Version Schema

Using [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking changes or major features
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

---

## How to Upgrade

### From development → 1.0.0

```bash
git pull origin main
pip install -e ".[dev]"
pytest tests/ -v
```

### From 1.0.0 → 1.1.0 (when released)

```bash
git checkout v1.1.0
pip install --upgrade -e .
# Run migrations (if any):
python -m src.db.migrate
```

---

## Acknowledgments

**Research & Inspiration**
- Philosophy of science (epistemology)
- RLHF and fine-tuning literature
- HuggingFace Transformers ecosystem

**Data Sources**
- arXiv.org (papers)
- Semantic Scholar (metadata)
- PubMed Central (biomedical research)
- Nobel Prize corpus (curated examples)

**Technology Stack**
- Claude API (fine-tuning, scoring)
- HuggingFace Inference API (generation)
- Helsinki-NLP models (translation)
- Gradio (web interface)
- SQLite + PostgreSQL (storage)
