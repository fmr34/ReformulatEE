# LLM Quest — System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                      │
│              (Multilingual, Feedback Collection)             │
└──────────────────────┬──────────────────────────────────────┘
                       │ Portuguese ↔ English
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Translation Layer (MarianMT)                    │
│         Helsinki-NLP/opus-mt-{pt-en,en-ROMANCE}             │
└──────────────────────┬──────────────────────────────────────┘
                       │ English Research Question
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           Reformulation Pipeline (Best-of-N)                │
│                                                              │
│  1. GENERATION (8 parallel candidates)                      │
│     ├─ Backend 1: HF Inference API (Qwen2.5-1.5B) [FREE]   │
│     ├─ Backend 2: GGUF local (llama-cpp-python)             │
│     └─ Backend 3: Claude API (fallback)                     │
│                                                              │
│  2. SCORING (Epistemic Effectiveness)                       │
│     ├─ Respondibilidade (BM25 + semantic search)            │
│     ├─ Tratabilidade (Ridge classifier on embeddings)       │
│     └─ Não-trivialidade (semantic dissimilarity probe)      │
│                                                              │
│  3. FILTERING (Stage 1)                                     │
│     └─ Keep only: EE(q_cand) > EE(q_bad) + ε              │
│                                                              │
│  4. SELECTION                                               │
│     └─ Return highest-scoring candidate                     │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ English Reformulation
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Translation Layer (MarianMT)                    │
│                      (en → pt)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ Portuguese Reformulation
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Database (SQLite)                         │
│   ├─ historico: query + 8 candidates + best + feedback     │
│   ├─ cache_tratabilidade: memoized scores                  │
│   └─ cache_ee: full result memoization                     │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Core Components

### 1. Generation (`src/rl/generate_free.py`)

Produces N candidate reformulations.

**Backends:**
| Backend | Model | Speed | Cost | Memory |
|---------|-------|-------|------|--------|
| `hf_inference` | Qwen/Qwen2.5-1.5B | Fast | FREE | - |
| `gguf` | GGUF quantized | Medium | FREE | ~4 GB |
| `claude` | Claude Haiku | Fast | $0.05/1K | - |
| `local` | DPO fine-tuned PEFT | Slow | FREE | ~12 GB |

**Auto-detection logic:**
```python
if SPACE_ID env var → hf_inference  (HF Spaces detected)
elif GGUF file exists → gguf
else → hf_inference (default)
```

### 2. Translation (`src/ee/translate_local.py`)

Converts pt-br ↔ en using MarianMT.

**Models:**
- `Helsinki-NLP/opus-mt-tc-big-pt-en` (~300 MB)
- `Helsinki-NLP/opus-mt-en-ROMANCE` + `>>pt<<` prefix

**Cost:** FREE (CPU inference, 22ms per call)

**Fallback:** Claude API if `transformers` not installed

### 3. Epistemic Effectiveness Scoring (`src/ee/reward.py`)

Computes `EE(Q) = 0.05·R + 0.05·T + 0.90·NT`

#### 3a. Respondibilidade (R)

How well-established is the research area?

```python
# Uses BM25 + semantic search on corpus
from src.corpus.index import search_papers

papers = search_papers(question, top_k=50)
respondibilidade = (
    len(papers) / max_possible_papers  # normalized
)
```

- **Source:** 30K+ papers (arXiv, Semantic Scholar, PubMed, Nobel Prize corpus)
- **Speed:** ~200ms (BM25 + cosine sim re-ranking)
- **Cache:** Enabled (SQLite + in-memory)

#### 3b. Tratabilidade (T)

Can we answer this with existing tools?

```python
# Local Ridge regression on sentence embeddings
from src.classifier.tractability_local import predict_local

pred = predict_local(question)  # ∈ [0, 1]
```

- **Model:** Ridge(alpha=50.0) on all-MiniLM-L6-v2 embeddings
- **Training data:** 100 curated questions (binary labels)
- **Validation:** Accuracy=100%, RMSE CV=0.44
- **Speed:** ~22ms per call
- **Cache:** Enabled

#### 3c. Não-trivialidade (NT)

Is the reformulation significantly different from the original?

```python
# Uses semantic similarity probe (Claude API with caching)
from src.ee.nao_trivialidade import compute_nt

nt = compute_nt(q_original, q_reformulated)  # ∈ [0, 1]
```

- **Method:** Cosine distance between embeddings + semantic classification
- **Speed:** ~500ms (with prompt caching)
- **Cache:** Enabled

### 4. Stage 1 Filter (`src/ee/reward.py`)

Rejects candidates that don't improve over baseline.

```python
ε = 0.05  # threshold
passes = EE(candidate) > EE(original) + ε
```

- **Rejection rate:** ~30% at runtime
- **Fallback:** If 0 candidates pass, return highest-EE anyway

### 5. Database (`src/db/historico.py`)

Stores query history, candidates, user feedback.

**Schema:**
```sql
CREATE TABLE historico (
  id INTEGER PRIMARY KEY,
  ts TIMESTAMP,
  idioma TEXT,
  pergunta_orig TEXT,           -- original question
  pergunta_en TEXT,             -- English translation
  candidatos JSON,              -- [{"text": "...", "ee": 0.5}, ...]
  melhor TEXT,                  -- best selected
  melhor_pt TEXT,               -- best in Portuguese
  ee_antes FLOAT,               -- EE(original)
  ee_depois FLOAT,              -- EE(best)
  stage1_pass BOOLEAN,          -- passed filtering?
  feedback INTEGER              -- 1=👍, -1=👎, NULL=none
);

CREATE TABLE cache_tratabilidade (
  hash TEXT PRIMARY KEY,
  query TEXT,
  resultado JSON,               -- {"prob_tractable": 0.7, ...}
  ts TIMESTAMP
);
```

**Usage:**
```python
from src.db.historico import salvar, registrar_feedback, ultimas

record_id = salvar(pergunta_orig, candidatos, melhor, ...)
registrar_feedback(record_id, valor=1)  # 👍
history = ultimas(n=10)  # load recent queries
```

## 🔄 Data Flow Example

```
User Input: "O que é a consciência?"
    ↓
[Translate pt→en via MarianMT]
"What are the neural mechanisms of consciousness?"
    ↓
[Generate 8 candidates via HF Inference API]
{
  "candidates": [
    "What neural signatures predict conscious reports?",
    "How do synchronized neural patterns relate to awareness?",
    ...
  ]
}
    ↓
[Score each candidate via EE scoring]
{
  "candidates": [
    {"text": "...", "ee": 0.82, "resp": 0.7, "tract": 0.6, "nt": 0.85},
    ...
  ]
}
    ↓
[Stage 1 Filter: keep EE > 0.48 + 0.05 = 0.53]
    ↓
[Select best: max(score) = 0.82]
    ↓
[Translate en→pt via MarianMT]
"Quais sinais neurais predizem relatórios conscientes?"
    ↓
[Save to DB + collect feedback]
    ↓
User sees result + 👍/👎 buttons
```

## 🧠 Machine Learning Components

### Tractability Classifier

**Training:**
```bash
python -m src.classifier.train_tractability --api
```

- Trains Ridge regression on 100 curated questions
- Features: all-MiniLM-L6-v2 sentence embeddings (384-dim)
- Target: binary labels (0/1) or real scores from Claude API
- Output: `data/models/tractability/classifier.pkl`

**Evaluation:**
- R² = 0.9954
- RMSE CV = 0.4408 ± 0.3293
- Binary accuracy (threshold 0.4) = 100%

### DPO Fine-tuning (Onda 4)

**Data preparation:**
```bash
python -m src.dataset.prepare_dpo
```

Consolidates ~133 unique DPO pairs from multiple sources:
- `dpo_tier{1,2,3}.jsonl` — curated + adversarial pairs
- `batch_pairs.jsonl` — Batch API expanded pairs
- `historico.db` — user feedback pairs

**Training on Colab:**
```bash
# See notebooks/dpo_finetune_colab.ipynb
# Model: Qwen2.5-1.5B-Instruct
# Method: DPO + LoRA (4-bit QLoRA)
# Cost: FREE (Colab T4)
# Output: uploaded to HF Hub
```

## 🗄️ Caching Strategy

Three-level cache hierarchy for efficiency:

```
Level 1: In-Memory Dict
├─ Treated like: {"question_hash": score}
├─ TTL: session lifetime
└─ Speed: O(1)
          ↓
Level 2: SQLite (cross-session)
├─ Tables: cache_tratabilidade, etc.
├─ TTL: infinite (until manual clear)
└─ Speed: ~5ms
          ↓
Level 3: API (Claude with prompt caching)
├─ Type: ephemeral cache (TTL ~5 min)
├─ Savings: ~70% cost reduction
└─ Speed: ~500ms (first call), cached after
```

## ⚡ Performance Characteristics

| Operation | Speed | Cost | Notes |
|-----------|-------|------|-------|
| Generate 8 candidates | ~5s | FREE | HF Inference (parallel) |
| Translate pt→en | ~100ms | FREE | MarianMT local |
| Score 8 candidates | ~2s | FREE | classifier + semantics |
| Stage 1 Filter + select | ~50ms | FREE | local ranking |
| Translate en→pt | ~100ms | FREE | MarianMT local |
| **Total pipeline** | **~8s** | **FREE** | end-to-end |

## 🚀 Scalability & Deployment

### Single User (Local)
- Gradio app on laptop
- CPU-only (works on Intel Arc)
- Latency: ~8s per query

### Multi-User (HF Spaces)
- Gradio on free tier (ephemeral storage)
- SQLite history not persistent (use external DB for production)
- ~10 concurrent users on free tier

### Production Scale
- Docker container + load balancer
- PostgreSQL for history (replace SQLite)
- Redis for caching (replace in-memory dict)
- Async workers (celery) for parallelization
