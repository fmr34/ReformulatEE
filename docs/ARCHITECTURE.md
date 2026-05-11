# ReformulatEE — System Architecture

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Gradio Web Interface                            │
│   Rate limit: 10 req/min/session • Privacy notice shown     │
└──────────────────────┬──────────────────────────────────────┘
                       │ Portuguese ↔ English
                       ↓
┌─────────────────────────────────────────────────────────────┐
│              Translation Layer (MarianMT)                    │
│         Helsinki-NLP/opus-mt-{ROMANCE-en, en-ROMANCE}       │
│         Local CPU inference — zero cost                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ English Research Question
                       ↓
┌─────────────────────────────────────────────────────────────┐
│           Reformulation Pipeline (Best-of-N)                │
│                                                              │
│  1. GENERATION (8 parallel candidates)                      │
│     ├─ Backend: ollama        GGUF fine-tuned [local, FREE] │
│     ├─ Backend: claude        Claude Haiku [HF Space]       │
│     └─ Backend: hf_inference  HF Inference API [free]       │
│                                                              │
│  2. SCORING (Epistemic Effectiveness)                        │
│     ├─ Respondibilidade (BM25 + semantic search, 919 papers)│
│     ├─ Tratabilidade (Ridge classifier, local)              │
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
│                    Persistence Layer                         │
│   ├─ SQLite (local): historico + cache_tratabilidade        │
│   └─ HF Dataset (cross-session): fmr34/reformulatee-logs   │
│       └─ All queries logged; feedback merged for DPO        │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Core Components

### 1. Generation (`src/rl/generate_free.py`, `src/rl/inference.py`)

Produces N candidate reformulations. Backend is selected per environment:

**Backends:**
| Backend | Model | Speed | Cost | Used when |
|---------|-------|-------|------|-----------|
| `ollama` | Fine-tuned GGUF (reformulatee) | Fast | FREE | Local (recommended) |
| `claude` | Claude Haiku | Fast | ~$0.001/req | HF Space (auto) |
| `hf_inference` | Qwen/Qwen2.5-1.5B | Fast | FREE | Explicit config |
| `gguf` | GGUF via llama-cpp-python | Medium | FREE | Explicit config |
| `local` | DPO fine-tuned PEFT | Slow | FREE | Explicit config |

**Backend selection logic (`app.py`):**
```python
if SPACE_ID env var:
    INFERENCE_BACKEND = "claude"   # HF Space: always Claude
else:
    INFERENCE_BACKEND = "auto"     # Local: tries Ollama → Claude
```

All user inputs are wrapped in `<question>` XML tags before being sent to any model to delimit data from instructions (prompt injection mitigation).

### 2. Translation (`src/ee/translate_local.py`)

Converts pt-br ↔ en using MarianMT.

**Models:**
- `Helsinki-NLP/opus-mt-ROMANCE-en` (~300 MB, pt→en)
- `Helsinki-NLP/opus-mt-en-ROMANCE` + `>>pt<<` prefix (en→pt)

**Cost:** FREE (local CPU inference)

**Fallback:** Claude API if `transformers` not installed

### 3. Epistemic Effectiveness Scoring (`src/ee/reward.py`)

Computes `EE(Q) = 0.05·R + 0.05·T + 0.90·NT`

#### 3a. Respondibilidade (R)

How well-established is the research area?

- **Source:** 919 papers (arXiv, Semantic Scholar, PubMed, Nobel Prize corpus)
- **Method:** BM25 + cosine similarity re-ranking
- **Fallback:** If corpus missing, R = 0 (app warns but continues)
- **Speed:** ~200ms
- **Cache:** In-memory + SQLite

#### 3b. Tratabilidade (T)

Can we answer this with existing tools?

- **Primary:** Ridge(alpha=50.0) on all-MiniLM-L6-v2 embeddings (local, ~22ms, free)
- **Fallback:** Claude API with prompt caching if local classifier not trained
- **Cache:** In-memory + SQLite cross-session

#### 3c. Não-trivialidade (NT)

Is the reformulation significantly different from the original?

- **Method:** Cosine distance between sentence embeddings + semantic classification
- **Speed:** ~500ms (with prompt caching)
- **Cache:** In-memory + SQLite

### 4. Stage 1 Filter (`src/ee/reward.py`)

Rejects candidates that don't improve over baseline.

```python
ε = 0.05  # threshold
passes = EE(candidate) > EE(original) + ε
```

- **Rejection rate:** ~30% at runtime
- **Fallback:** If 0 candidates pass, return highest-EE anyway

### 5. Persistence (`src/db/historico.py`, `src/db/hf_logger.py`)

Two-layer persistence strategy:

**SQLite (local, ephemeral on HF Space):**
```sql
CREATE TABLE historico (
  id INTEGER PRIMARY KEY,
  ts TIMESTAMP,
  idioma TEXT,
  pergunta_orig TEXT,           -- original question
  pergunta_en TEXT,             -- English translation
  candidatos JSON,              -- [{"text": "...", "ee": 0.5}, ...]
  melhor TEXT,                  -- best selected (English)
  melhor_pt TEXT,               -- best in Portuguese
  ee_antes FLOAT,               -- EE(original)
  ee_depois FLOAT,              -- EE(best)
  stage1_pass BOOLEAN,          -- passed filtering?
  feedback INTEGER              -- 1=👍, -1=👎, NULL=none
);
```

**HF Dataset (`fmr34/reformulatee-logs`, cross-session):**
- Every query logged as `{"type": "record", ...}` via background thread (non-blocking)
- Feedback logged as `{"type": "feedback", "id": ..., "feedback": 1}` (urgent flush)
- `ultimas()` falls back to HF Dataset when SQLite is empty (e.g. after Space restart)
- Records validated (type/length/idioma) before display to prevent cache poisoning

**Usage:**
```python
from src.db.historico import salvar, registrar_feedback, ultimas

record_id = salvar(pergunta_orig, candidatos, melhor, ...)
registrar_feedback(record_id, valor=1)  # 👍
history = ultimas(n=10)  # SQLite → HF Dataset fallback
```

## 🔄 Data Flow Example

```
User Input: "O que é a consciência?"
    ↓
[Translate pt→en via MarianMT]
"What is consciousness?"
    ↓
[Generate 8 candidates via Claude Haiku (HF Space) / Ollama (local)]
Input wrapped: <question>What is consciousness?</question>
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
[Stage 1 Filter: keep EE > baseline + 0.05]
    ↓
[Select best: max(score)]
    ↓
[Translate en→pt via MarianMT]
"Quais sinais neurais predizem relatórios conscientes?"
    ↓
[Save to SQLite + async log to HF Dataset]
    ↓
[Audit log: {"action": "reformulate", "session": "hash...", "ee_antes": 0.15, "ee_depois": 0.89}]
    ↓
User sees result + 👍/👎 buttons
```

## 🧠 Machine Learning Components

### Tractability Classifier

**Training:**
```bash
python -m src.classifier.train_tractability --api
```

- Trains Ridge regression on curated questions
- Features: all-MiniLM-L6-v2 sentence embeddings (384-dim)
- Target: binary labels (0/1) or real scores from Claude API
- Output: `data/models/tractability/classifier.pkl`

### DPO Fine-tuning

**Data preparation:**
```bash
python -m src.dataset.prepare_dpo
```

Consolidates DPO pairs from multiple sources (in priority order):
- `dpo_tier3.jsonl` — adversarial cross-domain pairs (highest quality)
- `dpo_tier2.jsonl` — adversarial validated pairs
- `dpo_tier1.jsonl` — curated base pairs
- `batch_pairs.jsonl`, `batch_domains.jsonl`, `batch_large.jsonl` — API-expanded
- `historico.db` — local user feedback (👍)
- HF Dataset (`fmr34/reformulatee-logs`) — online user feedback (👍)

**Training on Colab:**
```bash
# See notebooks/dpo_finetune_colab.ipynb
# Model: Qwen2.5-1.5B-Instruct
# Method: DPO + LoRA (4-bit QLoRA)
# Cost: FREE (Colab T4)
# Output: uploaded to HF Hub as GGUF
```

## 🗄️ Caching Strategy

Three-level cache hierarchy for efficiency:

```
Level 1: In-Memory Dict
├─ TTL: session lifetime
└─ Speed: O(1)
          ↓
Level 2: SQLite (cross-session, local)
├─ Tables: cache_tratabilidade
├─ TTL: infinite (until manual clear)
└─ Speed: ~5ms
          ↓
Level 3: Claude API (with prompt caching)
├─ Type: ephemeral cache (TTL ~5 min)
├─ Savings: ~70% cost reduction
└─ Speed: ~500ms (first call), cached after
```

## 🔒 Security

- **Input sanitization:** User input wrapped in `<question>` tags in all backends (prompt injection mitigation)
- **Rate limiting:** 10 requests/min per session (sliding window, in-memory)
- **Audit logging:** Structured JSON to stderr — action, timestamp, session hash (SHA-256 truncated), EE scores
- **SQLite permissions:** chmod 600 applied on every connection
- **HF Dataset records:** Validated (type, length ≤ 1000 chars, idioma whitelist) before display
- **Startup validation:** ANTHROPIC_API_KEY checked at startup on HF Space (fails fast with clear error)

## ⚡ Performance Characteristics

| Operation | Speed | Cost (HF Space) | Cost (Local) |
|-----------|-------|-----------------|--------------|
| Generate 8 candidates | ~8s | Claude API | FREE (Ollama) |
| Translate pt→en | ~100ms | FREE | FREE |
| Score 8 candidates | ~2s | FREE | FREE |
| Stage 1 Filter + select | ~50ms | FREE | FREE |
| Translate en→pt | ~100ms | FREE | FREE |
| **Total pipeline** | **~10s** | **~$0.001** | **$0** |

## 🚀 Deployment Modes

### Local (Zero Cost)
- Ollama + fine-tuned GGUF model
- MarianMT for translation
- Ridge classifier for tractability
- CPU-only (works on standard laptop)
- Latency: ~10s per query

### HF Space (Public Demo)
- Claude Haiku for generation (forced when SPACE_ID present)
- MarianMT loaded on first request (~300 MB download)
- Questions persisted to HF Dataset (cross-session, cross-user)
- SQLite ephemeral (resets on restart; HF Dataset used as fallback)

### Production Scale
- Docker container + load balancer
- PostgreSQL for history (replace SQLite)
- Redis for caching (replace in-memory dict)
- Async workers for parallelization
