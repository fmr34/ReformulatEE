# ReformulatEE — Deployment Guide

## HuggingFace Spaces (Recommended)

The live demo runs at [huggingface.co/spaces/fmr34/reformulatee](https://huggingface.co/spaces/fmr34/reformulatee).

To deploy your own instance:

1. Fork the repository on GitHub.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space), select **Gradio**, and link it to your fork.
3. In Space **Settings → Secrets**, add:
   - `ANTHROPIC_API_KEY` — **required** — Claude Haiku is the generation backend on HF Space
   - `HF_TOKEN` — **required** — write-access token for logging queries to HF Dataset (cross-session history)
4. Restart the Space. It will auto-deploy on every push to `main`.

**Notes:**
- The app automatically sets `INFERENCE_BACKEND=claude` when `SPACE_ID` is detected.
- Do **not** add `INFERENCE_BACKEND` or `HF_MODEL` as secrets — they are unused on HF Space and add unnecessary surface area.
- SQLite history is ephemeral and resets on Space restart; the HF Dataset log (`fmr34/reformulatee-logs`) is the persistent cross-session store.
- The free CPU tier supports ~5 concurrent users (limited by `concurrency_limit=5` and `queue(max_size=20)`).
- Rate limiting: 10 requests per minute per session, enforced server-side.

---

## Local Setup (Zero Cost)

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e .
cp .env.example .env   # fill in API keys
```

Install and start [Ollama](https://ollama.com), then create the fine-tuned model:
```bash
# Download Modelfile + GGUF from HF Hub (fmr34/reformulatee-reformulator-merged)
ollama create reformulatee -f Modelfile
```

Set in `.env`:
```bash
INFERENCE_BACKEND=ollama
OLLAMA_MODEL=reformulatee
ANTHROPIC_API_KEY=sk-ant-...   # still needed for tractability fallback
```

Run the app:
```bash
python app.py   # opens http://localhost:7860
```

Local mode cost: **$0 per request** (Ollama + MarianMT + Ridge classifier, all local).

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
EXPOSE 7860
CMD ["python", "app.py"]
```

```bash
docker build -t reformulatee .
docker run -p 7860:7860 --env-file .env reformulatee
```

### Docker Compose (with PostgreSQL)

```yaml
services:
  app:
    build: .
    ports:
      - "7860:7860"
    environment:
      INFERENCE_BACKEND: ollama
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: reformulatee
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Space fails to start with `ANTHROPIC_API_KEY não configurada` | Missing required secret | Add `ANTHROPIC_API_KEY` in HF Space Settings → Secrets |
| Space fails to start with `EnvironmentError` | Missing required secret | Check both `ANTHROPIC_API_KEY` and `HF_TOKEN` are set |
| High latency (>20s) | MarianMT loading on first request | Expected on cold start (~600 MB models); subsequent requests are fast |
| Out of memory | MarianMT + embeddings loaded together | Upgrade Space hardware tier or reduce `N_CANDIDATES` |
| Empty "Últimas perguntas" after Space restart | SQLite reset, HF Dataset also empty | Normal on first deploy; questions appear after first use |
| `ollama: connection refused` locally | Ollama not running | Run `ollama serve` before starting the app |
| `ollama: model not found` | Model not created | Run `ollama create reformulatee -f Modelfile` |
| Local tractability always 0.5 | `classifier.pkl` not found | Train with `python -m src.classifier.train_tractability` |
