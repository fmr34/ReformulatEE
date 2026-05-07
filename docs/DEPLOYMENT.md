# ReformulatEE — Deployment Guide

## HuggingFace Spaces (Recommended)

The live demo runs at [huggingface.co/spaces/fmr34/reformulatee](https://huggingface.co/spaces/fmr34/reformulatee).

To deploy your own instance:

1. Fork the repository on GitHub.
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space), select **Gradio**, and link it to your fork.
3. In Space **Settings → Secrets**, add:
   - `HF_TOKEN` — HuggingFace token with `inference:serverless` scope
   - `ANTHROPIC_API_KEY` — optional, used as fallback if HF Inference fails
4. Restart the Space. It will auto-deploy on every push to `main`.

**Notes:**
- SQLite history is ephemeral and resets on Space restart.
- The free tier supports ~10 concurrent users with CPU-only inference.

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
      DATABASE_URL: postgresql://user:password@db:5432/reformulatee
      INFERENCE_BACKEND: hf_inference
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

## Local Setup

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e .
cp .env.example .env   # fill in API keys
python app.py          # opens http://localhost:7860
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Space fails to start | Missing dependency | Check logs → add to `requirements.txt` |
| High latency | Local inference active | Set `INFERENCE_BACKEND=hf_inference` |
| Out of memory | MarianMT + embeddings loaded together | Increase Space hardware tier |
| Empty "Últimas perguntas" | Fresh SQLite DB | Expected; example questions shown as fallback |
| 403 from HF Inference API | Token missing `inference:serverless` scope | Re-generate token with correct scope |
