# ReformulatEE — Deployment Guide

## 🚀 Quick Deploy to HuggingFace Spaces

### Step 1: Create GitHub Repository

```bash
git init
git add .
git commit -m "Initial commit: ReformulatEE open source"
git remote add origin https://github.com/yourusername/reformulatee.git
git branch -M main
git push -u origin main
```

### Step 2: Create HF Spaces

1. Go to **https://huggingface.co/new-space**
2. Choose **Gradio** runtime
3. Set **Repository URL** to your GitHub repo
4. Select **Public** space
5. Create space

→ Auto-deploys `app.py` on each GitHub push

### Step 3: Configure `.env` in Spaces

HF Spaces doesn't have `.env` secrets by default. Options:

#### Option A: No API keys needed (Recommended)
The app works with `INFERENCE_BACKEND=auto` (uses HF Inference API, free).

No configuration needed—just push and it works!

#### Option B: Optional Claude API
If you want fallback to Claude API for better quality:

1. In HF Spaces settings:
   - **Secrets** → Add `ANTHROPIC_API_KEY`
   - Paste your API key
2. In `app.py`, this will be automatically loaded

### Step 4: Monitor & Maintain

- **Logs** — HF Spaces dashboard shows runtime logs
- **Traffic** — Free tier has no billing, but has rate limits
- **Updates** — Auto-deploy on GitHub push
- **Storage** — SQLite database is ephemeral (resets on space restart)

For persistent storage, set up PostgreSQL connection:
```python
# src/db/historico.py — add PostgreSQL backend
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///historico.db")
```

## 🐳 Docker Deployment

### Local Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["python", "app.py"]
```

Build & run:
```bash
docker build -t reformulatee .
docker run -p 7860:7860 reformulatee
```

### Docker Compose (with PostgreSQL)

```yaml
# docker-compose.yml
version: "3.9"

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

Run:
```bash
docker-compose up
```

## ☁️ Cloud Deployment Options

### AWS Lambda + API Gateway

Create `lambda_handler.py`:
```python
from mangum import Mangum
from app import app as gradio_app

# Convert Gradio to ASGI
asgi_app = gradio_app.asgi

handler = Mangum(asgi_app)
```

Deploy with Serverless Framework:
```bash
pip install serverless-python-requirements
serverless deploy
```

### Google Cloud Run

```bash
gcloud run deploy reformulatee \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --allow-unauthenticated
```

### Azure Container Instances

```bash
az container create \
  --resource-group mygroup \
  --name reformulatee \
  --image yourusername/reformulatee:latest \
  --ports 7860 \
  --environment-variables INFERENCE_BACKEND=hf_inference
```

## 📊 Performance Tuning

### On HF Spaces Free Tier
- **Ephemeral storage** — SQLite resets on restart
- **CPU-only** — Use HF Inference API (fast enough)
- **Memory** — ~2 GB available
- **Rate limit** — ~30 req/min

### Optimization Tips

1. **Cache aggressively**
   ```bash
   # In app.py
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def _cached_score(question):
       return compute_ee(question, ...)
   ```

2. **Use CDN for assets**
   ```python
   # Gradio auto-caches CSS/JS
   # HF Spaces uses cloudflare CDN
   ```

3. **Batch API calls**
   ```bash
   # For multiple reformulations, use batch processing
   python -m src.rl.inference --batch questions.txt
   ```

## 🔒 Security

### Environment Variables
Never commit secrets:
```bash
# .env (gitignored)
ANTHROPIC_API_KEY=sk-ant-...
```

### CORS for API Usage
```python
# If serving as API
import gradio as gr

app.interface.allowed_paths = ["/file="]  # Restrict uploads
```

### Rate Limiting
```python
# Using slowapi
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@limiter.limit("30/minute")
def reformular_ui(...):
    ...
```

## 📈 Monitoring & Logging

### HF Spaces Logs
```bash
# View logs in dashboard
# Or via HF CLI:
huggingface-cli get-space-logs yourusername/reformulatee
```

### Custom Logging
```python
# In src/rl/inference.py
import logging

logger = logging.getLogger(__name__)
logger.info(f"Generated {len(candidates)} candidates")
```

### Metrics Tracking
```python
# Optional: send to wandb, datadog, etc.
import wandb

wandb.log({
    "ee_improvement": delta_ee,
    "stage1_pass_rate": (passed / total),
    "inference_time_ms": elapsed_ms,
})
```

## 🧪 Pre-Deployment Checklist

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Linting passes: `ruff check src/`
- [ ] Code formatted: `black src/`
- [ ] `.env` secrets not in git: `git status`
- [ ] README updated with usage
- [ ] CHANGELOG updated
- [ ] GitHub Actions passing (lint + test)
- [ ] Docker builds successfully
- [ ] HF Spaces connected to GitHub repo
- [ ] No hardcoded API keys in code

## 🔄 Continuous Deployment

### GitHub Actions → HF Spaces

```yaml
# .github/workflows/deploy-hf-spaces.yml
name: Deploy to HF Spaces

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Push to HF Spaces
        run: |
          git config user.email "ci@example.com"
          git config user.name "CI Bot"
          git remote add huggingface https://huggingface.co/spaces/yourusername/reformulatee
          git push huggingface main
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
```

## 🆘 Troubleshooting

### App fails to start on HF Spaces
- Check logs: **Space settings → Logs**
- Common issues:
  - Missing dependencies → add to `requirements.txt`
  - Wrong port (HF uses 7860)
  - Large model download timeout → use smaller model

### High latency
- Use HF Inference API (not local inference)
- Reduce `INFERENCE_N` from 8 → 4
- Cache more aggressively

### Out of memory on startup
- Switch from local transformer to HF Inference API
- Use `INFERENCE_BACKEND=hf_inference`

### Database grows too large
- Use PostgreSQL instead of SQLite
- Implement retention policy: `DELETE FROM historico WHERE ts < NOW() - INTERVAL 30 DAYS`

---

**Next Step:** Push to GitHub and create HF Spaces!
