# 🔬 ReformulatEE — Reformulação de Efetividade Epistêmica

[![Demo ao Vivo](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/fmr34/reformulatee)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Licença](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/fmr34/ReformulatEE/actions/workflows/test.yml/badge.svg)](https://github.com/fmr34/ReformulatEE/actions)

ReformulatEE transforma perguntas de pesquisa vagas em hipóteses concretas e testáveis, maximizando a **Efetividade Epistêmica (EE)** — uma pontuação composta que mede o quão operacionalizável, tratável e não-trivial uma pergunta é.

> **[Experimente ao vivo →](https://huggingface.co/spaces/fmr34/reformulatee)**

---

## Como Funciona

```
Pergunta de entrada  →  Gera 8 candidatos  →  Pontua cada um (EE)  →  Filtra  →  Melhor saída
```

**EE(Q) = 0,05 · Respondibilidade + 0,05 · Tratabilidade + 0,90 · Não-trivialidade**

| Componente | Descrição | Método |
|------------|-----------|--------|
| **Respondibilidade** | Existe um corpus de pesquisa ativo? | BM25 + busca semântica sobre 30K+ artigos |
| **Tratabilidade** | Pode ser respondida com ferramentas existentes? | Classificador Ridge local sobre embeddings de sentenças |
| **Não-trivialidade** | É significativamente diferente da original? | Sonda de dissimilaridade semântica |

**Exemplos:**

| Entrada | Saída | EE |
|---------|-------|----|
| "O que é consciência?" | "Quais correlatos neurais mensuráveis distinguem o processamento consciente do inconsciente?" | 0,137 → **0,926** |
| "O que causa o envelhecimento?" | "Como a taxa de dano ao DNA mitocondrial se correlaciona com biomarcadores de senescência?" | 0,153 → **0,891** |

---

## Início Rápido

### Interface Web

```bash
git clone https://github.com/fmr34/ReformulatEE.git
cd ReformulatEE
pip install -e .
cp .env.example .env   # adicione suas chaves de API
python app.py          # abre http://localhost:7860
```

### CLI

```bash
# Português
python -m src.rl.inference --pt "O que é a consciência?"

# Inglês
python -m src.rl.inference "Does free will exist?"

# Lote
python -m src.rl.inference --pt --batch perguntas.txt
```

---

## Configuração

Copie `.env.example` para `.env` e defina:

```bash
# Necessário para pontuação de tratabilidade
ANTHROPIC_API_KEY=sk-ant-...

# Backend de geração
INFERENCE_BACKEND=hf_inference   # hf_inference | claude | gguf | local
HF_TOKEN=hf_...                  # Token HuggingFace (Inference API)
HF_MODEL=fmr34/reformulatee-reformulator-merged   # modelo fine-tuned

# Opcional
CORPUS_DIR=data/corpus           # caminho para o corpus de artigos
```

---

## Arquitetura

```
Interface Web Gradio
        │
   Tradução (MarianMT pt ↔ en)
        │
   Geração — 8 candidatos paralelos
   ├─ hf_inference  HF Inference API (padrão, gratuito)
   ├─ claude        Claude Haiku (fallback)
   └─ gguf          Modelo quantizado local
        │
   Pontuação EE
   ├─ Respondibilidade  BM25 + re-ranking cossenoidal
   ├─ Tratabilidade     Ridge(α=50) sobre embeddings MiniLM
   └─ Não-trivialidade  Dissimilaridade semântica
        │
   Filtro Stage 1  EE(candidato) > EE(original) + ε
        │
   Melhor candidato → Banco de dados (SQLite) → Feedback do usuário
```

---

## Modelos

| Modelo | HuggingFace | Descrição |
|--------|-------------|-----------|
| Gerador | [fmr34/reformulatee-reformulator-merged](https://huggingface.co/fmr34/reformulatee-reformulator-merged) | Qwen2.5-1.5B fine-tuned com DPO |
| Adaptador LoRA | [fmr34/reformulatee-reformulator](https://huggingface.co/fmr34/reformulatee-reformulator) | Apenas o adaptador (17 MB) |
| Tradução pt→en | Helsinki-NLP/opus-mt-ROMANCE-en | Inferência local MarianMT |
| Tradução en→pt | Helsinki-NLP/opus-mt-en-ROMANCE | Inferência local MarianMT |
| Embeddings | all-MiniLM-L6-v2 | Embeddings de sentenças para pontuação |

---

## Instalação

### Requisitos

- Python 3.9+
- ~1 GB em disco (modelos baixados na primeira execução)

### Dependências

```bash
pip install -e .                     # núcleo
pip install -e ".[training]"         # + fine-tuning DPO (trl, peft)
pip install -e ".[dev]"              # + ferramentas de desenvolvimento
```

---

## Fine-tuning

O gerador foi ajustado com **DPO (Direct Preference Optimization)** sobre ~700 pares escolhidos/rejeitados de reformulações de perguntas de pesquisa.

Para replicar:
1. `python -m src.dataset.prepare_dpo` — consolidar dataset
2. Abra `notebooks/dpo_finetune_colab.ipynb` no Google Colab (GPU T4 gratuita)
3. Faça upload de `data/rl/dpo_final.jsonl`, execute o treinamento (~45 min)
4. Publique no HF Hub e atualize `HF_MODEL` no `.env`

---

## Documentação

- [Arquitetura](docs/ARCHITECTURE.md) — componentes e fluxo de dados
- [Implantação](docs/DEPLOYMENT.md) — HF Spaces, Docker, configuração local
- [Contribuição](docs/CONTRIBUTING.md) — diretrizes de contribuição

---

## Licença

Apache License 2.0 — veja [LICENSE](LICENSE)

## Citação

```bibtex
@software{reformulatee_2025,
  title   = {ReformulatEE: Epistemic Effectiveness Reformulation},
  author  = {fmr34},
  year    = {2025},
  url     = {https://github.com/fmr34/ReformulatEE},
  license = {Apache-2.0}
}
```
