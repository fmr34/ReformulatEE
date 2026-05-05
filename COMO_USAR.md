# ReformulatEE — Guia de Uso

Sistema de reformulação de perguntas de pesquisa para maior **Efetividade Epistêmica (EE)**.  
Transforma perguntas vagas em perguntas operacionalizáveis, metodologicamente fundamentadas e respondíveis.

---

## Pré-requisitos

- Python 3.11+ com o ambiente virtual `.venv` já configurado
- Chave de API Claude configurada no `.env` (`ANTHROPIC_API_KEY`)
- Conexão com a internet (para chamadas à API)

---

## Opção 1 — Interface visual (recomendado)

### 1. Abrir o Jupyter

No terminal, dentro da pasta do projeto:

```powershell
.venv\Scripts\jupyter.exe notebook reformulatee_interface.ipynb
```

O navegador abrirá automaticamente. Caso não abra, acesse `http://localhost:8888`.

### 2. Selecionar o kernel correto

No menu do notebook: **Kernel → Change Kernel → ReformulatEE (.venv)**

### 3. Executar as células

**Kernel → Restart Kernel and Run All Cells**

Aguarde a mensagem da interface Gradio aparecer (alguns segundos).

### 4. Usar a interface

| Campo | Descrição |
|-------|-----------|
| **Pergunta de pesquisa** | Digite sua pergunta em português ou inglês |
| **Idioma** | Selecione `Português` para tradução automática pt↔en |
| **Reformular** | Clica para processar (primeira vez demora ~10s para carregar os modelos) |

**O que aparece no resultado:**
- **Pergunta original** — sua entrada (e a tradução para inglês, se aplicável)
- **Reformulação epistêmica** — a melhor reformulação encontrada
- **EE antes / EE depois** — barras visuais de Efetividade Epistêmica (0 a 1)
- **Ganho** — percentual de melhora
- **Filtro Stage 1** — ✅ PASS = a reformulação é genuinamente melhor que a original
- **Ver N candidatos** — expande para ver todas as reformulações geradas e rankeadas

> **Dica:** use os **Exemplos** na parte inferior para testar rapidamente sem digitar.

---

## Opção 2 — Linha de comando

### Pergunta única (português)

```powershell
.venv\Scripts\python -m src.rl.inference --pt "O que causa o envelhecimento biológico?"
```

### Pergunta única (inglês)

```powershell
.venv\Scripts\python -m src.rl.inference "Does consciousness arise from physical processes?"
```

### Lote de perguntas

Crie um arquivo `.txt` com uma pergunta por linha (linhas com `#` são ignoradas):

```text
O que causa o envelhecimento biológico?
O livre-arbítrio existe?
# esta linha é ignorada
Qual é a natureza da consciência?
```

Execute:

```powershell
.venv\Scripts\python -m src.rl.inference --pt --batch perguntas.txt
```

Os resultados são salvos automaticamente em `perguntas.results.jsonl`.

---

## Como funciona

```
Sua pergunta (pt)
      ↓
  Tradução pt → en  (Claude Haiku)
      ↓
  Geração de 8 candidatos  (Claude Haiku, temperatura 1.1)
      ↓
  Score de cada candidato:
    EE = 0.05·Respondibilidade + 0.05·Tratabilidade + 0.90·Não-trivialidade
      ↓
  Filtro Stage 1: EE(candidato) > EE(original) + ε
      ↓
  Melhor candidato aprovado
      ↓
  Tradução en → pt  (Claude Haiku)
      ↓
Reformulação final (pt)
```

**Custo aproximado por pergunta:** ~R$ 0,05 (8 candidatos, com tradução)

---

## Interpretando os resultados

| EE | Significado |
|----|-------------|
| 0.00 – 0.39 | Pergunta vaga ou ontologicamente bloqueada ("qual a essência de X?") |
| 0.40 – 0.69 | Parcialmente tratável, metodologia limitada |
| 0.70 – 1.00 | Operacionalizável, com paradigma de pesquisa ativo |

**Stage 1 PASS** → a reformulação é genuinamente mais epistêmica que a original  
**Fallback** → nenhum candidato superou o threshold — o sistema retorna o melhor disponível

---

## Arquivos gerados

| Arquivo | Conteúdo |
|---------|----------|
| `data/rl/lote_resultados.jsonl` | Resultados do modo lote (interface) |
| `perguntas.results.jsonl` | Resultados do modo lote (CLI) |

---

## Solução de problemas

**Interface não abre / porta ocupada**  
O Gradio escolhe automaticamente uma porta livre (7860, 7861...). Verifique o output da célula para ver a URL.

**Erro de API / timeout na primeira pergunta**  
Normal na primeira chamada — os modelos de scoring estão carregando. Tente novamente.

**Resultado em inglês mesmo com idioma Português**  
Verifique se `ANTHROPIC_API_KEY` está no `.env` e se há conexão com a internet.

**Kernel "ReformulatEE (.venv)" não aparece**  
Execute no terminal:
```powershell
.venv\Scripts\python -m ipykernel install --user --name reformulatee_venv --display-name "ReformulatEE (.venv)"
```
