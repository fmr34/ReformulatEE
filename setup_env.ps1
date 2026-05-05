# LLM Quest - Environment Setup
# Run with: powershell -ExecutionPolicy Bypass -File setup_env.ps1

$ErrorActionPreference = "Stop"
$PROJECT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV_DIR = Join-Path $PROJECT_DIR ".venv"
$PYTHON_VERSION = "3.11"

Write-Host "`n=== LLM Quest - Setup de Ambiente ===" -ForegroundColor Cyan

# 1. Python 3.11
Write-Host "`n[1/5] Verificando Python $PYTHON_VERSION..." -ForegroundColor Yellow

$py311 = py -3.11 --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python 3.11 nao encontrado. Instalando via py launcher..." -ForegroundColor Yellow
    py install 3.11
    $py311 = py -3.11 --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERRO: Falha ao instalar Python 3.11. Instale manualmente em https://python.org" -ForegroundColor Red
        exit 1
    }
}
Write-Host "OK: $py311" -ForegroundColor Green

# 2. Criar venv
Write-Host "`n[2/5] Criando ambiente virtual em .venv..." -ForegroundColor Yellow

if (Test-Path $VENV_DIR) {
    Write-Host "  .venv ja existe - recriando..." -ForegroundColor DarkYellow
    Remove-Item $VENV_DIR -Recurse -Force
}

py -3.11 -m venv $VENV_DIR
Write-Host "OK: venv criado" -ForegroundColor Green

$PIP = Join-Path $VENV_DIR "Scripts\pip.exe"
$PYTHON = Join-Path $VENV_DIR "Scripts\python.exe"

# 3. Atualizar pip + wheel
Write-Host "`n[3/5] Atualizando pip..." -ForegroundColor Yellow
& $PIP install --upgrade pip wheel setuptools --quiet
Write-Host "OK" -ForegroundColor Green

# 4. Instalar dependencias
Write-Host "`n[4/5] Instalando dependencias..." -ForegroundColor Yellow

# PyTorch CPU (base - DirectML adicionado depois se Arc estiver OK)
Write-Host "  -> torch (CPU)..."
& $PIP install torch --index-url https://download.pytorch.org/whl/cpu --quiet

# Core ML
Write-Host "  -> transformers, sentence-transformers..."
& $PIP install transformers sentence-transformers --quiet

# RAG pipeline (Fase 0)
Write-Host "  -> rank_bm25, faiss-cpu..."
& $PIP install rank_bm25 faiss-cpu --quiet

# Cross-encoder (Respondibilidade)
Write-Host "  -> cross-encoder via sentence-transformers (ja incluido)..."

# Evaluation
Write-Host "  -> scikit-learn, numpy, pandas..."
& $PIP install scikit-learn numpy pandas --quiet

# Corpus access
Write-Host "  -> arxiv, semanticscholar, requests..."
& $PIP install arxiv semanticscholar requests --quiet

# Anthropic API (Tratabilidade stub)
Write-Host "  -> anthropic SDK..."
& $PIP install anthropic --quiet

# Utilities
Write-Host "  -> python-dotenv, tqdm, rich..."
& $PIP install python-dotenv tqdm rich --quiet

Write-Host "OK: todas as dependencias instaladas" -ForegroundColor Green

# 5. Verificacao final
Write-Host "`n[5/5] Verificando instalacao..." -ForegroundColor Yellow

$VERIFY_SCRIPT = Join-Path $PROJECT_DIR "verify_env.py"
& $PYTHON $VERIFY_SCRIPT

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Setup concluido com sucesso! ===" -ForegroundColor Green
    Write-Host "`nPara ativar o ambiente:" -ForegroundColor Cyan
    Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "`nProximo passo: configurar .env com ANTHROPIC_API_KEY" -ForegroundColor Cyan
} else {
    Write-Host "`nSetup concluido com erros - verifique os itens [FAIL] acima." -ForegroundColor Red
    exit 1
}
