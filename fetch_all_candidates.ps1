# Busca todos os candidatos em paralelo e aguarda conclusao.
# Uso: powershell -ExecutionPolicy Bypass -File fetch_all_candidates.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
$LOG = Join-Path $ROOT "data\pairs\logs"

New-Item -ItemType Directory -Force -Path $LOG | Out-Null

Write-Host "`n=== LLM Quest — Fetch de candidatos (paralelo) ===" -ForegroundColor Cyan
Write-Host "Logs em: $LOG`n"

$jobs = @(
    @{ Name = "arxiv";            Script = "src.dataset.fetch_arxiv_pairs" },
    @{ Name = "semantic_scholar"; Script = "src.dataset.fetch_semantic_scholar_pairs" },
    @{ Name = "pubmed";           Script = "src.dataset.fetch_pubmed_pairs" },
    @{ Name = "nobel";            Script = "src.dataset.fetch_nobel_pairs" },
    @{ Name = "sep";              Script = "src.dataset.fetch_sep_pairs" }
)

$procs = @()
foreach ($job in $jobs) {
    $log = Join-Path $LOG "$($job.Name).log"
    Write-Host "  Iniciando $($job.Name)..." -ForegroundColor Yellow
    $p = Start-Process -FilePath $PY `
        -ArgumentList "-m", $job.Script `
        -WorkingDirectory $ROOT `
        -RedirectStandardOutput $log `
        -RedirectStandardError ($log -replace '\.log$', '_err.log') `
        -PassThru -NoNewWindow
    $procs += @{ Process = $p; Name = $job.Name; Log = $log }
}

Write-Host "`nAguardando conclusao de todos os fetchers..." -ForegroundColor Cyan
Write-Host "(acompanhe os logs em $LOG)`n"

$start = Get-Date
while ($true) {
    $running = $procs | Where-Object { -not $_.Process.HasExited }
    $done    = $procs | Where-Object { $_.Process.HasExited }

    foreach ($d in $done) {
        if ($d.Reported) { continue }
        $d.Reported = $true
        $exit = $d.Process.ExitCode
        $color = if ($exit -eq 0) { "Green" } else { "Red" }
        $status = if ($exit -eq 0) { "OK" } else { "ERRO (exit $exit)" }
        Write-Host "  [$($d.Name)] $status — log: $($d.Log)" -ForegroundColor $color
    }

    if ($running.Count -eq 0) { break }
    Start-Sleep -Seconds 10
}

$elapsed = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)
Write-Host "`nTodos os fetchers concluidos em $elapsed min." -ForegroundColor Green

# Conta candidatos coletados
Write-Host "`n=== Candidatos coletados ===" -ForegroundColor Cyan
$files = @(
    "arxiv_candidates.jsonl",
    "s2_candidates.jsonl",
    "pubmed_candidates.jsonl",
    "nobel_candidates.jsonl",
    "sep_candidates.jsonl"
)
$total = 0
foreach ($f in $files) {
    $path = Join-Path $ROOT "data\pairs\$f"
    if (Test-Path $path) {
        $n = (Get-Content $path | Measure-Object -Line).Lines
        $total += $n
        Write-Host ("  {0,-28} {1,5} candidatos" -f $f, $n)
    } else {
        Write-Host "  $f — nao encontrado" -ForegroundColor DarkYellow
    }
}
Write-Host ("  {0,-28} {1,5} total" -f "TOTAL", $total) -ForegroundColor Cyan

Write-Host "`nProximo passo — extrair pares com Claude:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\python -m src.dataset.extract_pairs`n"
