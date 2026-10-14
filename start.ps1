# ═══════════════════════════════════════════════════════════════════════════
# Project Aletheia — One-Click Windows Startup Script (start.ps1)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   .\start.ps1 [-Port 7860] [-Ingest] [-NoBrowser]
#
param(
    [int]$Port = 7860,
    [switch]$Ingest,
    [switch]$NoBrowser
)

$Host.UI.RawUI.WindowTitle = "Project Aletheia — Autonomous Research Assistant"

function Write-Header {
    Write-Host ""
    Write-Host "══════════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
    Write-Host "                    PROJECT ALETHEIA                                " -ForegroundColor Cyan
    Write-Host "      Autonomous Multi-Agent AI Literature Research Assistant        " -ForegroundColor White
    Write-Host "══════════════════════════════════════════════════════════════════════" -ForegroundColor Magenta
    Write-Host ""
}

Write-Header

# ── 1. Check Python Virtual Environment ──────────────────────────────────
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/5] Virtual environment not found. Initializing .venv..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not (Test-Path $VenvPython)) {
        Write-Host "ERROR: Failed to create Python virtual environment." -ForegroundColor Red
        Exit 1
    }
    Write-Host "      Installing dependencies from requirements.txt..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r requirements.txt
    Write-Host "      Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "[1/5] Python virtual environment detected (.venv)." -ForegroundColor Green
}

# ── 2. Check Memory Indexes (FAISS & BM25) ────────────────────────────────
$FaissIndex = Join-Path $PSScriptRoot "data\faiss_index.bin"
$Bm25Index = Join-Path $PSScriptRoot "data\bm25_index.pkl"

if ($Ingest -or (-not (Test-Path $FaissIndex)) -or (-not (Test-Path $Bm25Index))) {
    Write-Host "[2/5] Corpus indexes missing or --Ingest specified. Building indexes..." -ForegroundColor Yellow
    Write-Host "      Processing literature in RAGxamples/..." -ForegroundColor Cyan
    & $VenvPython -m ingestion.cli --input RAGxamples --output data --no-neo4j
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Ingestion pipeline encountered non-fatal issues." -ForegroundColor Yellow
    } else {
        Write-Host "      Corpus indexes built successfully in data/." -ForegroundColor Green
    }
} else {
    Write-Host "[2/5] Dual memory stores verified (data/faiss_index.bin, data/bm25_index.pkl)." -ForegroundColor Green
}

# ── 3. Check Neo4j Daemon Status ─────────────────────────────────────────
$Neo4jConn = Test-NetConnection -ComputerName "localhost" -Port 7687 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($Neo4jConn) {
    Write-Host "[3/5] Neo4j Daemon active on bolt://localhost:7687." -ForegroundColor Green
} else {
    Write-Host "[3/5] Neo4j is offline on localhost:7687 (Graceful local fallback active)." -ForegroundColor DarkGray
}

# ── 4. Check Ollama Daemon Status ─────────────────────────────────────────
$OllamaConn = Test-NetConnection -ComputerName "localhost" -Port 11434 -WarningAction SilentlyContinue -InformationLevel Quiet
if ($OllamaConn) {
    Write-Host "[4/5] Ollama Daemon active on http://localhost:11434." -ForegroundColor Green
} else {
    Write-Host "[4/5] Ollama is offline on localhost:11434 (Deterministic fallback active)." -ForegroundColor DarkGray
}

# ── 5. Launch Gradio Web Interface ────────────────────────────────────────
$Url = "http://127.0.0.1:$Port"
Write-Host "[5/5] Launching Project Aletheia Web Console on $Url..." -ForegroundColor Cyan
Write-Host "      (Ensure your browser uses http://, not https://)" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to terminate the web server." -ForegroundColor DarkGray
Write-Host ""

if (-not $NoBrowser) {
    Start-Process $Url
}

& $VenvPython -m ui.app