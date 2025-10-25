param(
  [switch]$RecreateVenv,
  [switch]$SkipBackendDeps,
  [switch]$SkipSenderDeps
)

$ErrorActionPreference = 'Stop'

function Info($msg){ Write-Host "[bootstrap] $msg" -ForegroundColor Cyan }
function Warn($msg){ Write-Host "[bootstrap] $msg" -ForegroundColor Yellow }
function Fail($msg){ Write-Host "[bootstrap] $msg" -ForegroundColor Red; exit 1 }

if (-not (Test-Path -Path 'backend/requirements.txt')) {
  Fail 'Run from the repo root (backend/requirements.txt missing).'
}

$python = 'python'
try { & $python -V | Out-Null } catch { Fail 'Python not found in PATH.' }

if ($RecreateVenv -and (Test-Path '.venv')) {
  Info 'Removing existing .venv (RecreateVenv)'
  Remove-Item -Recurse -Force .venv
}

if (-not (Test-Path '.venv')) {
  Info 'Creating virtual environment (.venv)'
  & $python -m venv .venv
}

$venvPy = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (-not (Test-Path $venvPy)) { Fail 'Failed to create venv (missing .venv/Scripts/python.exe).' }

Info 'Upgrading pip and wheel'
& $venvPy -m pip install -U pip wheel

if (-not $SkipBackendDeps) {
  Info 'Installing backend requirements (with OS markers)'
  & $venvPy -m pip install -r backend/requirements.txt
}

if (-not $SkipSenderDeps) {
  Info 'Installing sender requirements'
  & $venvPy -m pip install -r scripts/requirements-sender.txt
}

Info 'Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1'

