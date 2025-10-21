[CmdletBinding()]
param(
    [switch]$Gpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Stage($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Ok($msg)    { Write-Host $msg -ForegroundColor Green }
function Write-Info($msg)  { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn2($msg) { Write-Warning $msg }

Write-Info "=== Fraud_One Environment Setup ==="

# Resolve repo root relative to this script so it works from anywhere
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir
if (-not (Test-Path $Root)) { $Root = (Get-Location).Path }
Set-Location $Root

# Choose Python entry and venv paths
$venvPath = Join-Path $Root '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
$venvPip    = Join-Path $venvPath 'Scripts\pip.exe'

function Ensure-Python {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if ($py) { return 'python' }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) { return 'py -3' }
    throw 'Python not found. Please install Python 3.10+ and re-run.'
}

function In-Venv {
    if (-not (Test-Path $venvPython)) { return $false }
    return $true
}

function Run-InVenv([string[]]$args) {
    & $venvPython @args
}

function Pip-InVenv([string[]]$args) {
    & $venvPip @args
}

# 0) Create a local virtual environment if missing
if (-not (In-Venv)) {
    Write-Stage "[0/7] Creating virtual environment at .venv..."
    $pyEntry = Ensure-Python
    & $pyEntry -m venv $venvPath
}
Write-Ok "Virtual environment ready: $venvPath"

# 1) Upgrade pip, setuptools, wheel
Write-Stage "[1/7] Upgrading pip and build tools..."
Pip-InVenv @('-m','pip','install','--upgrade','pip','setuptools','wheel')

# 2) Install scientific base packages (binary wheels)
Write-Stage "[2/7] Installing core scientific packages..."
Pip-InVenv @('install',
    'numpy==1.26.4',
    'pandas==2.2.1',
    'scikit-learn==1.4.1.post1',
    'scipy==1.12.0',
    '--only-binary=:all:'
)

# 3) Install ONNX Runtime (GPU optional)
Write-Stage "[3/7] Installing ONNX Runtime..."
if ($Gpu) {
    Write-Info 'GPU flag set: installing onnxruntime-gpu'
    Pip-InVenv @('install','onnxruntime-gpu==1.23.1','--only-binary=:all:')
} else {
    Pip-InVenv @('install','onnxruntime==1.23.1','--only-binary=:all:')
}

# 4) Install orjson (binary preferred, with Rust fallback on Windows)
Write-Stage "[4/7] Installing orjson..."
$orjsonInstalled = $false
try {
    Pip-InVenv @('install','orjson==3.9.15','--only-binary=:all:')
    $orjsonInstalled = $true
} catch {
    Write-Warn2 'orjson binary wheel not available. Attempting Rust toolchain install (Windows winget if present).'
    try {
        if ($IsWindows) {
            $winget = Get-Command winget -ErrorAction SilentlyContinue
            if ($winget) {
                & winget install --id Rustlang.Rustup -e --source winget --silent
            } else {
                Write-Warn2 'winget not found; skipping Rust installation.'
            }
        }
        Pip-InVenv @('install','orjson==3.9.15')
        $orjsonInstalled = $true
    } catch {
        Write-Warn2 'Failed to install orjson; continuing without it.'
    }
}
if ($orjsonInstalled) { Write-Ok 'orjson installed.' }

# 5) Backend/runtime dependencies
Write-Stage "[5/7] Installing backend dependencies..."
Pip-InVenv @('install',
    'fastapi==0.110.0', 'uvicorn==0.27.0', 'pydantic>=2,<3', 'pydantic-settings>=2,<3', 'python-dotenv==1.0.1',
    'httpx==0.27.0', 'redis==5.0.1', 'motor==3.5.1', 'pymongo==4.6.3', 'prometheus-client==0.20.0',
    'pyjwt==2.8.0', 'tenacity==8.2.3', 'psutil==5.9.8', 'gunicorn==21.2.0'
)

# 6) Verify installation
Write-Stage "[6/7] Verifying installed packages..."
try {
    Pip-InVenv @('show','fastapi','uvicorn','pydantic','numpy','pandas','scipy','scikit-learn','onnxruntime','redis','motor') | Select-String 'Name|Version'
} catch {
    Write-Warn2 'Verification encountered an issue; some packages may be missing.'
}

# 7) Done
Write-Ok "[7/7] Environment setup complete."
Write-Host "Activate the environment: `n  .\\.venv\\Scripts\\Activate.ps1" -ForegroundColor Cyan
Write-Host "Run tests or app after activation, e.g.:`n  pytest backend\\tests -v" -ForegroundColor Cyan
