$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# If a lock file exists, prefer pip-tools sync for reproducibility
if (Test-Path -Path "requirements.lock.txt") {
  Write-Host "[pip-tools] Using requirements.lock.txt to sync environment" -ForegroundColor Cyan
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install pip-tools
  python -m piptools sync requirements.lock.txt
  Write-Host "[done] Locked environment installed" -ForegroundColor Green
  # Still show key packages
  python -m pip show fastapi uvicorn pydantic numpy pandas scikit-learn scipy onnxruntime orjson | Write-Host
  exit 0
}

Write-Host "[1/7] Upgrading pip/setuptools/wheel" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "[2/7] Installing core scientific wheels (binary-only)" -ForegroundColor Cyan
python -m pip install numpy==1.26.4 --only-binary=:all:
python -m pip install pandas==2.2.1 --only-binary=:all:
python -m pip install scikit-learn==1.4.1.post1 --only-binary=:all:
python -m pip install scipy==1.12.0 --only-binary=:all:

Write-Host "[3/7] Installing ONNX Runtime" -ForegroundColor Cyan
python -m pip install onnxruntime==1.23.1 --only-binary=:all:

Write-Host "[4/7] Installing orjson (binary if available; otherwise install Rust toolchain and retry)" -ForegroundColor Cyan
$orjsonOk = $false
try {
  python -m pip install orjson==3.9.15 --only-binary=:all:
  $orjsonOk = $true
} catch {
  $orjsonOk = $false
}
if (-not $orjsonOk) {
  Write-Warning "Binary wheel for orjson not found; installing Rust toolchain for build"
  if ($IsWindows) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
      winget install --id Rustlang.Rustup -e --silent | Out-Null
    } else {
      Write-Warning "winget not found; skipping Rust install. Please install Rust manually if orjson build fails."
    }
  } else {
    try {
      bash -lc "curl -sSf https://sh.rustup.rs | sh -s -- -y" | Out-Null
      $env:PATH = "$HOME/.cargo/bin;$env:PATH"
    } catch {
      Write-Warning "Failed to auto-install Rust; proceeding to try orjson anyway."
    }
  }
  python -m pip install orjson==3.9.15
}

Write-Host "[5/7] Installing backend core dependencies (compatible versions)" -ForegroundColor Cyan
python -m pip install fastapi==0.110.0 uvicorn==0.27.0 "pydantic>=2,<3" "pydantic-settings>=2,<3" python-dotenv==1.0.1
# Common runtime deps used by backend/inference
python -m pip install httpx==0.27.0 redis==5.0.1 motor==3.5.1 pymongo==4.6.3 prometheus-client==0.20.0 pyjwt==2.8.0 tenacity==8.2.3 psutil==5.9.8

Write-Host "[6/7] Verifying key packages" -ForegroundColor Cyan
python -m pip show fastapi uvicorn pydantic numpy pandas scikit-learn scipy onnxruntime orjson | Write-Host

Write-Host "[7/7] Environment setup complete." -ForegroundColor Green
