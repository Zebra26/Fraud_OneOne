#!/usr/bin/env bash
set -euo pipefail

echo "[0/7] Python: $(python --version 2>/dev/null || true)"

if [[ -f requirements.lock.txt ]]; then
  echo "[pip-tools] Using requirements.lock.txt to sync environment"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install pip-tools
  pip-sync requirements.lock.txt
  python -m pip show fastapi uvicorn pydantic numpy pandas scikit-learn scipy onnxruntime orjson || true
  exit 0
fi

echo "[1/7] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

echo "[2/7] Installing core scientific wheels (binary-only)"
python -m pip install numpy==1.26.4 --only-binary=:all:
python -m pip install pandas==2.2.1 --only-binary=:all:
python -m pip install scikit-learn==1.4.1.post1 --only-binary=:all:
python -m pip install scipy==1.12.0 --only-binary=:all:

echo "[3/7] Installing ONNX Runtime"
python -m pip install onnxruntime==1.23.1 --only-binary=:all:

echo "[4/7] Installing orjson (binary preferred)"
if ! python -m pip install orjson --only-binary=:all:; then
  echo "orjson binary not available; installing Rust toolchain via rustup"
  curl -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null 2>&1 || true
  export PATH="$HOME/.cargo/bin:$PATH"
  python -m pip install orjson==3.9.15
fi

echo "[5/7] Installing backend core dependencies"
python -m pip install fastapi==0.110.0 uvicorn==0.27.0 pydantic==1.10.13 python-dotenv==1.0.1
python -m pip install httpx==0.27.0 redis==5.0.1 motor==3.5.1 prometheus-client==0.20.0 pyjwt==2.8.0 tenacity==8.2.3 psutil==5.9.8

echo "[6/7] Verifying key packages"
python -m pip show fastapi uvicorn pydantic numpy pandas scikit-learn scipy onnxruntime orjson || true

echo "[7/7] Environment setup complete."

