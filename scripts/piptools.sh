#!/usr/bin/env bash
set -euo pipefail

SYNC=0
UPGRADE=0
for arg in "$@"; do
  case "$arg" in
    --sync) SYNC=1 ;;
    --upgrade) UPGRADE=1 ;;
  esac
done

python -m pip install --upgrade pip setuptools wheel >/dev/null
python -m pip install --upgrade pip-tools >/dev/null

ARGS=()
if [[ $UPGRADE -eq 1 ]]; then ARGS+=(--upgrade); fi
ARGS+=(-o requirements.lock.txt requirements.in)
python -m piptools compile "${ARGS[@]}"

if [[ $SYNC -eq 1 ]]; then
  python -m piptools sync requirements.lock.txt
fi

echo "pip-tools done. Updated requirements.lock.txt"

