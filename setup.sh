#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo
echo " ========================================"
echo "  EventSync AI - First-time Setup"
echo " ========================================"
echo

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo ""
  fi
}

PYTHON="$(pick_python)"
if [[ -z "$PYTHON" ]]; then
  echo "[ERROR] Python was not found. Install Python 3.10+ and try again."
  exit 1
fi

echo "[1/3] Checking Python..."
"$PYTHON" --version

VENV="$ROOT/.venv"
if [[ -x "$VENV/bin/python" ]]; then
  echo "[2/3] Virtual environment already exists at .venv"
else
  echo "[2/3] Creating virtual environment in .venv ..."
  "$PYTHON" -m venv "$VENV"
fi

echo "[3/3] Installing dependencies from requirements.txt ..."
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r requirements.txt

echo
echo " Setup complete! Run ./start.sh to launch EventSync AI."
echo
