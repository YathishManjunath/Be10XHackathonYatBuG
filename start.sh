#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo
echo " ========================================"
echo "  EventSync AI - Starting Application"
echo " ========================================"
echo

VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found. Running setup first..."
  echo
  bash "$ROOT/setup.sh"
fi

if [[ ! -x "$VENV/bin/streamlit" && ! -f "$VENV/bin/streamlit" ]]; then
  echo "Streamlit not installed. Running setup..."
  echo
  bash "$ROOT/setup.sh"
fi

echo "Launching EventSync AI in your browser..."
echo "Press Ctrl+C in this terminal to stop the server."
echo

exec "$PYTHON" -m streamlit run app.py --server.headless false
