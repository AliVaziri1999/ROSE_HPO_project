#!/usr/bin/env bash
# env-rose-hpo-setup.sh
# One-shot environment setup for ROSE_HPO_project on Linux / WSL.

set -euo pipefail

echo "[ROSE_HPO] Detecting project root..."
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="${PROJECT_ROOT}/rose_venv"

echo "[ROSE_HPO] Python version:"
python3 --version || python --version || true
echo

# 1) Create virtual environment if it does not exist
if [ ! -d "$VENV_DIR" ]; then
  echo "[ROSE_HPO] Creating virtualenv at: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
else
  echo "[ROSE_HPO] Virtualenv already exists at: $VENV_DIR"
fi

# 2) Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[ROSE_HPO] Using Python from:"
which python
echo

# 3) Upgrade pip
echo "[ROSE_HPO] Upgrading pip..."
pip install --upgrade pip

# 4) Install dependencies from requirements-rose-hpo.txt if present
REQ_FILE="${PROJECT_ROOT}/requirements-rose-hpo.txt"
if [ -f "$REQ_FILE" ]; then
  echo "[ROSE_HPO] Installing dependencies from $REQ_FILE..."
  pip install -r "$REQ_FILE"
else
  echo "[ROSE_HPO] WARNING: $REQ_FILE not found, skipping requirements install."
fi

# 5) Install ROSE_HPO_project itself in editable mode
echo "[ROSE_HPO] Installing project in editable mode (pip install -e .)..."
pip install -e .

echo
echo "[ROSE_HPO] Environment setup complete."
echo "To use it later, run:"
echo "  cd \"$PROJECT_ROOT\""
echo "  source rose_venv/bin/activate"
