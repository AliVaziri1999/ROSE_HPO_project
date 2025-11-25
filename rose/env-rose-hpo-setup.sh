#!/usr/bin/env bash
set -e

echo "=== ROSE HPO Setup (Linux/WSL/Mac) ==="

python3 -m venv rose_venv
source rose_venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete."
echo "Activate using:"
echo "   source rose_venv/bin/activate"
