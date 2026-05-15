#!/bin/bash
set -e  # exit on any error

VENV_DIR=".venv"
PYTHON=${PYTHON:-python3}   # override with: PYTHON=python3.11 ./setup_dev.sh

echo "==> Creating virtual environment in $VENV_DIR..."
$PYTHON -m venv $VENV_DIR

echo "==> Activating venv..."
source $VENV_DIR/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing oceanicospy with dev dependencies..."
pip install -e ".[dev]"   

echo "==> Registering Jupyter kernel..."
python -m ipykernel install --user \
  --name=oceanicospy-dev \
  --display-name "oceanicospy-dev"

echo ""
echo "Activate your environment with:"
echo "source $VENV_DIR/bin/activate"