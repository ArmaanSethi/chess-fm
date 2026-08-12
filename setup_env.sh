#!/bin/bash
set -e

echo "Setting up ChessFM Environment..."

# Install Stockfish based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Installing Stockfish via Homebrew..."
        brew install stockfish || true
    fi
elif command -v apt-get &> /dev/null; then
    echo "Installing Stockfish via apt..."
    sudo apt-get update && sudo apt-get install -y stockfish || true
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install editable package and dependencies
echo "Installing Python dependencies and editable package..."
pip install --upgrade pip
pip install -e .[dev]

# Verify installation
echo "Verifying installation..."
stockfish --version || echo "Stockfish binary not in standard path"
python -c "import chess; import trl; print('✅ Python dependencies verified')"

echo "Setup complete."
