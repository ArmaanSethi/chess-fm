#!/bin/bash
set -e

echo "Setting up ChessFM Environment..."

# Install Stockfish based on OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    if command -v brew &> /dev/null; then
        echo "Installing Stockfish via Homebrew..."
        brew install stockfish
    fi
elif command -v apt-get &> /dev/null; then
    echo "Installing Stockfish via apt..."
    sudo apt-get update
    sudo apt-get install -y stockfish
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
if command -v stockfish &> /dev/null; then
    stockfish --version || true
    echo "✅ Stockfish available"
else
    echo "❌ Stockfish binary not found in PATH"
    echo "   Stages 1/2 of GRPO and Stockfish-based evaluation require it."
    echo "   Install via: brew install stockfish  (or: sudo apt install stockfish)"
    exit 1
fi
python -c "import chess; import trl; print('✅ Python dependencies verified')"

echo "Setup complete."
