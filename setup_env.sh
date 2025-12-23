#!/bin/bash

echo "Setting up ChessFM Environment..."

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Please install Homebrew."
    exit 1
fi

# Install Stockfish
echo "Installing Stockfish..."
brew install stockfish

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
echo "Stockfish version:"
stockfish --version || echo "Stockfish not found"

echo "Python version:"
python --version

echo "Setup complete."
