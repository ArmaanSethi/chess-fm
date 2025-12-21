#!/bin/bash

# Update and install system dependencies
apt-get update && apt-get install -y stockfish git wget

# Install Python dependencies
# Unsloth (optimized for 4090)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# vLLM for fast inference
pip install vllm

# Chess logic
pip install python-chess

# Other ML essentials
pip install torch transformers datasets wandb

# Verify installations
echo "Verifying Stockfish..."
stockfish uci

echo "Verifying Python packages..."
python3 -c "import torch; print(f'Torch: {torch.__version__}'); import unsloth; print('Unsloth installed'); import chess; print(f'Python-Chess: {chess.__version__}')"

echo "Setup complete!"
