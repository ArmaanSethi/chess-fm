# ChessFM Training Scripts

Scripts for training the ChessFM model.

## Files

| Script | Purpose |
|--------|---------|
| `train_sft.py` | Supervised fine-tuning on reasoning data |
| (coming) `train_grpo.py` | Reinforcement learning with GRPO |

## SFT Training

### Prerequisites

```bash
# On RunPod or GPU machine:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets trl accelerate
```

### Usage

```bash
# First, generate and convert data
cd ../data_generation
python generate_sft_data_proxy.py --samples 1000
python convert_to_training.py

# Then train
cd ../training
python train_sft.py --data ../data_generation/data/sft_train.jsonl

# With custom settings
python train_sft.py \
    --data ../data_generation/data/sft_train.jsonl \
    --model Qwen/Qwen2.5-Math-1.5B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --lora-r 32 \
    --output ./checkpoints
```

### Dry Run (Validate Data)

```bash
python train_sft.py --data ../data_generation/data/sft_train.jsonl --dry-run
```

## Expected Output

After training, you'll have:
```
checkpoints/
├── final/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
└── checkpoint-*/
```

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 3 | Usually sufficient for SFT |
| Batch size | 4 | Increase if you have more VRAM |
| Learning rate | 2e-5 | Standard for LoRA |
| LoRA rank | 32 | Balance between capacity and efficiency |
