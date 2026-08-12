# ChessFM Training Scripts

Scripts for training the ChessFM model.

## Files

| Script | Purpose |
|--------|---------|
| `train_sft.py` | Supervised fine-tuning on reasoning data |
| `../rl/train_grpo.py` | Reinforcement learning with GRPO |

## SFT Training

### Prerequisites

```bash
# On RunPod or GPU machine:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -e ".[dev]"
```

### Usage

```bash
# First, convert data to training format (dedupes by FEN, fixes tags, writes manifest)
python data_generation/convert_to_training.py

# Then train (from the project root)
python training/train_sft.py --data data_generation/sft_train.jsonl

# With custom settings
python training/train_sft.py \
    --data data_generation/sft_train.jsonl \
    --model Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --output ./checkpoints
```

### Data Format

The converted dataset uses TRL's `prompt`/`completion` columns, so the loss
is computed **on completion tokens only** (the user prompt is never trained
on). The train/validation split is done at the **FEN level**, so no position
appears in both sets.

### Dry Run (Validate Data)

```bash
# Works on any machine, no GPU needed
python training/train_sft.py --data data_generation/sft_train.jsonl --dry-run
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
| LoRA rank | 16 | Balance between capacity and efficiency |
