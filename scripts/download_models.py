#!/usr/bin/env python3
"""
Download all candidate models locally for chess-fm training.
Models are cached in ~/.cache/huggingface/hub/
"""

import os
import sys

MODELS = [
    {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "description": "Selected base model (from Phase 0 benchmarks)"
    },
]

def download_model(model_info):
    """Download a single model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    name = model_info["name"]
    desc = model_info["description"]
    
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Description: {desc}")
    print(f"{'='*60}")
    
    try:
        # Download tokenizer
        print("  → Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        print(f"    ✓ Tokenizer downloaded (vocab size: {tokenizer.vocab_size})")
        
        # Download model weights
        print("  → Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Don't load into GPU yet
        )
        
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        print(f"    ✓ Model downloaded ({param_count/1e9:.2f}B parameters)")
        
        # Free memory
        del model
        del tokenizer
        
        return True
        
    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download ChessFM base models to the HuggingFace cache")
    parser.add_argument("--model", type=str, default=None,
                        help="Single model ID to download (defaults to the selected base model)")
    args = parser.parse_args()

    models = MODELS
    if args.model:
        models = [{"name": args.model, "description": "Requested model"}]

    print("ChessFM Model Downloader")
    print("========================")
    print("Models will be cached in: ~/.cache/huggingface/hub/")

    results = []

    for model_info in models:
        success = download_model(model_info)
        results.append((model_info["name"], success))
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results:
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    # Check if any failed
    failed = [name for name, success in results if not success]
    if failed:
        print(f"\n⚠️  {len(failed)} model(s) failed to download.")
        print("Note: Llama models may require HF_TOKEN authentication.")
        print("Set: export HF_TOKEN=your_token_here")
        return 1
    else:
        print(f"\n✓ All {len(MODELS)} models downloaded successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
