#!/usr/bin/env python3
"""Shared model/tokenizer loading for SFT, GRPO, and evaluation.

All entry points use the same loader so the LoRA configuration, quantization,
and inference mode never diverge between phases.
"""

from chessfm.config import LORA_ALPHA, LORA_R, MAX_SEQ_LENGTH

# Same target modules on both the unsloth and PEFT fallback paths so a LoRA
# trained on one backend is structurally identical to one trained on the other.
_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_model(model_name: str, for_inference: bool = False) -> tuple:
    """Load ``model_name`` with 4-bit LoRA (unsloth) or a PEFT fallback.

    Returns a ``(model, tokenizer)`` tuple. ``for_inference`` switches 4-bit
    mode off and enables fast inference mode when using unsloth.
    """
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=not for_inference,
        )
        if for_inference:
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            target_modules=_TARGET_MODULES,
            lora_alpha=LORA_ALPHA,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        return model, tokenizer
    except ImportError:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # bf16 matches Qwen2.5's native training dtype on modern GPUs; fp16 is
        # the fallback for pre-Ampere hardware.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto"
        )
        if for_inference:
            model = model.eval()
            return model, tokenizer
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=_TARGET_MODULES,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        return model, tokenizer
