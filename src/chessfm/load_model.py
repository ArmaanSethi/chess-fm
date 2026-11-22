#!/usr/bin/env python3
"""Shared model/tokenizer loading for SFT, GRPO, and evaluation.

All entry points use the same loader so the LoRA configuration, quantization,
and inference mode never diverge between phases.
"""

import os
from pathlib import Path

from chessfm.config import LORA_ALPHA, LORA_R, MAX_SEQ_LENGTH, is_gemma4_model

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


def is_peft_adapter_checkpoint(model_name: str | Path) -> bool:
    """Return True if ``model_name`` is a local directory containing a PEFT adapter."""
    return (Path(model_name) / "adapter_config.json").is_file()


def _auth_kwargs() -> dict[str, str]:
    """Return Hugging Face auth kwargs when ``HF_TOKEN`` is set in the environment."""
    token = os.environ.get("HF_TOKEN")
    return {"token": token} if token else {}


def _get_auto_model_cls(model_name: str):
    """Select the Transformers AutoModel class for ``model_name``."""
    from transformers import AutoModelForCausalLM

    if is_gemma4_model(model_name):
        try:
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        except ImportError:
            return AutoModelForCausalLM
    return AutoModelForCausalLM


def _load_tokenizer(model_name: str, auth_kwargs: dict[str, str]):
    """Load tokenizer (or processor's tokenizer for Gemma 4) with safe fallback."""
    from transformers import AutoTokenizer

    if is_gemma4_model(model_name):
        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_name, **auth_kwargs)
            return getattr(processor, "tokenizer", processor)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(model_name, **auth_kwargs)


def load_model(
    model_name: str,
    for_inference: bool = False,
    merge_peft: bool = False,
) -> tuple:
    """Load ``model_name`` with 4-bit LoRA (unsloth) or a PEFT fallback.

    Returns a ``(model, tokenizer)`` tuple. ``for_inference`` switches 4-bit
    mode off and enables fast inference mode when using unsloth.

    When ``model_name`` is a local PEFT adapter checkpoint (contains
    ``adapter_config.json``), the base model is loaded and the adapter is
    attached via ``PeftModel.from_pretrained``. If ``merge_peft=True`` (the
    recommended mode for SFT -> GRPO handoff), the adapter is merged into the
    base weights via ``merge_and_unload()`` before attaching any new LoRA
    adapter, avoiding silent ``lora_B`` zeroing.
    """
    is_adapter = is_peft_adapter_checkpoint(model_name)
    auth = _auth_kwargs()

    # Use the Transformers + PEFT path for PEFT adapter checkpoints (to cleanly
    # merge or continue training without silent lora_B zeroing) and for Gemma 4
    # multimodal checkpoints.
    use_unsloth = not is_adapter and not is_gemma4_model(model_name)

    if use_unsloth:
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=not for_inference,
                **auth,
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
            pass

    import torch
    from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model

    # bf16 matches modern LLM native training dtype on Ampere+ GPUs; fp16 is
    # the fallback for pre-Ampere GPUs, and fp32 for CPU.
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(str(model_name))
        base_model_name = peft_cfg.base_model_name_or_path
        model_cls = _get_auto_model_cls(base_model_name)
        try:
            tokenizer = _load_tokenizer(str(model_name), auth)
        except Exception:
            tokenizer = _load_tokenizer(base_model_name, auth)

        base_model = model_cls.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
            **auth,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(model_name),
            is_trainable=not for_inference and not merge_peft,
        )
        if merge_peft:
            model = model.merge_and_unload()
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

        if for_inference:
            model = model.eval()
        return model, tokenizer

    model_cls = _get_auto_model_cls(model_name)
    tokenizer = _load_tokenizer(model_name, auth)
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        **auth,
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
