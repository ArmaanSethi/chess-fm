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

_ATTN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def resolve_lora_config(
    lora_r: int = LORA_R,
    lora_alpha: int | None = None,
    all_linear: bool = True,
) -> dict:
    """Resolve effective LoRA rank, alpha (defaults to ``2 * lora_r``), and target modules."""
    if lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    effective_alpha = (
        lora_alpha if lora_alpha is not None else (LORA_ALPHA if lora_r == LORA_R else 2 * lora_r)
    )
    if effective_alpha <= 0:
        raise ValueError(f"lora_alpha must be positive, got {effective_alpha}")
    target_modules = list(_TARGET_MODULES if all_linear else _ATTN_TARGET_MODULES)
    return {
        "r": int(lora_r),
        "lora_alpha": int(effective_alpha),
        "target_modules": target_modules,
    }


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
    load_in_4bit: bool = False,
    *,
    lora_r: int = LORA_R,
    lora_alpha: int | None = None,
    all_linear: bool = True,
    full_ft: bool = False,
) -> tuple:
    """Load ``model_name`` with LoRA (unsloth/PEFT) or native Full Fine-Tuning.

    Returns a ``(model, tokenizer)`` tuple. ``for_inference`` switches 4-bit
    mode off and enables fast inference mode when using unsloth.

    When ``full_ft=True``:
    - Raises ``ValueError`` if ``load_in_4bit=True`` (cannot full-FT a 4-bit
      quantized model).
    - Bypasses Unsloth and PEFT/LoRA wrapping entirely, loading native weights
      in ``bf16`` (Ampere+/H100), ``fp16`` (pre-Ampere), or ``fp32`` (CPU).
    - If ``model_name`` is a PEFT adapter and ``merge_peft=True``, merges and
      unloads the adapter without re-attaching LoRA.

    When ``full_ft=False`` (default):
    - Applies LoRA with rank ``lora_r``, alpha ``lora_alpha`` (default
      ``2 * lora_r``), and all-linear targets (``q/k/v/o/gate/up/down_proj``)
      when ``all_linear=True`` or attention-only targets when ``all_linear=False``.
    """
    if full_ft and load_in_4bit:
        raise ValueError(
            "full_ft=True cannot be combined with load_in_4bit=True; "
            "full fine-tuning requires non-quantized weights (bf16/fp16/fp32)."
        )

    lora_cfg = None if full_ft else resolve_lora_config(lora_r, lora_alpha, all_linear)
    is_adapter = is_peft_adapter_checkpoint(model_name)
    auth = _auth_kwargs()

    # Use the Transformers + PEFT path for Full FT, PEFT adapter checkpoints
    # (to cleanly merge or continue training without silent lora_B zeroing),
    # and Gemma 4 multimodal checkpoints.
    use_unsloth = not full_ft and not is_adapter and not is_gemma4_model(model_name)

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
            assert lora_cfg is not None
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_cfg["r"],
                target_modules=lora_cfg["target_modules"],
                lora_alpha=lora_cfg["lora_alpha"],
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

    use_bnb_4bit = bool(load_in_4bit and torch.cuda.is_available() and not for_inference)
    quant_kwargs: dict = {}
    if use_bnb_4bit and not (is_adapter and merge_peft):
        from transformers import BitsAndBytesConfig

        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    if is_adapter:
        peft_cfg = PeftConfig.from_pretrained(str(model_name))
        base_model_name = peft_cfg.base_model_name_or_path
        model_cls = _get_auto_model_cls(base_model_name)
        try:
            tokenizer = _load_tokenizer(str(model_name), auth)
        except Exception:
            tokenizer = _load_tokenizer(base_model_name, auth)

        base_kwargs = {"device_map": "auto", **auth, **quant_kwargs}
        if "quantization_config" not in base_kwargs:
            base_kwargs["torch_dtype"] = dtype
        base_model = model_cls.from_pretrained(base_model_name, **base_kwargs)
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
            if full_ft:
                for p in model.parameters():
                    p.requires_grad = True
                return model, tokenizer
            assert lora_cfg is not None
            peft_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                target_modules=lora_cfg["target_modules"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            return model, tokenizer

        if for_inference:
            model = model.eval()
        return model, tokenizer

    model_cls = _get_auto_model_cls(model_name)
    tokenizer = _load_tokenizer(model_name, auth)
    load_kwargs = {"device_map": "auto", **auth, **quant_kwargs}
    if "quantization_config" not in load_kwargs:
        load_kwargs["torch_dtype"] = dtype
    model = model_cls.from_pretrained(model_name, **load_kwargs)
    if for_inference:
        model = model.eval()
        return model, tokenizer
    if full_ft:
        for p in model.parameters():
            p.requires_grad = True
        return model, tokenizer
    if "quantization_config" in load_kwargs:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    assert lora_cfg is not None
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer
