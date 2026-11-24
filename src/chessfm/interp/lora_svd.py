"""Exact LoRA weight-diff SVD geometry across transformer layers and modules.

Uses QR-based thin SVD on ``(alpha / r) * B @ A`` so exact singular values,
Frobenius norms, and effective rank are computed in milliseconds on CPU without
materializing full ``(d_out, d_in)`` matrices.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch

_LAYER_MODULE_RE = re.compile(
    r"layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.(lora_A|lora_B)"
)

ATTN_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})


def parse_lora_module_key(key: str) -> tuple[int, str, str, str] | None:
    """Parse a PEFT state-dict key into ``(layer_idx, block_type, proj_name, side)``."""
    m = _LAYER_MODULE_RE.search(key)
    if not m:
        return None
    return int(m.group(1)), m.group(2), m.group(3), m.group(4)


def compute_exact_lora_svd(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float = 1.0,
) -> dict[str, Any]:
    """Compute exact singular values, Frobenius norm, and effective rank of ``scaling * B @ A``.

    Uses QR decompositions ``B = Q_B R_B`` and ``A^T = Q_A R_A`` so the non-zero
    singular values of ``B @ A`` equal the singular values of the tiny ``(r, r)``
    matrix ``R_B @ R_A^T``.
    """
    a = lora_a.detach().cpu().float()
    b = lora_b.detach().cpu().float()
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError(f"Incompatible LoRA shapes: B={tuple(b.shape)}, A={tuple(a.shape)}")

    _, r_b = torch.linalg.qr(b, mode="reduced")
    _, r_a = torch.linalg.qr(a.T, mode="reduced")
    core = (r_b @ r_a.T) * float(scaling)
    svals = torch.linalg.svdvals(core)

    energy = svals.pow(2)
    total_energy = float(energy.sum().item())
    frob = math.sqrt(max(total_energy, 0.0))

    if total_energy > 1e-18:
        probs = (energy / total_energy).clamp(min=1e-18)
        entropy = float(-(probs * probs.log()).sum().item())
        eff_rank = float(math.exp(entropy))
        top1_energy_frac = float((energy[0] / total_energy).item())
    else:
        eff_rank = 0.0
        top1_energy_frac = 0.0

    return {
        "frobenius_norm": round(frob, 6),
        "spectral_norm": round(float(svals[0].item()), 6) if svals.numel() > 0 else 0.0,
        "effective_rank": round(eff_rank, 4),
        "top1_energy_fraction": round(top1_energy_frac, 4),
        "top5_singular_values": [round(float(x), 6) for x in svals[:5].tolist()],
        "rank": int(a.shape[0]),
    }


def _load_adapter_tensors(adapter_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")
    config = json.loads(cfg_path.read_text(encoding="utf-8"))

    safetensors_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        tensors = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        tensors = torch.load(bin_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or .bin in {adapter_dir}")
    return tensors, config


def analyze_lora_adapter(adapter_dir: str | Path) -> dict[str, Any]:
    """Summarize exact per-layer and Attention-vs-MLP LoRA weight diffs."""
    path = Path(adapter_dir)
    tensors, config = _load_adapter_tensors(path)
    r = int(config.get("r", 16))
    alpha = float(config.get("lora_alpha", 2 * r))
    scaling = alpha / max(r, 1)

    pairs: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        parsed = parse_lora_module_key(key)
        if parsed is None:
            continue
        layer_idx, block_type, proj_name, side = parsed
        pairs.setdefault((layer_idx, block_type, proj_name), {})[side] = tensor

    per_layer: dict[int, dict[str, Any]] = {}
    attn_energy_total = 0.0
    mlp_energy_total = 0.0
    eff_ranks: list[float] = []
    top1_fracs: list[float] = []

    for (layer_idx, _block_type, proj_name), sides in sorted(pairs.items()):
        if "lora_A" not in sides or "lora_B" not in sides:
            continue
        stats = compute_exact_lora_svd(sides["lora_A"], sides["lora_B"], scaling=scaling)
        energy = stats["frobenius_norm"] ** 2
        eff_ranks.append(stats["effective_rank"])
        top1_fracs.append(stats["top1_energy_fraction"])

        entry = per_layer.setdefault(
            layer_idx,
            {"layer": layer_idx, "attn_energy": 0.0, "mlp_energy": 0.0, "modules": {}},
        )
        entry["modules"][proj_name] = stats
        if proj_name in ATTN_MODULES:
            entry["attn_energy"] += energy
            attn_energy_total += energy
        else:
            entry["mlp_energy"] += energy
            mlp_energy_total += energy

    layer_rows: list[dict[str, Any]] = []
    for layer_idx in sorted(per_layer):
        e = per_layer[layer_idx]
        attn_frob = math.sqrt(e["attn_energy"])
        mlp_frob = math.sqrt(e["mlp_energy"])
        tot_frob = math.sqrt(e["attn_energy"] + e["mlp_energy"])
        layer_rows.append(
            {
                "layer": layer_idx,
                "attn_frobenius": round(attn_frob, 5),
                "mlp_frobenius": round(mlp_frob, 5),
                "total_frobenius": round(tot_frob, 5),
                "mlp_share": round(
                    e["mlp_energy"] / max(e["attn_energy"] + e["mlp_energy"], 1e-18), 4
                ),
            }
        )

    total_energy = attn_energy_total + mlp_energy_total
    peak_layer = max(layer_rows, key=lambda x: x["total_frobenius"])["layer"] if layer_rows else -1
    # Split layers into early / middle / late thirds for clean depth attribution
    n_layers = len(layer_rows)
    third = max(n_layers // 3, 1)
    early_e = sum(r["total_frobenius"] ** 2 for r in layer_rows[:third])
    mid_e = sum(r["total_frobenius"] ** 2 for r in layer_rows[third : 2 * third])
    late_e = sum(r["total_frobenius"] ** 2 for r in layer_rows[2 * third :])
    depth_tot = max(early_e + mid_e + late_e, 1e-18)

    return {
        "adapter_dir": str(path),
        "lora_r": r,
        "lora_alpha": alpha,
        "scaling": scaling,
        "num_layers": n_layers,
        "num_modules": len(eff_ranks),
        "total_frobenius": round(math.sqrt(total_energy), 5),
        "attn_frobenius": round(math.sqrt(attn_energy_total), 5),
        "mlp_frobenius": round(math.sqrt(mlp_energy_total), 5),
        "mlp_energy_share": round(mlp_energy_total / max(total_energy, 1e-18), 4),
        "attn_energy_share": round(attn_energy_total / max(total_energy, 1e-18), 4),
        "depth_energy_shares": {
            "early": round(early_e / depth_tot, 4),
            "middle": round(mid_e / depth_tot, 4),
            "late": round(late_e / depth_tot, 4),
        },
        "peak_layer": peak_layer,
        "mean_effective_rank": round(sum(eff_ranks) / max(len(eff_ranks), 1), 3),
        "mean_top1_energy_fraction": round(sum(top1_fracs) / max(len(top1_fracs), 1), 4),
        "per_layer": layer_rows,
    }


def compare_lora_profiles(profile_a: dict[str, Any], profile_b: dict[str, Any]) -> dict[str, Any]:
    """Compare layer-norm profiles of two RL adapters via cosine similarity and layer deltas."""
    map_a = {r["layer"]: r["total_frobenius"] for r in profile_a.get("per_layer", [])}
    map_b = {r["layer"]: r["total_frobenius"] for r in profile_b.get("per_layer", [])}
    common = sorted(set(map_a) & set(map_b))
    if not common:
        return {"cosine_similarity": 0.0, "norm_ratio_b_over_a": 0.0, "top_diff_layers": []}

    va = torch.tensor([map_a[i] for i in common], dtype=torch.float32)
    vb = torch.tensor([map_b[i] for i in common], dtype=torch.float32)
    cos = float(torch.dot(va, vb) / (va.norm().clamp(min=1e-8) * vb.norm().clamp(min=1e-8)))

    diffs = [
        {
            "layer": i,
            "a_norm": round(map_a[i], 5),
            "b_norm": round(map_b[i], 5),
            "delta": round(map_b[i] - map_a[i], 5),
        }
        for i in common
    ]
    diffs.sort(key=lambda x: abs(x["delta"]), reverse=True)
    return {
        "cosine_similarity": round(cos, 4),
        "norm_ratio_b_over_a": round(
            float(profile_b.get("total_frobenius", 0.0))
            / max(float(profile_a.get("total_frobenius", 0.0)), 1e-8),
            4,
        ),
        "top_diff_layers": diffs[:5],
    }
