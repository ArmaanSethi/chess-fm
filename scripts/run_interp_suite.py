#!/usr/bin/env python3
"""Run all 4 Mechanistic Interpretability experiments (P4C-INTERP) in one warm GPU session.

Outputs four self-contained JSON reports + manifest to ``--output-dir``:
  1. ``interp1_board_probes_and_steering.json``:
     64-square linear board-state probes across layers [0, 6, 12, 18, 24, 30, 35]
     for Base (ASCII), E0 (ASCII), E1a (Piece-List), E2a (r=64), and E3b (sharp_wdl),
     plus causal board surgery steering vs. norm-matched orthogonal random control.
  2. ``interp2_lora_svd_and_slop_steering.json``:
     Exact QR-SVD per-layer Attention-vs-MLP weight-diff profiles for unmerged RL
     adapters (E3a A-Pawn Hack, E3b sharp_wdl, Arm A V2-GRPO, Arm C V1-GRPO), plus
     causal Goodhart/Slop vector steering sweep alpha in [-2, -1, 0, +1, +2].
  3. ``interp3_logit_lens_and_cot_sabotage.json``:
     Token-position logit lens across <think> progress (0..100%) and Anthropic/Lanham
     corrupted-<think> prefill faithfulness intervention on solved puzzles.
  4. ``interp4_attention_circuits_and_ablation.json``:
     Piece-Tracker attention-head selectivity ranking from generated UCI move tokens
     back to prompt square tokens, plus causal zero-ablation of top-5 heads vs. 5
     layer-matched random control heads.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chessfm.coach.coach_rewards import verify_claims_v1
from chessfm.coach.socratic import contains_move_spoiler
from chessfm.coach.tactical_facts import extract_tactical_facts
from chessfm.coach.verifier import detect_boilerplate_slop, verify_claims
from chessfm.config import build_chat_prompt
from chessfm.data_generation.distill_open_teacher import (
    SYSTEM_PROMPT as COACH_SYSTEM_PROMPT,
)
from chessfm.data_generation.distill_open_teacher import (
    build_teacher_prompt,
    infer_task_type,
)
from chessfm.interp import (
    PIECE_TO_CLASS,
    analyze_lora_adapter,
    build_corrupted_think_prefill,
    build_random_control_vector,
    build_square_steering_vector,
    compare_lora_profiles,
    fen_to_board_targets,
    find_square_token_indices,
    rank_piece_tracker_heads,
    sample_layer_matched_random_heads,
    sample_think_checkpoints,
    score_attention_heads,
    summarize_logit_lens,
    train_board_linear_probe,
)
from chessfm.rl.move_utils import parse_move_from_output


def _read_jsonl(path: str | Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _load_model_and_tok(
    model_path: str,
    *,
    attn_implementation: str = "sdpa",
) -> tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    return model, tok


def _free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _extract_last_token_hidden_states(
    model: Any,
    tok: Any,
    prompts: list[str],
    layers: list[int],
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """Extract residual-stream hidden states at the final prompt token for requested layers."""
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    device = next(model.parameters()).device
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            device
        )
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # With left-padding, the final prompt token is index -1 for every row.
        # hidden_states[0] is embedding output; hidden_states[layer + 1] is block `layer` output.
        for layer in layers:
            h = out.hidden_states[layer + 1][:, -1, :].detach().float().cpu()
            collected[layer].append(h)
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in collected.items()}


def _generate_greedy(
    model: Any,
    tok: Any,
    prompts: list[str],
    *,
    max_new_tokens: int = 128,
    batch_size: int = 16,
) -> list[str]:
    device = next(model.parameters()).device
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(
            device
        )
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for row in gen:
            text = tok.decode(row[prompt_len:], skip_special_tokens=True)
            outputs.append(text)
    return outputs


def _make_residual_add_hook(delta_vec: torch.Tensor):
    """Forward hook adding ``delta_vec`` to the residual stream at every token step."""

    def _hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        vec = delta_vec.to(device=hidden.device, dtype=hidden.dtype)
        steered = hidden + vec.view(1, 1, -1)
        if is_tuple:
            return (steered, *output[1:])
        return steered

    return _hook


# ---------------------------------------------------------------------------
# INTERP #1: Linear Board-State Probes + Causal Board Surgery
# ---------------------------------------------------------------------------
def run_interp1_board_probes(
    fens_path: str,
    checkpoints: list[dict[str, Any]],
    *,
    layers: list[int] | None = None,
    n_fens: int = 1000,
    n_train: int = 800,
    n_steer: int = 50,
    batch_size: int = 32,
) -> dict[str, Any]:
    if layers is None:
        layers = [0, 6, 12, 18, 24, 30, 35]
    print("\n=== INTERP #1: Linear Board-State Probes + Causal Board Surgery ===", flush=True)
    rows = _read_jsonl(fens_path, limit=n_fens)
    fens = [r["fen"] for r in rows]
    y_all = torch.tensor([fen_to_board_targets(f) for f in fens], dtype=torch.long)
    y_train, y_test = y_all[:n_train], y_all[n_train:]

    heatmap: dict[str, dict[str, Any]] = {}
    steering_result: dict[str, Any] = {}

    for spec in checkpoints:
        name = spec["name"]
        path = spec["path"]
        use_ascii = bool(spec.get("include_ascii", False))
        use_piece_list = bool(spec.get("include_piece_list", False))
        print(f"  [INTERP #1] Probing {name} ({path})...", flush=True)
        model, tok = _load_model_and_tok(path)
        prompts = [
            build_chat_prompt(
                f,
                tokenizer=tok,
                include_ascii=use_ascii,
                include_piece_list=use_piece_list,
                enable_thinking=True,
            )
            for f in fens
        ]
        h_by_layer = _extract_last_token_hidden_states(
            model, tok, prompts, layers, batch_size=batch_size
        )

        layer_metrics: dict[str, Any] = {}
        best_layer = layers[0]
        best_occ = -1.0
        best_probe_weight: torch.Tensor | None = None
        best_res_norm = 1.0

        for layer in layers:
            x_all = h_by_layer[layer].to("cuda" if torch.cuda.is_available() else "cpu")
            res = train_board_linear_probe(
                x_all[:n_train],
                y_train.to(x_all.device),
                x_all[n_train:],
                y_test.to(x_all.device),
                epochs=80,
                lr=1e-2,
                seed=42,
            )
            layer_metrics[str(layer)] = {
                "overall_accuracy": res["overall_accuracy"],
                "occupied_accuracy": res["occupied_accuracy"],
                "empty_accuracy": res["empty_accuracy"],
            }
            if res["occupied_accuracy"] > best_occ:
                best_occ = res["occupied_accuracy"]
                best_layer = layer
                best_probe_weight = res["weight"]
                best_res_norm = float(h_by_layer[layer].norm(dim=-1).mean().item())

        heatmap[name] = {
            "model_path": path,
            "include_ascii": use_ascii,
            "include_piece_list": use_piece_list,
            "peak_layer": best_layer,
            "peak_occupied_accuracy": round(best_occ, 4),
            "layers": layer_metrics,
        }
        print(
            f"  [INTERP #1] {name} peak layer={best_layer} occupied_acc={best_occ:.4f}",
            flush=True,
        )

        # Run Causal Board Surgery on the primary RL model (E3b)
        if spec.get("run_steering") and best_probe_weight is not None:
            print(
                f"  [INTERP #1] Causal Board Surgery on {name} at layer {best_layer} (n={n_steer})...",
                flush=True,
            )
            test_fens = fens[n_train : n_train + n_steer]
            test_prompts = prompts[n_train : n_train + n_steer]
            baseline_texts = _generate_greedy(
                model, tok, test_prompts, max_new_tokens=96, batch_size=batch_size
            )
            baseline_moves = [parse_move_from_output(t) for t in baseline_texts]

            # Scale steering vector to 35% of mean residual norm (same norm for random control)
            steer_scale = 0.35 * best_res_norm
            probe_flips = 0
            random_flips = 0
            probe_from_sq_dropped = 0
            random_from_sq_dropped = 0
            valid_cases = 0
            examples: list[dict[str, Any]] = []

            for idx, (fen, prompt, base_mv) in enumerate(
                zip(test_fens, test_prompts, baseline_moves, strict=True)
            ):
                board = chess.Board(fen)
                # Pick the source square of baseline move if legal, else highest-value mover piece
                target_sq: int | None = None
                if base_mv:
                    try:
                        mv_obj = chess.Move.from_uci(base_mv)
                        if mv_obj in board.legal_moves and board.piece_at(mv_obj.from_square):
                            target_sq = mv_obj.from_square
                    except ValueError:
                        pass
                if target_sq is None:
                    for sq in range(64):
                        pc = board.piece_at(sq)
                        if pc and pc.color == board.turn and pc.piece_type != chess.KING:
                            target_sq = sq
                            break
                if target_sq is None:
                    continue
                pc = board.piece_at(target_sq)
                if pc is None:
                    continue
                valid_cases += 1
                from_cls = PIECE_TO_CLASS[pc.symbol()]
                probe_vec = build_square_steering_vector(best_probe_weight, target_sq, from_cls, 0)
                probe_delta = probe_vec * steer_scale
                rand_delta = build_random_control_vector(probe_delta, seed=100 + idx)

                target_block = model.model.layers[best_layer]

                h1 = target_block.register_forward_hook(_make_residual_add_hook(probe_delta))
                try:
                    steered_mv = parse_move_from_output(
                        _generate_greedy(model, tok, [prompt], max_new_tokens=96, batch_size=1)[0]
                    )
                finally:
                    h1.remove()

                h2 = target_block.register_forward_hook(_make_residual_add_hook(rand_delta))
                try:
                    rand_mv = parse_move_from_output(
                        _generate_greedy(model, tok, [prompt], max_new_tokens=96, batch_size=1)[0]
                    )
                finally:
                    h2.remove()

                if steered_mv != base_mv:
                    probe_flips += 1
                if rand_mv != base_mv:
                    random_flips += 1
                sq_name = chess.square_name(target_sq)
                if base_mv and base_mv.startswith(sq_name):
                    if not (steered_mv and steered_mv.startswith(sq_name)):
                        probe_from_sq_dropped += 1
                    if not (rand_mv and rand_mv.startswith(sq_name)):
                        random_from_sq_dropped += 1

                if len(examples) < 5:
                    examples.append(
                        {
                            "fen": fen,
                            "erased_square": sq_name,
                            "erased_piece": pc.symbol(),
                            "baseline_move": base_mv,
                            "probe_steered_move": steered_mv,
                            "random_control_move": rand_mv,
                        }
                    )

            steering_result = {
                "model": name,
                "layer": best_layer,
                "steer_norm": round(steer_scale, 4),
                "valid_cases": valid_cases,
                "probe_move_flip_rate": round(probe_flips / max(valid_cases, 1), 4),
                "random_control_move_flip_rate": round(random_flips / max(valid_cases, 1), 4),
                "causal_excess_flip_pp": round(
                    100.0 * (probe_flips - random_flips) / max(valid_cases, 1), 2
                ),
                "probe_erased_square_abandoned_count": probe_from_sq_dropped,
                "random_erased_square_abandoned_count": random_from_sq_dropped,
                "examples": examples,
            }

        _free_model(model)

    return {
        "schema_version": "1.0",
        "layers_probed": layers,
        "n_train": n_train,
        "n_test": len(fens) - n_train,
        "heatmap": heatmap,
        "causal_board_surgery": steering_result,
    }


# ---------------------------------------------------------------------------
# INTERP #2: RL Weight-Diffing (LoRA SVD) + Goodhart "Slop Vector" Steering
# ---------------------------------------------------------------------------
def _build_coach_prompt_text(row: dict[str, Any], tok: Any) -> str:
    fen = row["fen"]
    task_type = infer_task_type(row)
    student_move = str(row.get("student_move") or row.get("played_move_uci") or "").strip()
    best_move = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
    pv = row.get("pv") or row.get("principal_variation") or []
    tf = row.get("tactical_facts")
    if not isinstance(tf, dict):
        tf = extract_tactical_facts(
            fen=fen,
            student_move=student_move or None,
            best_move=best_move or None,
            pv=pv if isinstance(pv, list) else None,
            cp_loss=row.get("cp_loss"),
            eval_before_cp=row.get("eval_before_cp"),
            eval_after_cp=row.get("eval_after_cp"),
        )
    prompt_row = {**row, "tactical_facts": tf}
    user_msg = build_teacher_prompt(
        prompt_row,
        task_type=task_type,
        include_tactical_facts=True,
    )
    messages = [
        {"role": "system", "content": COACH_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_interp2_lora_svd_and_slop_steering(
    adapters: dict[str, str],
    arm_a_merged: str,
    arm_c_merged: str,
    coach_test_path: str,
    *,
    n_diff_prompts: int = 60,
    n_sweep_prompts: int = 50,
    alphas: list[float] | None = None,
) -> dict[str, Any]:
    if alphas is None:
        alphas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    print(
        "\n=== INTERP #2: LoRA SVD Weight-Diffing + Goodhart Slop Vector Steering ===", flush=True
    )
    profiles: dict[str, Any] = {}
    for name, path in adapters.items():
        if Path(path).exists():
            profiles[name] = analyze_lora_adapter(path)
            print(
                f"  [INTERP #2] {name}: frob={profiles[name]['total_frobenius']:.4f} "
                f"mlp_share={profiles[name]['mlp_energy_share']:.3f} "
                f"peak_layer={profiles[name]['peak_layer']} "
                f"eff_rank={profiles[name]['mean_effective_rank']:.2f}",
                flush=True,
            )

    comparisons: dict[str, Any] = {}
    if "e3a_apawn_hack" in profiles and "e3b_sharp_wdl" in profiles:
        comparisons["e3b_vs_e3a"] = compare_lora_profiles(
            profiles["e3a_apawn_hack"], profiles["e3b_sharp_wdl"]
        )
    if "arm_a_v2_grpo" in profiles and "arm_c_v1_grpo" in profiles:
        comparisons["arm_c_vs_arm_a"] = compare_lora_profiles(
            profiles["arm_a_v2_grpo"], profiles["arm_c_v1_grpo"]
        )

    # Pick peak diff layer between Arm C (V1) and Arm A (V2), default 20
    peak_diff_layer = 20
    if "arm_c_vs_arm_a" in comparisons and comparisons["arm_c_vs_arm_a"]["top_diff_layers"]:
        peak_diff_layer = int(comparisons["arm_c_vs_arm_a"]["top_diff_layers"][0]["layer"])

    coach_rows = _read_jsonl(coach_test_path)
    diff_rows = coach_rows[:n_diff_prompts]
    sweep_rows = coach_rows[n_diff_prompts : n_diff_prompts + n_sweep_prompts]

    print(
        f"  [INTERP #2] Extracting Goodhart diff vector at layer {peak_diff_layer} "
        f"(n_diff={len(diff_rows)}, n_sweep={len(sweep_rows)})...",
        flush=True,
    )

    # Extract mean activation from Arm C (Weak V1)
    model_c, tok = _load_model_and_tok(arm_c_merged)
    diff_prompts = [_build_coach_prompt_text(r, tok) for r in diff_rows]
    sweep_prompts = [_build_coach_prompt_text(r, tok) for r in sweep_rows]
    h_c = _extract_last_token_hidden_states(
        model_c, tok, diff_prompts, [peak_diff_layer], batch_size=32
    )[peak_diff_layer]
    _free_model(model_c)

    # Extract mean activation from Arm A (Strict V2), compute diff = mean(h_c) - mean(h_a)
    model_a, tok = _load_model_and_tok(arm_a_merged)
    h_a = _extract_last_token_hidden_states(
        model_a, tok, diff_prompts, [peak_diff_layer], batch_size=32
    )[peak_diff_layer]
    slop_vec = (h_c.mean(dim=0) - h_a.mean(dim=0)).float()
    # Normalize to 25% of Arm A residual norm at alpha=1.0 so alpha in [-2..+2] spans strong steering
    res_norm = float(h_a.norm(dim=-1).mean().item())
    unit_slop = slop_vec / slop_vec.norm().clamp(min=1e-8)
    calibrated_slop = unit_slop * (0.25 * res_norm)

    sweep_results: list[dict[str, Any]] = []
    target_block = model_a.model.layers[peak_diff_layer]

    for alpha in alphas:
        if alpha == 0.0:
            texts = _generate_greedy(model_a, tok, sweep_prompts, max_new_tokens=160, batch_size=25)
        else:
            hook = target_block.register_forward_hook(
                _make_residual_add_hook(calibrated_slop * alpha)
            )
            try:
                texts = _generate_greedy(
                    model_a, tok, sweep_prompts, max_new_tokens=160, batch_size=25
                )
            finally:
                hook.remove()

        v1_pass = 0
        v2_pass = 0
        spoiler_free = 0
        l12_total = 0
        for row, txt in zip(sweep_rows, texts, strict=True):
            fen = row["fen"]
            r1 = verify_claims_v1(fen, txt)
            r2 = verify_claims(fen, txt, engine=None, check_boilerplate=True)
            if r1.get("passed"):
                v1_pass += 1
            if r2.get("passed"):
                v2_pass += 1
            tt = infer_task_type(row)
            if tt in ("socratic_l1", "socratic_l2"):
                l12_total += 1
                best_uci = str(row.get("best_move_uci") or row.get("best_move") or "").strip()
                if not best_uci or not contains_move_spoiler(txt, best_uci, fen):
                    spoiler_free += 1

        slop_audit = detect_boilerplate_slop(texts)
        n = max(len(sweep_rows), 1)
        sweep_results.append(
            {
                "alpha": alpha,
                "v1_pass_rate": round(v1_pass / n, 4),
                "v2_pass_rate": round(v2_pass / n, 4),
                "v1_minus_v2_gap": round((v1_pass - v2_pass) / n, 4),
                "distinct_4gram_ratio": round(
                    float(slop_audit.get("distinct_4gram_ratio", 0.0)), 4
                ),
                "slop_flagged_rate": round(float(slop_audit.get("flagged_rate", 0.0)), 4),
                "l12_spoiler_free_rate": round(spoiler_free / max(l12_total, 1), 4),
                "mean_words": round(sum(len(t.split()) for t in texts) / n, 2),
                "sample_output_0": texts[0][:240] if texts else "",
            }
        )
        print(
            f"  [INTERP #2] alpha={alpha:+.1f} v1={v1_pass / n:.2%} v2={v2_pass / n:.2%} "
            f"distinct4={slop_audit.get('distinct_4gram_ratio', 0.0):.3f}",
            flush=True,
        )

    _free_model(model_a)

    return {
        "schema_version": "1.0",
        "lora_svd_profiles": profiles,
        "adapter_comparisons": comparisons,
        "slop_vector_steering": {
            "peak_diff_layer": peak_diff_layer,
            "raw_diff_norm": round(float(slop_vec.norm().item()), 4),
            "calibrated_unit_scale": round(0.25 * res_norm, 4),
            "n_diff_prompts": len(diff_rows),
            "n_sweep_prompts": len(sweep_rows),
            "sweep": sweep_results,
        },
    }


# ---------------------------------------------------------------------------
# INTERP #3: Logit Lens + Causal <think> Sabotage (Anthropic/Lanham)
# INTERP #4: Attention-Head Circuit Hunting + Causal Zero-Ablation Control
# ---------------------------------------------------------------------------
def run_interp3_and_interp4(
    model_path: str,
    puzzles_path: str,
    *,
    target_solved: int = 100,
    max_scan_puzzles: int = 750,
    batch_size: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("\n=== INTERP #3 & #4: Logit Lens, CoT Sabotage & Attention Circuits ===", flush=True)
    puzzles = _read_jsonl(puzzles_path, limit=max_scan_puzzles)
    model, tok = _load_model_and_tok(model_path, attn_implementation="sdpa")
    device = next(model.parameters()).device

    prompts = [
        build_chat_prompt(
            p["fen"],
            tokenizer=tok,
            include_ascii=False,
            include_piece_list=True,
            enable_thinking=True,
        )
        for p in puzzles
    ]
    print(
        f"  [INTERP #3] Scanning up to {len(puzzles)} puzzles to collect {target_solved} solved...",
        flush=True,
    )
    solved_items: list[dict[str, Any]] = []
    for start in range(0, len(puzzles), batch_size):
        batch_p = puzzles[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        texts = _generate_greedy(
            model, tok, batch_prompts, max_new_tokens=160, batch_size=batch_size
        )
        for p, pr, txt in zip(batch_p, batch_prompts, texts, strict=True):
            mv = parse_move_from_output(txt)
            target = p.get("best_move") or p.get("solution", [""])[0]
            if mv == target and "</think>" in txt:
                solved_items.append(
                    {
                        "fen": p["fen"],
                        "target_move": target,
                        "prompt": pr,
                        "raw_output": txt,
                        "theme": p.get("primary_theme", ""),
                    }
                )
        if len(solved_items) >= target_solved:
            break

    solved_items = solved_items[:target_solved]
    print(f"  [INTERP #3] Collected {len(solved_items)} solved puzzles with </think>.", flush=True)

    # 3A: Token-Position Logit Lens across <think> progress + post_</think> decision token
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    lens_records: list[dict[str, Any]] = []
    post_think_probs: list[float] = []
    post_think_top1: list[float] = []

    for item in solved_items:
        raw = item["raw_output"]
        think_end_char = raw.find("</think>")
        think_body = raw[:think_end_char]
        # Advance past </think> and whitespace to the exact position predicting the UCI move
        after_close = think_end_char + len("</think>")
        mv_pos = raw.find(item["target_move"], after_close)
        pre_move_raw = raw[:mv_pos] if mv_pos != -1 else raw[:after_close]

        prompt_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
        think_ids = tok(item["prompt"] + think_body, add_special_tokens=False)["input_ids"]
        full_ids = tok(item["prompt"] + pre_move_raw, add_special_tokens=False)["input_ids"]
        n_think = max(len(think_ids) - len(prompt_ids), 1)
        offsets = sample_think_checkpoints(n_think, fractions=fractions)

        # First token of the target UCI move (check both bare and newline-prefixed)
        target_tok_ids = tok(item["target_move"], add_special_tokens=False)["input_ids"]
        if not target_tok_ids:
            continue
        first_target_id = int(target_tok_ids[0])

        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(input_ids=inp, output_hidden_states=True, use_cache=False)
            last_h = out.hidden_states[-1][0]  # (seq_len, hidden_dim)
            prob_by_f: dict[str, float] = {}
            top1_by_f: dict[str, bool] = {}
            for f, off in zip(fractions, offsets, strict=True):
                pos = min(len(prompt_ids) + off, last_h.shape[0] - 1)
                logits = lm_head(final_norm(last_h[pos : pos + 1]))[0].float()
                probs = torch.softmax(logits, dim=-1)
                prob_by_f[str(f)] = float(probs[first_target_id].item())
                top1_by_f[str(f)] = int(logits.argmax().item()) == first_target_id

            # Decision token right after </think> + whitespace
            dec_logits = lm_head(final_norm(last_h[-1:]))[0].float()
            dec_probs = torch.softmax(dec_logits, dim=-1)
            post_think_probs.append(float(dec_probs[first_target_id].item()))
            post_think_top1.append(
                1.0 if int(dec_logits.argmax().item()) == first_target_id else 0.0
            )

        lens_records.append(
            {
                "fen": item["fen"],
                "target_move": item["target_move"],
                "n_think_tokens": n_think,
                "prob_by_fraction": prob_by_f,
                "top1_by_fraction": top1_by_f,
            }
        )

    lens_summary = summarize_logit_lens(lens_records, fractions=fractions)
    lens_summary["post_think_mean_prob"] = round(
        sum(post_think_probs) / max(len(post_think_probs), 1), 5
    )
    lens_summary["post_think_top1_match"] = round(
        sum(post_think_top1) / max(len(post_think_top1), 1), 4
    )

    # 3B: Anthropic/Lanham Corrupted-<think> Prefill Intervention
    sabotage_prompts: list[str] = []
    sabotage_meta: list[tuple[str, str]] = []
    for item in solved_items:
        try:
            prefill, distractor = build_corrupted_think_prefill(item["fen"], item["target_move"])
        except ValueError:
            continue
        # Strip leading <think> if prompt already ends with <think>\n
        if item["prompt"].rstrip().endswith("<think>") and prefill.startswith("<think>"):
            prefill = prefill[len("<think>") :].lstrip("\n")
        sabotage_prompts.append(item["prompt"] + prefill)
        sabotage_meta.append((item["target_move"], distractor))

    sabotage_outs = _generate_greedy(
        model, tok, sabotage_prompts, max_new_tokens=16, batch_size=batch_size
    )
    flipped_from_target = 0
    adopted_distractor = 0
    still_target = 0
    sabotage_examples: list[dict[str, Any]] = []
    for (tgt, dist), out_txt in zip(sabotage_meta, sabotage_outs, strict=True):
        mv = parse_move_from_output(out_txt)
        if mv == tgt:
            still_target += 1
        else:
            flipped_from_target += 1
        if mv == dist:
            adopted_distractor += 1
        if len(sabotage_examples) < 5:
            sabotage_examples.append(
                {
                    "target_move": tgt,
                    "distractor_move": dist,
                    "post_sabotage_move": mv,
                    "raw_tail": out_txt.strip()[:40],
                }
            )

    n_sab = max(len(sabotage_meta), 1)
    interp3_report = {
        "schema_version": "1.0",
        "model_path": model_path,
        "num_solved_analyzed": len(solved_items),
        "logit_lens": lens_summary,
        "causal_cot_sabotage": {
            "num_tested": len(sabotage_meta),
            "causal_move_flip_rate": round(flipped_from_target / n_sab, 4),
            "adopted_distractor_rate": round(adopted_distractor / n_sab, 4),
            "resisted_sabotage_rate": round(still_target / n_sab, 4),
            "examples": sabotage_examples,
        },
    }
    print(
        f"  [INTERP #3] Logit lens P(0%->100%->post): "
        f"{lens_summary['mean_prob_by_fraction']['0.0']:.4f} -> "
        f"{lens_summary['mean_prob_by_fraction']['1.0']:.4f} -> "
        f"{lens_summary['post_think_mean_prob']:.4f} | "
        f"Sabotage flip={flipped_from_target / n_sab:.2%}, adopted_distractor={adopted_distractor / n_sab:.2%}",
        flush=True,
    )

    _free_model(model)

    # ------------------------------------------------------------------
    # INTERP #4: Reload with eager attention for exact attention weights & head zero-ablation
    # ------------------------------------------------------------------
    print(
        f"  [INTERP #4] Reloading {model_path} with eager attention for circuit hunting...",
        flush=True,
    )
    model_e, tok = _load_model_and_tok(model_path, attn_implementation="eager")
    device = next(model_e.parameters()).device
    num_heads = int(model_e.config.num_attention_heads)
    head_dim = int(getattr(model_e.config, "head_dim", model_e.config.hidden_size // num_heads))

    per_example_scores: list[dict[tuple[int, int], float]] = []
    # Run eager attention extraction on up to 60 solved puzzles (fast, exact, low VRAM)
    attn_items = solved_items[:60]
    for item in attn_items:
        mv_uci = item["target_move"]
        board = chess.Board(item["fen"])
        target_squares = {mv_uci[:2], mv_uci[2:4]}
        # Also include opponent king square (tactical forks/checks)
        opp_king = board.king(not board.turn)
        if opp_king is not None:
            target_squares.add(chess.square_name(opp_king))

        raw = item["raw_output"]
        close_idx = raw.find("</think>")
        after_close = close_idx + len("</think>")
        mv_pos = raw.find(mv_uci, after_close)
        prefix_to_move = item["prompt"] + (raw[:mv_pos] if mv_pos != -1 else raw[:after_close])

        prompt_ids = tok(item["prompt"], add_special_tokens=False)["input_ids"]
        prompt_tok_strs = [tok.decode([tid]) for tid in prompt_ids]
        tgt_idx, bg_idx = find_square_token_indices(prompt_tok_strs, target_squares)
        if not tgt_idx:
            continue

        full_ids = tok(prefix_to_move, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model_e(input_ids=inp, output_attentions=True, use_cache=False)
        query_pos = len(full_ids) - 1
        scores = score_attention_heads(
            out.attentions,
            query_pos=query_pos,
            target_indices=tgt_idx,
            background_indices=bg_idx,
        )
        per_example_scores.append(scores)
        del out
        torch.cuda.empty_cache()

    top_heads = rank_piece_tracker_heads(per_example_scores, top_k=5)
    random_heads = sample_layer_matched_random_heads(top_heads, num_heads, seed=42)
    print(
        f"  [INTERP #4] Scored {len(per_example_scores)} puzzles | Top-5 Piece-Tracker Heads: {top_heads}",
        flush=True,
    )

    def _eval_with_zero_ablation(head_list: list[dict[str, Any]]) -> dict[str, Any]:
        by_layer: dict[int, list[int]] = {}
        for h in head_list:
            by_layer.setdefault(int(h["layer"]), []).append(int(h["head"]))

        handles = []
        for layer_idx, heads in by_layer.items():
            o_proj = model_e.model.layers[layer_idx].self_attn.o_proj

            def _make_pre_hook(h_indices: list[int]):
                def _pre_hook(_mod, args):
                    x = args[0].clone()
                    for hi in h_indices:
                        x[..., hi * head_dim : (hi + 1) * head_dim] = 0.0
                    return (x, *args[1:])

                return _pre_hook

            handles.append(o_proj.register_forward_pre_hook(_make_pre_hook(heads)))

        try:
            eval_prompts = [it["prompt"] for it in solved_items]
            outs = _generate_greedy(model_e, tok, eval_prompts, max_new_tokens=160, batch_size=32)
        finally:
            for hdl in handles:
                hdl.remove()

        still_solved = 0
        still_legal = 0
        for it, txt in zip(solved_items, outs, strict=True):
            mv = parse_move_from_output(txt)
            if mv == it["target_move"]:
                still_solved += 1
            if mv:
                try:
                    if chess.Move.from_uci(mv) in chess.Board(it["fen"]).legal_moves:
                        still_legal += 1
                except ValueError:
                    pass
        n = max(len(solved_items), 1)
        return {
            "retained_solve_rate": round(still_solved / n, 4),
            "solve_drop_pp": round(100.0 * (1.0 - still_solved / n), 2),
            "retained_legal_rate": round(still_legal / n, 4),
        }

    top_ablation = _eval_with_zero_ablation(top_heads)
    rand_ablation = _eval_with_zero_ablation(random_heads)
    _free_model(model_e)

    interp4_report = {
        "schema_version": "1.0",
        "model_path": model_path,
        "num_attention_examples": len(per_example_scores),
        "num_ablation_puzzles": len(solved_items),
        "baseline_solve_rate_on_subset": 1.0,
        "top5_piece_tracker_heads": top_heads,
        "layer_matched_random_heads": random_heads,
        "top5_zero_ablation": top_ablation,
        "random5_zero_ablation": rand_ablation,
        "causal_excess_drop_pp": round(
            top_ablation["solve_drop_pp"] - rand_ablation["solve_drop_pp"], 2
        ),
    }
    print(
        f"  [INTERP #4] Top-5 ablation drop={top_ablation['solve_drop_pp']:.2f} pp vs "
        f"Random-5 drop={rand_ablation['solve_drop_pp']:.2f} pp",
        flush=True,
    )
    return interp3_report, interp4_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChessFM Mechanistic Interpretability Suite")
    parser.add_argument("--output-dir", default="/vol/results/interp")
    parser.add_argument("--fens", default="/vol/data/phase2/test_fens_1000.jsonl")
    parser.add_argument("--puzzles", default="/vol/data/phase2/test_puzzles_900.jsonl")
    parser.add_argument("--coach-test", default="/vol/data/phase2/test_coach_300.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B")
    parser.add_argument("--e0-model", default="/vol/checkpoints/sft_qwen3_4b/final_merged")
    parser.add_argument("--e1a-model", default="/vol/checkpoints/sft_e1a_piece_list/final_merged")
    parser.add_argument("--e2a-model", default="/vol/checkpoints/sft_e2a_20k_r64/final_merged")
    parser.add_argument(
        "--e3b-model", default="/vol/checkpoints/grpo_e3b_sharp_wdl/stage1_final_merged"
    )
    parser.add_argument("--e3a-adapter", default="/vol/checkpoints/grpo_e3a_dapo_wdl/stage1_final")
    parser.add_argument("--e3b-adapter", default="/vol/checkpoints/grpo_e3b_sharp_wdl/stage1_final")
    parser.add_argument("--arm-a-adapter", default="/vol/checkpoints/e5_arm_a_grpo/stage0_final")
    parser.add_argument("--arm-c-adapter", default="/vol/checkpoints/e5_arm_c_v1_grpo/stage0_final")
    parser.add_argument(
        "--arm-a-merged", default="/vol/checkpoints/e5_arm_a_grpo/stage0_final_merged"
    )
    parser.add_argument(
        "--arm-c-merged", default="/vol/checkpoints/e5_arm_c_v1_grpo/stage0_final_merged"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip stages whose output JSON already exists in --output-dir.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    p1 = out_dir / "interp1_board_probes_and_steering.json"
    t1 = time.monotonic()
    if args.resume and p1.exists():
        print(f"  [INTERP #1] --resume: skipping existing {p1}", flush=True)
    else:
        r1 = run_interp1_board_probes(
            args.fens,
            checkpoints=[
                {"name": "Base_ASCII", "path": args.base_model, "include_ascii": True},
                {"name": "E0_ASCII", "path": args.e0_model, "include_ascii": True},
                {"name": "E1a_PieceList", "path": args.e1a_model, "include_piece_list": True},
                {"name": "E2a_r64", "path": args.e2a_model, "include_piece_list": True},
                {
                    "name": "E3b_SharpWDL",
                    "path": args.e3b_model,
                    "include_piece_list": True,
                    "run_steering": True,
                },
            ],
        )
        p1.write_text(json.dumps(r1, indent=2) + "\n", encoding="utf-8")
    dt1 = round(time.monotonic() - t1, 2)

    p2 = out_dir / "interp2_lora_svd_and_slop_steering.json"
    t2 = time.monotonic()
    if args.resume and p2.exists():
        print(f"  [INTERP #2] --resume: skipping existing {p2}", flush=True)
    else:
        r2 = run_interp2_lora_svd_and_slop_steering(
            adapters={
                "e3a_apawn_hack": args.e3a_adapter,
                "e3b_sharp_wdl": args.e3b_adapter,
                "arm_a_v2_grpo": args.arm_a_adapter,
                "arm_c_v1_grpo": args.arm_c_adapter,
            },
            arm_a_merged=args.arm_a_merged,
            arm_c_merged=args.arm_c_merged,
            coach_test_path=args.coach_test,
        )
        p2.write_text(json.dumps(r2, indent=2) + "\n", encoding="utf-8")
    dt2 = round(time.monotonic() - t2, 2)

    t34 = time.monotonic()
    r3, r4 = run_interp3_and_interp4(args.e3b_model, args.puzzles, target_solved=100)
    (out_dir / "interp3_logit_lens_and_cot_sabotage.json").write_text(
        json.dumps(r3, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "interp4_attention_circuits_and_ablation.json").write_text(
        json.dumps(r4, indent=2) + "\n", encoding="utf-8"
    )
    dt34 = round(time.monotonic() - t34, 2)

    total_s = round(time.monotonic() - t0, 2)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    rate = 2.50 / 3600.0 if "A100" in gpu_name else 0.80 / 3600.0
    manifest = {
        "schema_version": "1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gpu": gpu_name,
        "timings_seconds": {
            "interp1": dt1,
            "interp2": dt2,
            "interp3_and_4": dt34,
            "total": total_s,
        },
        "estimated_active_cost_usd": round(total_s * rate, 4),
    }
    (out_dir / "interp_suite_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== INTERP SUITE COMPLETE ===\n" + json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
