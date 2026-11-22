#!/usr/bin/env python3
"""Run reproducible local CPU + Stockfish-16 benchmarks for ChessFM.

Evaluates deterministic baselines and the full ChessFM Coach stack across all
4 tiers (Static Engine, Board-State Probing, ACT-Eval Coaching Faithfulness,
and Paired Color-Balanced Matches) without requiring a GPU.

Usage:
    venv/bin/python scripts/run_local_benchmarks.py --benchmark data_generation/eval_benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import chess

from chessfm.coach import (
    CoachOrchestrator,
    MaiaSidecar,
    generate_socratic_hint,
    review_pgn_game,
)
from chessfm.eval.suite import (
    evaluate_board_probes,
    evaluate_coaching_faithfulness,
    evaluate_static_positions,
    summarize_paired_match,
)
from chessfm.fens import read_jsonl
from chessfm.rl.engine import close_engine, open_engine

SAMPLE_PGNS = [
    """[Event "Benchmark Game 1"]
[White "Student"]
[Black "Opponent"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. Ng5 Qxg5 5. d3 Qxg2 6. Rf1 d6 7. Qh5 Bh3 8. Qxf7+ Kd8 9. Nd2 Nh6 10. Qh5 Rf8 0-1
""",
    """[Event "Benchmark Game 2"]
[White "Student"]
[Black "Opponent"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0
""",
]


def _maia_predict_fn(sidecar: MaiaSidecar, elo: int):
    def _predict(fen: str) -> str:
        pred = sidecar.predict_human_moves(fen, student_elo=elo, top_k=1)
        move = pred["top_moves"][0][0] if pred["top_moves"] else "0000"
        return f"<think>Selecting {elo}-calibrated candidate {move}.</think>\n{move}"

    return _predict


def _random_legal_predict_fn(seed: int = 42):
    rng = random.Random(seed)

    def _predict(fen: str) -> str:
        moves = [m.uci() for m in chess.Board(fen).legal_moves]
        move = rng.choice(moves) if moves else "0000"
        return f"<think>Sampling a legal move.</think>\n{move}"

    return _predict


def _oracle_probe_answer(probe: dict) -> str:
    """Deterministic python-chess oracle validating Tier-2 probe ground truth."""
    return str(probe["expected"])


def _run_paired_stockfish_match(
    sidecar: MaiaSidecar,
    engine,
    games: int = 10,
    student_elo: int = 1800,
    opponent_depth: int = 1,
    max_plies: int = 80,
) -> dict:
    fast_sidecar = MaiaSidecar(use_neural=False, engine=None)
    records = []
    for idx in range(games):
        model_color = "white" if idx % 2 == 0 else "black"
        board = chess.Board()
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < max_plies:
            turn_color = "white" if board.turn == chess.WHITE else "black"
            if turn_color == model_color:
                pred = fast_sidecar.predict_human_moves(
                    board.fen(), student_elo=student_elo, top_k=1
                )
                move = pred["top_moves"][0][0]
            else:
                res = engine.play(board, chess.engine.Limit(depth=opponent_depth))
                move = res.move.uci()
            board.push_uci(move)
            plies += 1
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
        records.append({"model_color": model_color, "result": result, "plies": plies})
    return summarize_paired_match(
        records,
        opponent_name=f"Stockfish-depth-{opponent_depth}",
        opponent_elo=1000,
        model_id=f"maia-sidecar-{student_elo}-{sidecar.backend}",
        decoding_mode="constrained",
    )


def run_benchmarks(benchmark_path: Path, depth: int = 8, match_games: int = 10) -> dict:
    t0 = time.perf_counter()
    rows = read_jsonl(benchmark_path)
    fens = [r["fen"] for r in rows]

    engine = open_engine()
    if engine is None:
        raise SystemExit("Stockfish binary is required to run local benchmarks.")

    try:
        sidecar = MaiaSidecar(use_neural=False, engine=None)
        orchestrator = CoachOrchestrator(engine=engine, maia_sidecar=sidecar)

        # Tier 1: Static Engine Metrics across baselines
        tier1_random = evaluate_static_positions(
            rows, _random_legal_predict_fn(seed=42), engine=engine, depth=depth
        )
        tier1_maia_1200 = evaluate_static_positions(
            rows, _maia_predict_fn(sidecar, elo=1200), engine=engine, depth=depth
        )
        tier1_maia_2000 = evaluate_static_positions(
            rows, _maia_predict_fn(sidecar, elo=2000), engine=engine, depth=depth
        )

        # Tier 2: Board-State Probing
        tier2_oracle = evaluate_board_probes(fens, _oracle_probe_answer)

        # Tier 3: Coaching Faithfulness (ACT-Eval) + Socratic 0%-spoiler check
        coach_cases = []
        socratic_cases = []
        for row in rows:
            fen = row["fen"]
            best_move = row["best_move"]
            coach_cases.append({"fen": fen, "best_move": best_move})
            socratic_cases.append({"fen": fen, "best_move": best_move, "level": 1})
            socratic_cases.append({"fen": fen, "best_move": best_move, "level": 2})

        def _coach_fn(case: dict) -> str:
            res = orchestrator.coach(
                case["fen"],
                user_question="Explain the best plan in this position.",
                student_elo=1400,
            )
            return res["response"]

        def _socratic_fn(case: dict) -> str:
            hint = generate_socratic_hint(
                case["fen"],
                case["best_move"],
                level=case["level"],
            )
            return hint["hint"]

        tier3_coach = evaluate_coaching_faithfulness(coach_cases, _coach_fn, engine=engine)
        tier3_socratic = evaluate_coaching_faithfulness(socratic_cases, _socratic_fn, engine=None)

        # Tier 4: Paired Color-Balanced Match
        tier4_match = _run_paired_stockfish_match(
            sidecar, engine, games=match_games, student_elo=1800, opponent_depth=1
        )

        # PGN Game Review latency & output check
        review_summaries = []
        for pgn in SAMPLE_PGNS:
            rev = review_pgn_game(pgn, engine=engine, student_color="white", depth=depth)
            review_summaries.append(
                {
                    "accuracy": rev["accuracy"],
                    "counts": rev["counts"],
                    "critical_moments": len(rev["critical_moments"]),
                }
            )

        elapsed = round(time.perf_counter() - t0, 2)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmark_file": str(benchmark_path),
            "positions_evaluated": len(rows),
            "stockfish_depth": depth,
            "elapsed_seconds": elapsed,
            "tier1_static": {
                "random_legal": tier1_random,
                "maia_fallback_1200": tier1_maia_1200,
                "maia_fallback_2000": tier1_maia_2000,
            },
            "tier2_probing_oracle": tier2_oracle,
            "tier3_coaching_faithfulness": {
                "coach_explanations": {k: v for k, v in tier3_coach.items() if k != "cases"},
                "socratic_l1_l2": {k: v for k, v in tier3_socratic.items() if k != "cases"},
            },
            "tier4_paired_match": tier4_match,
            "pgn_review_smoke": review_summaries,
        }
    finally:
        close_engine(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ChessFM CPU + Stockfish benchmarks")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data_generation/eval_benchmark.jsonl"),
        help="Path to Stockfish-verified eval benchmark JSONL",
    )
    parser.add_argument("--depth", type=int, default=8, help="Stockfish evaluation depth")
    parser.add_argument("--match-games", type=int, default=10, help="Paired match games for Tier 4")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report",
    )
    args = parser.parse_args()
    report = run_benchmarks(args.benchmark, depth=args.depth, match_games=args.match_games)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
