"""Unit tests for chessfm.data_generation.coaching_dataset."""

import pytest

from chessfm.config import CHATML_ASSISTANT_CLOSE
from chessfm.data_generation.coaching_dataset import (
    audit_dataset_rows,
    build_best_line_sft_example,
    build_coaching_example,
    detect_tactical_motifs,
)


def test_detect_check_and_checkmate():
    # Fool's mate: Qh5#
    fen = "rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3"
    motifs = detect_tactical_motifs(fen, "d1h5")
    assert "check" in motifs
    assert "checkmate" in motifs


def test_detect_capture_and_hanging_piece_capture():
    # Undefended Black rook on e5 captured by White knight on f3? Use White bishop c3 capturing undefended rook e5.
    hanging_fen = "4k3/8/8/4r3/8/2B5/8/4K3 w - - 0 30"
    motifs = detect_tactical_motifs(hanging_fen, "c3e5")
    assert "capture" in motifs
    assert "hanging_piece_capture" in motifs

    # Defended Black rook on e5 (defended by d6 pawn) -> capture, NOT hanging_piece_capture.
    defended_fen = "4k3/8/3p4/4r3/8/2B5/8/4K3 w - - 0 30"
    motifs_def = detect_tactical_motifs(defended_fen, "c3e5")
    assert "capture" in motifs_def
    assert "hanging_piece_capture" not in motifs_def

    # En passant capture of an undefended pawn.
    ep_fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20"
    motifs_ep = detect_tactical_motifs(ep_fen, "e5d6")
    assert "capture" in motifs_ep
    assert "hanging_piece_capture" in motifs_ep


def test_detect_promotion():
    fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 40"
    motifs = detect_tactical_motifs(fen, "a7a8q")
    assert "promotion" in motifs
    assert "check" in motifs


def test_detect_castling():
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 10"
    motifs = detect_tactical_motifs(fen, "e1g1")
    assert "castling" in motifs
    assert "discovered_check" not in motifs


def test_detect_forks():
    # Knight fork: Nc7+ forks Black king on e8 and Black rook on a8.
    knight_fork_fen = "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 30"
    motifs = detect_tactical_motifs(knight_fork_fen, "d5c7")
    assert "fork" in motifs
    assert "check" in motifs

    # Pawn fork: e4-e5 forks Black knight on d6 and Black bishop on f6.
    pawn_fork_fen = "4k3/8/3n1b2/8/4P3/8/8/4K3 w - - 0 25"
    assert "fork" in detect_tactical_motifs(pawn_fork_fen, "e4e5")

    # Queen fork: Qd5+ attacks Black king on g8 and undefended Black rook on a2 along diagonals.
    queen_fork_fen = "6k1/8/8/8/8/8/r7/3QK3 w - - 0 30"
    assert "fork" in detect_tactical_motifs(queen_fork_fen, "d1d5")


def test_detect_pins_absolute_and_relative():
    # Absolute pin: d7 is vacant (Black played d6), so Bb5 pins Black knight on c6 to king on e8.
    abs_pin_fen = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    assert "pin" in detect_tactical_motifs(abs_pin_fen, "f1b5")

    # Relative pin: White bishop c1-g5 pins Black knight on f6 to Black queen on d8.
    rel_pin_fen = "r1bq1rk1/pppp1ppp/2n2n2/4p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQ - 0 5"
    assert "pin" in detect_tactical_motifs(rel_pin_fen, "c1g5")


def test_detect_discovered_check():
    # White rook on e1 behind White bishop on e4; Black king on e8.
    # Moving Be4-b7 reveals check from Re1!
    disc_fen = "4k3/8/8/8/4B3/8/8/4R1K1 w - - 0 30"
    motifs = detect_tactical_motifs(disc_fen, "e4b7")
    assert "check" in motifs
    assert "discovered_check" in motifs


def test_detect_tactical_motifs_invalid_raises():
    with pytest.raises(ValueError):
        detect_tactical_motifs("bad-fen", "e2e4")
    with pytest.raises(ValueError, match="Illegal move"):
        detect_tactical_motifs("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e5")
    with pytest.raises(ValueError, match="Invalid UCI"):
        detect_tactical_motifs(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "not-uci"
        )


def test_build_best_line_sft_example_valid():
    fen = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    row = build_best_line_sft_example(
        fen=fen,
        best_move_uci="f1b5",
        pv_ucis=["f1b5", "a7a6", "b5a4"],
        eval_cp=35.0,
        include_ascii=True,
    )
    assert row["fen"] == fen
    assert row["move"] == "f1b5"
    assert row["eval_cp"] == 35.0
    assert row["is_reasonable"] is True
    assert "pin" in row["motifs"]
    assert "Board:\n" in row["prompt"]
    assert row["completion"].startswith("<think>\n")
    assert row["completion"].endswith(f"</think>\nf1b5{CHATML_ASSISTANT_CLOSE}")
    assert "Bb5 a6 Ba4" in row["completion"]

    # Empty PV defaults to [best_move_uci] and include_ascii=False omits board.
    row_no_ascii = build_best_line_sft_example(
        fen=fen,
        best_move_uci="f1b5",
        pv_ucis=[],
        eval_cp=-15.0,
        include_ascii=False,
    )
    assert "Board:\n" not in row_no_ascii["prompt"]
    assert row_no_ascii["pv"] == ["f1b5"]


def test_build_best_line_sft_example_rejects_illegal_moves():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(ValueError, match="Illegal best_move"):
        build_best_line_sft_example(fen, "e2e5", ["e2e5"], 0.0)
    with pytest.raises(ValueError, match="must start with best_move_uci"):
        build_best_line_sft_example(fen, "e2e4", ["d2d4"], 0.0)
    with pytest.raises(ValueError, match="Illegal pv"):
        build_best_line_sft_example(fen, "e2e4", ["e2e4", "e2e5"], 0.0)


@pytest.mark.parametrize(
    "mode",
    ["postmortem", "socratic_l1", "socratic_l2", "socratic_l3", "why_not"],
)
def test_build_coaching_example_all_modes(mode):
    # Knight fork position: student plays quiet Ke2 instead of Nc7+ fork!
    fen = "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 30"
    row = build_coaching_example(
        fen=fen,
        student_move_uci="e1e2",
        best_move_uci="d5c7",
        pv_ucis=["d5c7", "e8d7", "c7a8"],
        cp_loss=480.0,
        student_elo=1200,
        maia_prob=0.38,
        mode=mode,
    )
    assert row["mode"] == mode
    assert row["student_move"] == "e1e2"
    assert row["best_move"] == "d5c7"
    assert row["severity"] == "blunder"
    assert "fork" in row["best_motifs"]
    assert row["completion"].endswith(CHATML_ASSISTANT_CLOSE)
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"

    assistant_text = row["messages"][1]["content"]
    if mode in ("socratic_l1", "socratic_l2"):
        # Socratic levels 1 & 2 must NOT leak the exact best move!
        assert "d5c7" not in assistant_text
        assert "Nc7" not in assistant_text
    if mode in ("postmortem", "why_not"):
        assert "d5c7" in assistant_text
        assert "38%" in assistant_text


def test_build_coaching_example_refutation_pv_and_tiers():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # PV starting *after* student_move f2f3 (refutation line e7e5) should be accepted.
    row_beg = build_coaching_example(
        fen=fen,
        student_move_uci="f2f3",
        best_move_uci="e2e4",
        pv_ucis=["e7e5", "g2g4"],
        cp_loss=75.0,
        student_elo=850,
        mode="postmortem",
    )
    assert row_beg["severity"] == "inaccuracy"
    assert "beginner" in row_beg["prompt"]

    row_adv = build_coaching_example(
        fen=fen,
        student_move_uci="d2d4",
        best_move_uci="e2e4",
        pv_ucis=[],
        cp_loss=10.0,
        student_elo=1950,
        mode="why_not",
    )
    assert row_adv["severity"] == "accurate"
    assert "advanced" in row_adv["prompt"]

    row_mid = build_coaching_example(
        fen=fen,
        student_move_uci="a2a3",
        best_move_uci="e2e4",
        pv_ucis=[],
        cp_loss=120.0,
        student_elo=1300,
        mode="socratic_l1",
    )
    assert row_mid["severity"] == "mistake"


def test_build_coaching_example_validation_errors():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    with pytest.raises(ValueError, match="Unsupported coaching mode"):
        build_coaching_example(fen, "e2e4", "e2e4", [], 0.0, mode="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cp_loss"):
        build_coaching_example(fen, "e2e4", "e2e4", [], -5.0)
    with pytest.raises(ValueError, match="maia_prob"):
        build_coaching_example(fen, "e2e4", "e2e4", [], 10.0, maia_prob=1.5)
    with pytest.raises(ValueError, match="Illegal student_move"):
        build_coaching_example(fen, "e2e5", "e2e4", [], 10.0)
    with pytest.raises(ValueError, match="Illegal pv"):
        build_coaching_example(fen, "e2e4", "e2e4", ["a1a8"], 10.0)


def test_audit_dataset_rows_passes_clean_rows():
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    sft_row = build_best_line_sft_example(fen, "f1b5", ["f1b5", "a7a6"], 30.0)
    coach_row = build_coaching_example(
        fen, "a2a3", "f1b5", ["f1b5", "a7a6"], cp_loss=55.0, mode="postmortem"
    )
    report = audit_dataset_rows([sft_row, coach_row])
    assert report["passed"] is True
    assert report["total_rows"] == 2
    assert report["illegal_moves"] == 0
    assert report["duplicate_trailing_moves"] == 0
    assert report["malformed_think_tags"] == 0
    assert report["blunders_labeled_reasonable"] == 0
    assert report["issues"] == []


def test_audit_dataset_rows_detects_all_failure_modes():
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    bad_rows = [
        # 1. Illegal move + illegal completion tail
        {
            "fen": start_fen,
            "move": "e2e5",
            "completion": f"<think>Bad move</think>\ne2e5{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 0,
        },
        # 2. Duplicate trailing move after </think>
        {
            "fen": start_fen,
            "move": "e2e4",
            "completion": f"<think>Central control</think>\ne2e4\ne2e4{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 0,
        },
        # 3. Duplicate trailing move pasted inside </think>
        {
            "fen": start_fen,
            "move": "e2e4",
            "completion": f"<think>Central control\ne2e4</think>\ne2e4{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 0,
        },
        # 4. Malformed think tags (unbalanced) + blunder > 200 cp labeled reasonable
        {
            "fen": start_fen,
            "move": "f2f3",
            "completion": f"<think><think>Oops</think>\nf2f3{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": True,
            "centipawn_loss": 250,
        },
        # 5. Reversed think tags + missing FEN
        {
            "fen": "",
            "move": "e2e4",
            "completion": f"</think>Oops<think>\ne2e4{CHATML_ASSISTANT_CLOSE}",
        },
        # 6. Empty think body + invalid FEN
        {
            "fen": "invalid-fen",
            "move": "e2e4",
            "completion": f"<think>   </think>\ne2e4{CHATML_ASSISTANT_CLOSE}",
        },
        # 7. Missing think tags on SFT row
        {
            "fen": start_fen,
            "move": "e2e4",
            "completion": f"e2e4{CHATML_ASSISTANT_CLOSE}",
            "is_reasonable": False,
            "centipawn_loss": 300,  # Not flagged because is_reasonable=False
        },
    ]

    report = audit_dataset_rows(bad_rows)
    assert report["passed"] is False
    assert report["total_rows"] == len(bad_rows)
    assert report["illegal_moves"] >= 3
    assert report["duplicate_trailing_moves"] == 2
    assert report["malformed_think_tags"] == 4
    assert report["blunders_labeled_reasonable"] == 1
    assert len(report["issues"]) >= 10
