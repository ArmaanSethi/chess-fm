#!/usr/bin/env python3
"""Comprehensive unit tests for deterministic python-chess tactical fact extraction."""

from __future__ import annotations

import json

import chess
import pytest

from chessfm.coach.tactical_facts import (
    HangingPieceFact,
    PieceSquareRef,
    TacticalFacts,
    compute_tactical_fact_coverage,
    extract_tactical_facts,
    format_tactical_facts_for_prompt,
)
from chessfm.coach.verifier import _has_center_tension
from chessfm.rl.rewards import cp_to_expected_score

EXPECTED_KEYS = {
    "fen",
    "side_to_move",
    "piece_moved",
    "piece_moved_type",
    "from_square",
    "to_square",
    "best_move_uci",
    "best_move_san",
    "student_move_uci",
    "student_move_san",
    "captured_piece",
    "promotion_piece",
    "is_castling",
    "is_en_passant",
    "gives_check",
    "is_checkmate",
    "is_stalemate",
    "has_center_tension",
    "board_zone",
    "hanging_pieces_with_attackers",
    "student_hanging_after_move",
    "pieces_attacked_by_best_move",
    "pieces_defended",
    "tactical_motifs",
    "tactical_motif",
    "cp_loss",
    "wdl_loss",
    "wdl_loss_estimated",
    "best_pv_san",
}


class TestDataclassesAndValidation:
    def test_piece_square_ref_and_hanging_fact_helpers(self):
        ref = PieceSquareRef(color="black", piece="bishop", square="b5")
        assert ref.label() == "black bishop on b5"
        h = HangingPieceFact(
            color="white",
            piece="queen",
            square="f6",
            attackers=("black bishop on e5",),
            defenders=(),
            reason="undefended",
        )
        d = h.to_dict()
        assert d["piece"] == "white queen on f6"
        assert d["reason"] == "undefended"
        assert d["attackers"] == ["black bishop on e5"]

    def test_schema_keys_and_uci_or_san_acceptance(self):
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        facts_uci = extract_tactical_facts(
            fen,
            student_move="a2a3",
            best_move="f3g5",
            pv=["f3g5", "d7d5", "e4d5"],
            cp_loss=85,
        )
        facts_san = extract_tactical_facts(
            fen,
            student_move="a3",
            best_move="Ng5",
            pv=["Ng5", "d5", "exd5"],
            cp_loss=85,
        )
        assert set(facts_uci.keys()) == EXPECTED_KEYS
        assert facts_uci == facts_san
        assert facts_uci["best_move_san"] == "Ng5"
        assert facts_uci["student_move_san"] == "a3"
        assert facts_uci["best_pv_san"] == ["Ng5", "d5", "exd5"]
        assert "black pawn on f7" in facts_uci["pieces_attacked_by_best_move"]
        assert isinstance(TacticalFacts(**facts_uci), TacticalFacts)

    def test_invalid_fen_and_illegal_best_move_raise(self):
        with pytest.raises(ValueError):
            extract_tactical_facts("not-a-fen", best_move="e2e4")
        with pytest.raises(ValueError):
            extract_tactical_facts(chess.STARTING_FEN, best_move="e2e5")
        with pytest.raises(ValueError):
            extract_tactical_facts(chess.STARTING_FEN, best_move=None, pv=None)

    def test_student_move_none_or_invalid_tolerated(self):
        facts_none = extract_tactical_facts(chess.STARTING_FEN, student_move=None, best_move="e2e4")
        assert facts_none["student_move_uci"] is None
        assert facts_none["student_move_san"] is None
        assert facts_none["student_hanging_after_move"] == []

        facts_bad = extract_tactical_facts(
            chess.STARTING_FEN, student_move="z9z9", best_move="e2e4"
        )
        assert facts_bad["student_move_uci"] is None
        assert facts_bad["student_hanging_after_move"] == []

    def test_wdl_loss_exact_vs_proxy_vs_none(self):
        exact = extract_tactical_facts(
            chess.STARTING_FEN,
            best_move="e2e4",
            cp_loss=150,
            eval_before_cp=200,
            eval_after_cp=50,
        )
        expected_exact = round(max(0.0, cp_to_expected_score(200) - cp_to_expected_score(50)), 4)
        assert exact["wdl_loss"] == pytest.approx(expected_exact)
        assert exact["wdl_loss_estimated"] is False

        proxy = extract_tactical_facts(chess.STARTING_FEN, best_move="e2e4", cp_loss=150)
        expected_proxy = round(
            max(0.0, cp_to_expected_score(0.0) - cp_to_expected_score(-150.0)), 4
        )
        assert proxy["wdl_loss"] == pytest.approx(expected_proxy)
        assert proxy["wdl_loss_estimated"] is True

        none_wdl = extract_tactical_facts(chess.STARTING_FEN, best_move="e2e4")
        assert none_wdl["wdl_loss"] is None
        assert none_wdl["wdl_loss_estimated"] is False

    @pytest.mark.parametrize(
        "fen",
        [
            chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "2b3k1/1p2rp1p/r3p1p1/p7/8/2P3P1/PP1R1PBP/3R2K1 w - - 0 21",
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        ],
    )
    def test_has_center_tension_matches_verifier(self, fen: str):
        board = chess.Board(fen)
        move = next(iter(board.legal_moves))
        facts = extract_tactical_facts(fen, best_move=move.uci())
        assert facts["has_center_tension"] is _has_center_tension(board)


class TestElevenTacticalPatterns:
    def test_01_castling_kingside_and_queenside(self):
        # Kingside castling: rook lands on f1 attacking f7 pawn and defending f2 pawn.
        fen_oo = "r3k2r/pppp1ppp/8/8/8/8/PPP2PPP/R3K2R w KQkq - 0 1"
        oo = extract_tactical_facts(fen_oo, best_move="e1g1")
        assert oo["is_castling"] == "kingside"
        assert "castling" in oo["tactical_motifs"]
        assert "king_safety" in oo["tactical_motifs"]
        assert "white pawn on f2" in oo["pieces_defended"]

        # Queenside castling: rook lands on d1 attacking d7 pawn directly on open d-file.
        ooo = extract_tactical_facts(fen_oo, best_move="O-O-O")
        assert ooo["is_castling"] == "queenside"
        assert "castling" in ooo["tactical_motifs"]
        assert "black pawn on d7" in ooo["pieces_attacked_by_best_move"]
        assert "white pawn on c2" in ooo["pieces_defended"]

    def test_02_en_passant_and_discovered_attack(self):
        # White pawn on e5 captures Black pawn on d5 en passant (e5d6),
        # vacating BOTH e5 and d5 on the 5th rank so White rook on h5 discovers
        # an attack on Black queen on a5!
        fen = "4k3/8/8/q2pP2R/8/8/8/6K1 w - d6 0 1"
        facts = extract_tactical_facts(fen, best_move="e5d6")
        assert facts["is_en_passant"] is True
        assert facts["captured_piece"] == "black pawn on d5 (en passant)"
        assert "en_passant" in facts["tactical_motifs"]
        assert "black queen on a5" in facts["pieces_attacked_by_best_move"]
        assert "discovered_attack" in facts["tactical_motifs"]

    def test_03_promotion_and_underpromotion(self):
        # Queen promotion attacks enemy rook on h8 and king on a8.
        fen_q = "k6r/4P3/8/8/8/8/8/4K3 w - - 0 1"
        fq = extract_tactical_facts(fen_q, best_move="e7e8q")
        assert fq["piece_moved_type"] == "pawn"
        assert fq["promotion_piece"] == "queen"
        assert "promotion" in fq["tactical_motifs"]
        assert "black rook on h8" in fq["pieces_attacked_by_best_move"]
        assert fq["gives_check"] is True

        # Underpromotion to knight forks king on c7 and queen on g7.
        fen_n = "8/2k1P1q1/8/8/8/8/8/4K3 w - - 0 1"
        fn = extract_tactical_facts(fen_n, best_move="e7e8n")
        assert fn["promotion_piece"] == "knight"
        assert "underpromotion" in fn["tactical_motifs"]
        assert "promotion" in fn["tactical_motifs"]
        assert "fork" in fn["tactical_motifs"]
        assert "black king on c7" in fn["pieces_attacked_by_best_move"]
        assert "black queen on g7" in fn["pieces_attacked_by_best_move"]

    def test_04_checkmate_back_rank_mate_and_stalemate(self):
        # Classic back-rank mate: Re8# against boxed king on g8.
        fen_mate = "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1"
        fm = extract_tactical_facts(fen_mate, best_move="e1e8", pv=["e1e8"])
        assert fm["gives_check"] is True
        assert fm["is_checkmate"] is True
        assert fm["is_stalemate"] is False
        assert "back_rank_mate" in fm["tactical_motifs"]
        assert "checkmate" in fm["tactical_motifs"]
        assert fm["best_pv_san"] == ["Re8#"]

        # Stalemate: Qg6 leaves Black king on h8 with no legal moves and not in check.
        fen_stale = "7k/5Q2/8/8/8/8/8/6K1 w - - 0 1"
        fs = extract_tactical_facts(fen_stale, best_move="f7g6")
        assert fs["is_stalemate"] is True
        assert fs["gives_check"] is False
        assert fs["is_checkmate"] is False
        assert "stalemate" in fs["tactical_motifs"]

    def test_05_absolute_and_relative_pin(self):
        # Absolute pin: with d7 pawn on d6, Bb5 pins Black knight on c6 to Black king on e8.
        fen_abs = "r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
        fa = extract_tactical_facts(fen_abs, best_move="f1b5")
        assert "pin" in fa["tactical_motifs"]

        # Relative pin: Bg5 pins Black knight on f6 against Black queen on d8 (king on g8).
        fen_rel = "r1bq1rk1/pppp1ppp/2n2n2/4p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQ - 0 5"
        fr = extract_tactical_facts(fen_rel, best_move="c1g5")
        assert "pin" in fr["tactical_motifs"]

    def test_06_skewer(self):
        # Clean rook skewer on rank 8: Ra8+ skewers Black king on d8 and Black queen on g8.
        fen = "3k2q1/8/8/8/8/8/8/R6K w - - 0 1"
        facts = extract_tactical_facts(fen, best_move="a1a8")
        assert "skewer" in facts["tactical_motifs"]
        assert facts["gives_check"] is True

    def test_07_fork(self):
        # Knight on d4 moves to e2? Classic royal fork: Nd4-e2+ forks king on g1 and queen on c3.
        fen = "6k1/5p1p/4p3/4q3/3n4/2Q3P1/PP1N1P1P/6K1 b - - 3 37"
        facts = extract_tactical_facts(fen, best_move="d4e2")
        assert "fork" in facts["tactical_motifs"]
        assert facts["gives_check"] is True
        assert "white king on g1" in facts["pieces_attacked_by_best_move"]
        assert "white queen on c3" in facts["pieces_attacked_by_best_move"]

    def test_08_discovered_attack_and_discovered_check(self):
        # Discovered attack: Ne4->c5 discovers Re1 attack on Black queen on e7.
        fen_da = "4k3/4q3/8/8/4N3/8/8/4R1K1 w - - 0 1"
        fda = extract_tactical_facts(fen_da, best_move="e4c5")
        assert "discovered_attack" in fda["tactical_motifs"]
        assert "black queen on e7" in fda["pieces_attacked_by_best_move"]

        # Discovered check: Ne4->c5 discovers Re1 check on Black king on e8.
        fen_dc = "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1"
        fdc = extract_tactical_facts(fen_dc, best_move="e4c5")
        assert "discovered_check" in fdc["tactical_motifs"]
        assert fdc["gives_check"] is True
        assert "black king on e8" in fdc["pieces_attacked_by_best_move"]

    def test_09_hanging_pieces_root_and_student_blunder(self):
        # Real Case #185: Qf6?? puts White queen on f6 where Black bishop on e5 captures it!
        fen = "r1r3k1/q4ppp/8/p2pb3/N2p4/PP5P/2P2QP1/2BR2K1 w - - 0 23"
        facts = extract_tactical_facts(fen, student_move="f2f6", best_move="a4b2", cp_loss=600)
        assert facts["student_move_san"] == "Qf6"
        assert any(
            h["piece"] == "white queen on f6" and "black bishop on e5" in h["attackers"]
            for h in facts["student_hanging_after_move"]
        )
        # Also verify root hanging pieces with attackers & defenders (knight attacked by pawn).
        fen_root_hang = "4k3/8/8/3p4/4N3/8/8/4K3 w - - 0 1"
        frh = extract_tactical_facts(fen_root_hang, best_move="e4c3")
        assert any(
            h["piece"] == "white knight on e4"
            and "black pawn on d5" in h["attackers"]
            and h["reason"] == "undefended"
            for h in frh["hanging_pieces_with_attackers"]
        )

    def test_10_back_rank_threat(self):
        # Black king boxed on g8 behind f7/g7/h7; White plays Rd1-d7? No, Rd8 threatens/attacks back rank.
        # White rook moves to d7? Better: White rook on d2 moves to d8? If Black rook on a8 covers d8, it's not mate but back_rank_threat!
        fen = "r5k1/5ppp/8/8/8/8/3R4/6K1 w - - 0 1"
        facts = extract_tactical_facts(fen, best_move="d2d8")
        assert facts["is_checkmate"] is False
        assert "back_rank_threat" in facts["tactical_motifs"]

    def test_11_trapped_piece(self):
        # Classic Noah's Ark trap: Black plays c5-c4 attacking White bishop on b3;
        # a2/c2 are blocked by White pawns and a4/c4 are covered by Black's b5 pawn.
        fen_trap = "4k3/8/8/1pp5/p7/1B6/P1P5/4K3 b - - 0 1"
        facts = extract_tactical_facts(fen_trap, best_move="c5c4")
        assert "trapped_piece" in facts["tactical_motifs"]


class TestPromptFormattingAndCoverage:
    def test_format_tactical_facts_includes_explicit_booleans(self):
        fen = "3r1r1k/p1nb2R1/1p1p1n2/2pPpN2/2P1P3/P1NB3P/1P5K/6R1 b - - 1 28"
        facts = extract_tactical_facts(fen, student_move="b6b5", best_move="c7e8", cp_loss=180)
        block = format_tactical_facts_for_prompt(facts, task_type="postmortem")
        assert block.startswith("Tactical Facts (Verified):\n")
        parsed = json.loads(block.split("\n", 1)[1])
        assert parsed["gives_check"] is False
        assert parsed["is_checkmate"] is False
        assert "has_center_tension" in parsed
        assert parsed["side_to_move"] == "black"
        assert parsed["best_move_san"] == "Nce8"

    def test_coverage_concrete_and_spoiler_safe_socratic(self):
        fen = "6k1/5p1p/4p3/4q3/3n4/2Q3P1/PP1N1P1P/6K1 b - - 3 37"
        facts = extract_tactical_facts(
            fen,
            student_move="e5c3",
            best_move="d4e2",
            pv=["d4e2", "g1f1", "e2c3"],
            cp_loss=300,
        )
        # Concrete postmortem mentioning best move, student move, fork motif, and target square c3.
        good_pm = "Instead of Qxc3, play Ne2+ to fork the white king on g1 and queen on c3."
        assert compute_tactical_fact_coverage(good_pm, facts, task_type="postmortem") == 1.0

        # Socratic L3 referencing best move, PV reply Kf1, fork, and c3.
        good_l3 = "Play Ne2+ to fork the king and queen; after Kf1 you win the queen on c3."
        assert compute_tactical_fact_coverage(good_l3, facts, task_type="socratic_l3") == 1.0

        # Socratic L1 spoiler MUST receive 0.0 coverage!
        spoiled_l1 = "Play Ne2+ to fork the white king and queen."
        assert compute_tactical_fact_coverage(spoiled_l1, facts, task_type="socratic_l1") == 0.0

        # Clean Socratic L1 referencing knight/fork and queen/center without leaking squares.
        clean_l1 = "Look for a knight fork that targets the opponent's queen and king."
        assert compute_tactical_fact_coverage(clean_l1, facts, task_type="socratic_l1") == 1.0

        # Socratic L2 spoiler MUST receive 0.0 coverage!
        spoiled_l2 = "Jump your knight from d4 to e2 to attack the queen."
        assert compute_tactical_fact_coverage(spoiled_l2, facts, task_type="socratic_l2") == 0.0

        # Clean Socratic L2 referencing knight/fork, c3 target square, and queen.
        clean_l2 = "Look for a knight fork targeting the white queen on c3."
        assert compute_tactical_fact_coverage(clean_l2, facts, task_type="socratic_l2") == 1.0

        # Empty text -> 0.0
        assert compute_tactical_fact_coverage("", facts, task_type="postmortem") == 0.0
