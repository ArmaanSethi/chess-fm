"""Unit tests for chessfm.data_generation.phase_sampler."""

import chess
import pytest

from chessfm.data_generation.phase_sampler import (
    classify_game_phase,
    normalize_board_fen,
    parse_lichess_puzzle_row,
    phase_distribution,
    sample_balanced_positions,
    verify_zero_leakage,
)

# Representative FENs across all three game phases.
OPENING_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
]

MIDDLEGAME_FENS = [
    # Move 18, queens + rooks + minors still on board.
    "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 2 18",
    "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/R3KB1R w KQ - 1 12",
    "r2q1rk1/1pp2ppp/p1np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 15",
    "2rq1rk1/pp1bppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 4 14",
    # Queens still on board with 6 minor/major (8 non-pawn) -> middlegame.
    "3q1rk1/ppp2ppp/2n5/8/8/2N5/PPP2PPP/3Q1RK1 w - - 0 22",
]

ENDGAME_FENS = [
    # K+R vs K+R (4 non-pawn pieces).
    "8/5pk1/6p1/8/8/6P1/5PK1/3R3r w - - 0 40",
    # K+Q+R vs K+Q+R (6 non-pawn pieces -> <= 6 non-pawn endgame).
    "3q2k1/5ppp/8/8/8/8/5PPP/3Q1RK1 w - - 0 35",
    # No queens, 6 minor/major pieces (R+B+N vs R+B+N).
    "2r2bk1/pp3ppp/2n5/8/8/2N2B2/PP3PPP/2R3K1 w - - 0 30",
    # Pure pawn endgame (2 non-pawn pieces).
    "8/5pk1/8/5P2/8/8/6K1/8 w - - 0 45",
    # Early move number (< 10) with massive trades -> still endgame!
    "8/5pk1/6p1/8/8/6P1/5PK1/4R2r w - - 0 8",
]


def test_normalize_board_fen_strips_move_counters():
    fen1 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    fen2 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 7 19"
    assert normalize_board_fen(fen1) == normalize_board_fen(fen2)
    assert normalize_board_fen(fen1) == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3"


def test_normalize_board_fen_invalid_raises():
    with pytest.raises(ValueError, match="Invalid FEN"):
        normalize_board_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")


@pytest.mark.parametrize("fen", OPENING_FENS)
def test_classify_game_phase_opening(fen):
    assert classify_game_phase(fen) == "opening"


@pytest.mark.parametrize("fen", MIDDLEGAME_FENS)
def test_classify_game_phase_middlegame(fen):
    assert classify_game_phase(fen) == "middlegame"


@pytest.mark.parametrize("fen", ENDGAME_FENS)
def test_classify_game_phase_endgame(fen):
    assert classify_game_phase(fen) == "endgame"


def test_classify_game_phase_invalid_fen_raises():
    with pytest.raises(ValueError):
        classify_game_phase("not-a-valid-fen")


def test_phase_distribution_empty_and_nonempty():
    assert phase_distribution([]) == {"opening": 0.0, "middlegame": 0.0, "endgame": 0.0}

    mixed = OPENING_FENS[:2] + MIDDLEGAME_FENS[:5] + ENDGAME_FENS[:3]
    dist = phase_distribution(mixed)
    assert dist["opening"] == pytest.approx(0.2)
    assert dist["middlegame"] == pytest.approx(0.5)
    assert dist["endgame"] == pytest.approx(0.3)
    assert sum(dist.values()) == pytest.approx(1.0)


def test_sample_balanced_positions_stratifies_and_deduplicates():
    # Include duplicates with different move counters to verify board-FEN deduplication.
    dup_opening = OPENING_FENS[0].replace("0 1", "4 5")
    pool = OPENING_FENS + [dup_opening] + MIDDLEGAME_FENS + ENDGAME_FENS

    sampled = sample_balanced_positions(pool, total=10, ratios=(0.30, 0.50, 0.20), seed=42)
    assert len(sampled) == 10
    # All returned positions must have distinct normalized board FENs.
    assert len({normalize_board_fen(f) for f in sampled}) == 10

    dist = phase_distribution(sampled)
    assert dist["opening"] == pytest.approx(0.30)
    assert dist["middlegame"] == pytest.approx(0.50)
    assert dist["endgame"] == pytest.approx(0.20)

    # Deterministic given same seed.
    assert sampled == sample_balanced_positions(pool, total=10, ratios=(0.30, 0.50, 0.20), seed=42)


def test_sample_balanced_positions_redistributes_shortfall():
    # Only 1 opening available, but ratio requests 50% of 6 (=3).
    pool = OPENING_FENS[:1] + MIDDLEGAME_FENS[:4] + ENDGAME_FENS[:4]
    sampled = sample_balanced_positions(pool, total=6, ratios=(0.50, 0.25, 0.25), seed=7)
    assert len(sampled) == 6
    assert len({normalize_board_fen(f) for f in sampled}) == 6


def test_sample_balanced_positions_edge_cases():
    assert sample_balanced_positions([], total=5) == []
    assert sample_balanced_positions(OPENING_FENS, total=0) == []
    # Requesting more than unique pool returns all unique positions.
    sampled = sample_balanced_positions(OPENING_FENS[:3], total=10)
    assert len(sampled) == 3

    with pytest.raises(ValueError, match="non-negative"):
        sample_balanced_positions(OPENING_FENS, total=-1)
    with pytest.raises(ValueError, match="ratios"):
        sample_balanced_positions(OPENING_FENS, total=2, ratios=(0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="ratios"):
        sample_balanced_positions(OPENING_FENS, total=2, ratios=(-0.1, 0.6, 0.5))


def test_verify_zero_leakage_clean_and_leaky():
    clean = verify_zero_leakage(OPENING_FENS, MIDDLEGAME_FENS, ENDGAME_FENS, strict=True)
    assert clean["zero_leakage"] is True
    assert clean["total_overlap"] == 0
    assert clean["overlap_counts"] == {"train_val": 0, "train_test": 0, "val_test": 0}

    # Same board position with different move counters must still trigger leakage!
    transposed_leak = OPENING_FENS[0].replace("0 1", "12 25")
    leaky = verify_zero_leakage(
        OPENING_FENS,
        MIDDLEGAME_FENS + [transposed_leak],
        ENDGAME_FENS + [MIDDLEGAME_FENS[0]],
    )
    assert leaky["zero_leakage"] is False
    assert leaky["overlap_counts"]["train_val"] == 1
    assert leaky["overlap_counts"]["val_test"] == 1
    assert leaky["overlap_counts"]["train_test"] == 0
    assert leaky["total_overlap"] == 2

    with pytest.raises(ValueError, match="Position leakage detected"):
        verify_zero_leakage(OPENING_FENS, [transposed_leak], ENDGAME_FENS, strict=True)


def test_parse_lichess_puzzle_row_applies_setup_move():
    # Setup: Black plays f7f6 (blunder), White responds with d1h5 checkmate!
    pre_setup_fen = "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
    row = {
        "PuzzleId": "00001",
        "FEN": pre_setup_fen,
        "Moves": "g7g5 d1h5",
        "Rating": "850",
        "Themes": "mateIn1 opening short",
    }
    parsed = parse_lichess_puzzle_row(row)
    assert parsed is not None
    # Setup move g7g5 must already be applied to the returned puzzle FEN.
    expected_board = chess.Board(pre_setup_fen)
    expected_board.push_uci("g7g5")
    assert parsed["fen"] == expected_board.fen()
    assert parsed["best_move"] == "d1h5"
    assert parsed["line"] == ["d1h5"]
    assert parsed["rating"] == 850
    assert parsed["themes"] == ["mateIn1", "opening", "short"]
    assert parsed["phase"] == "opening"


def test_parse_lichess_puzzle_row_lowercase_and_list_inputs():
    row = {
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "moves": ["e2e4", "e7e5", "g1f3"],
        "rating": 1400,
        "themes": ["opening", "development"],
    }
    parsed = parse_lichess_puzzle_row(row)
    assert parsed is not None
    assert parsed["best_move"] == "e7e5"
    assert parsed["line"] == ["e7e5", "g1f3"]
    assert parsed["rating"] == 1400
    assert parsed["themes"] == ["opening", "development"]


@pytest.mark.parametrize(
    "bad_row",
    [
        {},
        {"FEN": "invalid-fen", "Moves": "e2e4 e7e5"},
        {"FEN": OPENING_FENS[0], "Moves": "e2e4"},  # Only 1 move (no solution)
        {"FEN": OPENING_FENS[0], "Moves": "e2e9 e7e5"},  # Malformed UCI setup move
        {"FEN": OPENING_FENS[0], "Moves": "e7e5 e2e4"},  # Illegal setup move for White
        {"FEN": OPENING_FENS[0], "Moves": "e2e4 e2e5"},  # Illegal solution move
        {"FEN": OPENING_FENS[0], "Moves": "e2e4 notuci"},  # Malformed solution UCI
        {"FEN": OPENING_FENS[0], "Moves": 12345},  # Non-string/non-list moves
        {"FEN": OPENING_FENS[0], "Moves": "e2e4 e7e5", "Rating": "not-an-int"},
    ],
)
def test_parse_lichess_puzzle_row_rejects_invalid(bad_row):
    assert parse_lichess_puzzle_row(bad_row) is None
