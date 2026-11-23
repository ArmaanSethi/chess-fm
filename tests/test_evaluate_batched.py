"""Unit tests for batched decoding, rolling batched matches, and --stockfish-elo."""

from __future__ import annotations

import json
import math
from typing import Any

import chess
import chess.engine
import chess.pgn
import pytest
import torch

import chessfm.rl.evaluate as evaluate
from chessfm.rl.evaluate import (
    MATCH_ENGINE_MAX_RESTARTS,
    MIN_STOCKFISH_ELO,
    BatchedLegalMoveLogitsProcessor,
    _play_engine_move,
    generate_completions_batched,
    play_games_batched,
)

START_FEN = chess.STARTING_FEN
# White king on h1, rook on a2 blocking? No: White king h1, black king h3, black queen g3 -> only legal move is h1g1?
# Let's use a verified single-legal-move FEN: White King h1, Black King f2, Black Rook a2 -> only legal move is h1g1 (Wait: g1 attacked? No, f2 attacks g1! Let's use Black King f1, Black Pawn g3? Let's verify cleanly:
# White King h1, Black King f2, Black Pawn g2 -> h1g2 is illegal (defended by f2), h1g1 is illegal (attacked by f2), wait.
# White King h1, Black King h3, Black Pawn g3 -> h1g1 is the ONLY legal move (g2 attacked by h3 and g3, h2 attacked by h3 and g3).
SINGLE_LEGAL_FEN = "8/8/8/8/8/6pk/8/7K w - - 0 1"
# White pawn on e7 with no obstructions, White King a1, Black King a8 -> e7e8q/r/b/n + king moves.
PROMOTION_FEN = "k7/4P3/8/8/8/8/8/K7 w - - 0 1"


class _CharTokenizer:
    """Deterministic character-level tokenizer supporting left-padded batching."""

    def __init__(self, *, pad_token_id: int | None = None, padding_side: str = "right"):
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789<> /.,:-_\n()|+=#?!")
        self._tok2id: dict[str, int] = {"<eos>": 0, "<pad>": 1}
        for ch in chars:
            if ch not in self._tok2id:
                self._tok2id[ch] = len(self._tok2id)
        self._id2tok: dict[int, str] = {v: k for k, v in self._tok2id.items()}
        self.eos_token_id = 0
        self.eos_token = "<eos>"
        self.pad_token_id = pad_token_id
        self.pad_token = "<pad>" if pad_token_id is not None else None
        self.padding_side = padding_side
        self.vocab_size = len(self._tok2id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for ch in text.lower():
            if ch in self._tok2id:
                ids.append(self._tok2id[ch])
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        out = []
        for idx in ids:
            idx_int = int(idx)
            if skip_special_tokens and idx_int in (self.eos_token_id, self.pad_token_id, 1):
                continue
            tok = self._id2tok.get(idx_int, "")
            if skip_special_tokens and tok in ("<eos>", "<pad>"):
                continue
            out.append(tok)
        return "".join(out)

    def __call__(self, prompts, return_tensors=None, padding: bool = False):
        if isinstance(prompts, str):
            encoded = [self.encode(prompts)]
        else:
            encoded = [self.encode(p) for p in prompts]

        pad_id = self.pad_token_id if self.pad_token_id is not None else self.eos_token_id
        max_len = max((len(seq) for seq in encoded), default=0)
        padded_rows = []
        attn_rows = []
        for seq in encoded:
            pad_len = max_len - len(seq)
            if self.padding_side == "left":
                padded_rows.append([pad_id] * pad_len + seq)
                attn_rows.append([0] * pad_len + [1] * len(seq))
            else:
                padded_rows.append(seq + [pad_id] * pad_len)
                attn_rows.append([1] * len(seq) + [0] * pad_len)

        input_ids = torch.tensor(padded_rows, dtype=torch.long)
        attention_mask = torch.tensor(attn_rows, dtype=torch.long)

        class _Batch(dict):
            def __init__(self, ids_t, mask_t):
                super().__init__(input_ids=ids_t, attention_mask=mask_t)
                self.input_ids = ids_t
                self.attention_mask = mask_t

            def to(self, _device):
                return self

        return _Batch(input_ids, attention_mask)


class _AdversarialBatchedModel:
    """Greedy batched model that prefers an illegal move unless masked by logits_processor."""

    def __init__(
        self,
        tokenizer: _CharTokenizer,
        preferred_text: str = "<think>go</think>z9z9",
    ) -> None:
        self.device = "cpu"
        self.tokenizer = tokenizer
        self.preferred_ids = tokenizer.encode(preferred_text) + [tokenizer.eos_token_id]
        self.generate_calls: list[dict[str, Any]] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        input_ids: torch.Tensor = kwargs["input_ids"].clone()
        max_new_tokens: int = int(kwargs.get("max_new_tokens", 32))
        logits_processors = kwargs.get("logits_processor") or []
        eos_id = self.tokenizer.eos_token_id
        batch_size = input_ids.shape[0]
        finished = [False] * batch_size

        for step in range(max_new_tokens):
            if all(finished):
                break
            scores = torch.zeros((batch_size, self.tokenizer.vocab_size), dtype=torch.float32)
            # Strongly prefer the adversarial token at this step.
            pref = self.preferred_ids[step] if step < len(self.preferred_ids) else eos_id
            scores[:, pref] = 10.0
            # Give slight deterministic tie-breaking preference across vocab.
            scores += torch.arange(self.tokenizer.vocab_size, dtype=torch.float32) * 1e-4

            for proc in logits_processors:
                scores = proc(input_ids, scores)

            assert not torch.isnan(scores).any(), "Logits processor produced NaN scores!"
            next_tokens = scores.argmax(dim=-1)
            for row in range(batch_size):
                if finished[row]:
                    next_tokens[row] = eos_id
                elif int(next_tokens[row].item()) == eos_id:
                    finished[row] = True
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)

        return input_ids


class _DeterministicScriptedModel:
    """Batched and single-item compatible model returning scripted completions per prompt FEN."""

    def __init__(self, tokenizer: _CharTokenizer, reply_fn):
        self.device = "cpu"
        self.tokenizer = tokenizer
        self.reply_fn = reply_fn
        self.batch_sizes_seen: list[int] = []

    def generate(self, **kwargs):
        input_ids: torch.Tensor = kwargs["input_ids"]
        batch_size = int(input_ids.shape[0])
        self.batch_sizes_seen.append(batch_size)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        completions_ids = []
        for row in range(batch_size):
            prompt_text = self.tokenizer.decode(input_ids[row], skip_special_tokens=True)
            reply = self.reply_fn(prompt_text)
            completions_ids.append(self.tokenizer.encode(reply) + [self.tokenizer.eos_token_id])
        max_gen = max(len(c) for c in completions_ids)
        gen_rows = [c + [pad_id] * (max_gen - len(c)) for c in completions_ids]
        gen_tensor = torch.tensor(gen_rows, dtype=torch.long)
        return torch.cat([input_ids, gen_tensor], dim=1)


# ---------------------------------------------------------------------------
# BatchedLegalMoveLogitsProcessor Tests
# ---------------------------------------------------------------------------


def test_batched_logits_processor_mixed_states_and_zero_leakage():
    tok = _CharTokenizer(pad_token_id=1, padding_side="left")
    # Confirm SINGLE_LEGAL_FEN really has only h1g1.
    assert [m.uci() for m in chess.Board(SINGLE_LEGAL_FEN).legal_moves] == ["h1g1"]

    fens = [START_FEN, SINGLE_LEGAL_FEN, PROMOTION_FEN, SINGLE_LEGAL_FEN]
    prompt_len = 6
    proc = BatchedLegalMoveLogitsProcessor(
        fens, tok, prompt_length=prompt_len, require_think_close=True
    )

    # Row 0: still inside <think> -> completely unconstrained.
    # Row 1: after </think> with prefix "h1" on SINGLE_LEGAL_FEN -> only 'g' allowed!
    # Row 2: after </think> with complete move "e7e8q" on PROMOTION_FEN -> only EOS allowed!
    # Row 3: already emitted complete move "h1g1" + EOS + pad -> no NaN, EOS fallback!
    seqs = [
        [1, 1, 2, 3, 4, 5] + tok.encode("<think>aaaaaaaaaaaaaa"),
        [1, 2, 3, 4, 5, 6] + tok.encode("<think>aaaa</think>h1"),
        [2, 3, 4, 5, 6, 7] + tok.encode("<think>a</think>e7e8q"),
        [1, 1, 1, 2, 3, 4] + tok.encode("<think></think>h1g1") + [tok.eos_token_id, 1],
    ]
    assert len({len(s) for s in seqs}) == 1
    input_ids = torch.tensor(seqs, dtype=torch.long)
    scores = torch.zeros((4, tok.vocab_size), dtype=torch.float32)

    out = proc(input_ids, scores)
    assert out.shape == scores.shape
    assert not torch.isnan(out).any()

    # Row 0: unconstrained (all finite).
    assert torch.isfinite(out[0]).all()

    # Row 1: ONLY 'g' is finite (not 'e', 'a', etc. from START_FEN/PROMOTION_FEN).
    g_id = tok.encode("g")[0]
    e_id = tok.encode("e")[0]
    assert math.isfinite(out[1, g_id].item())
    assert math.isinf(out[1, e_id].item()) and out[1, e_id].item() < 0
    finite_row1 = torch.where(torch.isfinite(out[1]))[0].tolist()
    assert finite_row1 == [g_id]

    # Row 2: move complete ("e7e8q"), only EOS allowed.
    finite_row2 = torch.where(torch.isfinite(out[2]))[0].tolist()
    assert finite_row2 == [tok.eos_token_id]

    # Row 3: finished row (EOS + pad), no NaN, falls back to EOS.
    finite_row3 = torch.where(torch.isfinite(out[3]))[0].tolist()
    assert finite_row3 == [tok.eos_token_id]


def test_batched_logits_processor_require_think_close_false_and_mismatch():
    tok = _CharTokenizer(pad_token_id=1, padding_side="left")
    proc = BatchedLegalMoveLogitsProcessor(
        [SINGLE_LEGAL_FEN, START_FEN], tok, prompt_length=4, require_think_close=False
    )

    input_ids = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], dtype=torch.long)
    scores = torch.zeros((2, tok.vocab_size), dtype=torch.float32)
    out = proc(input_ids, scores)

    # Row 0 (SINGLE_LEGAL_FEN): only 'h' (plus optional leading whitespace) is allowed immediately!
    h_id = tok.encode("h")[0]
    finite_row0 = torch.where(torch.isfinite(out[0]))[0].tolist()
    assert h_id in finite_row0
    assert tok.encode("e")[0] not in finite_row0
    assert all(tok.decode([tid]).strip() in ("", "h") for tid in finite_row0)
    # Row 1 (START_FEN): 'e' (e2e4) is allowed, 'h' (h2h3/h2h4) is allowed, 'z' is masked.
    assert math.isfinite(out[1, tok.encode("e")[0]].item())
    assert math.isinf(out[1, tok.encode("z")[0]].item())

    # Batch size mismatch raises ValueError.
    with pytest.raises(ValueError, match="Batch size mismatch"):
        proc(input_ids[:1], scores[:1])


# ---------------------------------------------------------------------------
# generate_completions_batched Tests
# ---------------------------------------------------------------------------


def test_generate_completions_batched_left_pads_and_restores_tokenizer_state():
    tok = _CharTokenizer(pad_token_id=None, padding_side="right")
    model = _AdversarialBatchedModel(tok, preferred_text="<think>hi</think>e2e4")

    prompts = ["short", "a much longer prompt string here"]
    completions = generate_completions_batched(
        model, tok, prompts, max_new_tokens=30, decoding_mode="strict"
    )
    assert len(completions) == 2
    assert all("e2e4" in c for c in completions)
    # Tokenizer state must be restored exactly.
    assert tok.padding_side == "right"
    assert tok.pad_token_id is None
    assert tok.pad_token is None

    # Also verify restoration on exception.
    class _FailingModel:
        device = "cpu"

        def generate(self, **_kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        generate_completions_batched(_FailingModel(), tok, prompts, decoding_mode="strict")
    assert tok.padding_side == "right"
    assert tok.pad_token_id is None


def test_generate_completions_batched_strict_vs_constrained_and_validation():
    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _AdversarialBatchedModel(tok, preferred_text="<think>bad</think>z9z9")
    fens = [SINGLE_LEGAL_FEN, START_FEN, PROMOTION_FEN]
    prompts = ["p1", "longer prompt 2", "longest prompt number 3"]

    strict_out = generate_completions_batched(
        model, tok, prompts, max_new_tokens=30, decoding_mode="strict"
    )
    assert all("z9z9" in c for c in strict_out)

    constrained_out = generate_completions_batched(
        model, tok, prompts, max_new_tokens=30, decoding_mode="constrained", fens=fens
    )
    assert len(constrained_out) == 3
    for fen, comp in zip(fens, constrained_out, strict=True):
        move_str = evaluate.parse_move_from_output(comp)
        assert move_str is not None
        assert chess.Move.from_uci(move_str) in chess.Board(fen).legal_moves
    # Row 0 only has h1g1.
    assert evaluate.parse_move_from_output(constrained_out[0]) == "h1g1"

    # Edge cases & validation.
    assert generate_completions_batched(model, tok, [], decoding_mode="strict") == []
    with pytest.raises(ValueError, match="Unsupported decoding_mode"):
        generate_completions_batched(model, tok, prompts, decoding_mode="beam")
    with pytest.raises(ValueError, match="fens must be provided"):
        generate_completions_batched(
            model, tok, prompts, decoding_mode="constrained", fens=[START_FEN]
        )


# ---------------------------------------------------------------------------
# play_games_batched & evaluate_model Parity + Rolling Refill Tests
# ---------------------------------------------------------------------------


def test_evaluate_model_batched_parity_with_sequential(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    openings_file = tmp_path / "openings.txt"
    fen_1 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    fen_2 = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    openings_file.write_text(f"{fen_1}\n{fen_2}\n", encoding="utf-8")

    tok = _CharTokenizer(pad_token_id=1, padding_side="right")

    # Deterministic model that plays g1f3 when white can, else illegal z9z9 -> quick forfeit.
    def _reply(_prompt: str) -> str:
        return "<think>develop</think>g1f3"

    model_seq = _DeterministicScriptedModel(tok, _reply)
    manifest_seq = tmp_path / "manifest_seq.json"
    res_seq = evaluate.evaluate_model(
        model_seq,
        tok,
        "fake-parity",
        num_games=4,
        use_random_opponent=True,
        openings=openings_file,
        seed=17,
        batch_size=1,
        manifest_output=manifest_seq,
    )

    model_bat = _DeterministicScriptedModel(tok, _reply)
    manifest_bat = tmp_path / "manifest_bat.json"
    res_bat = evaluate.evaluate_model(
        model_bat,
        tok,
        "fake-parity",
        num_games=4,
        use_random_opponent=True,
        openings=openings_file,
        seed=17,
        batch_size=4,
        manifest_output=manifest_bat,
    )

    assert res_bat["batch_size"] == 4
    assert res_seq["batch_size"] == 1
    assert max(model_bat.batch_sizes_seen) > 1
    assert res_bat["num_games"] == res_seq["num_games"] == 4
    assert res_bat["color_balanced"] is True
    assert res_bat["white"]["games"] == 2
    assert res_bat["black"]["games"] == 2
    assert res_bat["wins"] == res_seq["wins"]
    assert res_bat["losses"] == res_seq["losses"]
    assert res_bat["draws"] == res_seq["draws"]

    # Check PGN headers and paired opening order.
    with open(res_bat["pgn_file"], encoding="utf-8") as f:
        games = []
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)
    assert len(games) == 4
    assert [g.headers["Event"] for g in games] == [
        "ChessFM Eval game 1",
        "ChessFM Eval game 2",
        "ChessFM Eval game 3",
        "ChessFM Eval game 4",
    ]
    assert [g.headers["FEN"] for g in games] == [fen_1, fen_1, fen_2, fen_2]
    assert [g.headers["ModelColor"] for g in games] == ["white", "black", "white", "black"]

    m_bat = json.loads(manifest_bat.read_text(encoding="utf-8"))
    assert m_bat["tiers"]["tier4_match"]["color_balanced"] is True
    assert m_bat["tiers"]["tier4_match"]["stockfish_elo"] is None


def test_play_games_batched_rolling_refill_and_index_order():
    tok = _CharTokenizer(pad_token_id=1, padding_side="right")

    # Game 0: white starts from SINGLE_LEGAL_FEN; model plays "z9z9" -> forfeits immediately on ply 1.
    # Game 1: white starts from START_FEN; model plays "e2e4" then "d2d4" then forfeits -> longer game.
    # Game 2: black starts from START_FEN; model plays "z9z9" -> forfeits after white's random opener.
    # Game 3: white starts from SINGLE_LEGAL_FEN; model plays "h1g1" (legal) then forfeits on next ply.
    def _reply(prompt: str) -> str:
        if "6pk/8/7k" in prompt:
            return "<think>only move</think>h1g1"
        if "rnbqkbnr/pppppppp/8/8/8/8/pppppppp/rnbqkbnr" in prompt:
            return "<think>open</think>e2e4"
        return "<think>illegal</think>z9z9"

    model = _DeterministicScriptedModel(tok, _reply)
    specs = [
        (0, "white", "8/8/8/8/8/8/4P3/K6k w - - 0 1"),  # immediate forfeit (z9z9)
        (1, "white", START_FEN),  # plays e2e4, then forfeits on turn 2
        (2, "black", START_FEN),  # forfeits on turn 1
        (3, "white", SINGLE_LEGAL_FEN),  # plays h1g1, then forfeits
    ]

    results = play_games_batched(
        model,
        tok,
        specs,
        engine=None,
        batch_size=2,
        use_random_opponent=True,
    )
    assert len(results) == 4
    # Game 0 forfeited immediately (0 legal moves, 1 total move).
    assert results[0][1]["legal_moves"] == 0
    assert results[0][1]["illegal_by_model"] == 1
    # Game 1 played e2e4 legally then forfeited.
    assert results[1][1]["legal_moves"] == 1
    assert results[1][1]["illegal_by_model"] == 1
    # Game 2 forfeited as black -> 1-0.
    assert results[2][1]["result"] == "1-0"
    # Game 3 played h1g1 legally.
    assert results[3][1]["legal_moves"] >= 1


def test_play_games_batched_constrained_no_forfeits():
    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _AdversarialBatchedModel(tok, preferred_text="<think>bad</think>z9z9")

    specs = [
        (0, "white", SINGLE_LEGAL_FEN),
        (1, "white", SINGLE_LEGAL_FEN),
    ]
    orig_init = evaluate._BatchedGameState.__init__

    def _short_init(self, *args, **kwargs):
        kwargs["max_moves"] = 4
        orig_init(self, *args, **kwargs)

    evaluate._BatchedGameState.__init__ = _short_init
    try:
        results = play_games_batched(
            model,
            tok,
            specs,
            engine=None,
            batch_size=2,
            use_random_opponent=True,
            decoding_mode="constrained",
        )
    finally:
        evaluate._BatchedGameState.__init__ = orig_init

    assert len(results) == 2
    for _game, stats in results:
        assert stats["illegal_by_model"] == 0
        assert stats["legal_moves"] == stats["total_moves"]
        assert stats["legal_moves"] >= 1


# ---------------------------------------------------------------------------
# --stockfish-elo & CLI Validation Tests
# ---------------------------------------------------------------------------


def test_evaluate_model_stockfish_elo_configures_engine_and_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    configured_options: list[dict[str, Any]] = []

    class _FakeEngine:
        def configure(self, opts):
            configured_options.append(dict(opts))

        def play(self, board, _limit):
            move = next(iter(board.legal_moves))
            return chess.engine.PlayResult(move=move, ponder=None)

    class _FakeEnv:
        def __init__(self):
            self.engine = _FakeEngine()
            self.board = chess.Board()

        def reset(self, fen=None):
            self.board = chess.Board(fen) if fen else chess.Board()
            return self.board.fen()

        def get_fen(self):
            return self.board.fen()

        def get_turn(self):
            return "white" if self.board.turn == chess.WHITE else "black"

        def get_legal_moves(self):
            return [m.uci() for m in self.board.legal_moves]

        def step(self, move_uci):
            from chessfm.rl.chess_env import StepResult

            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                return StepResult(done=True, legal=False, info={})
            if move not in self.board.legal_moves:
                return StepResult(done=True, legal=False, info={})
            self.board.push(move)
            return StepResult(done=False, legal=True, info={})

        def close(self):
            pass

    monkeypatch.setattr(evaluate, "ChessEnv", _FakeEnv)

    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _DeterministicScriptedModel(tok, lambda _p: "<think>x</think>z9z9")
    manifest_path = tmp_path / "sf_elo_manifest.json"

    res = evaluate.evaluate_model(
        model,
        tok,
        "fake-sf-elo",
        num_games=2,
        stockfish_elo=1320,
        opponent_depth=6,
        batch_size=2,
        manifest_output=manifest_path,
    )

    assert configured_options == [{"UCI_LimitStrength": True, "UCI_Elo": 1320}]
    assert res["stockfish_elo"] == 1320
    assert res["batch_size"] == 2
    assert "Stockfish Elo 1320" in res["opponent"]
    assert res["estimated_elo"] is not None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tier4 = manifest["tiers"]["tier4_match"]
    assert tier4["opponent_elo"] == 1320
    assert tier4["stockfish_elo"] == 1320
    assert "Stockfish Elo 1320" in tier4["opponent_name"]


def test_evaluate_model_and_cli_validation_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _DeterministicScriptedModel(tok, lambda _p: "e2e4")

    # stockfish_elo < 1320 rejected in Python API and CLI.
    with pytest.raises(ValueError, match=f"stockfish_elo must be >= {MIN_STOCKFISH_ELO}"):
        evaluate.evaluate_model(model, tok, "fake", num_games=1, stockfish_elo=1000)

    monkeypatch.setattr("sys.argv", ["evaluate", "--stockfish-elo", "1000"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2

    # Mutually exclusive opponent flags rejected in Python API and CLI.
    with pytest.raises(ValueError, match="at most one opponent override"):
        evaluate.evaluate_model(model, tok, "fake", num_games=1, stockfish_elo=1320, maia_elo=1100)

    with pytest.raises(ValueError, match="at most one opponent override"):
        evaluate.evaluate_model(
            model, tok, "fake", num_games=1, stockfish_elo=1320, use_random_opponent=True
        )

    monkeypatch.setattr("sys.argv", ["evaluate", "--stockfish-elo", "1320", "--maia-elo", "1100"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2

    monkeypatch.setattr("sys.argv", ["evaluate", "--stockfish-elo", "1320", "--random-opponent"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2

    # batch_size < 1 rejected in Python API and CLI.
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        evaluate.evaluate_model(model, tok, "fake", num_games=1, batch_size=0)

    monkeypatch.setattr("sys.argv", ["evaluate", "--batch-size", "0"])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Engine Crash Resilience Tests
# ---------------------------------------------------------------------------


def test_play_engine_move_and_batched_survive_five_consecutive_crashes():
    class _FlakyEngine:
        def __init__(self, fail_times: int, exc_cls=chess.engine.EngineTerminatedError):
            self.remaining_failures = fail_times
            self.exc_cls = exc_cls
            self.restart_calls = 0
            self.play_calls = 0

        def _restart(self) -> None:
            self.restart_calls += 1

        def play(self, board: chess.Board, _limit):
            self.play_calls += 1
            if self.remaining_failures > 0:
                self.remaining_failures -= 1
                raise self.exc_cls("engine process died unexpectedly (exit code: -8)")
            move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
            return chess.engine.PlayResult(move=move, ponder=None)

    # Direct helper survives a 5-consecutive EngineTerminatedError burst.
    engine = _FlakyEngine(fail_times=5)
    board = chess.Board()
    move_uci = _play_engine_move(engine, board, opponent_depth=4)
    assert chess.Move.from_uci(move_uci) in board.legal_moves
    assert engine.restart_calls == 5
    assert engine.play_calls == 6

    # Also recovers from BrokenPipeError.
    pipe_engine = _FlakyEngine(fail_times=2, exc_cls=BrokenPipeError)
    move_uci_pipe = _play_engine_move(pipe_engine, board, opponent_depth=4)
    assert chess.Move.from_uci(move_uci_pipe) in board.legal_moves
    assert pipe_engine.restart_calls == 2

    # Exhausting retries re-raises EngineTerminatedError.
    doomed_engine = _FlakyEngine(fail_times=10)
    with pytest.raises(chess.engine.EngineTerminatedError):
        _play_engine_move(doomed_engine, board, opponent_depth=4, max_retries=3)
    assert doomed_engine.restart_calls == 3
    assert doomed_engine.play_calls == 4

    # Full play_games_batched survives a 5-crash burst on the opponent's opening ply.
    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _DeterministicScriptedModel(tok, lambda _p: "<think>x</think>z9z9")
    batched_engine = _FlakyEngine(fail_times=5)
    specs = [(0, "black", START_FEN), (1, "black", START_FEN)]
    results = play_games_batched(
        model,
        tok,
        specs,
        engine=batched_engine,
        batch_size=2,
        use_random_opponent=False,
        opponent_depth=4,
    )
    assert len(results) == 2
    assert batched_engine.restart_calls == 5
    assert all(stats["result"] == "1-0" for _, stats in results)


def test_evaluate_model_bumps_engine_max_restarts(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "PGNS_DIR", tmp_path / "pgns")
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path)

    created_engines: list[Any] = []

    class _ResilientLikeEngine:
        def __init__(self, initial_restarts: int = 2):
            self._max_restarts = initial_restarts

        def configure(self, _opts):
            pass

        def play(self, board, _limit):
            move = next(iter(board.legal_moves))
            return chess.engine.PlayResult(move=move, ponder=None)

    class _FakeEnv:
        next_initial_restarts = 2

        def __init__(self):
            self.engine = _ResilientLikeEngine(self.next_initial_restarts)
            created_engines.append(self.engine)
            self.board = chess.Board()

        def reset(self, fen=None):
            self.board = chess.Board(fen) if fen else chess.Board()
            return self.board.fen()

        def get_fen(self):
            return self.board.fen()

        def get_turn(self):
            return "white" if self.board.turn == chess.WHITE else "black"

        def get_legal_moves(self):
            return [m.uci() for m in self.board.legal_moves]

        def step(self, move_uci):
            from chessfm.rl.chess_env import StepResult

            try:
                move = chess.Move.from_uci(move_uci)
            except ValueError:
                return StepResult(done=True, legal=False, info={})
            if move not in self.board.legal_moves:
                return StepResult(done=True, legal=False, info={})
            self.board.push(move)
            return StepResult(done=False, legal=True, info={})

        def close(self):
            pass

    monkeypatch.setattr(evaluate, "ChessEnv", _FakeEnv)
    tok = _CharTokenizer(pad_token_id=1, padding_side="right")
    model = _DeterministicScriptedModel(tok, lambda _p: "<think>x</think>z9z9")

    # Default _max_restarts=2 is bumped to MATCH_ENGINE_MAX_RESTARTS.
    evaluate.evaluate_model(
        model,
        tok,
        "fake-bump",
        num_games=2,
        stockfish_elo=1320,
        opponent_depth=4,
        batch_size=2,
    )
    assert created_engines[-1]._max_restarts == MATCH_ENGINE_MAX_RESTARTS

    # Higher pre-configured _max_restarts is preserved (never downgraded).
    _FakeEnv.next_initial_restarts = MATCH_ENGINE_MAX_RESTARTS + 8
    evaluate.evaluate_model(
        model,
        tok,
        "fake-keep-higher",
        num_games=2,
        stockfish_elo=1320,
        opponent_depth=4,
        batch_size=2,
    )
    assert created_engines[-1]._max_restarts == MATCH_ENGINE_MAX_RESTARTS + 8
