"""Unit tests for legal-move constrained decoding and DualModeGenerator."""

from __future__ import annotations

import chess
import pytest
import torch

from chessfm.config import build_chat_prompt
from chessfm.rl.constrained_decoding import (
    DualModeGenerator,
    LegalMoveLogitsProcessor,
    LegalMovePrefixTrie,
    legal_uci_moves,
)

# ---------------------------------------------------------------------------
# Lightweight mock tokenizer & model for deterministic unit tests
# ---------------------------------------------------------------------------


class MockChessTokenizer:
    """Character + special-tag tokenizer for fast, deterministic tests."""

    SPECIAL_TOKENS = (
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
    )

    def __init__(self, split_think_close: bool = False) -> None:
        self.split_think_close = split_think_close
        vocab: list[str] = ["<pad>", "<unk>"]
        vocab.extend(self.SPECIAL_TOKENS)
        vocab.extend(["</", "think", ">", ".</think>"])
        # Printable ASCII characters + newline
        vocab.append("\n")
        for code in range(32, 127):
            ch = chr(code)
            if ch not in vocab:
                vocab.append(ch)
        self.id_to_token: dict[int, str] = dict(enumerate(vocab))
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in self.id_to_token.items()}
        self.eos_token_id: int = self.token_to_id["<|im_end|>"]
        self.vocab_size: int = len(vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            if not self.split_think_close and text.startswith(".</think>", i):
                # Allow testing BPE merge into </think> explicitly when present.
                ids.append(self.token_to_id[".</think>"])
                i += len(".</think>")
                continue
            matched = False
            for tok in self.SPECIAL_TOKENS:
                if tok == "</think>" and self.split_think_close:
                    continue
                if text.startswith(tok, i):
                    ids.append(self.token_to_id[tok])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            if self.split_think_close and text.startswith("</think>", i):
                ids.extend(
                    [
                        self.token_to_id["</"],
                        self.token_to_id["think"],
                        self.token_to_id[">"],
                    ]
                )
                i += len("</think>")
                continue
            ch = text[i]
            ids.append(self.token_to_id.get(ch, self.token_to_id["<unk>"]))
            i += 1
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        pieces: list[str] = []
        for tid in token_ids:
            tok = self.id_to_token.get(int(tid), "")
            if skip_special_tokens and tok in ("<pad>", "<|im_start|>", "<|im_end|>"):
                continue
            pieces.append(tok)
        return "".join(pieces)

    def __call__(self, text: str, return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        ids = self.encode(text, add_special_tokens=False)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class AdversarialMockModel:
    """Autoregressive mock model that always prefers illegal move tokens unless constrained.

    Emits a fixed ``<think>...</think>`` prefix first, then chooses the highest-logit
    token at each step using ``logits_processor`` if supplied.
    """

    def __init__(
        self,
        tokenizer: MockChessTokenizer,
        think_prefix: str = "<think>Analyzing position carefully.</think>\n",
        preferred_illegal_text: str = "e2e5",
    ) -> None:
        self.tokenizer = tokenizer
        self.think_prefix_ids = tokenizer.encode(think_prefix, add_special_tokens=False)
        self.preferred_illegal_ids = tokenizer.encode(
            preferred_illegal_text, add_special_tokens=False
        )
        self.last_generate_kwargs: dict = {}

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_generate_kwargs = dict(kwargs)
        logits_processor = kwargs.get("logits_processor")
        max_new_tokens = kwargs.get("max_new_tokens", 32)
        vocab_size = self.tokenizer.vocab_size
        eos_id = self.tokenizer.eos_token_id

        seq = input_ids.clone()
        # Step 1: emit think prefix token by token (running logits_processor if present)
        for tid in self.think_prefix_ids:
            scores = torch.zeros((1, vocab_size), dtype=torch.float32)
            scores[0, tid] = 50.0
            if logits_processor is not None:
                scores = logits_processor(seq, scores)
            next_id = int(torch.argmax(scores, dim=-1).item())
            seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long)], dim=1)

        # Step 2: emit move tokens, strongly preferring illegal move tokens
        for step in range(max_new_tokens):
            scores = torch.zeros((1, vocab_size), dtype=torch.float32)
            if step < len(self.preferred_illegal_ids):
                scores[0, self.preferred_illegal_ids[step]] = 100.0
            else:
                scores[0, eos_id] = 100.0
            # Give slight deterministic preference among other tokens so argmax is well-defined
            scores[0] += torch.linspace(0.0, 0.01, steps=vocab_size)
            if logits_processor is not None:
                scores = logits_processor(seq, scores)
            next_id = int(torch.argmax(scores, dim=-1).item())
            seq = torch.cat([seq, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
            if next_id == eos_id:
                break
        return seq


class ScriptedBatchModel:
    """Mock model returning a fixed list of completions for Best-of-N tests."""

    def __init__(self, tokenizer: MockChessTokenizer, completions: list[str]) -> None:
        self.tokenizer = tokenizer
        self.completions = completions
        self.last_generate_kwargs: dict = {}

    def generate(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        self.last_generate_kwargs = dict(kwargs)
        num_return = kwargs.get("num_return_sequences", 1)
        prompt_ids = input_ids[0].tolist()
        eos_id = self.tokenizer.eos_token_id

        rows: list[list[int]] = []
        for idx in range(num_return):
            comp = self.completions[idx % len(self.completions)]
            comp_ids = self.tokenizer.encode(comp, add_special_tokens=False) + [eos_id]
            rows.append(prompt_ids + comp_ids)

        max_len = max(len(r) for r in rows)
        pad_id = self.tokenizer.token_to_id["<pad>"]
        padded = [r + [pad_id] * (max_len - len(r)) for r in rows]
        return torch.tensor(padded, dtype=torch.long)


# ---------------------------------------------------------------------------
# Fixtures / test FENs
# ---------------------------------------------------------------------------

START_FEN = chess.STARTING_FEN
# Midgame position (Italian Game after 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5 4.c3 Nf6)
MIDGAME_FEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2P2N2/PP1P1PPP/RNBQK2R w KQkq - 1 5"
# White pawn on e7 can promote to q/r/b/n (and king on h1 has moves too)
PROMOTION_FEN = "8/4P3/8/8/8/8/7k/K7 w - - 0 1"
# En passant available: White pawn on e5 can capture Black pawn on d5 via e5d6
EN_PASSANT_FEN = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
# Castling available both kingside (e1g1) and queenside (e1c1)
CASTLING_FEN = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
# Check evasion: White king on h1 is in check from Rook on h8; only g1 and g2 are legal (2 moves)
CHECK_EVASION_FEN = "4k2r/8/8/8/8/8/8/7K w - - 0 1"
# Single legal move: White king on h1 in check from Rook on h8, g2 covered by pawn on f3 -> only h1g1
SINGLE_LEGAL_FEN = "4k2r/8/8/8/8/5p2/8/7K w - - 0 1"
# Checkmate (0 legal moves): Fool's mate
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"


# ---------------------------------------------------------------------------
# 1. legal_uci_moves & LegalMovePrefixTrie tests
# ---------------------------------------------------------------------------


def test_legal_uci_moves_positions():
    start_moves = legal_uci_moves(START_FEN)
    assert len(start_moves) == 20
    assert start_moves == sorted(start_moves)
    assert "e2e4" in start_moves and "g1f3" in start_moves

    promo_moves = legal_uci_moves(PROMOTION_FEN)
    assert {"e7e8q", "e7e8r", "e7e8b", "e7e8n"}.issubset(set(promo_moves))
    assert "e7e8" not in promo_moves

    ep_moves = legal_uci_moves(EN_PASSANT_FEN)
    assert "e5d6" in ep_moves

    castle_moves = legal_uci_moves(CASTLING_FEN)
    assert "e1g1" in castle_moves
    assert "e1c1" in castle_moves

    evasion_moves = legal_uci_moves(CHECK_EVASION_FEN)
    assert evasion_moves == ["h1g1", "h1g2"]

    single_moves = legal_uci_moves(SINGLE_LEGAL_FEN)
    assert single_moves == ["h1g1"]

    assert legal_uci_moves(CHECKMATE_FEN) == []


def test_prefix_trie_bare_and_whitespace_variants():
    tok = MockChessTokenizer()
    trie = LegalMovePrefixTrie.from_fen(START_FEN, tok)

    # Bare, leading newline, and leading space variants must all be complete.
    for variant in ("e2e4", "\ne2e4", " e2e4"):
        ids = tok.encode(variant)
        assert trie.is_complete(ids), f"Expected {variant!r} to be complete"
        assert trie.allowed_next_tokens(ids) == set()

    # Partial prefix is not complete and restricts next tokens strictly.
    partial_ids = tok.encode("e2")
    assert trie.has_prefix(partial_ids)
    assert not trie.is_complete(partial_ids)
    assert trie.allowed_next_tokens(partial_ids) == {tok.token_to_id["e"]}

    # Illegal move is rejected.
    assert not trie.is_complete(tok.encode("e2e5"))
    assert not trie.has_prefix(tok.encode("e2e5"))


def test_prefix_trie_promotion_requires_piece_suffix():
    tok = MockChessTokenizer()
    trie = LegalMovePrefixTrie.from_fen(PROMOTION_FEN, tok)

    prefix_e7e8 = tok.encode("e7e8")
    allowed, is_complete = trie.step_info(prefix_e7e8)
    assert not is_complete
    assert allowed == {tok.token_to_id[p] for p in ("q", "r", "b", "n")}

    for piece in ("q", "r", "b", "n"):
        full_ids = tok.encode(f"e7e8{piece}")
        assert trie.is_complete(full_ids)


# ---------------------------------------------------------------------------
# 2. LegalMoveLogitsProcessor tests
# ---------------------------------------------------------------------------


def test_processor_unconstrained_before_think_close():
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(START_FEN, tok)

    input_ids = torch.tensor([tok.encode("<think>White can develop the center")], dtype=torch.long)
    scores = torch.randn((1, tok.vocab_size), dtype=torch.float32)

    out = proc(input_ids, scores)
    assert torch.equal(out, scores)
    assert not torch.isinf(out).any()


def test_processor_ignores_think_close_in_prompt_via_prompt_length():
    tok = MockChessTokenizer()
    prompt = build_chat_prompt(START_FEN)
    assert "</think>" in prompt  # ChessFM prompt itself contains </think>

    prompt_ids = tok.encode(prompt)
    gen_ids = tok.encode("<think>Thinking about e2e4")
    proc = LegalMoveLogitsProcessor(START_FEN, tok, prompt_length=len(prompt_ids))

    input_ids = torch.tensor([prompt_ids + gen_ids], dtype=torch.long)
    scores = torch.randn((1, tok.vocab_size), dtype=torch.float32)

    out = proc(input_ids, scores)
    assert torch.equal(out, scores)


@pytest.mark.parametrize(
    ("fen", "legal_move", "illegal_next_char"),
    [
        (START_FEN, "e2e4", "5"),
        (MIDGAME_FEN, "e1g1", "9"),
        (EN_PASSANT_FEN, "e5d6", "7"),
        (CASTLING_FEN, "e1g1", "a"),
        (CHECK_EVASION_FEN, "h1g1", "h"),
        (SINGLE_LEGAL_FEN, "h1g1", "2"),
    ],
)
def test_processor_masks_illegal_tokens_to_neg_inf_and_allows_eos_at_end(
    fen: str, legal_move: str, illegal_next_char: str
):
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(fen, tok)

    # Walk through every prefix of the legal move after </think>\n
    base = tok.encode("<think>Good move.</think>\n")
    move_ids = tok.encode(legal_move)

    for step in range(len(move_ids)):
        prefix = base + move_ids[:step]
        input_ids = torch.tensor([prefix], dtype=torch.long)
        scores = torch.ones((1, tok.vocab_size), dtype=torch.float32)
        out = proc(input_ids, scores)

        # Next legal token must keep its finite score
        expected_next = move_ids[step]
        assert out[0, expected_next].item() == 1.0
        # EOS must be -inf before the move is complete
        assert out[0, tok.eos_token_id].item() == float("-inf")

        # All illegal tokens must have exact 0.0 probability after softmax
        probs = torch.softmax(out, dim=-1)[0]
        allowed_now = proc.trie.allowed_next_tokens(tok.encode("\n") + move_ids[:step])
        for tid in range(tok.vocab_size):
            if tid not in allowed_now:
                assert out[0, tid].item() == float("-inf")
                assert probs[tid].item() == 0.0

    # Check illegal continuation at the last character specifically
    almost_done = base + move_ids[:-1]
    scores = torch.ones((1, tok.vocab_size), dtype=torch.float32)
    out = proc(torch.tensor([almost_done], dtype=torch.long), scores)
    assert out[0, tok.token_to_id[illegal_next_char]].item() == float("-inf")

    # Once the full move is emitted, ONLY EOS is allowed
    done_ids = torch.tensor([base + move_ids], dtype=torch.long)
    out_done = proc(done_ids, scores)
    assert out_done[0, tok.eos_token_id].item() == 1.0
    probs_done = torch.softmax(out_done, dim=-1)[0]
    assert probs_done[tok.eos_token_id].item() == pytest.approx(1.0)
    for tid in range(tok.vocab_size):
        if tid != tok.eos_token_id:
            assert out_done[0, tid].item() == float("-inf")
            assert probs_done[tid].item() == 0.0


def test_processor_promotion_masks_non_promotion_and_eos():
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(PROMOTION_FEN, tok)

    seq = tok.encode("<think>Promote pawn.</think>e7e8")
    scores = torch.zeros((1, tok.vocab_size), dtype=torch.float32)
    out = proc(torch.tensor([seq], dtype=torch.long), scores)

    allowed_promo = {tok.token_to_id[p] for p in ("q", "r", "b", "n")}
    assert out[0, tok.eos_token_id].item() == float("-inf")
    for tid in range(tok.vocab_size):
        if tid in allowed_promo:
            assert out[0, tid].item() == 0.0
        else:
            assert out[0, tid].item() == float("-inf")

    # After choosing 'q', EOS is allowed and everything else is -inf
    seq_done = tok.encode("<think>Promote pawn.</think>e7e8q")
    out_done = proc(torch.tensor([seq_done], dtype=torch.long), scores)
    assert out_done[0, tok.eos_token_id].item() == 0.0
    assert torch.isinf(out_done[0, tok.token_to_id["q"]]).item()


def test_processor_require_think_close_false():
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(SINGLE_LEGAL_FEN, tok, require_think_close=False)

    # Empty generated sequence immediately constrains to 'h' (or '\n' / ' ')
    input_ids = torch.zeros((1, 0), dtype=torch.long)
    scores = torch.zeros((1, tok.vocab_size), dtype=torch.float32)
    out = proc(input_ids, scores)

    expected = {tok.token_to_id["h"], tok.token_to_id["\n"], tok.token_to_id[" "]}
    for tid in range(tok.vocab_size):
        if tid in expected:
            assert out[0, tid].item() == 0.0
        else:
            assert out[0, tid].item() == float("-inf")


def test_processor_multi_token_and_merged_think_close():
    # Multi-token </think> -> ["</", "think", ">"]
    tok_split = MockChessTokenizer(split_think_close=True)
    proc_split = LegalMoveLogitsProcessor(SINGLE_LEGAL_FEN, tok_split)
    seq_split = tok_split.encode("<think>Only h1g1 works.</think>h1g")
    scores_split = torch.zeros((1, tok_split.vocab_size), dtype=torch.float32)
    out_split = proc_split(torch.tensor([seq_split], dtype=torch.long), scores_split)
    assert out_split[0, tok_split.token_to_id["1"]].item() == 0.0
    assert out_split[0, tok_split.token_to_id["2"]].item() == float("-inf")

    # Merged token ".</think>" fallback via decoded text
    tok_merged = MockChessTokenizer(split_think_close=False)
    proc_merged = LegalMoveLogitsProcessor(SINGLE_LEGAL_FEN, tok_merged)
    seq_merged = tok_merged.encode("<think>Only h1g1 works.</think>h1g")
    assert tok_merged.token_to_id[".</think>"] in seq_merged
    scores_merged = torch.zeros((1, tok_merged.vocab_size), dtype=torch.float32)
    out_merged = proc_merged(torch.tensor([seq_merged], dtype=torch.long), scores_merged)
    assert out_merged[0, tok_merged.token_to_id["1"]].item() == 0.0
    assert out_merged[0, tok_merged.token_to_id["2"]].item() == float("-inf")


def test_processor_batched_mixed_states():
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(SINGLE_LEGAL_FEN, tok)

    # Row 0: still inside <think> (unconstrained)
    # Row 1: after </think>h1g (only '1' allowed)
    # Row 2: after </think>h1g1 (only EOS allowed)
    r0 = tok.encode("<think>still thinking")
    r1 = tok.encode("<think>ok</think>h1g")
    r2 = tok.encode("<think>ok</think>h1g1")
    pad = tok.token_to_id["<pad>"]
    max_len = max(len(r0), len(r1), len(r2))
    batch = torch.tensor(
        [
            r0 + [pad] * (max_len - len(r0)),
            [pad] * (max_len - len(r1)) + r1,
            [pad] * (max_len - len(r2)) + r2,
        ],
        dtype=torch.long,
    )
    scores = torch.ones((3, tok.vocab_size), dtype=torch.float32)
    out = proc(batch, scores)

    assert torch.equal(out[0], scores[0])
    assert out[1, tok.token_to_id["1"]].item() == 1.0
    assert out[1, tok.token_to_id["2"]].item() == float("-inf")
    assert out[2, tok.eos_token_id].item() == 1.0
    assert out[2, tok.token_to_id["1"]].item() == float("-inf")


def test_processor_checkmate_falls_back_to_eos_without_nan():
    tok = MockChessTokenizer()
    proc = LegalMoveLogitsProcessor(CHECKMATE_FEN, tok)

    seq = tok.encode("<think>No moves.</think>")
    scores = torch.zeros((1, tok.vocab_size), dtype=torch.float32)
    out = proc(torch.tensor([seq], dtype=torch.long), scores)

    probs = torch.softmax(out, dim=-1)[0]
    assert not torch.isnan(probs).any()
    assert probs[tok.eos_token_id].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. DualModeGenerator tests
# ---------------------------------------------------------------------------


def test_dual_mode_generator_strict_vs_constrained():
    tok = MockChessTokenizer()
    model = AdversarialMockModel(tok, preferred_illegal_text="e2e5")
    gen = DualModeGenerator(model, tok, max_new_tokens=16)

    strict_res = gen.generate_move(START_FEN, mode="strict")
    assert strict_res["mode"] == "strict"
    assert strict_res["move"] == "e2e5"
    assert strict_res["is_legal"] is False
    assert "logits_processor" not in model.last_generate_kwargs

    constrained_res = gen.generate_move(START_FEN, mode="constrained")
    assert constrained_res["mode"] == "constrained"
    assert constrained_res["is_legal"] is True
    assert constrained_res["move"] in set(legal_uci_moves(START_FEN))
    assert "logits_processor" in model.last_generate_kwargs


@pytest.mark.parametrize(
    "fen",
    [START_FEN, MIDGAME_FEN, PROMOTION_FEN, EN_PASSANT_FEN, CASTLING_FEN, CHECK_EVASION_FEN],
)
def test_dual_mode_generator_constrained_always_legal_across_positions(fen: str):
    tok = MockChessTokenizer()
    model = AdversarialMockModel(tok, preferred_illegal_text="z9z9")
    gen = DualModeGenerator(model, tok, max_new_tokens=16)

    res = gen.generate_move(fen, mode="constrained")
    assert res["is_legal"] is True
    assert res["move"] in set(legal_uci_moves(fen))


def test_dual_mode_generator_best_of_n_with_custom_scorer():
    tok = MockChessTokenizer()
    completions = [
        "<think>Push a3.</think>\na2a3",
        "<think>Control center with e4.</think>\ne2e4",
        "<think>Develop knight.</think>\ng1f3",
    ]
    model = ScriptedBatchModel(tok, completions)
    gen = DualModeGenerator(model, tok, max_new_tokens=32)

    preference = {"a2a3": 0.1, "e2e4": 0.95, "g1f3": 0.7}

    def custom_scorer(fen: str, move: str | None, completion: str) -> float:
        assert fen == START_FEN
        assert "<think>" in completion
        return preference.get(move or "", -1.0)

    res = gen.generate_move(START_FEN, mode="constrained", num_samples=3, scorer=custom_scorer)
    assert res["move"] == "e2e4"
    assert res["is_legal"] is True
    assert res["mode"] == "constrained"
    assert model.last_generate_kwargs["do_sample"] is True
    assert model.last_generate_kwargs["num_return_sequences"] == 3


def test_dual_mode_generator_best_of_n_majority_vote_without_scorer():
    tok = MockChessTokenizer()
    completions = [
        "<think>Illegal.</think>\ne2e5",
        "<think>Play d4.</think>\nd2d4",
        "<think>Play e4.</think>\ne2e4",
        "<think>Play e4 again.</think>\ne2e4",
    ]
    model = ScriptedBatchModel(tok, completions)
    gen = DualModeGenerator(model, tok, max_new_tokens=32)

    res = gen.generate_move(START_FEN, mode="strict", num_samples=4)
    assert res["move"] == "e2e4"
    assert res["is_legal"] is True
    assert res["mode"] == "strict"


def test_dual_mode_generator_validation_errors():
    tok = MockChessTokenizer()
    model = ScriptedBatchModel(tok, ["<think>ok</think> e2e4"])
    gen = DualModeGenerator(model, tok)

    with pytest.raises(ValueError, match="Unsupported mode"):
        gen.generate_move(START_FEN, mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples"):
        gen.generate_move(START_FEN, num_samples=0)
