#!/usr/bin/env python3
"""
Legal-move constrained decoding and dual-mode generation for ChessFM.

Provides:
- :func:`legal_uci_moves`: sorted legal UCI moves for any FEN position.
- :class:`LegalMovePrefixTrie`: token-ID prefix trie over legal UCI moves
  (supporting bare moves as well as leading-newline/space variants).
- :class:`LegalMoveLogitsProcessor`: PyTorch / HuggingFace-compatible logits
  processor that leaves ``<think>...</think>`` reasoning unconstrained and
  masks all illegal move tokens to ``-inf`` after ``</think>``.
- :class:`DualModeGenerator`: unified wrapper supporting both ``"strict"``
  (unconstrained pure-LLM Pass@1) and ``"constrained"`` (100%-legal trie
  decoding), plus optional Best-of-N reranking via a custom ``scorer``.

Heavy ML libraries (``torch``, ``transformers``) are imported lazily so pure
helpers work in CPU-only or minimal environments without import overhead.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import chess

from chessfm.config import DEFAULT_MAX_NEW_TOKENS, build_chat_prompt
from chessfm.rl.move_utils import parse_move_from_output

DEFAULT_MOVE_PREFIXES: tuple[str, ...] = ("", "\n", " ")


def legal_uci_moves(fen: str) -> list[str]:
    """Return all legal UCI moves for ``fen`` in sorted order."""
    board = chess.Board(fen)
    return sorted(move.uci() for move in board.legal_moves)


def non_repeating_legal_uci_moves(board: chess.Board) -> tuple[list[str], bool]:
    """Return ``(allowed_uci_moves, filtered)`` avoiding 2-fold (then 3-fold) repetitions.

    Tests each legal move via ``push()``/``pop()`` without mutating ``board``.
    Tier 1 excludes moves causing >=2-fold repetition; if none remain, Tier 2
    excludes moves causing >=3-fold repetition; if none remain, Tier 3 falls
    back to all legal moves. ``filtered`` is ``True`` iff at least one
    repeating legal move was excluded.
    """
    legal_moves = sorted(board.legal_moves, key=lambda m: m.uci())
    if not legal_moves:
        return [], False

    non_2fold: list[str] = []
    non_3fold: list[str] = []
    for move in legal_moves:
        board.push(move)
        try:
            rep2 = board.is_repetition(2)
            rep3 = board.is_repetition(3) if rep2 else False
        finally:
            board.pop()
        uci = move.uci()
        if not rep2:
            non_2fold.append(uci)
        if not rep3:
            non_3fold.append(uci)

    if non_2fold:
        return non_2fold, len(non_2fold) < len(legal_moves)
    if non_3fold:
        return non_3fold, len(non_3fold) < len(legal_moves)
    return [m.uci() for m in legal_moves], False


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` to token IDs without special tokens."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(text)
    return [int(tid) for tid in ids]


def _normalize_eos_ids(tokenizer: Any, eos_token_ids: int | Sequence[int] | None) -> set[int]:
    """Resolve EOS token IDs from explicit argument or ``tokenizer.eos_token_id``."""
    raw = eos_token_ids if eos_token_ids is not None else getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return set()
    if isinstance(raw, int):
        return {raw}
    return {int(tid) for tid in raw}


def _find_subsequence_end(seq: Sequence[int], subseq: Sequence[int]) -> int | None:
    """Return the index immediately after the first occurrence of ``subseq`` in ``seq``."""
    sub_len = len(subseq)
    if sub_len == 0 or sub_len > len(seq):
        return None
    sub_list = list(subseq)
    for start in range(len(seq) - sub_len + 1):
        if list(seq[start : start + sub_len]) == sub_list:
            return start + sub_len
    return None


class _TrieNode:
    """Internal node for :class:`LegalMovePrefixTrie`."""

    __slots__ = ("children", "is_terminal")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.is_terminal: bool = False


class LegalMovePrefixTrie:
    """Prefix trie over tokenized legal UCI move ID sequences.

    Inserts each legal move with configurable leading prefixes (by default bare
    ``"e2e4"``, leading newline ``"\\ne2e4"``, and leading space ``" e2e4"``) so
    whitespace formatting immediately after ``</think>`` is preserved cleanly.
    """

    def __init__(
        self,
        legal_moves: Iterable[str],
        tokenizer: Any,
        prefixes: Sequence[str] = DEFAULT_MOVE_PREFIXES,
    ) -> None:
        self._root = _TrieNode()
        self.legal_moves: tuple[str, ...] = tuple(sorted(set(legal_moves)))
        self.prefixes: tuple[str, ...] = tuple(prefixes)
        self._num_sequences = 0

        for move in self.legal_moves:
            for prefix in self.prefixes:
                token_ids = _encode_text(tokenizer, f"{prefix}{move}")
                if token_ids:
                    self.insert(token_ids)

    @classmethod
    def from_fen(
        cls,
        fen: str,
        tokenizer: Any,
        prefixes: Sequence[str] = DEFAULT_MOVE_PREFIXES,
    ) -> "LegalMovePrefixTrie":
        """Build a prefix trie directly from a FEN string."""
        return cls(legal_uci_moves(fen), tokenizer, prefixes=prefixes)

    def insert(self, token_ids: Sequence[int]) -> None:
        """Insert a single token-ID sequence into the trie."""
        if not token_ids:
            return
        node = self._root
        for tid in token_ids:
            tid_int = int(tid)
            child = node.children.get(tid_int)
            if child is None:
                child = _TrieNode()
                node.children[tid_int] = child
            node = child
        if not node.is_terminal:
            node.is_terminal = True
            self._num_sequences += 1

    def _walk(self, prefix: Sequence[int]) -> _TrieNode | None:
        node = self._root
        for tid in prefix:
            node = node.children.get(int(tid))
            if node is None:
                return None
        return node

    def has_prefix(self, prefix: Sequence[int]) -> bool:
        """Return True if ``prefix`` is a valid path in the trie."""
        return self._walk(prefix) is not None

    def allowed_next_tokens(self, prefix: Sequence[int]) -> set[int]:
        """Return the set of token IDs that can validly follow ``prefix``."""
        node = self._walk(prefix)
        if node is None:
            return set()
        return set(node.children.keys())

    def is_complete(self, prefix: Sequence[int]) -> bool:
        """Return True if ``prefix`` forms a complete legal move."""
        node = self._walk(prefix)
        return bool(node is not None and node.is_terminal)

    def step_info(self, prefix: Sequence[int]) -> tuple[set[int], bool]:
        """Return ``(allowed_next_tokens, is_complete)`` in a single traversal."""
        node = self._walk(prefix)
        if node is None:
            return set(), False
        return set(node.children.keys()), node.is_terminal

    def __len__(self) -> int:
        return self._num_sequences


class LegalMoveLogitsProcessor:
    """HuggingFace/PyTorch-compatible logits processor enforcing legal UCI moves.

    - While ``require_think_close=True``, generation inside ``<think>...</think>``
      is completely unconstrained so reasoning is unaffected.
    - Once ``</think>`` appears in the newly generated tokens (or immediately
      when ``require_think_close=False``), every token ID that does not extend a
      legal UCI move in the prefix trie is masked to ``-inf``, guaranteeing 0.0
      probability on illegal moves.
    - Once a complete legal move has been generated, EOS token IDs are allowed
      so generation terminates cleanly.

    Note: ``prompt_length`` slices off prompt tokens before scanning for
    ``</think>`` because ChessFM's own instruction prompt contains the literal
    substring ``"</think>"``.
    """

    def __init__(
        self,
        fen: str,
        tokenizer: Any,
        prompt_length: int = 0,
        require_think_close: bool = True,
        think_close_tag: str = "</think>",
        eos_token_ids: int | Sequence[int] | None = None,
        prefixes: Sequence[str] = DEFAULT_MOVE_PREFIXES,
        trie: LegalMovePrefixTrie | None = None,
        allowed_moves: Iterable[str] | None = None,
    ) -> None:
        self.fen = fen
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.require_think_close = require_think_close
        self.think_close_tag = think_close_tag
        self.eos_token_ids = _normalize_eos_ids(tokenizer, eos_token_ids)
        if trie is not None:
            self.trie = trie
        elif allowed_moves is not None:
            self.trie = LegalMovePrefixTrie(allowed_moves, tokenizer, prefixes=prefixes)
        else:
            self.trie = LegalMovePrefixTrie.from_fen(fen, tokenizer, prefixes=prefixes)
        self._think_close_ids = _encode_text(tokenizer, think_close_tag)

    def _extract_move_prefix(self, generated_ids: list[int]) -> list[int] | None:
        """Return token IDs generated after ``</think>``, or None if unconstrained.

        When ``require_think_close`` is False, returns ``generated_ids`` directly.
        """
        if not self.require_think_close:
            return generated_ids

        # Fast path: exact token-subsequence match for the closing tag.
        end_idx = _find_subsequence_end(generated_ids, self._think_close_ids)
        if end_idx is not None:
            return generated_ids[end_idx:]

        # Fallback for BPE tokenizers where a preceding character merges into
        # the first token of ``</think>``.
        if generated_ids and hasattr(self.tokenizer, "decode"):
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            if self.think_close_tag in decoded:
                for k in range(len(generated_ids), 0, -1):
                    prefix_text = self.tokenizer.decode(
                        generated_ids[:k], skip_special_tokens=False
                    )
                    if self.think_close_tag not in prefix_text:
                        return generated_ids[k + 1 :]
                return generated_ids[1:]

        return None

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        """Mask illegal move tokens to ``-inf`` once in the move-generation phase."""
        import torch

        masked_scores = scores.clone()
        batch_size, vocab_size = scores.shape

        for row in range(batch_size):
            seq = input_ids[row].tolist()
            gen_ids = seq[self.prompt_length :] if self.prompt_length > 0 else seq
            move_prefix = self._extract_move_prefix(gen_ids)
            if move_prefix is None:
                continue

            allowed, is_complete = self.trie.step_info(move_prefix)
            valid_ids = set(allowed)
            if is_complete:
                valid_ids.update(self.eos_token_ids)
            if not valid_ids and self.eos_token_ids:
                # Avoid all -inf (NaN after softmax) in terminal positions with
                # zero legal moves (checkmate/stalemate).
                valid_ids.update(self.eos_token_ids)

            bounded_ids = sorted(tid for tid in valid_ids if 0 <= tid < vocab_size)
            if not bounded_ids:
                continue

            row_scores = scores[row]
            new_row = torch.full_like(row_scores, float("-inf"))
            idx = torch.tensor(bounded_ids, dtype=torch.long, device=scores.device)
            new_row[idx] = row_scores[idx]
            masked_scores[row] = new_row

        return masked_scores


def _select_best_candidate(
    fen: str,
    candidates: list[dict[str, Any]],
    scorer: Callable[[str, str | None, str], float] | None,
) -> dict[str, Any]:
    """Select the winning candidate from Best-of-N samples."""
    if len(candidates) == 1:
        return candidates[0]

    if scorer is not None:
        return max(
            candidates,
            key=lambda c: scorer(fen, c["move"], c["completion"]),
        )

    # Without an explicit scorer, prefer majority vote among legal moves
    # (breaking ties by first appearance), falling back to the first sample.
    legal_candidates = [c for c in candidates if c["is_legal"] and c["move"] is not None]
    if not legal_candidates:
        return candidates[0]

    counts = Counter(c["move"] for c in legal_candidates)
    best_move = max(counts, key=lambda mv: counts[mv])
    for candidate in legal_candidates:
        if candidate["move"] == best_move:
            return candidate
    return legal_candidates[0]


class DualModeGenerator:
    """Unified move generator supporting ``"strict"`` and ``"constrained"`` modes.

    - ``mode="strict"``: unconstrained generation measuring raw LLM Pass@1.
    - ``mode="constrained"``: attaches :class:`LegalMoveLogitsProcessor` so the
      emitted move is guaranteed legal (0.0 probability on illegal moves).
    - ``num_samples > 1``: enables sampling and Best-of-N reranking via
      ``scorer(fen, move, completion) -> float`` (or legal majority vote when
      ``scorer`` is None).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        prompt_builder: Callable[[str], str] = build_chat_prompt,
        think_close_tag: str = "</think>",
        require_think_close: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompt_builder = prompt_builder
        self.think_close_tag = think_close_tag
        self.require_think_close = require_think_close
        self.temperature = temperature
        self.top_p = top_p

    def _wrap_logits_processor(self, processor: LegalMoveLogitsProcessor) -> Any:
        """Wrap processor in ``transformers.LogitsProcessorList`` when available."""
        try:
            from transformers import LogitsProcessorList

            return LogitsProcessorList([processor])
        except ImportError:
            return [processor]

    def generate_move(
        self,
        fen: str,
        mode: Literal["strict", "constrained"] = "strict",
        num_samples: int = 1,
        scorer: Callable[[str, str | None, str], float] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> dict[str, Any]:
        """Generate a move for ``fen`` in strict or constrained mode.

        Returns a dict with keys ``"move"``, ``"completion"``, ``"is_legal"``,
        and ``"mode"``.
        """
        if mode not in ("strict", "constrained"):
            raise ValueError(f"Unsupported mode {mode!r}; expected 'strict' or 'constrained'.")
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")

        import torch

        prompt = self.prompt_builder(fen)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)

        input_ids = inputs["input_ids"]
        prompt_length = int(input_ids.shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": (
                max_new_tokens if max_new_tokens is not None else self.max_new_tokens
            ),
        }
        if num_samples == 1:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["num_return_sequences"] = num_samples
            gen_kwargs["temperature"] = temperature if temperature is not None else self.temperature
            gen_kwargs["top_p"] = top_p if top_p is not None else self.top_p

        if mode == "constrained":
            processor = LegalMoveLogitsProcessor(
                fen=fen,
                tokenizer=self.tokenizer,
                prompt_length=prompt_length,
                require_think_close=self.require_think_close,
                think_close_tag=self.think_close_tag,
            )
            gen_kwargs["logits_processor"] = self._wrap_logits_processor(processor)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        legal_set = set(legal_uci_moves(fen))
        candidates: list[dict[str, Any]] = []
        for row in range(len(outputs)):
            generated = outputs[row][prompt_length:]
            completion = self.tokenizer.decode(generated, skip_special_tokens=True)
            move = parse_move_from_output(completion)
            is_legal = move is not None and move in legal_set
            candidates.append(
                {
                    "move": move,
                    "completion": completion,
                    "is_legal": is_legal,
                    "mode": mode,
                }
            )

        return _select_best_candidate(fen, candidates, scorer)
