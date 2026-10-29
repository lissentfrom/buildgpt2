"""Unit tests for :mod:`tokenizer` (buildgpt2 ``src/tokenizer.py``).

"""

from unittest.mock import patch

import pytest

import tokenizer


# ---------------------------------------------------------------------------
# Behaviour: the helper reports the installed tiktoken version
# ---------------------------------------------------------------------------


class TestEncoding:
    """Tests for the observable behaviour of ``getTikTokenVersion``."""

    def test_unkownWord(self):
        """The helper must return a ``str`` (not bytes, ``None`` or an object)."""
        text = 'this is unknown jumbalaza'
        result = tokenizer.encode(text)
        print(result)

    def test_endOfText(self):
        text = 'this is part 1 <|endoftext|> this is part 2'
        result = tokenizer.encode(text)
        print(result)

class TestEncode:
    """Tests for the observable behaviour of ``encode``/``decode``."""

    def test_encode_returns_list_of_ints(self) -> None:
        result = tokenizer.encode("this is unknown jumbalaza")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)
        assert result

    def test_encode_known_text(self) -> None:
        # From the actual tiktoken output probed earlier.
        assert tokenizer.encode("hello") == [31373]
        assert tokenizer.encode("hello world") == [31373, 995]
        assert tokenizer.encode("this is unknown jumbalaza") == [
            5661, 318, 6439, 474, 2178, 282, 7056,
        ]

    def test_encode_empty_string(self) -> None:
        assert tokenizer.encode("") == []

    def test_encode_is_deterministic(self) -> None:
        text = "hello world"
        assert tokenizer.encode(text) == tokenizer.encode(text)

    def test_encode_round_trip(self) -> None:
        text = "hello world"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_encode_special_token(self) -> None:
        # The end-of-text token is encodable by default (id 50256).
        result = tokenizer.encode("hello" + tokenizer.EOT_TOKEN + " world")
        assert 50256 in result

    def test_encode_special_token_round_trip(self) -> None:
        text = "hello" + tokenizer.EOT_TOKEN + " world"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_encode_ordinary_equals_tiktoken(self) -> None:
        # encode() should behave like tiktoken with allowed_special="all".
        assert tokenizer.encode("hello") == tokenizer.tokenizer.encode("hello")


class TestDecode:
    """Tests for the observable behaviour of ``decode``."""

    def test_decode_returns_str(self) -> None:
        assert isinstance(tokenizer.decode([31373, 995]), str)

    def test_decode_empty(self) -> None:
        assert tokenizer.decode([]) == ""

