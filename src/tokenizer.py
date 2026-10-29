from __future__ import annotations

from importlib.metadata import version

import tiktoken

TIKTOKEN_ENCODING_MODEL = "gpt2"

'''tiktoken BPE gpt2 tokenizer

Byte Pair Encoding
Byte -> multi language, i.e. English, Chinese, Unicode
handle unknown words

The only GPT-2 special token is the end-of-text token (""), the document
separator. It is made encodable by default so real corpora round-trip
through encode / decode unchanged.
'''

# the loaded tiktoken encoding (GPT-2 BPE vocabulary).
tokenizer = tiktoken.get_encoding(TIKTOKEN_ENCODING_MODEL)


# tiktoken's single GPT-2 special token: end-of-text (document separator).
EOT_TOKEN: str = "<|endoftext|>"

def getTikTokenVersion() -> str:
    """Return the installed tiktoken distribution version as a string."""
    return version("tiktoken")


def encode(text: str, *, allowed_special: str = "all") -> list[int]:
    """Encode text into GPT-2 BPE token ids.

    The GPT-2 special token ("" / end-of-text, id 50256) is treated as a
    token by default, so encode("doc1

doc2") returns a list that decodes back to the original text.
    Pass allowed_special=set() to make special tokens raise instead.
    """
    return tokenizer.encode(text, allowed_special=allowed_special)


def decode(token_ids: list[int]) -> str:
    """Decode a list of token ids back into text."""
    return tokenizer.decode(token_ids)
