"""Token counting with the encoder's own tokenizer (chunk size is measured in its tokens)."""
from __future__ import annotations


class TokenCounter:
    def __init__(self, name: str, revision: str | None = None):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name, revision=revision)
        self._tok.model_max_length = 10**9          # we only count; silence length warnings
        self._cache: dict[str, int] = {}

    def __call__(self, text: str) -> int:
        n = self._cache.get(text)
        if n is None:
            n = len(self._tok(text, add_special_tokens=False)["input_ids"])
            if len(self._cache) < 500_000:
                self._cache[text] = n
        return n
