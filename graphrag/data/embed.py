"""Text encoder shared by chunks, graph-node text and queries (spec §2.5).

BAAI/bge-m3 dense head (1024-d, CLS pooling, no prefixes; chosen over multilingual-e5-base in
docs/m1/research-encoder-report.md). Prefixes come from config/model.yaml. Weights are
frozen at construction: the RL phase must see fixed features (spec §2.5, §12 item 4).
Fine-tuning (roadmap U8) produces a new model directory; indexes are keyed by `slug`.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from graphrag import config

log = logging.getLogger(__name__)


def _prefixes(model_name: str, cfg: dict) -> tuple[str, str]:
    """(query_prefix, passage_prefix): from config; the model-name check is only a fallback
    for configs without the keys (a fine-tuned e5 in models/u8_ft/ must keep the prefixes)."""
    if "query_prefix" in cfg or "passage_prefix" in cfg:
        return cfg.get("query_prefix", ""), cfg.get("passage_prefix", "")
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


class Encoder:
    def __init__(self, model_name: str | None = None, device: str | None = None,
                 batch_size: int | None = None, query_prefix: str | None = None,
                 passage_prefix: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = config.load("model")["encoder"]
        self.model_name = model_name or cfg["name"]
        dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        rev = cfg.get("revision") if self.model_name == cfg["name"] else None     # a fine-tuned dir has no HF revision
        self._m = SentenceTransformer(self.model_name, device=dev, revision=rev)
        # Chunks are sized to fit max_seq_length (chunker.ENCODER_MAX_TOKENS); the model must
        # accept that many tokens, and longer inputs are detected in _enc (never silent).
        self.max_seq_length = cfg["max_seq_length"]
        tok_max = self._m.tokenizer.model_max_length
        if tok_max < self.max_seq_length:
            raise RuntimeError(f"{self.model_name}: tokenizer max {tok_max} < "
                               f"max_seq_length {self.max_seq_length}")
        self._m.max_seq_length = self.max_seq_length
        self._m.eval()
        for p in self._m.parameters():
            p.requires_grad_(False)
        self.batch_size = batch_size or cfg["batch_size"]
        get_dim = getattr(self._m, "get_embedding_dimension", None) \
            or self._m.get_sentence_embedding_dimension
        self.dim = get_dim()
        if self.model_name == cfg["name"] and self.dim != cfg["dim"]:
            raise RuntimeError(f"{self.model_name}: dim {self.dim} != config dim {cfg['dim']}")
        qp, pp = _prefixes(self.model_name, cfg)
        self._qp = qp if query_prefix is None else query_prefix
        self._pp = pp if passage_prefix is None else passage_prefix

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.model_name.lower()).strip("_")

    def _check_lengths(self, texts: list[str], strict: bool) -> None:
        tok = self._m.tokenizer
        n_over = 0
        for i in range(0, len(texts), self.batch_size):
            ids = tok(texts[i:i + self.batch_size], add_special_tokens=True,
                      truncation=False)["input_ids"]
            n_over += sum(len(x) > self.max_seq_length for x in ids)
        if not n_over:
            return
        msg = f"{n_over} of {len(texts)} inputs exceed max_seq_length {self.max_seq_length} tokens"
        if strict:
            raise ValueError(msg)
        log.warning("%s; they will be truncated", msg)

    def _enc(self, texts: list[str], strict: bool = False,
             progress: bool | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._check_lengths(texts, strict)
        show = len(texts) > 512 if progress is None else progress
        v = self._m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=show)
        return v.astype(np.float32, copy=False)

    def encode_passages(self, texts: list[str], strict: bool = False,
                        progress: bool | None = None) -> np.ndarray:
        """strict=True (index builds) raises ValueError if any input would be truncated."""
        return self._enc([self._pp + t for t in texts], strict, progress)

    def encode_queries(self, texts: list[str], progress: bool | None = None) -> np.ndarray:
        return self._enc([self._qp + t for t in texts], False, progress)
