import logging

import numpy as np
import pytest

from graphrag import config
from graphrag.data.embed import Encoder, _prefixes

model = pytest.mark.model


@pytest.fixture(scope="module")
def enc():
    return Encoder()


@model
def test_dim_norm_and_frozen(enc):
    v = enc.encode_passages(["soliq kodeksi", "jinoyat kodeksi"])
    assert v.shape == (2, config.load("model")["encoder"]["dim"]) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-4)
    assert not any(p.requires_grad for p in enc._m.parameters())


@model
def test_query_matches_relevant_passage(enc):
    q = enc.encode_queries(["soliq kodeksiga oʼzgartirish"])[0]
    p = enc.encode_passages(["soliq kodeksiga oʼzgartish va qoʼshimchalar kiritish toʼgʼrisida",
                             "raislik qiluvchi: majlisni ochiq deb eʼlon qilaman"])
    assert p[0] @ q > p[1] @ q


@model
def test_deterministic(enc):
    a = enc.encode_passages(["qonun"])
    b = enc.encode_passages(["qonun"])
    assert np.allclose(a, b, atol=1e-5)


# --- no model needed: prefix selection, truncation detection ---

def test_prefixes_from_config_even_for_non_e5_name():
    cfg = {"query_prefix": "query: ", "passage_prefix": "passage: "}   # an e5-style config
    assert _prefixes("models/u8_ft", cfg) == ("query: ", "passage: ")
    assert _prefixes("intfloat/multilingual-e5-base", {"query_prefix": "", "passage_prefix": ""}) == ("", "")


def test_prefixes_name_fallback_without_config_keys():
    assert _prefixes("intfloat/multilingual-e5-base", {}) == ("query: ", "passage: ")
    assert _prefixes("models/u8_ft", {}) == ("", "")


class _Tok:
    """Whitespace tokenizer + 2 special tokens, mimicking the HF call signature."""
    def __call__(self, texts, add_special_tokens=True, truncation=False):
        extra = 2 if add_special_tokens else 0
        return {"input_ids": [[0] * (len(t.split()) + extra) for t in texts]}


class _Model:
    tokenizer = _Tok()

    def encode(self, texts, **kw):
        return np.ones((len(texts), 4), dtype=np.float64)


def _stub_encoder(max_len=6):
    e = Encoder.__new__(Encoder)
    e._m, e.max_seq_length, e.batch_size, e.dim = _Model(), max_len, 2, 4
    e._qp, e._pp = "query: ", "passage: "
    return e


def test_truncation_warns_once_or_raises_when_strict(caplog):
    e = _stub_encoder()
    ok, long = "a b c", "a b c d e f g"      # 1+3+2=6 fits; 1+7+2=10 exceeds
    with caplog.at_level(logging.WARNING, logger="graphrag.data.embed"):
        v = e.encode_passages([ok, long, long])
    assert v.shape == (3, 4) and v.dtype == np.float32
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and "2 of 3" in warns[0].getMessage() and "6" in warns[0].getMessage()
    with pytest.raises(ValueError):
        e.encode_passages([ok, long], strict=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="graphrag.data.embed"):
        e.encode_passages([ok], strict=True)
    assert not caplog.records


# --- M2 task 1: bge-m3 ---

def test_bge_m3_prefixes_are_empty():
    q, p = _prefixes("BAAI/bge-m3", config.load("model")["encoder"])
    assert q == "" and p == ""


@pytest.mark.model
def test_bge_m3_dim_and_cross_script_similarity():
    enc = Encoder()
    assert enc.dim == 1024
    v = enc.encode_passages(["Солиқ кодекси", "soliq kodeksi", "raislik qiluvchi: rahmat"], strict=True)
    assert v[0] @ v[1] > v[0] @ v[2]
