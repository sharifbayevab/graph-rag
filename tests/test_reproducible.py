"""Fail-loud offline builds (final review I3): vision API gate, degraded-build refusal, manifest info."""
import json

import pytest

from graphrag.data import extract
from graphrag.data.ingest import BUILD_INFO, build_corpus, read_records, write_records
from graphrag.data.pdf import VisionCacheMiss, VisionTranscriber
from tests.helpers import words
from tests.test_ingest import _corpus
from tests.test_pdf import _FakeClient


class _NoCalls:
    def __getattr__(self, name):
        raise AssertionError("API must not be called")


def test_vision_offline_miss_raises_without_calling(tmp_path):
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=False)
    with pytest.raises(VisionCacheMiss, match="GRL_VISION_API=1"):
        vt(b"img", page_text="")
    with pytest.raises(VisionCacheMiss):
        vt.prefetch([b"img2"], workers=2)
    with pytest.raises(VisionCacheMiss):
        vt.page_pairs(b"page", "k", ["1 2 3"])
    assert vt.stats["api_calls"] == 0 and not list(tmp_path.glob("*.json"))


def test_vision_offline_skip_mode_drops_and_counts(tmp_path):
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=False, on_miss="skip")
    vt.prefetch([b"img"], workers=1)
    assert vt(b"img", page_text="") is None
    assert vt.page_pairs(b"page", "k", ["1 2 3"]) is None
    assert vt.stats["offline_miss"] == 2 and vt.stats["api_calls"] == 0   # unique misses
    assert not list(tmp_path.glob("*.json"))                              # nothing cached


def test_vision_offline_uses_cache(tmp_path):
    online = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("12 та мурожаат келди", "12 — мурожаат"))
    want = online(b"img", page_text="")
    offline = VisionTranscriber("gpt-4.1", tmp_path, client=_NoCalls(), allow_api=False)
    assert offline(b"img", page_text="") == want and offline.stats["offline_miss"] == 0


def test_vision_factory_gate(monkeypatch, tmp_path):
    v = {"model": "gpt-4.1", "cache_dir": str(tmp_path), "api": False}
    for env, want in ((None, False), ("1", True)):
        monkeypatch.setattr(extract, "_VISION", None)
        if env:
            monkeypatch.setenv("GRL_VISION_API", env)
        else:
            monkeypatch.delenv("GRL_VISION_API", raising=False)
        vt = extract._vision(v)
        assert vt.allow_api is want and vt.on_miss == "raise"
    monkeypatch.setattr(extract, "_VISION", None)


def test_data_config_has_api_gates_off():
    from graphrag import config
    cfg = config.load("data")
    assert cfg["pdf"]["vision"]["api"] is False and cfg["agenda"]["api"] is False


def test_manifest_records_build_info_not_records(tmp_path):
    cfg = _corpus(tmp_path)
    records, _ = build_corpus(cfg, words)
    info = {r.doc_id: {"engine": "opendataloader", "agenda": "llm"} for r in records}
    out = tmp_path / "rec"
    write_records(records, out, info)
    lines = [json.loads(ln) for ln in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(m["engine"] == "opendataloader" and m["agenda"] == "llm" for m in lines)
    plain = tmp_path / "plain"
    write_records(records, plain)
    for f in plain.glob("*.json"):
        assert (out / f.name).read_bytes() == f.read_bytes()             # record files unaffected
    assert [r.to_json() for r in read_records(out)] == [r.to_json() for r in read_records(plain)]
    assert set(BUILD_INFO) >= {r.doc_id for r in records}                 # docx/md: no engine, no agenda
    assert all(v == {} for v in BUILD_INFO.values())


def _script(monkeypatch, tmp_path, records, skipped, misses=(0, 0), engines=None):
    import scripts.build_corpus as bc
    writes = []
    monkeypatch.setattr(bc, "write_records", lambda *a: writes.append("records"))
    monkeypatch.setattr(bc, "write_lexicon", lambda *a: writes.append("lexicon"))
    calls = iter([(0, 0), tuple(misses)])
    monkeypatch.setattr(bc, "_misses", lambda cfg: next(calls))

    def fake_build(cfg):
        extract.PDF_ENGINE.clear()
        extract.PDF_ENGINE.update(engines or {})
        return records, skipped
    monkeypatch.setattr(bc, "build_corpus", fake_build)
    return bc, writes


@pytest.mark.parametrize("misses,engines", [
    ((1, 0), {}),                                        # agenda offline miss
    ((0, 2), {}),                                        # vision miss (skip mode)
    ((0, 0), {"/x.pdf": "pypdf_fallback"}),              # no Java
])
def test_build_corpus_script_refuses_degraded(monkeypatch, tmp_path, misses, engines):
    records, skipped = build_corpus(_corpus(tmp_path), words)
    bc, writes = _script(monkeypatch, tmp_path, records, skipped, misses, engines)
    assert bc.main([]) == 1 and writes == []
    bc, writes = _script(monkeypatch, tmp_path, records, skipped, misses, engines)
    assert bc.main(["--allow-degraded"]) == 0 and writes == ["records", "lexicon"]
    extract.PDF_ENGINE.clear()


def test_build_corpus_script_accepts_clean_and_explicit_pypdf(monkeypatch, tmp_path):
    records, skipped = build_corpus(_corpus(tmp_path), words)
    bc, writes = _script(monkeypatch, tmp_path, records, skipped, (0, 0), {"/x.pdf": "pypdf"})
    assert bc.main([]) == 0 and writes == ["records", "lexicon"]
    extract.PDF_ENGINE.clear()


def test_apply_overrides():
    import scripts.build_corpus as bc
    cfg = {"records_dir": "data/records", "pdf": {"vision": {"cache_dir": "c"}}}
    bc.apply_overrides(cfg, ["records_dir=/tmp/r", "pdf.vision.cache_dir=/tmp/e", "max_tokens=12"])
    assert cfg == {"records_dir": "/tmp/r", "pdf": {"vision": {"cache_dir": "/tmp/e"}}, "max_tokens": 12}
    with pytest.raises(SystemExit):
        bc.apply_overrides(cfg, ["novalue"])
