import copy

from graphrag import config
from graphrag.data.ingest import (build_corpus, doc_id_for, read_records, validate_record,
                                  write_lexicon, write_records)
from tests.helpers import P, make_docx, words
from tests.test_structure_law import LAW


def _corpus(tmp_path):
    docs = tmp_path / "docs"
    for sec in ("QONUNLAR", "QARORLAR", "XULOSALAR"):
        (docs / sec).mkdir(parents=True)
    body = "".join(P(t) for t in LAW)
    make_docx(docs / "QONUNLAR" / "a.docx", body)
    make_docx(docs / "QONUNLAR" / "b (1).docx", body)                   # identical text
    make_docx(docs / "QARORLAR" / "empty.docx", P(""))                  # no text
    (docs / "XULOSALAR" / "x.md").write_text("Х У Л О С А\n\nХулоса матни.\n", encoding="utf-8")
    cfg = copy.deepcopy(config.load("data"))
    cfg.update(corpus_root=str(docs), sections=["QONUNLAR", "QARORLAR", "XULOSALAR"])
    return cfg


def test_build_corpus_dedups_skips_and_validates(tmp_path):
    cfg = _corpus(tmp_path)
    records, skipped = build_corpus(cfg, words)
    assert [r.source.file for r in records] == ["docs/QONUNLAR/a.docx", "docs/XULOSALAR/x.md"]
    assert records[0].source.aliases == ["docs/QONUNLAR/b (1).docx"]
    reasons = {s["file"]: s["reason"] for s in skipped}
    assert reasons["docs/QARORLAR/empty.docx"] == "empty"
    assert reasons["docs/QONUNLAR/b (1).docx"].startswith("duplicate of")
    for r in records:
        assert validate_record(r, cfg["max_tokens"]) == []
        assert all(c.parent for c in r.chunks)


def test_rebuild_is_byte_identical(tmp_path):
    cfg = _corpus(tmp_path)
    a, _ = build_corpus(cfg, words)
    b, _ = build_corpus(cfg, words)
    assert [r.to_json() for r in a] == [r.to_json() for r in b]


def test_doc_id_is_path_based_and_stable():
    assert doc_id_for("QONUNLAR", "docs/QONUNLAR/a.docx") == doc_id_for("QONUNLAR", "docs/QONUNLAR/a.docx")
    assert doc_id_for("QONUNLAR", "docs/QONUNLAR/a.docx").startswith("QONUNLAR-")


def test_write_read_roundtrip_and_lexicon(tmp_path):
    cfg = _corpus(tmp_path)
    records, _ = build_corpus(cfg, words)
    out = tmp_path / "records"
    write_records(records, out)
    (out / "stale.json").write_text("{}")
    write_records(records, out)                                          # removes stale files
    assert not (out / "stale.json").exists()
    assert read_records(out) == records
    lx = tmp_path / "lexicon.json"
    write_lexicon(records, lx)
    assert "qonun" in lx.read_text(encoding="utf-8")


def test_validate_record_catches_broken_chunk(tmp_path):
    cfg = _corpus(tmp_path)
    rec = build_corpus(cfg, words)[0][0]
    rec.chunks[0].text = rec.chunks[0].text + "x"
    assert any("not an exact substring" in e for e in validate_record(rec, cfg["max_tokens"]))


# ── changes since the plan: merged stenogram turns, index_text ────────────────
from tests.test_chunker_regress import STENO  # noqa: E402


def _steno_record(tmp_path):
    docs = tmp_path / "docs"
    (docs / "STENOGRAMMA").mkdir(parents=True)
    (docs / "STENOGRAMMA" / "s.md").write_text("\n\n".join(STENO) + "\n", encoding="utf-8")
    cfg = copy.deepcopy(config.load("data"))
    cfg.update(corpus_root=str(docs), sections=["STENOGRAMMA"])
    records, skipped = build_corpus(cfg, words)
    assert skipped == [] and len(records) == 1
    return records[0], cfg


def test_merged_stenogram_turns_validate(tmp_path):
    rec, cfg = _steno_record(tmp_path)
    by_id = {n.node_id: n for n in rec.structure}
    merged = [c for c in rec.chunks if by_id[c.parent].type == "вопрос"]
    assert merged                                           # tiny turns were attached
    turns = [n for n in rec.structure if n.type == "выступление"
             and any(c.start <= n.start and n.end <= c.end for c in merged)]
    assert len(turns) >= 2                                  # one chunk spans several units
    assert validate_record(rec, cfg["max_tokens"]) == []


def test_validate_record_checks_tokens_and_index_text(tmp_path):
    rec, cfg = _steno_record(tmp_path)
    rec.chunks[0].tokens = cfg["max_tokens"] + 1
    rec.chunks[1].index_text = "breadcrumb only"
    rec.chunks[-1].index_text = ""
    errs = validate_record(rec, cfg["max_tokens"])
    assert any(e.startswith(rec.chunks[0].chunk_id) and "tokens >" in e for e in errs)
    assert any(e.startswith(rec.chunks[1].chunk_id) and "index_text" in e for e in errs)
    assert any(e.startswith(rec.chunks[-1].chunk_id) and "index_text" in e for e in errs)


def test_validate_record_catches_uncovered_unit(tmp_path):
    rec, cfg = _steno_record(tmp_path)
    by_id = {n.node_id: n for n in rec.structure}
    victim = next(c for c in rec.chunks if by_id[c.parent].type == "вопрос")
    rec.chunks.remove(victim)
    assert any("unit text not fully covered" in e for e in validate_record(rec, cfg["max_tokens"]))


# ── fix round 1 ──────────────────────────────────────────────────────────────
import unicodedata  # noqa: E402

from graphrag.data.ingest import build_record, doc_id_for as _did  # noqa: E402

_REQ = "11.ПС (Парламент сўрови).docx"


def _one_file(tmp_path, name):
    sec = tmp_path / "docs" / "QARORLAR"
    sec.mkdir(parents=True)
    p = make_docx(sec / name, P("Сенаторлар мурожаати матни.") + P("Иккинчи хатбоши."))
    cfg = copy.deepcopy(config.load("data"))
    cfg.update(corpus_root=str(tmp_path / "docs"), sections=["QARORLAR"])
    return cfg, p


def test_nfd_filename_gives_same_record_as_nfc(tmp_path):
    cfg_c, p_c = _one_file(tmp_path / "c", unicodedata.normalize("NFC", _REQ))
    cfg_d, p_d = _one_file(tmp_path / "d", unicodedata.normalize("NFD", _REQ))
    assert unicodedata.normalize("NFD", _REQ) != _REQ
    rel = f"docs/QARORLAR/{_REQ}"
    rc = build_record("QARORLAR", p_c, rel, cfg_c, words)
    rd = build_record("QARORLAR", p_d, rel, cfg_d, words)
    assert rc.meta.type == "resolution"
    assert rd.to_json() == rc.to_json()
    assert build_corpus(cfg_d, words)[0][0].to_json() == rc.to_json()


def test_validate_record_requires_deepest_parent(tmp_path):
    cfg = _corpus(tmp_path)
    rec = build_corpus(cfg, words)[0][0]
    root = next(n for n in rec.structure if n.parent_id is None)
    c = next(c for c in rec.chunks if c.parent != root.node_id)
    c.parent = root.node_id
    assert any(e.startswith(c.chunk_id) and "not the deepest" in e
               for e in validate_record(rec, cfg["max_tokens"]))


def test_dedup_prefers_file_without_download_suffix(tmp_path):
    docs = tmp_path / "docs" / "QONUNLAR"
    docs.mkdir(parents=True)
    body = "".join(P(t) for t in LAW)
    for n in ("3. Қонун матни (1).docx", "3. Қонун матни.docx", "3. Қонун матни(2).docx"):
        make_docx(docs / n, body)
    cfg = copy.deepcopy(config.load("data"))
    cfg.update(corpus_root=str(tmp_path / "docs"), sections=["QONUNLAR"])
    records, skipped = build_corpus(cfg, words)
    assert [r.source.file for r in records] == ["docs/QONUNLAR/3. Қонун матни.docx"]
    assert records[0].doc_id == _did("QONUNLAR", "docs/QONUNLAR/3. Қонун матни.docx")
    assert records[0].source.aliases == ["docs/QONUNLAR/3. Қонун матни (1).docx",
                                         "docs/QONUNLAR/3. Қонун матни(2).docx"]
    assert all(s["reason"] == "duplicate of docs/QONUNLAR/3. Қонун матни.docx" for s in skipped)
    assert len(skipped) == 2


def test_build_corpus_script_writes_nothing_when_invalid(tmp_path, monkeypatch):
    import scripts.build_corpus as bc
    cfg = _corpus(tmp_path)
    records, skipped = build_corpus(cfg, words)
    writes = []
    monkeypatch.setattr(bc, "write_records", lambda *a: writes.append("records"))
    monkeypatch.setattr(bc, "write_lexicon", lambda *a: writes.append("lexicon"))
    monkeypatch.setattr(bc, "build_corpus", lambda cfg: (records, skipped))
    assert bc.main() == 0 and writes == ["records", "lexicon"]
    writes.clear()
    monkeypatch.setattr(bc, "build_corpus",
                        lambda cfg: (records, skipped + [{"file": "f", "reason": "error: X"}]))
    assert bc.main() == 1 and writes == []
    records[0].chunks[0].text += "x"
    monkeypatch.setattr(bc, "build_corpus", lambda cfg: (records, skipped))
    assert bc.main() == 1 and writes == []
