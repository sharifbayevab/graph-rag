import json
import struct
import zlib
from pathlib import Path

import pytest

from graphrag.data.pdf import (VisionBudgetExceeded, VisionTranscriber, elements_to_paras,
                               flatten_elements, png_size, read_pdf)

FIX = Path(__file__).parent / "fixtures" / "odl_sample.json"
PDF12 = Path(__file__).resolve().parents[2] / "docs/QONUNLAR/12. 2024 фаолияти тўғрисида.pdf"
PDF13 = Path(__file__).resolve().parents[2] / "docs/QONUNLAR/13. Bolalar ombudsmani - ma_ruza.pdf"


def test_flatten_keeps_reading_order_and_pages():
    els = flatten_elements(json.loads(FIX.read_text(encoding="utf-8")))
    assert els and all(isinstance(e["page"], int) and e["page"] >= 1 for e in els)
    assert [e["page"] for e in els] == sorted(e["page"] for e in els)      # reading order is page-monotone
    assert not any(e["type"] in ("header", "footer") for e in els)


def test_elements_to_paras_tables_lists_images_and_running_headers():
    els = [{"type": "paragraph", "page": p, "text": "Бола ҳуқуқлари бўйича вакили", "image": None} for p in range(1, 6)]
    els += [{"type": "heading", "page": 1, "text": "1. Мурожаатлар", "image": None},
            {"type": "table_row", "page": 1, "text": "Қашқадарё вилояти | 2818 | 3274", "image": None},
            {"type": "list_item", "page": 2, "text": "а) биринчи", "image": None},
            {"type": "image", "page": 2, "text": "", "image": "img1.png"}]
    paras = elements_to_paras(els, image_text=lambda e: "898 та мурожаат" if e["image"] == "img1.png" else None)
    texts = [p.text for p in paras]
    assert "Бола ҳуқуқлари бўйича вакили" not in texts                    # running header dropped
    assert texts == ["1. Мурожаатлар", "Қашқадарё вилояти | 2818 | 3274", "а) биринчи", "[расм] 898 та мурожаат"]
    assert [p.page for p in paras] == [1, 1, 2, 2]


# ---- real-schema details (keys captured from opendataloader-pdf 2.5.11 output) ----

def test_flatten_fixture_tables_lists_images_and_no_running_header():
    els = flatten_elements(json.loads(FIX.read_text(encoding="utf-8")))
    texts = [e["text"] for e in els]
    rows = [e for e in els if e["type"] == "table_row"]
    assert [r["text"] for r in rows] == ["№ | Ҳудудлар | Мурожаатлар сони", "2023 йил | 2024 йил",
                                         "1. | Қашқадарё вилояти | 2818 | 3274"]
    assert all(r["page"] == 6 for r in rows)
    items = [e["text"][:4] for e in els if e["type"] == "list_item"]
    assert items == ["I. М", "1.1.", "1.2.", "II. ", "2.1.", "2.2."]            # nested list, document order
    imgs = [e for e in els if e["type"] == "image"]
    assert [e["page"] for e in imgs] == [1, 4, 6] and all(e["image"].endswith(".png") for e in imgs)
    hdr = "Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари бўйича вакилининг (омбудсманнинг) 2024 йилдаги фаолияти тўғрисида"
    assert texts.count(hdr) == 1 and "2" not in texts                       # title page only; no footer numbers
    caption = next(i for i, e in enumerate(els) if e["type"] == "caption" and e["page"] == 4)
    assert els[caption - 1]["type"] == "image"                               # array order, not id order


def test_flatten_text_blocks_list_item_kids_and_images_in_cells():
    doc = {"kids": [
        {"type": "text block", "page number": 3, "kids": [
            {"type": "heading", "page number": 3, "content": "  Сарлавҳа "},
            {"type": "text block", "page number": 3, "kids": [
                {"type": "paragraph", "page number": 3, "content": "Ички матн"}]}]},
        {"type": "list", "page number": 3, "list items": [
            {"type": "list item", "page number": 3, "content": "1) банд", "kids": [
                {"type": "paragraph", "page number": 3, "content": "давоми"}]}]},
        {"type": "table", "page number": 4, "rows": [
            {"type": "table row", "cells": [
                {"type": "table cell", "page number": 4, "kids": [
                    {"type": "paragraph", "page number": 4, "content": "A"},
                    {"type": "paragraph", "page number": 4, "content": "a2"}]},
                {"type": "table cell", "page number": 4, "kids": [
                    {"type": "image", "page number": 4, "source": "x/imageFile9.png"}]},
                {"type": "table cell", "page number": 4, "kids": []},
                {"type": "table cell", "page number": 4, "kids": [
                    {"type": "paragraph", "page number": 4, "content": "B"}]}]}]},
        {"type": "footer", "page number": 4, "kids": [{"type": "heading", "page number": 4, "content": "4"}]},
        {"type": "watermark", "page number": 4, "content": "DRAFT"},
    ]}
    got = [(e["type"], e["page"], e["text"], e["image"]) for e in flatten_elements(doc)]
    assert got == [("heading", 3, "Сарлавҳа", None), ("paragraph", 3, "Ички матн", None),
                   ("list_item", 3, "1) банд", None), ("paragraph", 3, "давоми", None),
                   ("table_row", 4, "A a2 | B", None), ("image", 4, "", "x/imageFile9.png")]


def test_elements_to_paras_drops_toc_leader_lines():
    els = [{"type": "paragraph", "page": 2, "text": "I. Мурожаатларни кўриб чиқиш ва", "image": None},  # wrapped TOC entry
           {"type": "list_item", "page": 2, "text": "1.2. Иқтисодий ҳуқуқлар ........... 11-14", "image": None},
           {"type": "paragraph", "page": 2, "text": "Мундарижа Кириш ........ 3 I. Мурожаатлар ...... 4-8", "image": None},
           {"type": "table_row", "page": 2, "text": "оширишга доир фаолият ........... | 58", "image": None},
           {"type": "table_row", "page": 2, "text": "6.1. Ишлар 6.2. Ҳамкорлик ........... V", "image": None},
           {"type": "paragraph", "page": 3, "text": "Жами 1.500 та.", "image": None}]
    assert [p.text for p in elements_to_paras(els, image_text=lambda e: None)] == ["Жами 1.500 та."]


def _png(w, h):
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    chunk = struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return b"\x89PNG\r\n\x1a\n" + chunk


def test_png_size_reads_header():
    assert png_size(_png(300, 90)) == (300, 90)
    assert png_size(b"not a png") is None


class _FakeClient:
    """Mimics openai.OpenAI().chat.completions.create for two prompts."""
    def __init__(self, a, b):
        self.calls, self._a, self._b = [], a, b
        self.chat = self
        self.completions = self

    def create(self, model, messages, temperature):
        self.calls.append(messages[0]["content"][0]["text"])
        txt = self._a if len(self.calls) % 2 == 1 else self._b

        class R:  # minimal response shape
            choices = [type("C", (), {"message": type("M", (), {"content": txt})()})()]
        return R()


def test_vision_accepts_only_self_consistent_numbers(tmp_path):
    ok = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("898 та назоратда, 93 та юборилди",
                                                                   "898 — назоратда; 93 — юборилди"))
    assert ok(b"img-a", page_text="") == "898 та назоратда, 93 та юборилди"
    bad = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("898 та, 95 та", "898 — ...; 93 — ..."))
    assert bad(b"img-b", page_text="") is None


def test_vision_skips_text_already_in_text_layer_and_caches(tmp_path):
    cl = _FakeClient("898 та мурожаат назоратга олинди", "898 — назоратга олинди")
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=cl)
    assert vt(b"img-c", page_text="2024 йилда 898 та мурожаат назоратга олинди") is None
    n = len(cl.calls)
    assert vt(b"img-c", page_text="2024 йилда 898 та мурожаат назоратга олинди") is None
    assert len(cl.calls) == n                                               # cache hit, no API call


def test_vision_none_answer_costs_one_call_and_stats(tmp_path):
    cl = _FakeClient("NONE", "unused")
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=cl)
    assert vt(b"photo", page_text="") is None
    assert len(cl.calls) == 1
    rec = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert rec["accepted"] is False and rec["model"] == "gpt-4.1" and rec["prompt_version"] >= 1
    assert vt.stats["api_calls"] == 1 and vt.stats["none"] == 1


def test_vision_budget_guard_raises(tmp_path):
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("12 та мурожаат", "12 — мурожаат"), max_calls=3)
    assert vt(b"one", page_text="") == "12 та мурожаат"                              # 2 calls
    with pytest.raises(VisionBudgetExceeded):
        vt(b"two", page_text="")                                            # would be calls 3 and 4
    assert vt(b"one", page_text="") == "12 та мурожаат"                              # cached: still fine


@pytest.mark.corpus
def test_real_pdf_has_pages_and_no_running_headers():
    paras = read_pdf(PDF12, vision=None)
    assert paras and all(p.page for p in paras)
    assert max(p.page for p in paras) >= 69          # text layer ends on p.69; pp.70-71 are scanned maps (vision)
    hdr = "Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари бўйича вакили"
    assert sum(p.text.strip() == hdr for p in paras) <= 1


@pytest.mark.corpus
def test_real_pdf_table_rows_and_infographic_pages():
    paras = read_pdf(PDF12, vision=None)
    assert "1. | Қашқадарё вилояти | 2818 | 3274" in [p.text for p in paras]
    p13 = read_pdf(PDF13, vision=None)
    assert all(p.page for p in p13) and max(p.page for p in p13) >= 52


@pytest.mark.corpus
def test_read_paragraphs_falls_back_to_pypdf_with_pages(monkeypatch):
    import graphrag.data.pdf as pdfmod
    from graphrag.data.extract import read_paragraphs

    def boom(*a, **k):
        raise pdfmod.PdfEngineError("no java")
    monkeypatch.setattr(pdfmod, "read_pdf", boom)
    paras = read_paragraphs(PDF12)
    assert paras and all(p.page for p in paras) and max(p.page for p in paras) >= 69
    hdr = "Ўзбекистон Республикаси Олий Мажлисининг Инсон ҳуқуқлари бўйича вакилининг (омбудсманнинг) 2024 йилдаги фаолияти тўғрисида"
    assert sum(hdr in p.text for p in paras) <= 1                           # 124-char running header


def test_line_breaks_keep_compound_hyphens():
    doc = {"kids": [{"type": "paragraph", "page number": 1,
                     "content": "вояга етмаганларга ижтимоий-\nҳуқуқий ёрдам\nкўрсатиш – ҳаракатланиш\n  эркинлиги"}]}
    assert flatten_elements(doc)[0]["text"] == \
        "вояга етмаганларга ижтимоий-ҳуқуқий ёрдам кўрсатиш – ҳаракатланиш эркинлиги"


def test_split_prose_paragraph_is_rejoined_across_page_and_image():
    long = "Мурожаатларнинг " + "таҳлили ва натижалари " * 10 + "шахсий ҳуқуқ ва"
    els = [{"type": "paragraph", "page": 8, "text": long, "image": None},
           {"type": "image", "page": 9, "text": "", "image": "i.png"},
           {"type": "paragraph", "page": 9, "text": "эркинликлар бўйича келиб тушган.", "image": None},
           {"type": "paragraph", "page": 9, "text": long, "image": None},
           {"type": "paragraph", "page": 9, "text": "(31,8 фоиз) ижтимоий ҳуқуқлар.", "image": None},
           {"type": "paragraph", "page": 9, "text": "2021 йил 62 ўрин", "image": None}]      # digit: infographic label
    paras = elements_to_paras(els, image_text=lambda e: "12 та")
    assert [p.page for p in paras] == [8, 9, 9, 9]
    assert paras[0].text == long + " эркинликлар бўйича келиб тушган."
    assert paras[1].text == "[расм] 12 та"
    assert paras[2].text == long + " (31,8 фоиз) ижтимоий ҳуқуқлар."
    assert paras[3].text == "2021 йил 62 ўрин"


def test_split_infographic_labels_are_rejoined_but_not_headings_or_finished_clauses():
    P = lambda t: {"type": "paragraph", "page": 29, "text": t, "image": None}
    els = [P("Босма газета ва журналларда 12 та мақола эълон"), P("қилинди"),
           P("оид 6 355 та медиа-"), P("материал жамоатчиликка эълон қилинди"),
           P("ЖИНОЯТЛАРНИНГ ГЕНДЕР БЎЙИЧА ТАСНИФИ"), P("ўғил болалар 60 фоиз."),
           P("4 та мурожаат келиб тушган"), P("етим болалар ҳақида.")]
    assert [p.text for p in elements_to_paras(els, image_text=lambda e: None)] == [
        "Босма газета ва журналларда 12 та мақола эълон қилинди",
        "оид 6 355 та медиа-материал жамоатчиликка эълон қилинди",
        "ЖИНОЯТЛАРНИНГ ГЕНДЕР БЎЙИЧА ТАСНИФИ", "ўғил болалар 60 фоиз.",
        "4 та мурожаат келиб тушган", "етим болалар ҳақида."]


def test_vision_prefetch_fills_cache(tmp_path):
    cl = _FakeClient("12 та мурожаат", "12 — мурожаат")
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=cl)
    vt.prefetch([b"p1", b"p1"], workers=1)                                  # duplicates fetched once
    assert len(cl.calls) == 2
    assert vt(b"p1", page_text="") == "12 та мурожаат" and len(cl.calls) == 2


def test_vision_rejects_fragments(tmp_path):
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("SCHOOL", "NONE"))
    assert vt(b"frag", page_text="") is None and vt.stats["rejected_short"] == 1


# ---- fix round 1 ----

def test_vision_b_is_always_called_and_numbers_must_match_as_multiset(tmp_path):
    cl = _FakeClient("Навоий Бухоро Самарқанд", "16 | Навоий\n11 | Бухоро")          # A dropped the values
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=cl)
    assert vt(b"map", page_text="") is None and len(cl.calls) == 2
    assert vt.stats["rejected_disagreement"] == 1
    cl2 = _FakeClient("898 та, 898 та, 93 та", "898 | назоратда\n93 | юборилди")      # 898 twice vs once
    assert VisionTranscriber("gpt-4.1", tmp_path, client=cl2)(b"dup", page_text="") is None
    cl3 = _FakeClient("898 та назоратда, 93 та юборилди", "- 898 | назоратда\n- 93 | 1-бандга кўра юборилди")
    assert VisionTranscriber("gpt-4.1", tmp_path, client=cl3)(b"ok", page_text="") == \
        "898 та назоратда, 93 та юборилди"                                             # label digits ignored


def test_vision_reuses_cached_a_when_only_policy_or_b_changed(tmp_path):
    import hashlib as _h
    from graphrag.data import pdf as pdfmod
    key = _h.sha256(b"legacy").hexdigest()
    (tmp_path / f"{key}.json").write_text(json.dumps(
        {"model": "gpt-4.1", "prompt_version": 3, "a": "12 та мурожаат", "b": "NONE", "accepted": True}),
        encoding="utf-8")
    cl = _FakeClient("12 | мурожаат", "unused")
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=cl)
    assert vt(b"legacy", page_text="") == "12 та мурожаат"
    assert cl.calls == [pdfmod.PROMPT_B]                                               # only B re-asked
    rec = json.loads((tmp_path / f"{key}.json").read_text(encoding="utf-8"))
    assert rec["prompt_version"] == pdfmod.PROMPT_VERSION and not list(tmp_path.glob("*.tmp"))


def _E(t, page, bbox, typ="paragraph"):
    return {"type": typ, "page": page, "text": t, "image": None, "bbox": bbox}


def test_number_only_elements_attach_to_nearest_label():
    els = [_E("61,4%", 9, [400, 700, 440, 715]),
           _E("3 таси тегишлилиги бўйича юборилган", 9, [600, 700, 780, 712]),
           _E("қиз бола", 9, [400, 684, 450, 696]),                                    # below 61,4%
           _E("1064", 9, [60, 650, 100, 668]),
           _E("жами мурожаат", 9, [110, 652, 190, 664]),                               # right of 1064
           _E("44 та", 9, [300, 400, 330, 420]),
           _E("Узоқдаги матн бу ерда", 9, [300, 100, 400, 112])]                       # too far: stays alone
    texts = [p.text for p in elements_to_paras(els, image_text=lambda e: None)]
    assert texts == ["3 таси тегишлилиги бўйича юборилган", "61,4% қиз бола", "1064 жами мурожаат",
                     "44 та", "Узоқдаги матн бу ерда"]


def test_side_by_side_short_labels_are_not_rejoined():
    els = [_E("ўғил болалар", 50, [100, 500, 160, 510]), _E("қиз болалар", 50, [300, 500, 360, 510])]
    assert [p.text for p in elements_to_paras(els, image_text=lambda e: None)] == ["ўғил болалар", "қиз болалар"]


def test_flatten_keeps_bbox():
    els = flatten_elements(json.loads(FIX.read_text(encoding="utf-8")))
    assert all(len(e["bbox"]) == 4 for e in els if e["type"] != "table_row")


def test_rejoined_paragraph_records_page_end_and_chunk_src():
    from graphrag.data.chunker import _Chunker
    from graphrag.data.structure import DocText
    long = "Мурожаатларнинг " + "таҳлили ва натижалари " * 10 + "шахсий ҳуқуқ ва"
    els = [{"type": "paragraph", "page": 8, "text": long, "image": None},
           {"type": "paragraph", "page": 9, "text": "эркинликлар бўйича келиб тушган.", "image": None},
           {"type": "paragraph", "page": 9, "text": "Кейинги абзац.", "image": None}]
    paras = elements_to_paras(els, image_text=lambda e: None)
    assert [(p.page, p.page_end) for p in paras] == [(8, 9), (9, None)]
    dt = DocText.build(paras)
    ch = _Chunker(dt, [], 1000, len)
    assert ch.src(0, dt.offsets[0][1]) == "с. 8-9, абз. 1"
    assert ch.src(0, dt.offsets[1][1]) == "с. 8-9, абз. 1-2"


def test_ligatures_and_nul_ti():
    doc = {"kids": [{"type": "paragraph", "page number": 1,
                     "content": "IDC (Interna\x00onal Development) ﬁnal eﬀort ﬂow"}]}
    assert flatten_elements(doc)[0]["text"] == "IDC (International Development) final effort flow"


def test_page_pairs_accepts_only_exact_number_multiset(tmp_path):
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("қиз бола: 61,4%\nўғил бола: 38,6%", "x"))
    assert vt.page_pairs(b"page-img", "k1", ["61.4% қиз бола ва 38,6% ўғил бола"]) == ["қиз бола: 61,4%", "ўғил бола: 38,6%"]
    vt2 = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("қиз бола: 61,4%", "x"))
    assert vt2.page_pairs(b"page-img2", "k2", ["61,4%", "38,6%"]) is None
    assert vt2.stats["page_rejected"] == 1


def test_page_pairs_are_inserted_after_the_page_text():
    from graphrag.data.extract import Para
    from graphrag.data.pdf import insert_page_pairs
    paras = [Para("a", None, 8), Para("b", None, 9), Para("[расм] c", None, 9), Para("d", None, 10)]
    got = insert_page_pairs(paras, {9: ["қиз бола: 61,4%"], 11: ["x: 1"]})
    assert [(p.page, p.text) for p in got] == [(8, "a"), (9, "b"), (9, "[расм] c"),
                                                (9, "[расм-жадвал] қиз бола: 61,4%"), (10, "d"),
                                                (11, "[расм-жадвал] x: 1")]


def test_page_pairs_normalizes_value_first_lines_and_drops_unlabelled(tmp_path):
    ans = "636 та мурожаат: аёллар\n84,3%:\nЖами: 1064 та"
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient(ans, "x"))
    assert vt.page_pairs(b"pg", "k3", ["636 та мурожаат аёллар", "84,3%", "Жами 1064 та"]) == \
        ["аёллар: 636 та мурожаат", "Жами: 1064 та"]


def test_vision_b_may_list_label_numbers_in_either_view(tmp_path):
    cl = _FakeClient("01 1666 таси ижтимоий ҳуқуқ 02 1197 таси иқтисодий ҳуқуқ",
                     "01 | 1666 таси ижтимоий ҳуқуқ\n02 | 1197 таси иқтисодий ҳуқуқ")
    assert VisionTranscriber("gpt-4.1", tmp_path, client=cl)(b"enum", page_text="") is not None


def test_page_pairs_drop_labels_not_in_text_layer(tmp_path):
    ans = "янги важлар: 53 таси\nянгли ваҳжлар ёки аниқдан: 20 таси"
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient(ans, "x"))
    assert vt.page_pairs(b"pg4", "k4", ["53 таси янги важлар ёки янгидан", "20 таси"]) == \
        ["янги важлар: 53 таси", "янги важлар ёки янгидан: 20 таси"]   # round 2: garbled label snapped, not dropped


# ---- fix round 2 ----

P6 = ["469 таси вакилнинг котибияти қабули орқали", "53 таси янги важлар ёки янгидан очилган ҳолатлар",
      "20 таси қонунчиликда белгиланган тартибда кўрмасдан қолдирилди", "2024 йилда жами 1064 та"]


def test_page_pair_labels_snap_to_verbatim_text_layer_span(tmp_path):
    ans = ("вакилнинг котибияти қабул орқали: 469 таси\n"
           "янгли ваҳжлар ёки аниқдан очилган ҳолатлар: 53 таси\n"
           "қонунчиликта белгиланган тартибда кўрмасдан қолдирилди: 20 таси\n"
           "2024 йилда жами: 1064 та")
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient(ans, "x"))
    assert vt.page_pairs(b"p6", "s1", P6) == [
        "вакилнинг котибияти қабули орқали: 469 таси",
        "янги важлар ёки янгидан очилган ҳолатлар: 53 таси",
        "қонунчиликда белгиланган тартибда кўрмасдан қолдирилди: 20 таси",
        "2024 йилда жами: 1064 та"]


def test_page_pair_without_matching_span_or_digit_value_is_dropped(tmp_path):
    ans = "мутлақо бошқа нарса ҳақида гап: 469 таси\nжами: мурожаат\n53 таси\n20 таси\n2024 йилда жами: 1064 та"
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient(ans, "x"))
    assert vt.page_pairs(b"p6b", "s2", P6) == ["2024 йилда жами: 1064 та"]


def test_page_read_skips_on_render_error_and_renders_only_on_cache_miss(tmp_path, monkeypatch):
    import subprocess as sp
    from graphrag.data import pdf as pdfmod
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient("жами: 1064 та", "x"))

    def boom(*a, **k):
        raise sp.CalledProcessError(1, "pdftoppm")
    monkeypatch.setattr(pdfmod, "render_page", boom)
    assert pdfmod.read_page_pairs(vt, Path("x.pdf"), 6, 150, "s3", ["жами 1064 та"]) is None   # no abort
    monkeypatch.setattr(pdfmod, "render_page", lambda *a: b"png")
    assert pdfmod.read_page_pairs(vt, Path("x.pdf"), 6, 150, "s3", ["жами 1064 та"]) == ["жами: 1064 та"]
    monkeypatch.setattr(pdfmod, "render_page", boom)                                         # cached now
    assert pdfmod.read_page_pairs(vt, Path("x.pdf"), 6, 150, "s3", ["жами 1064 та"]) == ["жами: 1064 та"]


def test_page_prompt_excludes_heading_numbers():
    from graphrag.data import pdf as pdfmod
    assert "heading" in pdfmod.PAGE_PROMPT.lower() and pdfmod.PAGE_PROMPT_VERSION >= 2


def test_page_pair_value_must_be_a_number_with_a_short_unit(tmp_path):
    ans = "Бола ҳуқуқлари бўйича вакилнинг: 35 та шахсий қабуллари ташкил этилган бўлиб\nаноним тарзда юборилган: 2 таси"
    vt = VisionTranscriber("gpt-4.1", tmp_path, client=_FakeClient(ans, "x"))
    assert vt.page_pairs(b"p6c", "s5", ["Бола ҳуқуқлари бўйича вакилнинг 35 та шахсий қабуллари",
                                        "2 таси аноним тарзда юборилган."]) == ["аноним тарзда юборилган: 2 таси"]


# ---- Task 11b: sentences split across ODL element types -------------------------------------
def _T(typ, text, page=7, bbox=None):
    return {"type": typ, "page": page, "text": text, "image": None, "bbox": bbox}


def test_sentence_split_across_element_types_is_rejoined():
    els = [_T("list_item", "Хусусан, 2024 йилда Омбудсман томонидан ташкилотлар ёки мансабдор", bbox=[56, 700, 538, 712]),
           _T("paragraph", "шахсларга 39 та тақдимнома киритилди.", bbox=[56, 686, 538, 699]),
           _T("heading", "Шунингдек, Женевада бўлиб ўтган БМТнинг Инсон ҳуқуқлари бўйича", bbox=[56, 600, 538, 612]),
           _T("paragraph", "кенгашининг 55-сессиясида Омбудсман иштирок этди.", bbox=[56, 586, 538, 599]),
           _T("paragraph", "Минтақавий вакилларга келиб тушган", bbox=[92, 500, 363, 512]),
           _T("paragraph", "мурожаатлар сонининг кўпайиши, авваломбор, ҳудудларда ташкил этилган сайёр қабуллар "
                           "ва “Омбудсман мактаби” лойиҳаси билан боғлиқ.", bbox=[56, 470, 538, 499])]
    assert [p.text for p in elements_to_paras(els, image_text=lambda e: None)] == [
        "Хусусан, 2024 йилда Омбудсман томонидан ташкилотлар ёки мансабдор шахсларга 39 та тақдимнома киритилди.",
        "Шунингдек, Женевада бўлиб ўтган БМТнинг Инсон ҳуқуқлари бўйича кенгашининг 55-сессиясида Омбудсман иштирок этди.",
        "Минтақавий вакилларга келиб тушган мурожаатлар сонининг кўпайиши, авваломбор, ҳудудларда ташкил этилган "
        "сайёр қабуллар ва “Омбудсман мактаби” лойиҳаси билан боғлиқ."]


def test_sentence_join_skips_tables_images_values_caps_and_far_boxes():
    els = [_T("table_row", "Қашқадарё вилояти | 2818"), _T("paragraph", "мурожаатлар сони."),
           _T("list_item", "– Ички ишлар вазирлигига 201 та"), _T("paragraph", "боланинг билим олиш ҳуқуқи бўйича"),
           _T("heading", "14. БОЛАНИНГ БИЛИМ ОЛИШ ҲУҚУҚИНИ ТАЪМИНЛАШ"), _T("paragraph", "унда тарбияланаётган болалар сони."),
           _T("heading", "мурожаатлари ҳамкорликда кўриб чиқилмоқда"), _T("caption", "мурожаатларни ўрганиш юзасидан"),
           _T("paragraph", "“А”ги қонун лойиҳаси", bbox=[143, 294, 774, 376]),
           _T("heading", "бола ҳуқуқларининг кафолатлари юзасидан", bbox=[143, 62, 774, 121])]
    texts = [p.text for p in elements_to_paras(els, image_text=lambda e: None)]
    assert len(texts) == 10
    els = [_T("paragraph", "Жами"), {"type": "image", "page": 7, "text": "", "image": "i.png", "bbox": None},
           _T("paragraph", "мурожаатлар")]
    assert [p.text for p in elements_to_paras(els, image_text=lambda e: "12 та")][1] == "[расм] 12 та"
