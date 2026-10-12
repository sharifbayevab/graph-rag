"""Task 11b: verified LLM agenda segmentation of stenograms (fake client, no network)."""
import json
from types import SimpleNamespace

import pytest

from graphrag.data.agenda import (AgendaBudgetExceeded, AgendaSegmenter, _cut_unopened_quote,
                                  build_input, excerpt, snap_title, verify)
from graphrag.data.extract import Para
from graphrag.data.structure import DocText, _is_chair, parse_stenogram

D = "X-0000abcd"
P = ["Oʼzbekiston Respublikasi Oliy Majlisi Senatining yalpi majlisi",
     "Raislik qiluvchi: Assalomu alaykum. Kun tartibi tasdiqlandi.",                         # turn 0
     "Raislik qiluvchi: Birinchi masala hayvonlarni identifikatsiya qilish, roʼyxatga olish "
     "va kuzatish toʼgʼrisidagi qonun haqida. Soʼz qoʼmita raisiga.",                        # turn 1
     "Sayyora Sapayevna Abdikarimova: Qonun chorvachilikni rivojlantirishga qaratilgan.",   # turn 2
     "Raislik qiluvchi: Rahmat. Savollar bormi? Ovozga qoʼyaman.",                            # turn 3
     "(ovoz beriladi)",
     "Raislik qiluvchi: Qaror qabul qilindi. “Elektron raqamli imzo toʼgʼrisida”gi Qonunni "
     "koʼrib chiqishga kirishamiz.",                                                        # turn 4
     "Odiljon Karimovich Iminov: Assalomu alaykum hurmatli senatorlar.",                     # turn 5
     "Raislik qiluvchi: Rahmat. Majlis yopiq deb eʼlon qilinadi."]                           # turn 6


def _turns(paras):
    from graphrag.data.structure import _ROLE, _SPEAKER
    out = []
    for j, t in enumerate(paras):
        m = _SPEAKER.match(t) or _ROLE.match(t)
        if m:
            out.append({"a": j, "b": j + 1, "speaker": m["name"]})
        elif out:
            out[-1]["b"] = j + 1
    return out


class FakeClient:
    def __init__(self, answer: dict):
        self.answer, self.calls = json.dumps(answer, ensure_ascii=False), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))])


GOOD = {"items": [{"start_turn": 1, "title": "hayvonlarni identifikatsiya qilish, roʼyxatga olish va "
                                             "kuzatish toʼgʼrisidagi qonun"},
                  {"start_turn": 4, "title": "Elektron raqamli imzo toʼgʼrisida"}]}


def test_snap_title_returns_verbatim_span_and_rejects_absent_titles():
    text = "Raislik qiluvchi: Navbatdagi masala “Oʼzbekiston Respublikasining Maʼmuriy javobgarlik " \
           "toʼgʼrisidagi kodeksiga qoʼshimcha kiritish haqida”gi Qonun toʼgʼrisida."
    # model spelling differs (apostrophe variant, case, one typo) → the text's own spelling comes back
    got = snap_title("O'zbekiston respublikasining Mamuriy javobgarlik toʼgʼrisidagi kodeksiga "
                     "qoʼshimcha kiritish haqida", text)
    assert got == "Oʼzbekiston Respublikasining Maʼmuriy javobgarlik toʼgʼrisidagi kodeksiga qoʼshimcha kiritish haqida”gi"
    assert _cut_unopened_quote(got) == ("Oʼzbekiston Respublikasining Maʼmuriy javobgarlik toʼgʼrisidagi "
                                        "kodeksiga qoʼshimcha kiritish haqida")
    assert snap_title("Soliq kodeksini tasdiqlash toʼgʼrisidagi qonun", text) is None


def test_verify_checks_turns_order_and_titles():
    chair = {0: "Raislik qiluvchi: Kun tartibi.", 3: "Raislik qiluvchi: Birinchi masala Suv kodeksi haqida.",
             7: "Raislik qiluvchi: Navbatdagi masala Yer kodeksi haqida.",
             9: "Raislik qiluvchi: Rahmat. Keyingi masala."}
    ctx = {9: "Maʼruzachi: Bojxona kodeksi toʼgʼrisida gapiraman.\nDavomi."}
    items = [{"start_turn": 3, "title": "Suv kodeksi"},
             {"start_turn": 8, "title": "Yer kodeksi"},          # presenter's turn → repaired to chair turn 7
             {"start_turn": 5, "title": "Suv kodeksi"},          # not increasing
             {"start_turn": 9, "title": "Bojxona kodeksi"},      # title only in the next paragraphs
             {"start_turn": 9, "title": "Soliq kodeksi"},        # nowhere in the text
             {"start_turn": 99, "title": "Yer kodeksi"}]         # repaired to 9, title not there
    got, why = verify(items, chair, ctx)
    assert got == [(3, "Suv kodeksi"), (7, "Yer kodeksi"), (9, "Bojxona kodeksi")]
    assert why["repaired_turn"] == 1 and why["not_increasing"] == 1
    assert why["title_not_found"] == 1 and why["bad_turn"] == 1


def test_verify_same_start_turn_keeps_the_later_item():
    chair = {4: "Raislik qiluvchi: Oliy sud tarkibi haqidagi masala hal qilindi. Endi Suv kodeksi haqida."}
    got, why = verify([{"start_turn": 4, "title": "Oliy sud tarkibi"}, {"start_turn": 4, "title": "Suv kodeksi"}],
                      chair, {})
    assert got == [(4, "Suv kodeksi")] and why["same_turn_replaced"] == 1


def test_input_lists_chair_turns_with_head_and_tail():
    long = "a" * 900 + " MIDDLE " + "b" * 900
    user = build_input("Senatning 7-yalpi majlisi", [(0, "Raislik qiluvchi: salom"), (5, long)])
    assert user.splitlines()[0] == "Session: Senatning 7-yalpi majlisi"
    assert "[0] Raislik qiluvchi: salom" in user and "[5] " in user
    assert "MIDDLE" not in user and excerpt(long).count(" … ") == 1


def test_segmenter_caches_and_parse_stenogram_uses_verified_items(tmp_path):
    client = FakeClient(GOOD)
    seg = AgendaSegmenter("gpt-4.1", tmp_path, client=client, allow_api=True, max_calls=5)
    dt = DocText.build([Para(t) for t in P])
    nodes = parse_stenogram(D, dt, "Senatning 7-yalpi majlisi", seg)
    qs = [n for n in nodes if n.type == "вопрос"]
    assert [q.title for q in qs] == ["", "hayvonlarni identifikatsiya qilish, roʼyxatga olish va kuzatish "
                                         "toʼgʼrisida", "Elektron raqamli imzo toʼgʼrisida"]   # fix round 1: trimmed
    assert [sum(n.parent_id == q.node_id for n in nodes) for q in qs] == [1, 3, 3]
    assert len(client.calls) == 1 and client.calls[0]["temperature"] == 0
    assert client.calls[0]["response_format"]["json_schema"]["strict"] is True
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["model"] == "gpt-4.1"
    # a second run is offline: same nodes from the cache, no call
    off = AgendaSegmenter("gpt-4.1", tmp_path, client=FakeClient({"items": []}), allow_api=False)
    assert parse_stenogram(D, dt, "Senatning 7-yalpi majlisi", off) == nodes
    assert off.stats["cache_hits"] == 1 and off._client.calls == []
    # another model is another cache key
    assert AgendaSegmenter("gpt-4o", tmp_path).cache_file("x") != seg.cache_file("x")


def test_offline_miss_and_unverified_answers_fall_back_to_regex(tmp_path):
    dt = DocText.build([Para(t) for t in P])
    regex = parse_stenogram(D, dt, "t")
    off = AgendaSegmenter("gpt-4.1", tmp_path, client=FakeClient(GOOD), allow_api=False)
    assert parse_stenogram(D, dt, "t", off) == regex
    assert off.stats["offline_miss"] == 1 and off._client.calls == [] and not list(tmp_path.glob("*.json"))
    bad = {"items": [{"start_turn": 1, "title": "Soliq kodeksi"}, {"start_turn": 2, "title": "Yer kodeksi"},
                     {"start_turn": 4, "title": "Elektron raqamli imzo toʼgʼrisida"}]}      # 1 of 3 verifies
    seg = AgendaSegmenter("gpt-4.1", tmp_path / "b", client=FakeClient(bad), allow_api=True)
    assert parse_stenogram(D, dt, "t", seg) == regex and seg.stats["fallback_regex"] == 1


def test_budget_is_a_hard_cap(tmp_path):
    seg = AgendaSegmenter("gpt-4.1", tmp_path, client=FakeClient(GOOD), allow_api=True, max_calls=1)
    turns = _turns(P)
    assert seg("s1", P, turns, _is_chair)
    with pytest.raises(AgendaBudgetExceeded):
        seg("s2", P, turns, _is_chair)                  # a different input: needs a second call
    assert seg.stats["api_calls"] == 1


# ---- fix round 1 ----------------------------------------------------------------------------
from graphrag.data import agenda as _agenda  # noqa: E402
from graphrag.data.agenda import tidy_title  # noqa: E402

PINNED_DIGESTS = {3: "3057fe96951848c39a4095a99a01ddbf7c126c951349336f18ec381d6034a8bf"}


def test_prompt_digest_is_pinned_to_the_prompt_version():
    # editing PROMPT or SCHEMA without bumping PROMPT_VERSION (and pinning the new digest) fails here
    assert PINNED_DIGESTS.get(_agenda.PROMPT_VERSION) == _agenda.PROMPT_DIGEST


def test_cache_record_with_another_prompt_digest_is_not_used(tmp_path):
    seg = AgendaSegmenter("gpt-4.1", tmp_path, client=FakeClient(GOOD), allow_api=True)
    assert seg.answer("u") is not None
    f = seg.cache_file("u")
    rec = json.loads(f.read_text())
    assert rec["prompt_digest"] == _agenda.PROMPT_DIGEST
    rec["prompt_digest"] = "0" * 64
    f.write_text(json.dumps(rec))
    off = AgendaSegmenter("gpt-4.1", tmp_path, allow_api=False)
    assert off.answer("u") is None and off.stats["offline_miss"] == 1


def test_tidy_title_restores_opening_quote_and_trims_trailing_noun():
    src = "Raislik qiluvchi: Navbatdagi masala ““Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi " \
          "Qonuniga qoʼshimcha kiritish haqida”gi Qonun toʼgʼrisida."
    t = tidy_title("Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuniga qoʼshimcha kiritish haqida", src)
    assert t == "“Davlat boji toʼgʼrisida”gi Oʼzbekiston Respublikasi Qonuniga qoʼshimcha kiritish haqida"
    src2 = "Raislik qiluvchi: Birinchi masala Davlat mudofaa buyurtmasi toʼgʼrisidagi qonun haqida."
    assert tidy_title("Davlat mudofaa buyurtmasi toʼgʼrisidagi qonun", src2) == "Davlat mudofaa buyurtmasi toʼgʼrisida"
    src3 = "Endi Markaziy saylov komissiyasi tarkibiga oʼzgartirishlar kiritish haqidagi masalani koʼramiz."
    assert tidy_title("Markaziy saylov komissiyasi tarkibiga oʼzgartirishlar kiritish haqidagi masalani", src3) == \
        "Markaziy saylov komissiyasi tarkibiga oʼzgartirishlar kiritish haqida"
    # the rest is more than -gi + noun: unchanged
    t4 = "Kengashning qarorlarini tasdiqlash toʼgʼrisidagi qaror loyihasi"
    assert tidy_title(t4, t4) == t4
    # no opening quote before the span in the source: unchanged, still a substring
    assert tidy_title("A toʼgʼrisida”gi Kodeks", "x A toʼgʼrisida”gi Kodeks") == "A toʼgʼrisida”gi Kodeks"
