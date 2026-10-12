from graphrag.types import Chunk, DocRecord, Meta, Node, Source


def _rec():
    text = "1-модда. Мақсад\n\nМатн."
    return DocRecord(
        doc_id="QONUNLAR-0000abcd",
        meta=Meta(type="law_text", date="2019-05-14", organ="senate", lang=["uz-Cyrl"],
                  status="принят", section="QONUNLAR", title="Тест"),
        source=Source(store="docs", file="docs/QONUNLAR/t.docx", pages=None),
        text=text,
        structure=[Node("QONUNLAR-0000abcd", "документ", "Тест", None, None, 0, len(text)),
                   Node("QONUNLAR-0000abcd/СТ.1", "статья", "Мақсад", "1", "QONUNLAR-0000abcd", 0, len(text))],
        chunks=[Chunk("QONUNLAR-0000abcd/СТ.1#1", "QONUNLAR-0000abcd/СТ.1", 1, 4, "абз. 1-2",
                      text, "1-modda. maqsad\n\nmatn.", 0, len(text))],
    )


def test_roundtrip():
    r = _rec()
    assert DocRecord.from_json(r.to_json()) == r


def test_json_is_deterministic_and_keeps_cyrillic():
    s1, s2 = _rec().to_json(), _rec().to_json()
    assert s1 == s2
    assert "модда" in s1          # ensure_ascii=False
    assert s1.endswith("\n")
