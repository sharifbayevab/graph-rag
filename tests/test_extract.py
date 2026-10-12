from graphrag.data.extract import Para, paras_from_md_text, paras_from_pdf_text, read_docx
from tests.helpers import P, R, make_docx

NUMBERING = (
    '<w:abstractNum w:abstractNumId="0">'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl>'
    '<w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="russianLower"/><w:lvlText w:val="%2)"/></w:lvl>'
    '</w:abstractNum><w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>'
)


def test_superscript_article_number_is_kept(tmp_path):
    f = make_docx(tmp_path / "a.docx", P(R("122"), R("1", sup=True), R("-модда. Болани назоратсиз қолдириш")))
    assert read_docx(f) == [Para("122¹-модда. Болани назоратсиз қолдириш")]


def test_auto_numbering_is_rendered(tmp_path):
    body = (P("Қонун маъқуллансин.", num=1) + P("кичик банд", num=1, ilvl=1)
            + P("Ушбу Қарор кучга киради.", num=1))
    f = make_docx(tmp_path / "q.docx", body, numbering_xml=NUMBERING)
    assert [p.text for p in read_docx(f)] == [
        "1. Қонун маъқуллансин.", "а) кичик банд", "2. Ушбу Қарор кучга киради."]
    assert [p.list_level for p in read_docx(f)] == [0, 1, 0]


def test_tables_tabs_and_empty_paragraphs(tmp_path):
    tbl = ('<w:tbl><w:tr><w:tc>' + P("A") + '</w:tc><w:tc>' + P("B") + '</w:tc></w:tr></w:tbl>')
    tab = '<w:p><w:r><w:t>x</w:t><w:tab/><w:t>y</w:t><w:br/><w:t>z</w:t></w:r></w:p>'
    f = make_docx(tmp_path / "t.docx", P("") + tbl + tab)
    assert [p.text for p in read_docx(f)] == ["A | B", "x y z"]


def test_headers_are_not_read(tmp_path):
    f = make_docx(tmp_path / "h.docx", P("Матн"), header_xml=P("10"))
    assert [p.text for p in read_docx(f)] == ["Матн"]


def test_pdf_reflow_drops_page_numbers_and_running_headers():
    header = "Бола ҳуқуқлари бўйича вакили"
    lines = []
    for page in range(5):
        lines += [header, f"Биринчи жумла {page} давоми", f"шу ерда {page} тугайди.", str(page + 1), ""]
    paras = paras_from_pdf_text("\n".join(lines))
    assert all(header not in p.text for p in paras)
    assert all(not p.text.strip().isdigit() for p in paras)
    assert paras[0].text == "Биринчи жумла 0 давоми шу ерда 0 тугайди."


def test_md_strips_images_and_tags():
    md = "# Сарлавҳа\n\n![](_page_1_Picture_2.jpeg)Матн<br>давоми\n| --- | --- |\n| a | b |"
    assert [p.text for p in paras_from_md_text(md)] == ["Сарлавҳа", "Матн давоми", "| a | b |"]



def test_pdf_reflow_handles_both_quote_types():
    # Curly quote U+201C should trigger paragraph break (from original brief)
    text_curly = "Биринчи жумла тугади.\n“Иккинчи жумла."
    paras = paras_from_pdf_text(text_curly)
    assert len(paras) == 2
    assert paras[0].text == "Биринчи жумла тугади."
    assert paras[1].text == "“Иккинчи жумла."
    
    # Straight quote should also trigger paragraph break
    text_straight = 'Биринчи жумла тугади.\n"Иккинчи жумла.'
    paras = paras_from_pdf_text(text_straight)
    assert len(paras) == 2
    assert paras[0].text == "Биринчи жумла тугади."
    assert paras[1].text == '"Иккинчи жумла.'
