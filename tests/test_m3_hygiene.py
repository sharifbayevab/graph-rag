"""Seed hygiene: unnameable documents (title heuristics) and low-content chunks (generator skip patterns)."""
from graphrag.dataset.seeds import bad_title, low_content


def test_bad_titles():
    assert bad_title("Tashkent Cotton Textile") == "generic_heading"
    assert bad_title("ЙЎЛ ХАРИТАСИ") and bad_title("Ақлли сув") == "generic_heading"
    assert bad_title("INSON HUQUQLARI BO'YICHA O'ZBEKISTON RESPUBLIKASI MILLIY MARKAZI THE NATIONAL CENTRE") == "letterhead"
    assert bad_title('"2030-йилга қадар Ўзбекистон Республикаси гендер тенгликка эришиш стратегиясини') == "truncated"
    assert bad_title("Biz Senat deganda, birinchi navbatda, milliy parlamentimizning siyosiy salohiyatini") == "sentence"
    for ok in ("Банклар ва банк фаолияти тўғрисида", "Senatning 6-yalpi majlisi 29-30.04.25",
               "Парламент сўрови (яйловлар деградациясининг озиқ-овқат хавфсизлигига таъсири бўйича)",
               "О санации и ликвидации банков"):
        assert bad_title(ok) is None, ok


def test_low_content_chunks():
    head = ("4-модда. Ўзбекистон Республикасининг Божхона кодекси (Ўзбекистон Республикаси Олий Мажлиси палаталарининг "
            "Ахборотномаси, 2016 йил, № 1, 2-модда) га қуйидаги ўзгартиш ва қўшимчалар киритилсин:")
    assert low_content(head) == "amending_header"
    assert low_content("ва бошқа ташкилотлар билан ҳамкорликда амалга оширади.") == "list_tail"
    assert low_content("Raislik qiluvchi: Savollar bormi? Yoʻq. Ovozga qoʻyaman.", "Raislik qiluvchi") == "chair_filler"
    assert low_content("12-модда. Давлат субсидиялари Олий Мажлис палаталари томонидан ажратилади.") is None
