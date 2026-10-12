"""Instructions v8: non-contiguous sitting days, relative years."""
from graphrag import config
from graphrag.dataset.generate import relative_year, session_range_gap

DEC = [{"session": {"date": "2024-12-18", "title": "Senatning 2-yalpi majlisi (2024-yil 18-19- va 21-dekabr)"}}]
MAR = [{"session": {"date": "2022-03-17", "title": "Senatning 24-yalpi majlisi (2022-yil 17-, 19-mart)"}}]
APR = [{"session": {"date": "2025-04-29", "title": "Senatning 6-yalpi majlisi 29-30.04.25"}}]


def test_a_range_must_not_span_a_missing_sitting_day():
    assert session_range_gap("На заседании 18–21 декабря 2024 года кто выступил?", DEC)
    assert session_range_gap("2022-yil 17–19-martdagi majlisda kim gapirdi?", MAR)
    assert not session_range_gap("На заседании 18–19 и 21 декабря 2024 года кто выступил?", DEC)
    assert not session_range_gap("2022-yil 17 va 19-martdagi majlisda kim gapirdi?", MAR)
    assert not session_range_gap("2025 йил 29–30 апрелдаги мажлисда ким гапирди?", APR)


def test_relative_years_need_the_actual_year():
    assert relative_year("Ўзбекистон маърузани жорий йилнинг июль ойида тақдим этади.")
    assert relative_year("Отчёт представят в текущем году.") and relative_year("В прошлом году выдано 23 кредита.")
    assert not relative_year("Ўзбекистон маърузани жорий, яъни 2023 йилнинг июль ойида тақдим этади.")
    assert not relative_year("Отчёт представят в 2023 году.")


def test_instructions_v8():
    o = config.load("train")["dataset"]["opus"]
    t = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["generator"]["version"] == 8 and o["generator_history"][8] == o["generator"]["digest"]
    assert "Non-contiguous sittings" in t and "No relative years" in t and "18–19 ва 21 декабрдаги" in t
