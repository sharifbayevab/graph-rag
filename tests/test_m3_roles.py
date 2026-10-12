"""Instructions v7: a speaker's role / title only when the fragment text or meta states it."""
from graphrag import config
from graphrag.dataset.generate import unsupported_roles

FR = [{"act": "Senatning 52-yalpi majlisi (2024-yil 18-may)", "where": "x",
       "text": "Tadbirkor qonunni takroran buzsa jarima solinadi.", "speaker": "Mubin Muxidinovich Mirzayev"}]


def test_role_words_need_support_in_text_or_meta():
    assert unsupported_roles("2024 йил 18 майдаги мажлисда сенатор Мубин Мирзаев нима деди?", FR) == ["senator"]
    assert unsupported_roles("Что сказал депутат Мирзаев на заседании?", FR) == ["deputat"]
    assert unsupported_roles("2024 йил 18 майдаги мажлисда Мубин Мирзаев нима деди?", FR) == []
    said = [dict(FR[0], text="Senator sifatida aytamanki, jarima solinadi.")]
    assert unsupported_roles("Сенатор Мирзаев нима деди?", said) == []
    chair = [dict(FR[0], speaker="Raislik qiluvchi")]
    assert unsupported_roles("Раислик қилувчи нима деди?", chair) == []


def test_instructions_v7_rule_and_example():
    o = config.load("train")["dataset"]["opus"]
    t = config.path(o["generator"]["file"]).read_text(encoding="utf-8")
    assert o["generator"]["version"] >= 7 and o["generator_history"][o["generator"]["version"]] == o["generator"]["digest"]
    assert "A speaker's title, position, role or stance" in t and "Мубин Муҳидинович Мирзаевнинг" in t
