from graphrag import config


def test_load_data_yaml():
    cfg = config.load("data")
    assert cfg["max_tokens"] == 384
    assert "QONUNLAR" in cfg["sections"]


def test_path_resolves_relative_to_package_root():
    assert config.path("data/records") == config.ROOT / "data" / "records"
    assert config.path("../docs").name == "docs"


def test_legacy_parse_is_importable():
    from graphrag.data.legacy.parse import parse_document
    pd = parse_document("x.docx", "QARORLAR", "Қ А Р О Р\n\nТест матни", "x.docx")
    assert pd.language in {"uz-Cyrl", "unknown"}
