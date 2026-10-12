import pytest

from graphrag import config
from graphrag.data.ingest import build_corpus, validate_record
from graphrag.data.tokens import TokenCounter

pytestmark = [pytest.mark.corpus, pytest.mark.model]


@pytest.fixture(scope="module")
def built():
    cfg = config.load("data")
    count = TokenCounter(cfg["tokenizer"])
    return cfg, count, build_corpus(cfg, count)


def test_all_records_valid_and_have_parents(built):
    cfg, _, (records, skipped) = built
    assert not [s for s in skipped if s["reason"].startswith("error")], skipped
    assert len(records) >= 300
    for r in records:
        assert validate_record(r, cfg["max_tokens"]) == [], r.source.file


def test_qonunlar_articles_found(built):
    _, _, (records, _) = built
    arts = [n for r in records if r.meta.section == "QONUNLAR" for n in r.structure if n.type == "статья"]
    assert len(arts) >= 800                         # ~892 own articles minus duplicate files
    # own article numbers (without superscript inserts) run 1..N in (almost) every law file
    runs = []
    for r in records:
        nums = [int(n.number) for n in r.structure if n.type == "статья" and n.number.isdigit()]
        if nums:
            runs.append(nums == list(range(1, len(nums) + 1)))
    assert sum(runs) / len(runs) >= 0.9, f"{sum(runs)}/{len(runs)} contiguous"


def test_rebuild_gives_identical_records(built):
    cfg, count, (records, _) = built
    again, _ = build_corpus(cfg, count)
    assert [r.to_json() for r in again] == [r.to_json() for r in records]
