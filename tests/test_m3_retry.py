"""Retry packets: which generator skips count as 'law / amended act not named' (then eligible once graph meta names it)."""
import importlib.util
import sys

from graphrag import config


def _bd():
    spec = importlib.util.spec_from_file_location("bd", config.path("scripts/build_dataset.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["bd"] = m
    spec.loader.exec_module(m)
    return m


def test_skip_reason_classes():
    bd = _bd()
    law = ["the speech names no law or draft: an agency official's report",
           "the discussed law is not named ('muhokama qilinayotgan qonun')",
           "the law the speaker endorses is never named in the fragment ('mazkur qonun')"]
    amended = ["amended act not named in the fragment: items 20-22 change articles 32(3)",
               "amending-law fragment that names neither the amended code nor the amended article",
               "the new wording of art. 12 never says which law it belongs to"]
    other = ["procedural chair line; filler", "letterhead-titled report: the document cannot be named",
             "a table row of the road map without the measure it belongs to"]
    assert all(bd._SKIP_LAW.search(x) for x in law)
    assert all(bd._SKIP_AMENDED.search(x) for x in amended)
    assert not any(bd._SKIP_LAW.search(x) or bd._SKIP_AMENDED.search(x) for x in other)
