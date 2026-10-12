"""M3 Task 12 (D6): fill Q-real GoldItem.support with the verified gold_complete pass.
    GRL_GOLD_API=1 python scripts/complete_gold.py     # live (train.yaml gold_llm budget)
    python scripts/complete_gold.py                    # offline replay"""
import json
import os
import sys

from graphrag import config
from graphrag.data.ingest import read_records
from graphrag.env.data import load_env_data
from graphrag.eval.gold import read_gold, write_gold
from graphrag.eval.gold_complete import complete
from graphrag.kg.llm import KgLLM
from graphrag.retrieval.rankings import QREAL_LATEST, load_rankings


def main(argv=None) -> int:
    tcfg = config.load("train")
    data = load_env_data()
    ranks = load_rankings(QREAL_LATEST, graph_version=data.version)
    records = read_records(config.path(config.load("data")["records_dir"]))
    f = config.path(tcfg["gold"]["file"])
    llm = KgLLM(cfg={"llm": tcfg["gold_llm"]}, cache_root=tcfg["gold_llm"]["cache_dir"],
                allow_api=os.environ.get("GRL_GOLD_API") == "1")
    items, st = complete(read_gold(f), records, ranks, data, llm)
    write_gold(f, items)
    st["llm"] = llm.stats_json()
    config.path("data/gold/complete_stats.json").write_text(json.dumps(st, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in st.items() if k != "core_not_supporting"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
