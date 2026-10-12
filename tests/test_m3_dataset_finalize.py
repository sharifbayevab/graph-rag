import numpy as np

from graphrag.dataset.facts import Fact
from graphrag.dataset.finalize import Pair, dedup_and_leak, manual_sample, ngram_jaccard, template
from graphrag.dataset.generate import Candidate
from graphrag.dataset.seeds import Seed
from graphrag.dataset.trajectories import expert_trajectory, select
from graphrag.env.mdp import Env
from graphrag.env.reward import Weights
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank

CFG = {"dedup_cos_max": 0.90, "ngram_jaccard_max": 0.5, "qreal_question_cos_max": 0.90}


def _c(i, q, chunks=("D#1",), lang="ru", pattern="single", split="train"):
    return Candidate(Seed(f"{pattern}-{i:010d}", pattern, lang, chunks, ("D",), split, "QONUNLAR"), q, "a",
                     [Fact(f"{chunks[0]}@0", chunks[0], "", "v", "q" * 20)], {})


def test_dedup_ngram_and_leakage():
    cs = [_c(1, "Кто утверждает бюджет страны на год?"), _c(2, "Кто утверждает бюджет страны на год вперёд?"),
          _c(3, "Какой срок полномочий Сената?"), _c(4, "Сколько членов в Сенате?", chunks=("Q#7",)),
          _c(5, "Где хранится архив?")]
    qv = np.eye(5)
    qv[4] = [0, 0, 0.6, 0, 0.8]
    kept, dropped = dedup_and_leak(cs, qv, np.array([[0, 0, 0.6, 0, 0.8]]), {"Q#7"}, CFG, 7)
    assert ngram_jaccard(cs[0].question, cs[1].question) >= 0.5
    assert "duplicate" in {dropped.get("single-0000000001"), dropped.get("single-0000000002")}
    assert dropped["single-0000000004"] == "qreal_fragment" and dropped["single-0000000005"] == "qreal_question"


def test_manual_sample_covers_every_cell():
    cs = [_c(i, f"q{i}?", lang=["ru", "uz-Cyrl"][i % 2], pattern=["single", "amends"][(i // 2) % 2],
             split=["train", "test"][(i // 4) % 2]) for i in range(200)]
    s = manual_sample(cs, 16, 7)
    assert len(s) == 16 and s == manual_sample(cs, 16, 7)
    assert {(c.seed.split, c.seed.pattern, c.seed.lang) for c in s} == \
        {(sp, p, l) for sp in ("train", "test") for p in ("single", "amends") for l in ("ru", "uz-Cyrl")}


def test_template_is_the_first_two_content_words():
    # ruling M3-R18 (b) replaces the brief's "first two words": the opening is the first 3 internal-form tokens
    assert template("Кто утверждает бюджет?") == template("кто  утверждает, бюджет!")
    assert template("Кто утверждает бюджет?") != template("кто  утверждает закон?")
    assert template("Ким тасдиқлайди бюджетни?") == template("Kim tasdiqlaydi byudjetni qachon?") == "kim tasdiqlaydi byudjetni"


def _p(i, split, pattern, lang="ru", doc="D", q=None):
    return Pair(f"syn-{i:04d}", q or f"Вопрос{i} номер{i} о чём-то?", lang, pattern,
                "single" if pattern == "single" else "multi", {"text": "a", "sources": []}, [f"{doc}#{i}"], [], [],
                [doc], split, "QONUNLAR", f"s{i}")


SPL = ["train"] * 7 + ["val"] * 2 + ["test"]


def test_select_exact_cells_doc_cap_and_template_cap_without_backfill():
    ps = [_p(i, SPL[i % 10], "single" if (i // 10) % 2 else "amends", doc=f"D{i}") for i in range(300)]
    ratios, hops, langs = {"train": 0.7, "val": 0.15, "test": 0.15}, {"single": 0.5, "amends": 0.5}, {"ru": 1.0}
    chosen, short = select(ps, 100, ratios, hops, langs, 7, 3, 0.08)
    assert short == {} and len(chosen) == 100 and chosen == select(ps, 100, ratios, hops, langs, 7, 3, 0.08)[0]
    _, short = select([p for p in ps if not (p.split == "test" and p.pattern == "amends")], 100, ratios, hops, langs, 7, 3, 0.08)
    assert short == {"amends|test|ru": 8}
    same_doc = [_p(i, "train", "single", doc="D") for i in range(20)]
    chosen, short = select(same_doc, 10, {"train": 1.0}, {"single": 1.0}, langs, 7, 3, 1.0)
    assert len(chosen) == 3 and short == {"single|train|ru": 7}                  # ≤ 3 pairs per document per cell
    same_q = [_p(i, "train", "single", doc=f"D{i}", q="Кто утверждает что-то?") for i in range(40)]
    chosen, _ = select(same_q, 25, {"train": 1.0}, {"single": 1.0}, langs, 7, 99, 0.08)
    assert len(chosen) == 2                                                     # ⌈0.08·25⌉ = 2 per opening


def test_expert_trajectory_criterion_first_then_threshold():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    s, parts, why = expert_trajectory(env, q, frozenset({4}), Weights(), 0.55, width=8)
    assert why == "ok" and s.reason == "stop" and {4} <= s.ctx_set and parts.total >= 0.55
    s2, parts2, why2 = expert_trajectory(env, q, frozenset({4}), Weights(), 0.9, width=8)
    assert why2 == "below_threshold" and s2 is not None


# --- added over the brief (Task 16 hard requirements) --------------------------------------------------------------
import json  # noqa: E402
from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402

from graphrag.dataset.finalize import (facts_cache_keys, provenance_missing, read_pairs, to_pair,  # noqa: E402
                                       write_pairs)
from graphrag.dataset.trajectories import TrajectoryReplayError, replay, traj_one  # noqa: E402
from graphrag.env.parallel import run  # noqa: E402
from graphrag.eval.gold import Anchor  # noqa: E402


def test_leakage_split_groups_and_every_qreal_chunk_class():
    """D8 / M3-R3 п. 13: a pair whose document is assigned to another split (group split) is dropped; any C_ref chunk in
    qreal_exclusion (core, support, alt, context, linked same path) is dropped; question cos to Q-real >= 0.90."""
    cs = [_c(1, "Кто назначает судей?"), _c(2, "Кем утверждается смета?", split="test"),
          _c(3, "Сколько дней длится сессия?", chunks=("D#1", "D/П.9#2")), _c(4, "Где публикуются законы?")]
    qv = np.eye(4)
    real = np.array([[0, 0, 0, 0.95]])
    kept, dropped = dedup_and_leak(cs, qv, real, {"D/П.9#2"}, dict(CFG, qreal_question_cos_max=0.90), 7,
                                   split_of={"D": "train"})
    assert dropped == {"single-0000000002": "split_leak", "single-0000000003": "qreal_fragment",
                       "single-0000000004": "qreal_question"}
    assert [c.seed.seed_id for c in kept] == ["single-0000000001"]
    kept, dropped = dedup_and_leak(cs, qv, real * 0.9, {"D/П.9#2"}, CFG, 7, split_of={"D": "train"})
    assert "single-0000000004" not in dropped                                    # 0.855 < 0.90


def test_dedup_exact_and_cosine_and_determinism():
    cs = [_c(1, "Кто утверждает бюджет?"), _c(2, "кто  утверждает   бюджет?"), _c(3, "Какие права у депутата?"),
          _c(4, "Каковы права депутата парламента?")]
    qv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0.1, 0.995]])
    kept, dropped = dedup_and_leak(cs, qv, np.zeros((0, 3)), set(), CFG, 7)
    assert len(kept) == 2 and sorted(dropped.values()) == ["duplicate", "duplicate"]
    assert (kept, dropped) == dedup_and_leak(list(reversed(cs)), qv[::-1], np.zeros((0, 3)), set(), CFG, 7)


def _prov():
    return {"seed_id": "single-0000000001", "chunks": ["D#1"], "walk": [], "fact_ids": ["D#1@0"],
            "facts": {"pass": "ds_facts", "prompt_digest": "f" * 64, "cache_keys": {"D#B0": "k" * 64}},
            "generate": {"pass": "ds_generate", "prompt_digest": "g" * 64, "cache_key": "c" * 64, "prompt_version": 1},
            "verify": {"pass": "ds_verify", "prompt_digest": "v" * 64, "cache_key": "w" * 64, "prompt_version": 1,
                       "verdicts": {"full": {}}, "report": {"facts_recovered": [True], "ablation": {}}},
            "copy": {"max_run": 2, "overlap": 0.3}}


def test_provenance_must_be_complete():
    p = replace(_p(1, "train", "single"), provenance={**_prov(), "seed": {}})
    assert provenance_missing(p) == []
    for path in (("facts", "cache_keys"), ("verify", "cache_key"), ("verify", "verdicts"), ("generate", "cache_key"),
                 ("copy", "max_run"), ("verify", "report")):
        pr = json.loads(json.dumps(p.provenance))
        del pr[path[0]][path[1]]
        assert ".".join(path) in provenance_missing(replace(p, provenance=pr))
    pr = json.loads(json.dumps(p.provenance))
    pr["facts"]["cache_keys"] = {}
    assert "facts.cache_keys" in provenance_missing(replace(p, provenance=pr))


def test_facts_cache_keys_are_the_keys_extract_block_used(tmp_path):
    """The Task 15 concern: provenance names the ds_facts cache files the pair's facts came from."""
    from graphrag.dataset.facts import extract_block
    from graphrag.kg.llm import KgLLM
    T2 = "5-модда. Марказий банк банкларни санация қилиш тўғрисида қарор ўттиз кун ичида қабул қилади."

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    body = json.dumps({"chunks": [{"id": "C1", "facts": []}, {"id": "C2", "facts": []}]})
                    return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": body})})]})
    cfg = {"llm": {"model": "gpt-4.1", "max_calls_total": 2, "passes": {"ds_facts": {"prompt_version": 1, "max_calls": 2}}}}
    ch = {"D/СТ.5#1": type("Ch", (), {"text": T2, "parent": "D/СТ.5"})(),
          "D/СТ.5#2": type("Ch", (), {"text": T2 + " II", "parent": "D/СТ.5"})()}
    ix = type("Ix", (), {"act": {"D": "Қонун"}, "format": staticmethod(lambda n: "Қонун, ст. 5")})()
    llm = KgLLM(cfg, cache_root=tmp_path, client=Client(), allow_api=True)
    extract_block(llm, "D#B0", ["D/СТ.5#1", "D/СТ.5#2"], ch, ix, 6)
    files = sorted(p.stem for p in (tmp_path / "ds_facts").glob("*.json"))
    cand = _c(1, "q?", chunks=("D/СТ.5#2",))
    keys = facts_cache_keys(cand, {"D/СТ.5#1": "D#B0", "D/СТ.5#2": "D#B0"}, {"D#B0": ["D/СТ.5#1", "D/СТ.5#2"]}, ch, ix,
                            KgLLM(cfg, cache_root=tmp_path, allow_api=False))
    assert keys == {"D#B0": files[0]} and len(files) == 1


def test_pair_roundtrip_and_to_pair(tmp_path):
    class Ch:
        def __init__(self, s, e, t):
            self.start, self.end, self.text = s, e, t
    chunks = {"D/СТ.5#1": Ch(0, 5, "abcde"), "D/СТ.3#1": Ch(9, 12, "xyz")}
    ix = type("Ix", (), {"sources_for": staticmethod(lambda data, ids: [f"Қонун, {i}" for i in ids])})()
    c = Candidate(Seed("amends-0000000001", "amends", "ru", ("D/СТ.5#1", "D/СТ.3#1"), ("D",), "val", "QONUNLAR",
                       (("D/СТ.5", "AMENDS", "D/СТ.3"),)), "Что изменено?", "a b", [], _prov())
    p = to_pair(c, ix, None, chunks)
    assert p.C_ref == ["D/СТ.3#1", "D/СТ.5#1"] and p.hop == "multi" and p.split == "val" and p.origin == "synthetic"
    assert p.C_ref_anchors[0] == Anchor("D", 9, 12, __import__("hashlib").sha256(b"xyz").hexdigest())
    assert p.provenance["seed"]["seed_id"] == "amends-0000000001" and p.y_ref["sources"]
    write_pairs(tmp_path / "p.jsonl", [p])
    assert read_pairs(tmp_path / "p.jsonl") == [p]


class _Ranks:
    def get(self, key):
        return toy_rank(key)


def _toy_pair(C_ref=("D2/СТ.5#2",)):
    return replace(_p(1, "train", "single"), id="syn-toy", C_ref=list(C_ref))


def test_traj_one_writes_a_replayable_stop_ended_8_1_record():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    p, why, r, ops, cell = traj_one({"env": env, "ranks": _Ranks(), "w": Weights(), "thr": 0.55, "width": 8}, _toy_pair())
    assert why == "ok" and r >= 0.55 and cell == "train|single|ru"
    assert p.trajectory[-1] == {"t": len(p.trajectory), "op": "stop"} and p.actions[-1] == 0
    assert all("arg" in x for x in p.trajectory[:-1])
    assert len(p.action_equiv) == len(p.action_masks) == len(p.actions)
    assert all(a in e and a in m for a, e, m in zip(p.actions, p.action_equiv, p.action_masks))
    assert p.reward["total"] == r
    q = env.query(toy_rank("syn-toy|q"))
    s, parts = replay(env, q, p.actions, frozenset({4}), Weights(), 0.55)
    assert s.reason == "stop" and 4 in s.ctx_set and parts.total == r
    # Task 13 parallel runner, serial path: the same record
    out = run(traj_one, [_toy_pair()], 1, env=env, ranks=_Ranks(), w=Weights(), thr=0.55, width=8)
    assert out[0][0] == p


def test_replay_refuses_a_trajectory_that_misses_c_ref_or_does_not_stop():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    q = env.query(toy_rank())
    s, parts, _ = expert_trajectory(env, q, frozenset({4}), Weights(), 0.55, width=8)
    acts = [h.action for h in s.history]
    with pytest.raises(TrajectoryReplayError):
        replay(env, q, acts[:-1], frozenset({4}), Weights(), 0.55)              # no STOP
    with pytest.raises(TrajectoryReplayError):
        replay(env, q, [0], frozenset({4}), Weights(), 0.55)                    # C_ref not reached
    with pytest.raises(TrajectoryReplayError):
        replay(env, q, acts, frozenset({4}), Weights(), 0.99)                   # R below the threshold


def test_below_threshold_pairs_carry_no_trajectory():
    env = Env(toy_data(), dict(TOY_CFG), TOY_START)
    p, why, r, ops, _ = traj_one({"env": env, "ranks": _Ranks(), "w": Weights(), "thr": 0.99, "width": 8}, _toy_pair())
    assert why == "below_threshold" and r is not None and p.trajectory == [] and p.actions == []


D7 = {"hops": {"single": 0.55, "refers_internal": 0.10, "refers_cross": 0.15, "amends": 0.20},
      "langs": {"uz-Cyrl": 0.45, "uz-Latn": 0.25, "ru": 0.30}, "ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}
TRANSFER = {"splits": ["val", "test"], "from": "amends", "to": "refers_internal"}


def test_cells_match_the_task14_final_need():
    """Review I3: pattern -> split -> language, as cmd_seeds allocates (the two cells that were off by one)."""
    from graphrag.dataset.trajectories import final_needs
    f = final_needs(1000, D7["hops"], D7["ratios"], D7["langs"])
    assert f["refers_cross|val|uz-Latn"] == 5 and f["single|test|uz-Latn"] == 21 and sum(f.values()) == 1000


def _pool(scarce_amends: dict):
    """Plenty of pairs in every D7 cell except amends val/test, which get scarce_amends[(split, lang)]."""
    from graphrag.dataset.trajectories import final_needs
    out, i = [], 0
    for k, n in final_needs(1000, D7["hops"], D7["ratios"], D7["langs"]).items():
        pat, sp, lang = k.split("|")
        have = scarce_amends.get((sp, lang), 3 * n) if pat == "amends" else 3 * n + 30
        for _ in range(have):
            out.append(_p(i, sp, pat, lang=lang, doc=f"D{i}"))
            i += 1
    return out


def test_select_transfers_scarce_val_test_amends_to_refers_internal_same_language():
    """M3-R15: amends val/test take what passes, refers_internal of the same split and language fills the rest;
    train stays exact; the transfer is recorded; total 1000."""
    scarce = {("val", "ru"): 2, ("test", "uz-Cyrl"): 5}
    rep = {}
    chosen, short = select(_pool(scarce), 1000, D7["ratios"], D7["hops"], D7["langs"], 7, 3, 0.08,
                           transfer=TRANSFER, report=rep)
    f = rep["final_need_by_cell"]
    got = __import__("collections").Counter(f"{p.pattern}|{p.split}|{p.lang}" for p in chosen)
    assert short == {} and len(chosen) == 1000
    assert got["amends|val|ru"] == 2 and got["amends|test|uz-Cyrl"] == 5
    assert got["refers_internal|val|ru"] == f["refers_internal|val|ru"] + f["amends|val|ru"] - 2
    assert got["refers_internal|test|uz-Cyrl"] == f["refers_internal|test|uz-Cyrl"] + f["amends|test|uz-Cyrl"] - 5
    assert rep["transfer"]["applied"]["val|ru"] == f["amends|val|ru"] - 2
    assert rep["transfer"]["total"] == (f["amends|val|ru"] - 2) + (f["amends|test|uz-Cyrl"] - 5)
    assert all(got[k] == n for k, n in f.items() if k.split("|")[1] == "train")
    for sp in ("val", "test"):                                                # multi-hop share per split unchanged
        multi = sum(v for k, v in got.items() if k.split("|")[1] == sp and not k.startswith("single"))
        assert multi == sum(v for k, v in f.items() if k.split("|")[1] == sp and not k.startswith("single"))


def test_scarce_train_amends_is_still_a_stop():
    chosen, short = select(_pool({("train", "ru"): 10}), 1000, D7["ratios"], D7["hops"], D7["langs"], 7, 3, 0.08,
                           transfer=TRANSFER)
    assert list(short) == ["amends|train|ru"]


def test_without_transfer_scarce_val_amends_is_a_stop():
    from graphrag.dataset.trajectories import final_needs
    need = final_needs(1000, D7["hops"], D7["ratios"], D7["langs"])["amends|val|ru"]
    _, short = select(_pool({("val", "ru"): 2}), 1000, D7["ratios"], D7["hops"], D7["langs"], 7, 3, 0.08)
    assert short == {"amends|val|ru": need - 2}


def test_dedup_keeps_the_pair_of_the_scarcer_cell():
    cs = [_c(i, f"Уникальный вопрос номер {i} про сенат?") for i in range(5)] + \
        [_c(9, "Кто утверждает бюджет страны на год?", pattern="amends", split="test"),
         _c(10, "Кто утверждает бюджет страны на год вперёд?")]
    qv = np.eye(7)
    kept, dropped = dedup_and_leak(cs, qv, np.zeros((0, 7)), set(), CFG, 7)
    assert dropped == {"single-0000000010": "duplicate"}


def test_transfer_does_not_stop_falsely_on_the_template_cap():
    """Re-review m3: the destination holds exactly final + transfer with two pairs sharing an opening; ru / uz-Cyrl have
    surplus -- the transfer must settle on a feasible split of the need, not STOP."""
    from graphrag.dataset.trajectories import final_needs
    f = final_needs(1000, D7["hops"], D7["ratios"], D7["langs"])
    ps = [p for p in _pool({("val", "ru"): 0, ("val", "uz-Cyrl"): 0, ("val", "uz-Latn"): 0})
          if not (p.pattern == "refers_internal" and p.split == "val" and p.lang == "uz-Latn")]
    n = f["refers_internal|val|uz-Latn"] + f["amends|val|uz-Latn"]
    for k in range(n):
        ps.append(_p(90000 + k, "val", "refers_internal", lang="uz-Latn", doc=f"X{k}",
                     q="Кто утверждает бюджет сейчас?" if k < 2 else None))
    rep = {}
    chosen, short = select(ps, 1000, D7["ratios"], D7["hops"], D7["langs"], 7, 3, 0.08, transfer=TRANSFER, report=rep)
    assert short == {} and len(chosen) == 1000
