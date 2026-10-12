"""M4/M5 toy fixtures on top of the M3 toy (tests/m3_toy.py): the toy graph as a pyg.pt payload (M2 report §14 field
set), a second toy query, picklable rankings / env init for the process vec-env, and oracle-labelled §8.1 pairs."""
import json

import numpy as np
import torch

from graphrag.env.mdp import Env
from graphrag.retrieval.rankings import QueryRanking
from tests.m3_toy import EDGE_TYPES, EDGES, NODE_X, NODES, TOY_CFG, TOY_START, toy_data, toy_rank

OPS = ["replace", "add", "delete", "rename", "repeal"]
MODEL_CFG = {"in_dim": 4, "hidden": 256, "heads": 8, "layers": 3, "edge_dim": 32, "dropout": 0.0, "encoder": "gat",
             "graph": {"reverse_edges": True, "drop_types": [], "hub_cap": None},
             "state": {"pool_hidden": 128, "cand_dim": 32, "preview": 3, "cand_top": 64},
             "policy": {"hidden": [512, 256]}}
SMALL_CFG = {**MODEL_CFG, "hidden": 32, "heads": 4, "edge_dim": 8,          # training tests: fast, tiny checkpoints
             "state": {"pool_hidden": 16, "cand_dim": 8, "preview": 3, "cand_top": 64}, "policy": {"hidden": [64, 32]}}
DKW = {"preview": 3, "cand_top": 64}


def toy_payload(x_dim: int = 4, relations: int = 2, node_types: int = 3) -> dict:
    row = {n: i for i, n in enumerate(NODES)}
    rows = []
    for s, t, d in EDGES:
        ti = EDGE_TYPES.index(t)
        rows.append((ti, row[s], row[d]))
        if t == "RELATED":
            rows.append((ti, row[d], row[s]))
    rows.sort()
    E = len(rows)
    et = torch.tensor([r[0] for r in rows], dtype=torch.long)
    is_ref = et == EDGE_TYPES.index("REFERS_TO")
    is_am = et == EDGE_TYPES.index("AMENDS")
    ops = torch.zeros((E, len(OPS)), dtype=torch.bool)
    ops[is_am, 1] = True
    if x_dim == 4:
        x = torch.from_numpy(NODE_X.copy())
    else:
        x = torch.nn.functional.normalize(torch.randn(len(NODES), x_dim, generator=torch.Generator().manual_seed(0)), dim=1)
    return {
        "x": x.float(),
        "edge_index": torch.tensor([[r[1] for r in rows], [r[2] for r in rows]], dtype=torch.long),
        "edge_type": et,
        "edge_weight": torch.where(et == EDGE_TYPES.index("RELATED"), 0.8, 1.0).float(),
        "edge_attr_relation": torch.where(is_ref, 0, -1).long(),
        "edge_attr_op": torch.where(is_am, 1, -1).long(),
        "edge_attr_ops": ops,
        "node_kind": torch.tensor([0, 1, 1, 0, 1, 2, 0, 1]),
        "node_type": torch.tensor([1, 2, 2, 1, 2, -1, 0, 2]),
        "entity_type": torch.tensor([-1, -1, -1, -1, -1, 0, -1, -1]),
        "is_external": torch.tensor([False] * 6 + [True, True]),
        "node_ids": list(NODES), "edge_types": list(EDGE_TYPES),
        "node_kinds": ["document", "element", "entity", "concept"],
        "edge_relations": ["title", "article"] + [f"r{i}" for i in range(relations - 2)],
        "amend_ops": list(OPS),
        "node_types": ["document:external", "document:law", "element:article"] + [f"t{i}" for i in range(node_types - 3)],
        "entity_types": ["organ", "position", "procedure", "legal_category"],
    }


def toy_graph(**kw):
    from grl.model.graph import graph_tensors
    return graph_tensors(toy_payload(), **kw)


def toy_env() -> Env:
    return Env(toy_data(), TOY_CFG, TOY_START)


def toy_rank2(qkey: str = "toy|q2") -> QueryRanking:
    """e_q = (0, 1, 0, 0): fragment cosines .6 0 1 .8 0."""
    return QueryRanking(qkey, "toy question 2", np.array([0, 1, 0, 0], np.float32), np.array([2, 3]),
                        np.array([2.0, 1.0]), np.array([2, 3, 0, 1, 4]), np.array([1.0, 0.8, 0.6, 0.0, 0.0]),
                        np.zeros(0, np.int64), np.zeros(0, np.float64))


class ToyRanks:
    """keys() / get() like graphrag.retrieval.rankings.Rankings; module-level so a spawn worker can rebuild it."""

    def __init__(self):
        self._r = {"toy|q": toy_rank("toy|q"), "toy|q2": toy_rank2("toy|q2"),
                   "p1|q": toy_rank("p1|q"), "p2|q": toy_rank2("p2|q")}

    def keys(self):
        return sorted(self._r)

    def get(self, k):
        return self._r[k]


TOY_RANKS = ToyRanks()


def toy_init() -> dict:
    return {"data": toy_data(), "ranks": ToyRanks(), "env_cfg": TOY_CFG, "start_cfg": TOY_START, "describe_kw": DKW}


def _pair(pid, split, c_ref, actions=(), equiv=(), masks=()) -> dict:
    return {"id": pid, "q": "?", "lang": "uz-Cyrl", "pattern": "single", "hop": "single",
            "y_ref": {"text": "", "sources": []}, "C_ref": list(c_ref), "C_ref_anchors": [], "facts": [],
            "docs": ["D1"], "split": split, "section": "QONUNLAR", "seed_id": pid, "actions": list(actions),
            "action_equiv": [list(e) for e in equiv], "action_masks": [list(m) for m in masks]}


def toy_pairs(path, p2_split: str = "val"):
    """Two oracle-labelled §8.1 pairs (M3 BeamSearch + trajectory_labels, STOP ∧ full C_ref) and one test pair."""
    from graphrag.env.policies import BeamSearch, d5_accept, trajectory_labels
    from graphrag.env.reward import Weights
    env, w, rows = toy_env(), Weights(), []
    for pid, split, gold in (("p1", "train", ("D2/СТ.5#1", "D2/СТ.5#2")), ("p2", p2_split, ("D1/СТ.2#1",))):
        q = env.query(TOY_RANKS.get(f"{pid}|q"))
        g = frozenset(env.data.chunk_row[c] for c in gold)
        s = BeamSearch(env, q, g, w, width=8, graph_share=0.25, lookahead=True).best(d5_accept(g))
        assert s is not None, pid
        equiv, masks = trajectory_labels(env, q, s)
        rows.append(_pair(pid, split, gold, [h.action for h in s.history], equiv, masks))
    rows.append(_pair("p3", "test", ("D1/СТ.1#1",), [0], [[0]], [[0, 1]]))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path
