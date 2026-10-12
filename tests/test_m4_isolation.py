"""M4/M5 code lives in grl/ and its config in config/train.yaml: both outside the graph version digest (M3 Task 1),
so the pinned graph 40d525a97cb9 does not move while M3 is still building the dataset (plan decision D9)."""
import json

import pytest

from graphrag import config
from graphrag.kg import build as B


def test_grl_is_outside_the_graph_version_inputs(tmp_path):
    import grl  # noqa: F401  -- the package exists
    assert not config.path("grl").resolve().is_relative_to(config.path("graphrag").resolve())
    for rel in ("graphrag/kg/a.py", "grl/model/gnn.py", "grl/train/grpo.py", "grl/threads.py",
                "config/graph.yaml", "config/train.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x: 1\n", encoding="utf-8")
    assert sorted(B.code_digest(tmp_path)) == ["graphrag/kg/a.py"]          # grl/ never enters the digest
    assert sorted(B.config_digest(tmp_path)) == ["config/graph.yaml"]         # train.yaml is skipped
    assert all(not p.startswith("grl/") for p in B.files_digest(config.path("."), B.VERSION_CODE_GLOBS))
    assert "config/train.yaml" in B.VERSION_CONFIG_SKIP


@pytest.mark.corpus
def test_the_worktree_recomputes_the_pinned_graph_version_inputs():
    """The code / config / library digest of this checkout equals the inputs the pinned graph 40d525a97cb9 was
    built from (its stats.json), so neither grl/ nor a new library version (Task 0 Step 4) moved the version."""
    latest = json.loads(config.path("data/graph/latest.json").read_text(encoding="utf-8"))
    assert latest["version"] == "40d525a97cb9"
    vin = json.loads((config.path(latest["dir"]) / "stats.json").read_text(encoding="utf-8"))["version_inputs"]
    root = config.path(".")
    assert B.code_digest(root) == vin["code"]
    assert B.config_digest(root) == vin["config"]
    assert B.library_versions() == vin["libraries"]


def test_no_new_config_files():
    names = sorted(p.name for p in config.path("config").glob("*.yaml"))
    assert names == ["data.yaml", "graph.yaml", "model.yaml", "train.yaml"]


def test_training_sections_copy_the_spec_values():
    t = config.load("train")
    m = t["model"]
    assert (m["in_dim"], m["hidden"], m["heads"], m["layers"]) == (1024, 256, 8, 3) and m["hidden"] // m["heads"] == 32
    assert m["state"]["pool_hidden"] == 128 and m["policy"]["hidden"] == [512, 256]
    assert m["graph"] == {"reverse_edges": True, "drop_types": [], "hub_cap": None}
    g = t["grpo"]
    assert (g["G_s"], g["lr"], g["beta_kl"], g["step_penalty"], g["freeze_gnn"]) == (4, 3.0e-4, 0.01, 0.0, False)
    assert t["sft"]["loss"] == "marginal" and t["sft"]["lr"] == 3.0e-4
    assert (t["env"]["T_max"], t["env"]["K_max"], t["env"]["L_max"]) == (10, 24, 8000)
    assert t["env"]["graph_version"] == "40d525a97cb9"
    assert t["run"]["keep_top"] == 3 and t["run"]["deterministic"] in ("strict", "warn", "off")
    assert t["m5_pilot"]["splits"] == ["train", "val"]
    s = t["spark"]
    assert s["remote_dir"].startswith("abdurakhmon/") and ".." not in s["remote_dir"]
    assert s["auth"] == "password" and (s["env_target"], s["env_password"]) == ("SHH", "PASSWORD")   # D7 b
    assert "root" not in t["run"]                                             # run dirs are always explicit
