"""M7 Task 4: observation, policy and expert of the 31-action `struct` variant + the 28 -> 31 weight transfer
(docs/m7 design §3.3). Toy: the M3 toy graph (tests/m3_toy.py) with STRUCT rows for its document / EXT roots
(extend_with_struct), so the GraphTensors of tests/m4_toy.py serve both the 28- and the 31-action policy."""
import numpy as np
import pytest
import torch

from graphrag.env.mdp import Env
from graphrag.env.policies import trajectory_labels
from grl.env.struct_data import extend_with_struct
from grl.model.observe import describe
from grl.model.policy import GraphPolicy, policy_dims
from grl.model.variants import StructEnv, struct_delta_v
from grl.train import expert as ex
from grl.train.fbeta import GRLWeights
from tests.m3_toy import TOY_CFG, TOY_START, toy_data, toy_rank
from tests.m4_toy import DKW, SMALL_CFG, toy_graph
from tests.test_m7_struct_env import make

ROOT_VECS = {"D1": [0.6, 0.8, 0, 0], "D2": [0.3, 0, np.sqrt(0.91), 0], "EXT/k": [0.8, 0, 0, 0.6]}


def struct_data(roots=("D1", "D2", "EXT/k"), base=None):
    ids = {f"STRUCT/{r}": r for r in roots}
    return extend_with_struct(base or toy_data(), {i: i for i in ids}, {i: np.array(ROOT_VECS[r], np.float32)
                                                                        for i, r in ids.items()}, {i: 40 for i in ids})


def _versioned(variant="full"):
    """Toy EnvData stamped with the pinned train.yaml graph version (setup / the expert workers check it); struct: +
    the STRUCT rows (added after the version stamp: dataclasses.replace drops them)."""
    from dataclasses import replace

    from graphrag import config
    base = replace(toy_data(), version=config.load("train")["env"]["graph_version"])
    return struct_data(base=base) if variant == "struct" else base


def struct_env(roots=("D1", "D2", "EXT/k")):
    return StructEnv(struct_data(roots), TOY_CFG, TOY_START)


def amends_m1(env):
    return next(i for i, a in enumerate(env.actions) if a.op == "graph" and a.r == "AMENDS" and a.m == 1)


def logits(pol, g, q, desc):
    with torch.no_grad():
        z, qz = pol.encode(g, torch.from_numpy(q.qvec[None]).float())
        return pol.head(pol.state(z, qz, [0], [desc]))


def batch(pol, g, env, s=None):
    q = env.query(toy_rank())
    d = describe(env, q, env.reset(q) if s is None else s, **DKW)
    return logits(pol, g, q, d), d


@pytest.fixture
def struct_toy_policy():
    torch.manual_seed(1)
    g = toy_graph()
    env = struct_env()
    pol = GraphPolicy(g, SMALL_CFG, n_actions=31, n_types=8).eval()
    lg, d = batch(pol, g, env)
    return pol, lg, d.mask


@pytest.fixture
def m3_toy_policy():
    torch.manual_seed(0)
    g = toy_graph()
    pol = GraphPolicy(g, SMALL_CFG).eval()
    for p in pol.head.parameters():                     # non-trivial biases: the transfer must carry them
        torch.nn.init.normal_(p, std=0.3)
    return pol, g


# ---- observation --------------------------------------------------------------------------------------------------

def test_forward_31(struct_toy_policy):
    pol, lg, mask = struct_toy_policy
    assert lg.shape[-1] == 31 and mask.shape == (31,) and mask[28:].any()


def test_describe_structure_is_type_7():
    env, m3 = struct_env(), Env(toy_data(), TOY_CFG, TOY_START)
    q, q3 = env.query(toy_rank()), m3.query(toy_rank())
    for steps in ((), (amends_m1(env),)):
        s, s3 = env.reset(q), m3.reset(q3)
        for a in steps:
            s, s3 = env.step(q, s, a), m3.step(q3, s3, a)
        d, d3 = describe(env, q, s, **DKW), describe(m3, q3, s3, **DKW)
        rows = env.struct_candidates(q, s)
        sc = struct_delta_v(env.data, q, rows, s.context, env.lambda_red, env.decimals)
        assert len(d.cand) == 8 and len(d3.cand) == 7
        assert d.cand[7].tolist() == env.data.chunk_node[rows].tolist()            # μ(STRUCT row) = its root
        assert d.cand_n[7] == len(rows) and d.cand_best[7] == pytest.approx(float(np.max(sc)))
        for r in range(7):
            assert d.cand[r].tolist() == d3.cand[r].tolist()
        assert d.cand_n[:7].tolist() == d3.cand_n.tolist() and d.cand_best[:7].tolist() == d3.cand_best.tolist()
        assert d.mask[:28].tolist() == d3.mask.tolist() and d.mask.shape == d.hist.shape == (31,)
        assert [p.tolist() for p in d.preview] == [p.tolist() for p in d3.preview]
        assert d.scalars.tolist() == d3.scalars.tolist() and d.preview_sim.tolist() == d3.preview_sim.tolist()
    d = describe(env, q, env.step(q, env.reset(q), 28), **DKW)
    assert d.hist[28] == pytest.approx(1 / env.limits.T_max) and d.hist[:28].sum() == 0


def test_describe_without_structure_candidates():
    env = struct_env(("EXT/k",))                          # V₀ of toy_rank has no EXT vertex: no candidate at reset
    q = env.query(toy_rank())
    d = describe(env, q, env.reset(q), **DKW)
    assert d.cand[7].size == 0 and d.cand_n[7] == 0 and d.cand_best[7] == 0 and not d.mask[28:].any()


def test_policy_dims():
    assert policy_dims(struct_env()) == {"n_actions": 31, "n_types": 8}
    assert policy_dims(Env(toy_data(), TOY_CFG, TOY_START)) == {"n_actions": 28, "n_types": 7}
    g = toy_graph()
    p = GraphPolicy(g, {**SMALL_CFG, **policy_dims(struct_env())})            # load_reference(g, mcfg) path
    assert p.n_actions == 31 and p.state.n_types == 8 and p.head.net[-1].out_features == 31
    assert GraphPolicy(g, SMALL_CFG).head.net[-1].out_features == 28


# ---- transfer -----------------------------------------------------------------------------------------------------

def test_transfer_preserves_old_logits(m3_toy_policy):
    from grl.model.transfer import expand_policy_state
    old, g = m3_toy_policy
    new = GraphPolicy(g, SMALL_CFG, n_actions=31, n_types=8).eval()
    new.load_state_dict(expand_policy_state(old.state_dict()), strict=True)
    m3 = Env(toy_data(), TOY_CFG, TOY_START)
    for roots in (("EXT/k",), ("D1", "D2", "EXT/k")):
        env = struct_env(roots)
        q, q3 = env.query(toy_rank()), m3.query(toy_rank())
        s, s3 = env.reset(q), m3.reset(q3)
        for a in (None, amends_m1(env)):
            if a is not None:
                s, s3 = env.step(q, s, a), m3.step(q3, s3, a)
            d = describe(env, q, s, **DKW)
            if roots == ("EXT/k",) and a is None:
                assert d.cand[7].size == 0                # the brief's case: a state with no structure candidate
            elif a is not None:
                assert d.cand[7].size > 0                 # zero columns: equal even with candidates
            torch.testing.assert_close(logits(new, g, q, d)[..., :28], logits(old, g, q3, describe(m3, q3, s3, **DKW)),
                                       rtol=1e-5, atol=1e-6)


def test_transfer_parameter_map(m3_toy_policy):
    from grl.model.transfer import expand_policy_state
    old, g = m3_toy_policy
    sd, env = old.state_dict(), struct_env()
    out = expand_policy_state(sd)
    new = GraphPolicy(g, SMALL_CFG, n_actions=31, n_types=8)
    assert set(out) == set(new.state_dict()) and all(out[k].shape == v.shape for k, v in new.state_dict().items())
    last = f"head.net.{len(new.head.net) - 1}"
    for k in sd:
        if k not in ("head.net.0.weight", f"{last}.weight", f"{last}.bias"):
            assert torch.equal(out[k], sd[k]), k
    for j, a in enumerate(env.actions[28:]):
        rows = [i for i, b in enumerate(env.actions[:28]) if b.op == "graph" and b.m == a.m]
        assert len(rows) == 7
        for suf in ("weight", "bias"):
            assert torch.allclose(out[f"{last}.{suf}"][28 + j], sd[f"{last}.{suf}"][rows].mean(0))
            assert torch.equal(out[f"{last}.{suf}"][:28], sd[f"{last}.{suf}"])
    # input layout (grl.model.state): q D | g f v0 3P | cand T·C | preview 2P | log1p n T | best T | psim 2 | 6 |
    # mask A | hist A
    D, P, C = SMALL_CFG["hidden"], SMALL_CFG["state"]["pool_hidden"], SMALL_CFG["state"]["cand_dim"]
    old_blocks = [D + 3 * P + 7 * C, 2 * P + 7, 7, 2 + 6 + 28, 28]
    new_pad = [C, 1, 1, 3, 3]                             # zeros appended after each old block
    keep, pos = [], 0
    for n, z in zip(old_blocks, new_pad):
        keep += list(range(pos, pos + n))
        pos += n + z
    w0, w1 = sd["head.net.0.weight"], out["head.net.0.weight"]
    assert w1.shape[1] == pos == new.state.out_dim and w0.shape[1] == sum(old_blocks) == old.state.out_dim
    zero = sorted(set(range(pos)) - set(keep))
    assert torch.equal(w1[:, keep], w0) and not w1[:, zero].any()
    with pytest.raises(ValueError):
        expand_policy_state(out)                          # already 31 actions


# ---- expert -------------------------------------------------------------------------------------------------------

def test_expert_depth5_reaches_struct():
    env, q = make()
    gold = frozenset({env.data.chunk_row["STRUCT/D"]})
    res = ex.expert_search(env, q, gold, GRLWeights(), depth=5)
    first_actions = trajectory_labels(env, q, res["final"])[0][0]     # the expert's first-action class
    assert res["depth"] == 5 and res["exact"]
    assert any(env.actions[a].op == "structure" for a in first_actions)
    assert ex.action_name(env.actions[28]) == "structure:1" and ex.action_name(env.actions[30]) == "structure:5"


def test_variant_data_loader(monkeypatch):
    import grl.env.struct_data as sd
    from grl.model import variants
    monkeypatch.setattr(sd, "load_struct_env_data", lambda *a, **k: "struct-data")
    monkeypatch.setattr("graphrag.env.data.load_env_data", lambda *a, **k: "base-data")
    assert variants.load_variant_data("struct") == "struct-data"
    assert variants.load_variant_data("full") == "base-data" and variants.load_variant_data("rerank") == "base-data"


def test_train_cli_struct():
    from tests.test_m5_expert import _cli
    cli = _cli()
    assert not cli._fixed_k_ok("struct") and cli._fixed_k_ok("full")


# ---- ruling R12: --set env.K_max reaches every env build, is recorded, and gates SFT ------------------------------

def _kmax_toy(monkeypatch):
    """Real-data loaders -> the toy (version = the pinned train.yaml graph, so the train.yaml env cfg accepts it)."""
    from dataclasses import replace
    from graphrag import config
    from tests.m4_toy import TOY_RANKS
    ver = config.load("train")["env"]["graph_version"]
    monkeypatch.setattr("grl.model.variants.load_variant_data", lambda *a, **k: replace(toy_data(), version=ver))
    monkeypatch.setattr("graphrag.retrieval.rankings.load_rankings", lambda *a, **k: TOY_RANKS)
    monkeypatch.setattr("grl.model.graph.build_graph", lambda *a, **k: toy_graph())


def _ns(**kw):
    import argparse
    base = {"set": [], "variant": "full", "cmd": "eval", "rankings": "unused", "split": "val"}
    return argparse.Namespace(**{**base, **kw})


def test_set_env_k_max_only():
    from tests.test_m5_expert import _cli
    cli = _cli()
    cli.check_set_keys(_ns(set=["env.K_max=5"]))
    for bad in ("env.T_max=3", "env.L_max=100", "env.K_max=0", "env.K_max=five", "env.K_max=true", "start.cap=3"):
        with pytest.raises(SystemExit):
            cli.check_set_keys(_ns(set=[bad]))


def test_k_max_reaches_setup_and_worker_env(tmp_path, monkeypatch):
    from tests.m4_toy import toy_pairs
    from tests.test_m5_expert import _cli
    _kmax_toy(monkeypatch)
    cli = _cli()
    a = _ns(set=["env.K_max=2", "run.vec_env.workers=1"], pairs=str(toy_pairs(tmp_path / "p.jsonl")))
    S = cli.setup(a)
    assert S["env"].limits.K_max == 2 and S["cfg"]["env"]["K_max"] == 2
    S["vec"].reset([("toy|q", 0)])                       # SerialVecEnv(**init()) -- the same call a ProcVecEnv worker
    assert S["vec"]._env["toy|q"].limits.K_max == 2     # makes (grl.train.rollout worker)
    assert cli.setup(_ns(set=["run.vec_env.workers=1"], pairs=a.pairs))["env"].limits.K_max == cli.legacy_k_max()


def _expert_file(tmp_path, k_max):
    from graphrag import config
    from tests.m4_toy import toy_pairs
    cfg = config.load("train")
    cfg["env"]["K_max"] = k_max
    out = tmp_path / f"expert-k{k_max}.jsonl"
    st = ex.run_expert(toy_pairs(tmp_path / "m3.jsonl"), "unused", "full", out, cfg, workers=1)
    return out, st


def test_expert_worker_env_records_k_max(tmp_path, monkeypatch):
    from graphrag import config
    from graphrag.dataset.finalize import read_pairs
    _kmax_toy(monkeypatch)
    env_cfg = {**config.load("train")["env"], "K_max": 2}
    assert ex.expert_context("unused", "full", {}, env_cfg=env_cfg)["env"].limits.K_max == 2   # the real worker init
    out, st = _expert_file(tmp_path, 2)                  # workers=1: parallel.run calls expert_context in-process
    assert st["K_max"] == 2
    assert all(p.provenance["meta"]["expert"]["K_max"] == 2 for p in read_pairs(out))


def test_sft_refuses_mismatched_k_max(tmp_path, monkeypatch, capsys):
    from grl.train.data import VariantMismatch, check_variant_file
    from tests.m4_toy import toy_pairs
    from tests.test_m5_expert import _cli
    _kmax_toy(monkeypatch)
    out, _ = _expert_file(tmp_path, 2)
    legacy = _cli().legacy_k_max()
    assert check_variant_file(out, "full", {}, True, None, 2, legacy) == "expert"
    with pytest.raises(VariantMismatch, match="K_max"):
        check_variant_file(out, "full", {}, True, None, 5, legacy)
    m3 = toy_pairs(tmp_path / "m3b.jsonl")              # no recorded K_max = built under the train.yaml K_max
    assert check_variant_file(m3, "full", {}, True, None, legacy, legacy) == "m3"
    with pytest.raises(VariantMismatch, match="K_max"):
        check_variant_file(m3, "full", {}, True, None, 5, legacy)
    cli = _cli()
    with pytest.raises(SystemExit):                      # the CLI gate, before any setup
        cli.main(["sft", "--pairs", str(out), "--rankings", "unused", "--run", str(tmp_path / "run"),
                  "--set", "env.K_max=5"])
    err = capsys.readouterr().err
    assert "labelled under env K_max ['2']" in err and "the run uses K_max 5" in err


# ---- part 2, Task 3 (R36): sft --variant struct --init-from <full SFT run> --------------------------------------

def test_transfer_refuses_single_layer_head():
    from grl.model.transfer import expand_policy_state
    g = toy_graph()
    one = GraphPolicy(g, {**SMALL_CFG, "policy": {"hidden": []}})       # head.net = one Linear: first == last
    with pytest.raises(ValueError, match="hidden"):
        expand_policy_state(one.state_dict())


def _sft_toy(tmp_path, monkeypatch, variant, pairs, run, init_from=None, captured=None):
    """setup() on the toy (variant-aware loaders: struct gets the STRUCT rows), SMALL_CFG shapes, 1 epoch of cmd_sft.
    captured: a list that receives the first-28 logits of the policy handed to run_sft, before any step."""
    from tests.m4_toy import TOY_RANKS
    from tests.test_m5_expert import _cli
    monkeypatch.setattr("grl.model.variants.load_variant_data", lambda v="full", *a, **k: _versioned(v))
    monkeypatch.setattr("graphrag.retrieval.rankings.load_rankings", lambda *a, **k: TOY_RANKS)
    monkeypatch.setattr("grl.model.graph.build_graph", lambda *a, **k: toy_graph())
    cli = _cli()
    a = _ns(cmd="sft", variant=variant, pairs=str(pairs), run=str(run), resume=False, keep_unreached=True,
            init_from=None if init_from is None else str(init_from),
            set=["env.K_max=2", "run.vec_env.workers=1", "run.device=cpu", "sft.max_epochs=1", "sft.loss=hard", "sft.batch_queries=2"])
    S = cli.setup(a)
    S["mcfg"] = {**SMALL_CFG, **policy_dims(S["env"])}
    if captured is not None:
        import grl.train.sft as sft_mod
        real = sft_mod.run_sft

        def spy(policy, *args, **kw):
            policy.eval()
            env3 = struct_env(("EXT/k",))                   # no structure candidate at reset
            q = env3.query(toy_rank())
            d = describe(env3, q, env3.reset(q), **DKW)
            assert d.cand[7].size == 0 and not d.mask[28:].any()
            captured.append(logits(policy, S["graph"], q, d)[..., :28])
            return real(policy, *args, **kw)
        monkeypatch.setattr(sft_mod, "run_sft", spy)
    try:
        return cli, cli.COMMANDS["sft"](a, S), S
    finally:
        S["vec"].close()


def _struct_labels(tmp_path, monkeypatch):
    from graphrag import config
    from tests.m4_toy import TOY_RANKS, toy_pairs
    monkeypatch.setattr("grl.model.variants.load_variant_data", lambda *a, **k: _versioned("struct"))
    monkeypatch.setattr("graphrag.retrieval.rankings.load_rankings", lambda *a, **k: TOY_RANKS)
    cfg = config.load("train")
    cfg["env"]["K_max"] = 2
    out = tmp_path / "struct.jsonl"
    ex.run_expert(toy_pairs(tmp_path / "m3s.jsonl"), "unused", "struct", out, cfg, workers=1, depth=3)
    return out


def test_sft_init_from_full_widens_and_keeps_old_logits(tmp_path, monkeypatch, capsys):
    import json

    from grl.hashing import sha256_file
    from grl.train.sft import load_reference
    _kmax_toy(monkeypatch)
    full_pairs, _ = _expert_file(tmp_path, 2)
    src = tmp_path / "full-sft"
    _, rc, S = _sft_toy(tmp_path, monkeypatch, "full", full_pairs, src)
    assert rc == 0 and (src / "pi_ref.json").is_file()
    assert json.loads((src / "manifest.json").read_text())["model"]["n_actions"] == 28
    struct_pairs = _struct_labels(tmp_path, monkeypatch)
    got = []
    run = tmp_path / "struct-sft"
    _, rc, S = _sft_toy(tmp_path, monkeypatch, "struct", struct_pairs, run, init_from=src, captured=got)
    assert rc == 0 and len(got) == 1
    old = load_reference(S["graph"], {**SMALL_CFG, "n_actions": 28, "n_types": 7}, src / "pi_ref.json")
    m3 = Env(toy_data(), TOY_CFG, TOY_START)
    q3 = m3.query(toy_rank())
    torch.testing.assert_close(got[0], logits(old, S["graph"], q3, describe(m3, q3, m3.reset(q3), **DKW)),
                               rtol=1e-5, atol=1e-6)
    man = json.loads((run / "manifest.json").read_text())
    w = src / json.loads((src / "pi_ref.json").read_text())["path"]
    assert man["init_from"] == {"run": str(src), "sha256": sha256_file(w), "variant": "full", "K_max": 2,
                                "K_max_recorded": True, "model_check": "recorded"}
    assert man["K_max"] == 2 and man["model"]["n_actions"] == 31


def test_sft_init_from_refuses_wrong_sources(tmp_path, monkeypatch):
    import json

    from tests.test_m5_expert import _cli
    cli = _cli()
    rr = tmp_path / "rr-sft"
    rr.mkdir()
    (rr / "manifest.json").write_text(json.dumps({"phase": "sft", "variant": "rerank", "K_max": 24}))
    (rr / "pi_ref.json").write_text(json.dumps({"path": "pi_ref.pt", "sha256": "0"}))
    with pytest.raises(SystemExit, match="rerank"):
        cli._init_from_source(rr)
    for argv in (["--variant", "struct", "--init-from", str(rr)],            # a rerank source: refused before setup
                 ["--variant", "full", "--init-from", str(rr)],              # --init-from is struct-only
                 ["--variant", "struct", "--init-from", str(tmp_path / "missing")]):
        with pytest.raises(SystemExit):
            cli.main(["sft", "--pairs", "p", "--rankings", "r", "--run", str(tmp_path / "run"), *argv])
    assert not (tmp_path / "run").exists()


def test_sft_init_from_checks_the_model(tmp_path, monkeypatch):
    """A recorded model cfg must equal the run's apart from n_actions / n_types; a legacy source (no recorded model,
    no K_max) is checked by shapes (strict load) and its K_max is the legacy train.yaml one."""
    import json

    from grl.train.sft import save_reference
    from tests.test_m5_expert import _cli
    cli = _cli()
    torch.manual_seed(0)
    src = tmp_path / "full"
    src.mkdir()
    save_reference(GraphPolicy(toy_graph(), SMALL_CFG), src)
    man = {"phase": "sft", "variant": "full"}
    (src / "manifest.json").write_text(json.dumps(man))
    g, mcfg = toy_graph(), {**SMALL_CFG, "n_actions": 31, "n_types": 8}
    sd, info = cli._init_from_state(cli._init_from_source(src), g, mcfg)
    assert info["model_check"] == "shapes" and info["K_max"] == cli.legacy_k_max() and not info["K_max_recorded"]
    GraphPolicy(g, mcfg).load_state_dict(sd, strict=True)
    with pytest.raises(SystemExit, match="model"):                         # legacy: shapes differ
        cli._init_from_state(cli._init_from_source(src), g, {**mcfg, "policy": {"hidden": [64, 16]}})
    (src / "manifest.json").write_text(json.dumps({**man, "K_max": 5, "model": {**SMALL_CFG, "n_actions": 28,
                                                                                    "n_types": 7}}))
    assert cli._init_from_state(cli._init_from_source(src), g, mcfg)[1]["model_check"] == "recorded"
    with pytest.raises(SystemExit, match="dropout"):                       # recorded: a shape-free setting differs
        cli._init_from_state(cli._init_from_source(src), g, {**mcfg, "dropout": 0.5})
    (src / "manifest.json").write_text(json.dumps({**man, "model": {**SMALL_CFG, "n_actions": 31, "n_types": 8}}))
    with pytest.raises(SystemExit, match="28"):
        cli._init_from_state(cli._init_from_source(src), g, mcfg)
