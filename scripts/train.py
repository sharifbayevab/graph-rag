"""M4/M5 training entry (plan Tasks 9–10, 13–14).
    python scripts/train.py sft  --pairs P --rankings R --run runs/<id> [--variant full] [--resume] [--set k=v ...]
    python scripts/train.py grpo --pairs P --rankings R --run runs/<id> --sft runs/<sft-id> [--resume] [--set ...]
    python scripts/train.py eval --pairs P --rankings R --ckpt F [--split val]
    python scripts/train.py baselines --pairs P --rankings R [--split val] [--curves] [--out F.json]
    python scripts/train.py expert --variant V --pairs P --rankings R --out F.jsonl [--depth 3] [--beam N]
                                   [--keep-zero] [--workers W] [--probe-n N]
    python scripts/train.py selector --variant V --pairs P --rankings R [--label expert|switch|best1]
                                     [--model logreg|tree] [--out F.json]
M5 pre-RL: `expert` searches the best trajectory of every train / val pair IN THE VARIANT ENV (grl.train.expert:
exact enumeration up to --depth retrieval steps + STOP, grl reward, ties shorter then lower ids) and writes §8.1 rows
with provenance.meta.variant; `sft --variant V` (also adaptive / fused) is accepted exactly when every train / val
row of --pairs was labelled in V with the run's dynamic_k kwargs AND the run's reward (reward.chunks_beta / weights;
grl.train.data.check_variant), and M3 fixed-k labels stay refused under adaptive / fused. Expert rows with score 0,
an immediate STOP or C* missed are dropped from the SFT train set (counts in the manifest; --keep-unreached keeps
them). train.yaml expert.max_states caps the exact search per query (then a beam of expert.cap_beam, flagged);
--probe-n N measures prune vs --keep-zero and the runtime on the first N train queries. `selector`: the gold-free one-action selector baseline
(grl.train.selector), trained on train, reported on train / val (JSON on stdout, --out also writes it).
--variant (grl.model.variants.VARIANTS) applies to eval / baselines / grpo. "hybrid" (STOP masked until a lexical, a
vector and a graph step were taken, or that became impossible): GRPO only, `grpo --variant hybrid --sft <full SFT
run>` -- θ₀ = π_ref = the unconstrained SFT policy, rollouts / log-probs / exact KL over the hybrid-masked
distribution. SFT under the hybrid mask is refused (the expert trajectories may STOP before using all three
families, so their labels are not valid under the mask). baselines under a `require` variant skip the oracles.
"adaptive" / "hybrid_adaptive" (M5 dynamic k: the lexical / vector slots are relative score thresholds p, see
grl.model.variants.AdaptiveEnv): GRPO only, from the unconstrained SFT π_ref as for hybrid (the oracle labels use
fixed k); train.yaml dynamic_k.normalise (default true) / norm_top_n (50): the agent thresholds min-max normalised
scores. baselines also report the dynamic-k baselines (rel:<kind>@p normalised, rel_raw:<kind>@p raw, elbow:<kind>,
agree@N, the fused rel:cc<w>@p and the ORACLE oracle_switch:dense|cc<w>@p, flagged oracle: true; every row has
recall@300tok / recall@600tok). "fused_adaptive": adaptive with the lexical slots on the convex BM25 + dense fusion
(train.yaml dynamic_k.fuse_w = dense weight, default 0.7); GRPO from the unconstrained π_ref like adaptive, or SFT on
variant-expert labels (see `expert` above). --curves prints only
the recall curves (bm25 / dense / rrf @ k = 1..K_max) and the dynamic-k baselines (JSON on stdout, a table on
stderr, --out also writes the JSON). --set reward.chunks_beta=2 makes R_chunks an F2 (grl.train.fbeta; default 1 =
the spec F1); the manifest records it.
"rerank" (M5 RL reranker, grl.model.rerank; user-approved spec deviation): its own action space -- STOP or select
candidate j of the fused top-N (train.yaml rerank.n, dynamic_k.fuse_w / norm_top_n) -- and a pointer policy
(RerankPolicy) over graph-aware per-candidate features; rerank.use_ce=true (+ rerank.ce_cache = scripts/ce_scores.py
output) adds the cross-encoder score as a feature. Sequence: `expert --variant rerank` (labels; ce-independent) ->
`sft --variant rerank` -> `eval` / `grpo --variant rerank --sft <that run>`; `baselines --variant rerank` reports
fused@ / ce@ / ce+fused:<a>@ selections and the ORACLE oracle@N (exact expert inside the candidates, cand_recall).
P: §8.1 pairs with expert trajectories (pilot: data/m5/pilot/<sha>/trajectories.jsonl; final: Task 14 snapshot);
R: the matching U9 rankings latest.json. Only train (fit) and val (selection) are read -- test / Q-real are refused.
"rank" (M5 pilot 6, grl.model.rank / grl.train.rank): the same pointer policy as a top-k RANKER -- picks without
replacement from the fused top-N pool (rerank.n) until rank.k_max are placed, no STOP; the ranked list is scored by
hit@k / recall@k (k = 1, 3, 5, 10, 15, 20), MRR and nDCG@10 / @20, the GRPO reward / val selection metric is
rank.reward (ndcg@20 | ndcg@10 | mix). `sft --variant rank` builds listwise labels from C* (no expert file; --pairs =
the snapshot trajectories.jsonl), `baselines --variant rank` reports dense / bm25 / rrf / fused / ce / ce+fused:<a>
rankings, the ORACLE pool_oracle and the pool ceilings for N = 20 / 50 / 100.
"struct" (M7, docs/m7 design §3.3; grl.model.variants.StructEnv): the 28 M3 actions + `structure` m ∈ {1, 3, 5}
(31) on EnvData with the STRUCT rows (grl.env.struct_data.load_struct_env_data, also in the rollout / expert workers);
the GraphPolicy gets 31 actions / 8 candidate types (grl.model.policy.policy_dims). `expert --variant struct
--depth 5` labels, `sft --variant struct` accepts only struct-expert labels (M3 rows refused), then eval / grpo /
baselines as for full. --set struct.amend_op=true (Task 6d-b) adds the `amendments` operator (34 actions / 9 types).
`sft --variant struct --init-from <full SFT run>` (R36) starts from that run's π_ref widened
28 -> 31 actions by grl.model.transfer.expand_policy_state (old logits unchanged; with amend_op 28 -> 31 -> 34, and a
31-action `struct` SFT run is also accepted as the source, widened 31 -> 34); the source must be a `full` SFT
run whose recorded model cfg equals this run's apart from n_actions / n_types (legacy runs: shapes, strict load), its
K_max may differ (recorded); the manifest gets init_from {run, sha256, variant, K_max} and every SFT manifest `model`.
Pilot 5 final report only: `eval` / `baselines --split test --pairs <ship>/test_eval/test_eval.jsonl --rankings
<ship>/test_eval/rankings/latest.json` evaluate on the strict consensus test (grl.train.data.load_test_items; no
train / val rows are loaded then, and every training loader refuses that file). --per-query F.jsonl (eval /
baselines) also writes one row per query (actions / context / metrics; baselines: per reference name).
Task GRPO-X (all default-off): `rft --ckpt F --out R.jsonl [--rft-g 32] [--rft-margin 0.05]` samples G episodes per
TRAIN query from the checkpoint and records the best one (grl.train.rft.sample_best; keep = R_best > R_greedy +
margin); `sft --init-ckpt F` starts from a checkpoint of the same model (strict load, sha256 in the manifest),
`sft --rft R.jsonl` adds the kept rows (replayed, grl.train.rft.rft_examples) to the SFT train examples and
`sft --expert-patterns P1,P2` keeps only the expert train rows of those patterns (val unchanged); `grpo` with
grpo.replay_lambda > 0 replays the expert train labels (after the SFT train filter) of grpo.replay_patterns."""
import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from grl.threads import pin_blas  # noqa: E402  -- stdlib only, before numpy / torch

pin_blas()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import yaml  # noqa: E402

from graphrag import config  # noqa: E402
from grl.train.cli_checks import baseline_names, check_grpo_k_max  # noqa: E402  -- light (json / pathlib)


def _set(cfg: dict, kv: str) -> None:
    k, v = kv.split("=", 1)
    node, *path = cfg, *k.split(".")
    for p in path[:-1]:
        node = node[p]
    if path[-1] not in node:
        from grl.train.grpo import GRPO_OPTIONAL           # pilot 5b: optional, default-off GRPO keys
        from grl.train.sft import SFT_OPTIONAL             # pilot 8: optional, default-off SFT keys
        if not ((path[:-1] == ["grpo"] and path[-1] in GRPO_OPTIONAL)
                or (path[:-1] == ["sft"] and path[-1] in SFT_OPTIONAL)):
            raise KeyError(f"--set {k}: no such config key")
    node[path[-1]] = yaml.safe_load(v)


ENV_SET_KEY = "env.K_max"                                 # the only env.* --set key (ruling R12)


def legacy_k_max() -> int:
    """K_max of the unmodified train.yaml: what M3 / pre-R12 expert labels (no recorded K_max) were built under."""
    return int(config.load("train")["env"]["K_max"])


def check_set_keys(a) -> None:
    """Refuse --set keys that would silently do nothing: env.* except env.K_max (ruling R12: the run's `env` cfg is
    passed to every env build -- setup, rollout and expert workers -- but only K_max is a supported override; the
    labels record it and sft refuses a mismatch), start.* (data.yaml, not train.yaml), dynamic_k.normalise outside an
    adaptive variant, dynamic_k.norm_top_n outside an adaptive variant except for baselines (rel:* uses it)."""
    from grl.model.variants import is_adaptive, is_fused, is_rerank
    for kv in a.set:
        k = kv.split("=", 1)[0].strip()
        top = k.split(".", 1)[0]
        if k == ENV_SET_KEY:
            v = yaml.safe_load(kv.split("=", 1)[1])
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise SystemExit(f"--set {k}: a positive integer is required, got {v!r}")
            continue
        if top in ("env", "start"):
            src = "train.yaml env" if top == "env" else "data.yaml start"
            raise SystemExit(f"--set {k}: not applied -- the env reads {src} directly (only --set {ENV_SET_KEY} is "
                             "passed on); edit the config file instead")
        rr = is_rerank(a.variant)
        from grl.model.variants import is_rank
        from grl.model.variants import is_struct
        if top == "struct" and not is_struct(a.variant):
            raise SystemExit(f"--set {k}: only the struct variant reads it (variant {a.variant!r})")
        if top == "rank" and not is_rank(a.variant):
            raise SystemExit(f"--set {k}: only the rank variant reads it (variant {a.variant!r})")
        if top == "rerank" and not rr:
            raise SystemExit(f"--set {k}: only the rerank variant reads it (variant {a.variant!r})")
        if k == "dynamic_k.normalise" and not is_adaptive(a.variant):
            raise SystemExit(f"--set {k}: only the adaptive variants read it (variant {a.variant!r})")
        if k == "dynamic_k.fuse_w" and not is_fused(a.variant) and not rr:
            raise SystemExit(f"--set {k}: only the fused variants read it (variant {a.variant!r}; the rel:cc<w> "
                             "baselines use their own w grid)")
        if k == "dynamic_k.norm_top_n" and not is_adaptive(a.variant) and not rr and a.cmd != "baselines":
            raise SystemExit(f"--set {k}: only the adaptive variants and baselines read it (variant {a.variant!r}, "
                             f"{a.cmd})")


def setup(a) -> dict:
    from graphrag.retrieval.rankings import load_rankings
    from grl.model.graph import build_graph
    from grl.model.variants import data_kw, env_spec, is_rerank, load_variant_data, variant_env_kw
    from grl.train.data import load_items
    from grl.train.log import pick_device, set_determinism
    from grl.train.rollout import ProcVecEnv, SerialVecEnv, real_init
    cfg = config.load("train")
    for kv in a.set:
        _set(cfg, kv)
    det = set_determinism(int(cfg["grpo" if a.cmd == "grpo" else "sft"]["seed"]), cfg["run"]["deterministic"])
    device = pick_device(cfg["run"]["device"])
    mcfg = _model_cfg(cfg, a.variant)
    akw = variant_env_kw(a.variant, cfg, observe=True)           # AdaptiveEnv dynamic_k / RerankEnv rerank / struct
    data = load_variant_data(a.variant, **data_kw(akw))     # struct (M7): + the STRUCT rows (6c: + article rows)
    ranks = load_rankings(a.rankings, graph_version=data.version)
    dkw = {"preview": int(mcfg["state"]["preview"]), "cand_top": int(mcfg["state"]["cand_top"])}
    ve = cfg["run"]["vec_env"]
    env_cfg = cfg["env"]                                    # incl. --set env.K_max (R12), for every env build
    init = partial(real_init, a.rankings, a.variant, dkw, akw, env_cfg=env_cfg)
    vec = (None if a.cmd in ("baselines", "selector") else    # these step the env directly: no rollout workers
           ProcVecEnv(init, int(ve["workers"]), ve["start_method"], float(ve["timeout_s"]))
           if int(ve["workers"]) > 1 else SerialVecEnv(**init()))
    cls, kw = env_spec(a.variant)
    env = cls(data, env_cfg, None, **kw, **akw)
    if not is_rerank(a.variant):                           # GraphPolicy shapes from the env: 28 / 7, struct 31 / 8
        from grl.model.policy import policy_dims
        mcfg.update(policy_dims(env))
    if getattr(a, "split", None) == "test":                                     # pilot 5 final report: the test-eval file only
        from grl.train.data import load_test_items
        tr, va, te = [], [], load_test_items(a.pairs, data)
    else:
        from grl.model.variants import is_rank
        load = load_items
        if is_rank(a.variant):                                # pilot 6: trajectory-free rows allowed (ship/pairs.jsonl)
            from grl.train.data import load_rank_items as load
        tr, va, te = load(a.pairs, data, ("train",)), load(a.pairs, data, ("val",)), None
    return {"cfg": cfg, "mcfg": mcfg, "device": device, "data": data, "ranks": ranks, "dkw": dkw, "det": det,
            "graph": None if a.cmd == "selector" else build_graph(mcfg["graph"]).to(device), "vec": vec, "env": env,
            "train": tr, "val": va, "test": te, "w": _reward(a, cfg)}


def _reward(a, cfg):
    """grl reward weights; the rank variant (pilot 6): grl.train.rank.RankReward (rank.reward, rank.k_max)."""
    from grl.model.variants import is_rank
    if is_rank(a.variant):
        from grl.train.rank import rank_reward
        return rank_reward(cfg)
    from grl.train.fbeta import grl_weights
    return grl_weights(cfg)


def _evaluator(S, expert=None):
    from grl.train.evaluate import make_evaluator
    return make_evaluator(S["graph"], S["vec"], S["ranks"], {"train": S["train"], "val": S["val"]}, S["w"],
                          S["env"].limits.T_max, S["device"], S["env"].actions, expert)


RERANK_SETTINGS = "rerank_settings.json"


def _rerank_settings(a, S) -> dict | None:
    """grl.model.rerank.rerank_settings of this run's env (None outside the rerank variant)."""
    from grl.model.variants import is_rerank
    if not is_rerank(a.variant):
        return None
    from grl.model.rerank import rerank_settings
    return rerank_settings(S["env"])


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _settings_near(ckpt) -> dict | None:
    """rerank_settings.json of the run a checkpoint belongs to: beside it (pi_ref.pt) or one level up
    (checkpoints/<name>.pt)."""
    ckpt = Path(ckpt)
    for d in (ckpt.parent, ckpt.parent.parent):
        f = d / RERANK_SETTINGS
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


def _manifest(a, S, phase: str, extra_inputs: dict | None = None, info: dict | None = None) -> dict:
    from grl.train.fbeta import chunks_beta, chunks_label
    from grl.train.log import run_manifest
    rs = _rerank_settings(a, S)
    if rs is not None:
        ce = (S["cfg"].get("rerank") or {}).get("ce_cache")
        extra_inputs = {**(extra_inputs or {}), **({"ce_cache": str(config.path(ce))} if ce else {})}
        info = {**(info or {}), "rerank_settings": rs}
        if getattr(S["w"], "is_rank", False):
            info["rank"] = dict(S["cfg"].get("rank") or {})
    return run_manifest(S["cfg"], {"pairs": a.pairs, "rankings": a.rankings, **(extra_inputs or {})},
                        {"phase": phase, "variant": a.variant, "determinism": S["det"], "device": str(S["device"]),
                         "n_train": len(S["train"]), "n_val": len(S["val"]),
                         "reward_chunks": chunks_label(S["w"]), "chunks_beta": chunks_beta(S["w"]),
                         "dynamic_k": S["cfg"].get("dynamic_k"), "K_max": int(S["env"].limits.K_max),
                         **(info or {})})


def _required(variant: str) -> tuple:
    from grl.model.variants import VARIANTS
    return tuple(VARIANTS[variant].get("require", ()))


def _env_kw(a, cfg) -> dict:
    """The label key's env kwargs (rerank: the candidate list only -- the ce feature never changes the labels)."""
    from grl.model.variants import variant_env_kw
    return variant_env_kw(a.variant, cfg)


def _fixed_k_ok(variant: str) -> bool:
    """M3 fixed-k oracle labels are SFT targets only where the variant's action space and search effects are M3's:
    not adaptive / fused / rerank, and not struct (its masks / classes have 31 actions; label with `expert` first)."""
    from grl.model.variants import is_adaptive, is_rerank, is_struct
    return not is_adaptive(variant) and not is_rerank(variant) and not is_struct(variant)


def _model_cfg(cfg: dict, variant: str) -> dict:
    from grl.model.variants import is_rerank, model_config
    m = model_config(cfg["model"], variant)
    if is_rerank(variant):
        m["rerank_head"] = dict((cfg.get("rerank") or {}).get("head") or {})
    return m


def _policy_cls(variant: str):
    from grl.model.variants import is_rerank
    if is_rerank(variant):
        from grl.model.rerank import RerankPolicy
        return RerankPolicy
    from grl.model.policy import GraphPolicy
    return GraphPolicy


INIT_SHAPE_KEYS = ("n_actions", "n_types")                # the only model cfg keys a 28 -> 31 transfer may change
INIT_SOURCE_VARIANTS = ("full", "struct")                 # R36 (28 / 7) and Task 6d-b (31 / 8 -> 34 / 9)


def _init_from_source(run_dir) -> dict:
    """R36: the source of `sft --variant struct --init-from <run_dir>` -- a finished `full` SFT run (28 actions /
    7 types): manifest.json (phase sft, variant full) + pi_ref.json whose weights still match its sha256. Its K_max
    is the recorded one, else (pre-R12 runs) the legacy train.yaml K_max; it may differ from the run's (the weights'
    shapes do not depend on it): both are recorded. SystemExit with the reason otherwise."""
    from grl.hashing import sha256_file
    d = Path(run_dir)
    man_f, ref_f = d / "manifest.json", d / "pi_ref.json"
    for f in (man_f, ref_f):
        if not f.is_file():
            raise SystemExit(f"sft --init-from {d}: no {f.name} -- not a finished SFT run")
    man = json.loads(man_f.read_text(encoding="utf-8"))
    if man.get("phase") != "sft" or man.get("variant") not in INIT_SOURCE_VARIANTS:
        raise SystemExit(f"sft --init-from {d}: phase {man.get('phase')!r}, variant {man.get('variant')!r} -- only a "
                         "`full` SFT run (28 actions, 7 candidate types; R36) or a `struct` SFT run (31 / 8, Task 6d-b: "
                         "widened to the 34-action amend_op policy) can be widened")
    meta = json.loads(ref_f.read_text(encoding="utf-8"))
    w = d / meta["path"]
    sha = sha256_file(w) if w.is_file() else None
    if sha != meta["sha256"]:
        raise SystemExit(f"sft --init-from {d}: {w.name} is missing or its sha256 differs from pi_ref.json")
    k = man.get("K_max")
    return {"run": str(d), "sha256": sha, "variant": man["variant"],
            "K_max": int(k) if k is not None else legacy_k_max(),
            "K_max_recorded": k is not None, "weights": str(w), "model": man.get("model")}


def _init_from_state(src: dict, graph, mcfg: dict) -> tuple[dict, dict]:
    """The widened state dict for GraphPolicy(graph, mcfg) (grl.model.transfer.widen_policy_state: 28 -> 31, and
    Task 6d-b 31 -> 34 / 28 -> 34 for struct.amend_op) and the manifest record {run, sha256, variant, K_max,
    K_max_recorded, model_check}. A source that recorded its model cfg must equal the run's apart from n_actions /
    n_types (and be 28 / 7 for `full`, 31 / 8 for `struct`); a legacy source (no recorded cfg) is checked by shapes:
    the widened dict must load strictly. SystemExit otherwise."""
    from grl.model.policy import GraphPolicy
    from grl.model.transfer import widen_policy_state
    rec, d = src.get("model"), src["run"]
    want = {"full": (28, 7), "struct": (31, 8)}[src.get("variant", "full")]
    if rec is not None:
        if (rec.get("n_actions", 28), rec.get("n_types", 7)) != want:
            raise SystemExit(f"sft --init-from {d}: the source model has {rec.get('n_actions')} actions / "
                             f"{rec.get('n_types')} types, expected {want[0]} / {want[1]}")
        a, b = ({k: v for k, v in json.loads(json.dumps(m)).items() if k not in INIT_SHAPE_KEYS} for m in (rec, mcfg))
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        if diff:
            raise SystemExit(f"sft --init-from {d}: the source model cfg differs from this run's in {diff} "
                             "(only n_actions / n_types may change)")
    sd = torch.load(src["weights"], map_location="cpu", weights_only=True)
    try:
        out = widen_policy_state(sd, want, (int(mcfg.get("n_actions", 28)), int(mcfg.get("n_types", 7))))
        GraphPolicy(graph, mcfg).load_state_dict(out, strict=True)
    except (ValueError, RuntimeError, KeyError) as e:
        raise SystemExit(f"sft --init-from {d}: the source weights do not fit this run's model: {e}") from None
    info = {k: src[k] for k in ("run", "sha256", "variant", "K_max", "K_max_recorded")}
    return out, {**info, "model_check": "shapes" if rec is None else "recorded"}


def cmd_sft(a, S) -> int:
    from grl.train.data import check_variant_items, replay_examples, sft_train_filter
    from grl.train.expert import reward_spec
    from grl.train.log import Checkpoints, RunLogger, plot_run
    from grl.train.sft import run_sft
    cfg = S["cfg"]
    from grl.model.variants import is_rank
    if is_rank(a.variant):
        return _cmd_sft_rank(a, S)
    labels = check_variant_items(S["train"] + S["val"], a.variant, _env_kw(a, cfg), _fixed_k_ok(a.variant),
                                 reward_spec(S["w"]), int(S["env"].limits.K_max), legacy_k_max())
    keep = bool(getattr(a, "keep_unreached", False))
    S["train"], drop = sft_train_filter(S["train"], keep)              # never silent: counts in the manifest
    _, val_counts = sft_train_filter(S["val"], True)                   # val is reported, never filtered
    info = {"sft_labels": labels, "sft_train_filter": drop, "sft_val_counts": val_counts, "model": S["mcfg"]}
    init_sd = None
    if getattr(a, "init_from", None):                       # R36: checked before the run dir is created
        init_sd, info["init_from"] = _init_from_state(_init_from_source(a.init_from), S["graph"], S["mcfg"])
    from grl.hashing import sha256_file
    if getattr(a, "init_ckpt", None):
        info["init_ckpt"] = {"path": a.init_ckpt, "sha256": sha256_file(a.init_ckpt)}
    if getattr(a, "rft", None):
        info["rft"] = {"path": a.rft, "sha256": sha256_file(a.rft)}
    if getattr(a, "expert_patterns", None):
        info["expert_patterns"] = a.expert_patterns
    if a.resume:                                            # the checkpoint carries the weights; the source must agree
        mf = Path(a.run) / "manifest.json"
        was = json.loads(mf.read_text(encoding="utf-8")).get("init_from") if mf.is_file() else None
        if (was or {}).get("sha256") != (info.get("init_from") or {}).get("sha256"):
            raise SystemExit(f"sft --resume {a.run}: --init-from differs from the run's ({(was or {}).get('run')})")
    print(json.dumps({"variant": a.variant, **info}), file=sys.stderr)
    logger = RunLogger(a.run, _manifest(a, S, "sft", info=info), resume=a.resume)
    rs = _rerank_settings(a, S)
    if rs is not None:
        _write_json(Path(a.run) / RERANK_SETTINGS, rs)
    ckpt = Checkpoints(Path(a.run) / "checkpoints", int(cfg["run"]["keep_top"]))
    tr_ex, tr_exp = replay_examples(S["env"], S["ranks"], S["train"], S["dkw"])
    pats = getattr(a, "expert_patterns", None)
    sft_items = S["train"] if not pats else [i for i in S["train"] if i.pattern in set(pats.split(","))]
    if pats:                                                # Task GRPO-X E1: expert rows of these patterns only
        keep_q = {i.qkey for i in sft_items}
        tr_ex = [e for e in tr_ex if e.qkey in keep_q]
    if getattr(a, "rft", None):                             # Task GRPO-X E1: the kept own samples
        from grl.train.rft import rft_examples
        rows = [json.loads(x) for x in Path(a.rft).read_text(encoding="utf-8").splitlines() if x.strip()]
        rex = rft_examples(S["env"], S["ranks"], rows, S["dkw"])
        tr_ex = tr_ex + rex
        print(json.dumps({"rft_rows": sum(1 for r in rows if r.get("keep")), "rft_steps": len(rex),
                          "expert_train_queries": len(sft_items), "expert_steps": len(tr_ex) - len(rex)}),
              file=sys.stderr)
    va_ex, va_exp = replay_examples(S["env"], S["ranks"], S["val"], S["dkw"])
    torch.manual_seed(int(cfg["sft"]["seed"]))
    policy = _policy_cls(a.variant)(S["graph"], S["mcfg"])
    if init_sd is not None:                                 # same seed / draws as a scratch run, then the weights
        policy.load_state_dict(init_sd, strict=True)
    if getattr(a, "init_ckpt", None):                       # Task GRPO-X E1: fine-tune from a checkpoint (strict)
        st = torch.load(a.init_ckpt, map_location="cpu", weights_only=True)
        policy.load_state_dict(st["model"] if "model" in st else st, strict=True)
    policy = policy.to(S["device"])
    res = run_sft(policy, S["graph"], S["ranks"], tr_ex, va_ex, cfg["sft"], logger, ckpt, S["device"],
                  _evaluator(S, {"train": tr_exp, "val": va_exp}), resume=a.resume)
    logger.export_csv()
    plot_run(a.run)
    logger.close()
    if rs is not None:                                        # π_ref carries the settings it was trained under
        pr = Path(a.run) / "pi_ref.json"
        _write_json(pr, {**json.loads(pr.read_text(encoding="utf-8")), "rerank": rs})
    print(json.dumps({"best": str(res.best_path), "pi_ref": res.ref, "epochs": res.epochs,
                      "best_val_reward": res.best_metric}, ensure_ascii=False))
    return 0


def _cmd_sft_rank(a, S) -> int:
    """Pilot 6: listwise SFT of the rank variant straight from C* (grl.train.rank.rank_sft_examples; no expert file),
    best epoch by the val ranking reward (rank.reward), early stop by the val loss."""
    from grl.train.log import Checkpoints, RunLogger, plot_run
    from grl.train.rank import rank_sft_examples
    from grl.train.sft import run_sft
    cfg = S["cfg"]
    m = int((cfg.get("rank") or {}).get("sft_neg_prefix", 0))
    tr_ex, tr_cnt = rank_sft_examples(S["env"], S["ranks"], S["train"], S["dkw"], m)
    va_ex, va_cnt = rank_sft_examples(S["env"], S["ranks"], S["val"], S["dkw"], m)
    info = {"rank_sft_train": tr_cnt, "rank_sft_val": va_cnt, "rank_sft_neg_prefix": m}
    print(json.dumps({"variant": a.variant, **info}), file=sys.stderr)
    logger = RunLogger(a.run, _manifest(a, S, "sft", info=info), resume=a.resume)
    rs = _rerank_settings(a, S)
    _write_json(Path(a.run) / RERANK_SETTINGS, rs)
    ckpt = Checkpoints(Path(a.run) / "checkpoints", int(cfg["run"]["keep_top"]))
    torch.manual_seed(int(cfg["sft"]["seed"]))
    policy = _policy_cls(a.variant)(S["graph"], S["mcfg"]).to(S["device"])
    res = run_sft(policy, S["graph"], S["ranks"], tr_ex, va_ex, cfg["sft"], logger, ckpt, S["device"], _evaluator(S),
                  resume=a.resume)
    logger.export_csv()
    plot_run(a.run)
    logger.close()
    pr = Path(a.run) / "pi_ref.json"
    _write_json(pr, {**json.loads(pr.read_text(encoding="utf-8")), "rerank": rs})
    print(json.dumps({"best": str(res.best_path), "pi_ref": res.ref, "epochs": res.epochs,
                      "best_val_reward": res.best_metric}, ensure_ascii=False))
    return 0


def _write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def cmd_eval(a, S) -> int:
    from grl.train.evaluate import evaluate_policy, per_query
    rs = _rerank_settings(a, S)
    if rs is not None:
        from grl.model.rerank import check_rerank_settings
        try:
            check_rerank_settings(_settings_near(a.ckpt), rs, f"eval {a.ckpt}")
        except ValueError as e:
            raise SystemExit(str(e)) from None
    policy = _policy_cls(a.variant)(S["graph"], S["mcfg"]).to(S["device"])
    st = torch.load(a.ckpt, map_location="cpu", weights_only=True)   # CUDA-saved files load anywhere
    policy.load_state_dict(st["model"] if "model" in st else st)
    s, eps, finals = evaluate_policy(policy, S["graph"], S["vec"], S["ranks"], S[a.split], S["w"],
                                     S["env"].limits.T_max, S["device"], S["env"].actions)
    s["split"] = a.split
    s["K_max"] = int(S["env"].limits.K_max)
    if getattr(a, "per_query", None):
        if getattr(S["w"], "is_rank", False):
            from grl.train.rank import rank_per_query
            rows = rank_per_query(S[a.split], eps, finals)
            tg = {i.qkey: i.gold for i in S[a.split]}
            for r in rows:
                r["reward"] = S["w"].value(r["ranking"], tg[r["qkey"]])
        else:
            rows = per_query(S["env"], S["ranks"], S[a.split], eps, finals, S["w"], S["env"].actions)
        _write_jsonl(a.per_query, rows)
        s["per_query_reward_mean"] = round(sum(r["reward"] for r in rows) / len(rows), 6) if rows else 0.0
    print(json.dumps(s, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_rft(a, S) -> int:
    """Task GRPO-X E1: best-of-G own samples per train query (grl.train.rft.sample_best) -> --out JSONL + a summary."""
    from grl.train.rft import sample_best
    policy = _policy_cls(a.variant)(S["graph"], S["mcfg"]).to(S["device"])
    st = torch.load(a.ckpt, map_location="cpu", weights_only=True)
    policy.load_state_dict(st["model"] if "model" in st else st)
    gen = torch.Generator().manual_seed(int(S["cfg"]["grpo"]["seed"]))
    rows = sample_best(policy, S["graph"], S["vec"], S["ranks"], S["train"], int(a.rft_g), gen, S["w"], S["device"],
                       float(a.rft_margin))
    _write_jsonl(a.out, rows)
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by["all"].append(r)
        by[r["pattern"] if r["pattern"] in ("count", "version") else "old"].append(r)
    summ = {g: {"n": len(v), "kept": sum(r["keep"] for r in v),
                **{k: round(sum(r[k] for r in v) / len(v), 6) for k in ("r_greedy", "r_mean", "r_best")}}
            for g, v in sorted(by.items())}
    print(json.dumps({"ckpt": a.ckpt, "G": int(a.rft_g), "margin": float(a.rft_margin), "out": a.out,
                      "summary": summ}, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_baselines(a, S) -> int:
    from grl.model.variants import is_rerank
    from grl.train.baselines import DYNAMIC_NAMES, curves_table, recall_curves, run_references
    from grl.train.fbeta import chunks_label
    ntop = int((S["cfg"].get("dynamic_k") or {}).get("norm_top_n", 50))
    pq = {} if getattr(a, "per_query", None) else None
    from grl.model.variants import is_rank
    if is_rank(a.variant):                                    # pilot 6: ranking baselines + pool ceilings
        from grl.train.rank import RANK_BASELINE_NAMES, run_rank_references
        if a.curves:
            raise SystemExit("baselines --curves is not defined for the rank variant")
        out = {"split": a.split, "rank_reward": S["w"].kind,
               **run_rank_references(RANK_BASELINE_NAMES, S["env"], S["ranks"], S[a.split], rows_out=pq)}
    elif is_rerank(a.variant):                                  # grl.train.rerank_baselines (fused / ce / oracle@N)
        from grl.train.rerank_baselines import RERANK_BASELINE_NAMES, run_rerank_references
        if a.curves:
            raise SystemExit("baselines --curves is not defined for the rerank variant")
        out = {"split": a.split, "reward_chunks": chunks_label(S["w"]),
               **run_rerank_references(RERANK_BASELINE_NAMES, S["env"], S["ranks"], S[a.split], S["w"],
                                       S["cfg"].get("expert"), rows_out=pq)}
    elif a.curves:
        out = {"split": a.split, "reward_chunks": chunks_label(S["w"]), "norm_top_n": ntop,
               "curves": recall_curves(S["env"], S["ranks"], S[a.split], S["w"]),
               "dynamic": run_references(DYNAMIC_NAMES, S["env"], S["ranks"], S[a.split], S["w"], ntop, rows_out=pq)}
        print(curves_table(out["curves"]), file=sys.stderr)
    else:
        # §9.3 + M5 + dynamic k; the oracles are skipped under a `require` mask (they STOP freely) and on struct
        # (BeamSearch's lookahead ignores `structure`: no upper_bound row), with the reason in the output
        names, skipped = baseline_names(a.variant, S["cfg"]["env"]["k_grid"])
        out = run_references(names, S["env"], S["ranks"], S[a.split], S["w"], ntop, rows_out=pq)
        if skipped:
            out["skipped"] = skipped
        bound = chunks_label(S["w"]) == "F1"           # the oracles maximise the spec F1 reward, not F-beta
        for n in ("greedy_oracle", "beam_oracle"):
            if n in out:
                out[n]["upper_bound"] = bound
                if not bound:
                    out[n]["note"] = f"oracle optimised the spec F1 reward; metrics scored with {chunks_label(S['w'])}"
    out["K_max"] = int(S["env"].limits.K_max)
    if pq is not None:
        _write_jsonl(a.per_query, [{"name": n, **r} for n in sorted(pq) for r in pq[n]])
    text = json.dumps(out, ensure_ascii=False, sort_keys=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def cmd_grpo(a, S) -> int:
    from grl.train.grpo import run_grpo
    from grl.train.log import Checkpoints, RunLogger, plot_run
    from grl.train.sft import load_reference
    if not a.sft:
        raise SystemExit("--sft runs/<sft-id> is required (π_ref and the initial policy)")
    cfg, sft = S["cfg"], Path(a.sft)
    meta = json.loads((sft / "pi_ref.json").read_text(encoding="utf-8"))
    check_grpo_k_max(sft, int(S["env"].limits.K_max), legacy_k_max())      # R12: π_ref was built under this K_max
    rs = _rerank_settings(a, S)
    alarm_cfg = cfg["alarms"]
    if rs is not None:
        from grl.model.rerank import check_rerank_settings
        from grl.train.log import expert_slot0_share, rerank_alarm_cfg
        try:
            check_rerank_settings(meta.get("rerank", _settings_near(sft / meta["path"])), rs, f"grpo --sft {sft}")
        except ValueError as e:
            raise SystemExit(str(e)) from None
        if getattr(S["w"], "is_rank", False):                 # pilot 6: §8.7 adapted to ranking
            from grl.train.rank import rank_alarm_cfg, target_slot0_share
            alarm_cfg = rank_alarm_cfg(cfg["alarms"], target_slot0_share(S["env"], S["ranks"], S["train"]))
        else:
            alarm_cfg = rerank_alarm_cfg(cfg["alarms"], expert_slot0_share(S["train"]))
    cls = _policy_cls(a.variant)
    ref = load_reference(S["graph"], S["mcfg"], sft / "pi_ref.json", device=S["device"], policy_cls=cls)   # frozen
    policy = cls(S["graph"], S["mcfg"]).to(S["device"])
    policy.load_state_dict(torch.load(sft / meta["path"], map_location="cpu", weights_only=True))   # θ₀ = π_ref (§8.5)
    logger = RunLogger(a.run, _manifest(a, S, "grpo", {"pi_ref": str(sft / meta["path"])},
                                        {"alarms": alarm_cfg} if rs is not None else None), resume=a.resume)
    if rs is not None:
        _write_json(Path(a.run) / RERANK_SETTINGS, rs)
    ckpt = Checkpoints(Path(a.run) / "checkpoints", int(cfg["run"]["keep_top"]))
    ev = _evaluator(S)
    replay_ex = None
    if float(cfg["grpo"].get("replay_lambda") or 0.0):         # Task GRPO-X: expert replay (SFT-filtered train rows)
        from grl.train.data import replay_examples, sft_train_filter
        pats = {x.strip() for x in str(cfg["grpo"].get("replay_patterns") or "").split(",") if x.strip()}
        kept, _ = sft_train_filter([i for i in S["train"] if i.pattern in pats])
        replay_ex, _ = replay_examples(S["env"], S["ranks"], kept, S["dkw"])
        print(json.dumps({"replay_queries": len(kept), "replay_steps": len(replay_ex)}), file=sys.stderr)
    if not a.resume:
        logger.log("grpo_val", 0, **{f"val_{k}": x for k, x in ev(policy, "val").items()})    # the SFT baseline
    res = run_grpo(policy, ref, S["graph"], S["vec"], S["ranks"], S["train"], cfg["grpo"], S["w"],
                   S["env"].limits.T_max, logger, ckpt, S["device"], ev, S["env"].actions, alarm_cfg,
                   resume=a.resume, **({"replay_ex": replay_ex} if replay_ex is not None else {}))
    logger.export_csv()
    plot_run(a.run)
    logger.close()
    print(json.dumps({"best": str(res.best_path), "best_val_reward": res.best_metric, "iterations": res.iterations,
                      "stopped_early": res.stopped_early}, ensure_ascii=False))
    return 0


def _load_cfg(a) -> dict:
    cfg = config.load("train")
    for kv in a.set:
        _set(cfg, kv)
    return cfg


def cmd_expert(a) -> int:
    """No setup(): no policy, no rollout workers; graphrag.env.parallel.run with EnvData once per worker."""
    from grl.train.expert import expert_config, run_expert
    cfg = _load_cfg(a)
    par = cfg.get("parallel") or {}
    ec = expert_config(cfg, a.variant)                       # struct: expert.struct_workers (R40)
    workers = int(a.workers if a.workers is not None else ec.get("workers", par.get("workers", 1)))
    prune = False if a.keep_zero else True if getattr(a, "prune_zero", False) else None   # None: variant default (R35)
    st = run_expert(a.pairs, a.rankings, a.variant, a.out, cfg, workers, par.get("start_method"), a.depth,
                    prune, a.beam, a.probe_n)
    keys = (("variant", "K_max", "depth", "max_states", "cap_beam", "probe_n", "modes", "n_best_differs",
             "n_score_differs", "keep_zero_better") if a.probe_n is not None
            else ("variant", "env_kw", "K_max", "depth", "seconds", "train", "val"))
    print(json.dumps({k: st[k] for k in keys}, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_selector(a, S) -> int:
    from grl.train.selector import run_selector
    out = run_selector(S["env"], S["ranks"], S["train"], S["val"], S["w"], a.label, a.model)
    out["variant"] = a.variant
    text = json.dumps(out, ensure_ascii=False, sort_keys=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


COMMANDS = {"sft": cmd_sft, "grpo": cmd_grpo, "eval": cmd_eval, "baselines": cmd_baselines, "selector": cmd_selector,
            "rft": cmd_rft}
CMDS = ["sft", "grpo", "eval", "baselines", "expert", "selector", "rft"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=CMDS)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--run")
    ap.add_argument("--sft")
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="test: eval / baselines on a test_eval.jsonl file only (pilot 5 final report)")
    ap.add_argument("--per-query", help="eval / baselines: also write per-query rows (JSONL) here")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-from", help="sft --variant struct: start from the π_ref of this `full` SFT run, widened "
                    "28 -> 31 actions (grl.model.transfer.expand_policy_state, R36), or of a `struct` SFT run, "
                    "widened 31 -> 34 for struct.amend_op (Task 6d-b)")
    ap.add_argument("--curves", action="store_true", help="baselines: recall curves k = 1..K_max + dynamic-k only")
    ap.add_argument("--out", help="baselines / selector: also write the JSON here; expert: the output .jsonl")
    ap.add_argument("--depth", type=int, help="expert: retrieval steps searched before STOP (train.yaml expert.depth)")
    ap.add_argument("--probe-n", type=int, help="expert: probe the first N train queries (prune vs keep-zero, "
                    "runtime); --out gets a JSON report, no trajectories")
    ap.add_argument("--keep-unreached", action="store_true", help="sft: keep expert rows with score 0, an "
                    "immediate STOP or C* missed in the train set (dropped by default, counted in the manifest)")
    ap.add_argument("--beam", type=int, help="expert: keep the best N live states per depth (default: exact)")
    ap.add_argument("--keep-zero", action="store_true", help="expert: do not prune steps that add no fragment "
                    "(the default for --variant struct, R35)")
    ap.add_argument("--prune-zero", action="store_true", help="expert: prune steps that add no fragment (the default "
                    "for every variant but struct; struct SFT refuses such labels)")
    ap.add_argument("--workers", type=int, help="expert: processes (default train.yaml parallel.workers)")
    ap.add_argument("--label", default="expert", choices=["expert", "switch", "best1"], help="selector label")
    ap.add_argument("--model", default="logreg", choices=["logreg", "tree"], help="selector model")
    ap.add_argument("--init-ckpt", help="sft: start from this checkpoint of the same model (Task GRPO-X E1)")
    ap.add_argument("--rft", help="sft: add the kept rows of this `rft` output to the train examples")
    ap.add_argument("--expert-patterns", help="sft: keep only the expert train rows of these patterns (P1,P2)")
    ap.add_argument("--rft-g", type=int, default=32, help="rft: samples per train query")
    ap.add_argument("--rft-margin", type=float, default=0.05, help="rft: keep when R_best > R_greedy + margin")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args(argv)
    if a.cmd in ("sft", "grpo") and not a.run:
        ap.error("--run is required for sft / grpo")
    if a.cmd == "grpo" and not a.sft:
        ap.error("--sft runs/<sft-id> is required for grpo (π_ref and the initial policy)")
    if a.cmd in ("eval", "rft") and not a.ckpt:
        ap.error(f"--ckpt is required for {a.cmd}")
    if a.cmd == "rft" and (not a.out or a.split != "train"):
        ap.error("rft needs --out F.jsonl and --split train (it samples train rows only)")
    if (a.init_ckpt or a.rft or a.expert_patterns) and a.cmd != "sft":
        ap.error("--init-ckpt / --rft / --expert-patterns apply to sft only")
    if a.init_ckpt and a.init_from:
        ap.error("--init-ckpt and --init-from exclude each other")
    from grl.model.variants import VARIANTS
    if a.variant not in VARIANTS:
        ap.error(f"--variant must be one of {sorted(VARIANTS)}")
    if a.curves and a.cmd != "baselines":
        ap.error("--curves applies to baselines only")
    if a.out and a.cmd not in ("baselines", "expert", "selector", "rft"):
        ap.error("--out applies to baselines / expert / selector / rft only")
    if a.split == "test" and a.cmd not in ("eval", "baselines"):
        ap.error("--split test is read only by eval / baselines (pilot 5 final report)")
    if a.per_query and a.cmd not in ("eval", "baselines"):
        ap.error("--per-query applies to eval / baselines only")
    if a.cmd == "expert" and not a.out:
        ap.error("--out F.jsonl is required for expert")
    if a.keep_zero and getattr(a, "prune_zero", False):
        ap.error("--keep-zero and --prune-zero exclude each other")
    check_set_keys(a)
    if a.init_from:
        from grl.model.variants import is_struct
        if a.cmd != "sft" or not is_struct(a.variant):
            ap.error("--init-from applies to sft --variant struct only (R36)")
        _init_from_source(a.init_from)                         # cheap refusal before setup()
    if a.cmd in ("sft", "expert", "selector") and _required(a.variant):
        ap.error(f"{a.cmd} --variant {a.variant} is not supported: the STOP mask of a `require` variant forces "
                 "families the expert / labels do not search; train SFT unconstrained and run grpo --variant "
                 f"{a.variant} --sft <that run>")
    from grl.model.variants import is_rank, is_rerank
    if a.cmd == "expert" and is_rank(a.variant):
        ap.error("expert: the rank variant needs no expert labels (sft --variant rank reads C* directly)")
    if a.cmd == "selector" and is_rerank(a.variant):
        ap.error("selector picks one of the 28 search actions: not defined for the rerank variant (see baselines)")
    if a.cmd == "sft" and not is_rank(a.variant):             # cheap label gate (rank SFT: labels from C* directly)
        from grl.train.data import check_variant_file
        try:
            from grl.train.expert import reward_spec
            from grl.train.fbeta import grl_weights
            cfg0 = _load_cfg(a)
            check_variant_file(a.pairs, a.variant, _env_kw(a, cfg0), _fixed_k_ok(a.variant),
                               reward_spec(grl_weights(cfg0)), int(cfg0["env"]["K_max"]), legacy_k_max())
        except (OSError, ValueError) as e:
            ap.error(f"sft --variant {a.variant} refused: {e}")
    if a.cmd == "expert":
        return cmd_expert(a)
    S = setup(a)
    try:
        return COMMANDS[a.cmd](a, S)
    finally:
        if S["vec"] is not None:
            S["vec"].close()


if __name__ == "__main__":
    sys.exit(main())
