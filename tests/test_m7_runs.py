"""Task 9 (final): the seed queue -- deterministic names, the pilot-9 recipe, never a test split."""
import hashlib
import json

from graphrag import config
from tests.scripts_import import load_script


def _cfg_sha(argv):
    t = load_script("train")
    cfg = config.load("train")
    for kv in (argv[i + 1] for i, x in enumerate(argv) if x == "--set"):
        t._set(cfg, kv)
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def test_queue_is_deterministic_and_val_only():
    r = load_script("m7_runs")
    a, b = r.jobs(), r.jobs()
    assert [j["name"] for j in a] == ["f9-s01chk", "f9-s02", "f9-s03"]
    assert a == b
    for j in a:
        for _, argv in j["steps"]:
            assert "test" not in argv
            assert all(y == "val" for x, y in zip(argv, argv[1:]) if x == "--split")
    s2 = dict(r.steps("s02"))
    assert "sft.seed=20261002" in s2["sft"] and "grpo.seed=20261002" in s2["grpo"]
    assert s2["grpo"][s2["grpo"].index("--sft") + 1] == "runs/f9-sft-s02"


def test_recipe_equals_pilot9_config():
    """With the pilot-9 seed the generated SFT / GRPO configs hash exactly like p9-sft-scratch / p9-grpo (their
    manifests' config_sha256), so only the seed differs between s01 (= pilot 9) and s02 / s03."""
    r = load_script("m7_runs")
    sft = [x.replace("20261002", "20261001") for x in dict(r.steps("s02"))["sft"]]
    assert _cfg_sha(sft).startswith("c3a7bbaa7d05")
    assert _cfg_sha(r.steps("s01chk")[0][1]).startswith("b77311a89bfb")


def test_acceptance_rule_one_se_and_no_late_alarm(tmp_path, monkeypatch):
    """Plan Task 9 step 2 (fix round 1, M2): > 1 paired SE over SFT on val R AND no alarm in the last 20 %."""
    r = load_script("m7_runs")

    def pq(name, rewards):
        f = tmp_path / name
        f.write_text("".join(json.dumps({"qkey": f"q{i}|q", "reward": x}) + "\n" for i, x in enumerate(rewards)))
        return str(f)
    sft = pq("sft.jsonl", [0.5, 0.5, 0.5, 0.5])
    up = pq("up.jsonl", [0.6, 0.6, 0.6, 0.62])            # +.105 +- .005 -> > 1 SE
    noisy = pq("noisy.jsonl", [0.9, 0.1, 0.9, 0.2])       # +.025 +- .2   -> not > 1 SE
    monkeypatch.setattr(r, "seed_summary", lambda s, local="runs/spark": {"grpo": {"alarms_last20": []}})
    assert r.acceptance("s03", sft, up)["candidate"] == "grpo"
    assert r.acceptance("s03", sft, noisy)["candidate"] == "sft"
    monkeypatch.setattr(r, "seed_summary", lambda s, local="runs/spark": {"grpo": {"alarms_last20": [[248, "stop"]]}})
    assert r.acceptance("s02", sft, up)["candidate"] == "sft"
