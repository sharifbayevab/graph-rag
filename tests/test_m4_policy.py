"""Policy (IMPLEMENTATION.md §6.4; U4; §11 step 7: "прямой проход даёт распределение по действиям нужной размерности")."""
import pytest
import torch

from grl.model.graph import graph_tensors
from grl.model.observe import describe
from grl.model.policy import GraphPolicy, masked_log_softmax, param_breakdown
from tests.m3_toy import toy_rank
from tests.m4_toy import MODEL_CFG, toy_env, toy_graph, toy_payload


def _policy(seed: int = 0):
    torch.manual_seed(seed)
    g = toy_graph()
    return GraphPolicy(g, MODEL_CFG).eval(), g


def test_forward_gives_a_masked_distribution_over_28_actions():
    pol, g = _policy()
    env = toy_env()
    q = env.query(toy_rank())
    d = describe(env, q, env.reset(q))
    with torch.no_grad():
        z, qz = pol.encode(g, torch.from_numpy(q.qvec[None]).float())
        lp = pol.log_probs(z, qz, [0], [d])
    p = lp.exp()
    assert lp.shape == (1, 28) and float(p.sum()) == pytest.approx(1.0, abs=1e-6)
    assert bool((p[0, ~torch.from_numpy(d.mask)] == 0).all())


def test_only_stop_valid_gives_logp_zero_and_never_samples_masked():
    m = torch.zeros(3, 28, dtype=torch.bool)
    m[:, 0] = True
    m[1, 5] = True
    m[2, 27] = True
    logits = torch.zeros(3, 28)
    logits[1, 6] = 100.0                                 # a huge logit on a masked action
    lp = masked_log_softmax(logits, m)
    assert float(lp[0, 0]) == 0.0 and int(lp[1].argmax()) in (0, 5)
    draws = torch.multinomial(torch.softmax(lp / 0.3, -1), 1000, replacement=True,
                              generator=torch.Generator().manual_seed(0))
    assert bool(m.gather(1, draws).all())
    with pytest.raises(ValueError):
        masked_log_softmax(torch.zeros(1, 28), torch.zeros(1, 28, dtype=torch.bool))


def test_parameter_count_at_spec_dims():
    g = graph_tensors(toy_payload(x_dim=1024, relations=9, node_types=26))
    b = param_breakdown(GraphPolicy(g, {**MODEL_CFG, "in_dim": 1024}))
    assert (b["encoder"], b["state"], b["head"], b["total"]) == (1_155_040, 139_808, 752_412, 2_047_260)
