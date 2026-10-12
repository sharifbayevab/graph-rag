"""Pilot 9 helpers: the ranker's no-STOP top-5 view, the reporting cells, the SFT-filter replica."""
import torch

from scripts.m7_pilot8 import _NoStop
from scripts.m7_pilot9 import _filtered, fine_group, subgroups


class _Fake:
    def __init__(self, logits, mask):
        self.lg, self.mask = logits, mask

    def encode(self, g, qvec):
        return None, None

    def logits(self, z, qz, qidx, descs):
        return self.lg, self.mask


def test_nostop_masks_stop_only_while_a_slot_is_valid():
    lg = torch.tensor([[5.0, 1.0, 2.0], [5.0, 1.0, 2.0]])
    mask = torch.tensor([[True, True, True], [True, False, False]])
    lp = _NoStop(_Fake(lg, mask)).log_probs(None, None, [0, 1], [None, None])
    assert int(lp[0].argmax()) == 2                  # STOP (logit 5) masked, best slot taken
    assert torch.isinf(lp[0, 0]) and lp[0, 0] < 0
    assert int(lp[1].argmax()) == 0                  # no valid slot: STOP stays available
    assert bool(mask[0, 0])                          # the caller's mask is not mutated


def _row(pattern, tpl=None, easy=None):
    m7 = {"template": tpl} if tpl else {}
    if easy is not None:
        m7["easy"] = easy
    return {"pattern": pattern, "provenance": {"m7": m7} if m7 else {}}


def test_groups_and_cells():
    assert fine_group(_row("single")) == "old"
    assert fine_group(_row("count", "count_amends_year")) == "Q1"
    assert fine_group(_row("count", "count_points_article", False)) == "Q1x"
    assert fine_group(_row("version", "what_law_changed")) == "Q2"
    c = subgroups(_row("count", "count_parts_article", True))
    assert {"all", "hard", "Q1x", "count", "Q1x/count_parts_article", "Q1x/count_parts_article/easy",
            "Q1x/article/easy"} == set(c)
    assert set(subgroups(_row("count", "count_articles"))) == {"all", "hard", "Q1", "count"}
    assert set(subgroups(_row("single"))) == {"all", "old"}


def test_filtered_replicates_sft_train_filter():
    def lab(score, reached, actions):
        return {"actions": actions, "provenance": {"meta": {"expert": {"score": score, "reached": reached}}}}
    assert not _filtered(lab(0.8, True, [28, 0]))
    assert _filtered(lab(0.8, False, [28, 0]))
    assert _filtered(lab(0.0, True, [28, 0]))
    assert _filtered(lab(0.5, True, [0]))
