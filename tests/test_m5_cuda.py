"""CUDA checks on the DGX Spark (plan Task 12): GAT on cuda == cpu, deterministic forward under strict mode; plus the
Task 2 carry-over: the real GAT encoder forward + backward on CUDA under torch.use_deterministic_algorithms(True)
(two runs bit-identical; a strict-mode RuntimeError fails the test and names the op)."""
import os

import pytest
import torch

from grl.model.gnn import GraphEncoder
from tests.m4_toy import MODEL_CFG, toy_graph

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # strict-mode cuBLAS (as grl.train.log.set_determinism)

pytestmark = pytest.mark.cuda


@pytest.fixture(autouse=True)
def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def test_gat_cuda_matches_cpu_and_is_deterministic():
    torch.manual_seed(0)
    g = toy_graph()
    enc = GraphEncoder(g, MODEL_CFG).eval()
    q = torch.eye(4)[:2]
    with torch.no_grad():
        zc, _ = enc(g, q)
        torch.use_deterministic_algorithms(True)
        try:
            enc_g, gg = enc.to("cuda"), g.to("cuda")
            z1, _ = enc_g(gg, q.cuda())
            z2, _ = enc_g(gg, q.cuda())
        finally:
            torch.use_deterministic_algorithms(False)
    assert torch.equal(z1, z2) and torch.allclose(z1.cpu(), zc, atol=1e-4)


def _fwd_bwd(seed: int):
    torch.manual_seed(seed)
    g = toy_graph()
    enc = GraphEncoder(g, MODEL_CFG).train().to("cuda")
    gg, q = g.to("cuda"), torch.eye(4)[:2].cuda()
    z, qz = enc(gg, q)
    (z.square().mean() + qz.sum()).backward()
    return z.detach().cpu(), {n: p.grad.detach().cpu() for n, p in enc.named_parameters() if p.grad is not None}


def test_gat_cuda_forward_backward_is_bit_identical_under_strict_mode():
    torch.use_deterministic_algorithms(True)
    try:
        z1, g1 = _fwd_bwd(0)
        z2, g2 = _fwd_bwd(0)
    finally:
        torch.use_deterministic_algorithms(False)
    assert g1 and set(g1) == set(g2)
    assert torch.equal(z1, z2)
    assert [n for n in g1 if not torch.equal(g1[n], g2[n])] == []
