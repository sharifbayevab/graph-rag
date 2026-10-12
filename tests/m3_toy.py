"""Hand-checked toy environment shared by the M3 tests (expected values derived in the plan).
Rows: nodes 0 D1, 1 D1/СТ.1, 2 D1/СТ.2, 3 D2, 4 D2/СТ.5, 5 ENT/organ/senat, 6 EXT/k, 7 EXT/k/СТ.9;
chunks 0 D1/СТ.1#1, 1 D1/СТ.1#2, 2 D1/СТ.2#1, 3 D2/СТ.5#1, 4 D2/СТ.5#2. e_q = (1, 0, 0, 0)."""
import numpy as np

from graphrag.env.data import EnvData
from graphrag.retrieval.rankings import QueryRanking

EDGE_TYPES = ["CONTAINS", "REFERS_TO", "AMENDS", "REPEALS", "REGULATES", "DEFINES", "RELATED"]
NODES = ["D1", "D1/СТ.1", "D1/СТ.2", "D2", "D2/СТ.5", "ENT/organ/senat", "EXT/k", "EXT/k/СТ.9"]
CHUNKS = ["D1/СТ.1#1", "D1/СТ.1#2", "D1/СТ.2#1", "D2/СТ.5#1", "D2/СТ.5#2"]
EDGES = [("D1", "CONTAINS", "D1/СТ.1"), ("D1", "CONTAINS", "D1/СТ.2"), ("D2", "CONTAINS", "D2/СТ.5"),
         ("EXT/k", "CONTAINS", "EXT/k/СТ.9"), ("D1/СТ.2", "REFERS_TO", "D2/СТ.5"),
         ("D2/СТ.5", "AMENDS", "EXT/k/СТ.9"), ("D1/СТ.1", "REGULATES", "ENT/organ/senat"),
         ("D2/СТ.5", "REGULATES", "ENT/organ/senat"), ("D1/СТ.1", "RELATED", "D2/СТ.5")]
CHUNK_PARENT = {"D1/СТ.1#1": "D1/СТ.1", "D1/СТ.1#2": "D1/СТ.1", "D1/СТ.2#1": "D1/СТ.2",
                "D2/СТ.5#1": "D2/СТ.5", "D2/СТ.5#2": "D2/СТ.5"}
MU_INV = {"D1": CHUNKS[:3], "D1/СТ.1": CHUNKS[:2], "D1/СТ.2": [CHUNKS[2]], "D2": CHUNKS[3:], "D2/СТ.5": CHUNKS[3:]}
TOKENS = {"D1/СТ.1#1": 100, "D1/СТ.1#2": 50, "D1/СТ.2#1": 400, "D2/СТ.5#1": 80, "D2/СТ.5#2": 60}
CHUNK_VECS = np.array([[0.8, 0.6, 0, 0], [0.6, 0, 0.8, 0], [0, 1, 0, 0], [0.6, 0.8, 0, 0],
                       [0.9, 0, 0, np.sqrt(0.19)]], np.float32)             # cos with e_q: .8 .6 0 .6 .9
NODE_X = np.array([[0, 0, 0, 1]] * 5 + [[0.5, 0, 0, np.sqrt(0.75)], [0.1, np.sqrt(0.99), 0, 0],
                                        [0.7, 0, np.sqrt(0.51), 0]], np.float32)   # chunkless: .5 .1 .7
QVEC = np.array([1, 0, 0, 0], np.float32)
TOY_CFG = {"graph_version": "toy", "T_max": 10, "K_max": 4, "L_max": 1000, "k_grid": [3, 5, 10],
           "m_grid": [1, 3, 5], "lambda_red": 0.5, "per_vertex_chunks": 3, "score_decimals": 6, "top": 200}
TOY_START = {"tau_lex": 0.5, "tau_sem": 0.85, "cap": 64}


def toy_data() -> EnvData:
    return EnvData.build(NODES, CHUNKS, EDGES, CHUNK_PARENT, MU_INV, TOKENS, CHUNK_VECS, NODE_X, EDGE_TYPES)


def toy_rank(qkey: str = "toy|q") -> QueryRanking:
    return QueryRanking(qkey, "toy question", QVEC, np.array([2, 0, 3]), np.array([5.0, 4.0, 1.0]),
                        np.array([4, 0, 1, 3, 2]), np.array([0.9, 0.8, 0.6, 0.6, 0.0]),
                        np.array([5]), np.array([2.0]))
