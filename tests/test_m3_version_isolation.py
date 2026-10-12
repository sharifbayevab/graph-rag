"""M3 Task 1: the graph version digest leaves out code the graph build never imports (retrievers, env, eval,
dataset); encoder / tokenizer revisions are pinned (final review m5); BM25 ties are deterministic (m8)."""
import ast
import importlib.metadata as md
from pathlib import Path

import pytest

from graphrag import config
from graphrag.kg import build as B

ROOT = config.path(".")


def _module_file(mod: str) -> Path | None:
    p = ROOT / Path(*mod.split("."))
    for f in (p.with_suffix(".py"), p / "__init__.py"):
        if f.exists():
            return f
    return None


def _imports(f: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):        # function-level imports included
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return {m for m in out if m.split(".")[0] in ("graphrag", "scripts")}


def _closure(start: list[Path]) -> set[Path]:
    seen, todo = set(), list(start)
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for m in _imports(f):
            parts = m.split(".")
            for i in range(1, len(parts) + 1):                               # packages run their __init__
                mf = _module_file(".".join(parts[:i]))
                if mf is not None and mf not in seen:
                    todo.append(mf)
    return seen


def test_graph_build_imports_no_excluded_package():
    files = _closure([ROOT / "scripts/build_graph.py", ROOT / "graphrag/kg/build.py",
                      ROOT / "graphrag/kg/pipeline.py"])
    rel = sorted(f.relative_to(ROOT).as_posix() for f in files)
    assert [r for r in rel if r.startswith(B.VERSION_CODE_EXCLUDE)] == [], rel


def test_digest_skips_m3_code_and_config(tmp_path):
    for rel in ("graphrag/kg/a.py", "graphrag/env/mdp.py", "graphrag/retrieval/x.py", "graphrag/eval/gold.py",
                "graphrag/dataset/seeds.py", "scripts/build_graph.py", "config/graph.yaml", "config/train.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x: 1\n", encoding="utf-8")
    (tmp_path / "config/data.yaml").write_text("max_tokens: 384\nstart:\n  tau_lex: 0.5\n", encoding="utf-8")
    assert sorted(B.code_digest(tmp_path)) == ["graphrag/kg/a.py", "scripts/build_graph.py"]
    cfg = B.config_digest(tmp_path)
    assert sorted(cfg) == ["config/data.yaml", "config/graph.yaml"]
    (tmp_path / "config/data.yaml").write_text("max_tokens: 384\nstart:\n  tau_lex: 0.7\n", encoding="utf-8")
    assert B.config_digest(tmp_path)["config/data.yaml"] == cfg["config/data.yaml"]
    (tmp_path / "config/data.yaml").write_text("max_tokens: 385\nstart:\n  tau_lex: 0.7\n", encoding="utf-8")
    assert B.config_digest(tmp_path)["config/data.yaml"] != cfg["config/data.yaml"]


def test_encoder_passes_the_pinned_revision(monkeypatch):
    import sentence_transformers
    seen = {}

    class FakeST:
        def __init__(self, name, device=None, revision=None):
            seen.update(name=name, revision=revision)
            self.tokenizer = type("T", (), {"model_max_length": 8192})()

        def eval(self):
            return self

        def parameters(self):
            return iter(())

        def get_embedding_dimension(self):
            return 1024

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeST)
    from graphrag.data.embed import Encoder
    Encoder(device="cpu")
    cfg = config.load("model")["encoder"]
    assert len(cfg["revision"]) == 40 and seen == {"name": cfg["name"], "revision": cfg["revision"]}


def test_token_counter_passes_the_pinned_revision(monkeypatch):
    import transformers
    seen = {}

    class FakeTok:
        model_max_length = 512

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    def fake_from_pretrained(name, revision=None):
        seen.update(name=name, revision=revision)
        return FakeTok()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained))
    from graphrag.data.tokens import TokenCounter
    d = config.load("data")
    assert TokenCounter(d["tokenizer"], d["tokenizer_revision"])("a b c") == 3
    assert seen == {"name": d["tokenizer"], "revision": d["tokenizer_revision"]}


def test_graph_build_tokenizer_passes_the_pinned_encoder_revision(monkeypatch):
    """M3-R2 (Task 1 review): the vertex-text token budget (kg/features.py) is cut with the encoder's own
    tokenizer; scripts/build_graph.LazyEncoder loads it at the pinned encoder revision."""
    import transformers
    seen = {}

    def fake_from_pretrained(name, revision=None):
        seen.update(name=name, revision=revision)
        return object()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_from_pretrained))
    import scripts.build_graph as BG
    BG.LazyEncoder(device="cpu")
    cfg = config.load("model")["encoder"]
    assert len(cfg["revision"]) == 40 and seen == {"name": cfg["name"], "revision": cfg["revision"]}


def test_lockfile_pins_the_installed_model_stack():
    lines = config.path("requirements.lock").read_text(encoding="utf-8").splitlines()
    lock = {a.lower().replace("_", "-"): b for a, _, b in (x.partition("==") for x in lines if "==" in x)}
    for dist in ("torch", "torch-geometric", "transformers", "sentence-transformers", "tokenizers", "numpy",
                 "pymorphy3-dicts-ru", "qdrant-client", "neo4j", "psycopg2-binary", "openai"):
        assert lock[dist] == md.version(dist), dist


@pytest.mark.integration
def test_bm25_ties_break_by_chunk_id_after_rounding():
    """M3-R2 (Task 1 review): the raw BM25 scores differ by < 1e-9 (the shorter chunk scores higher) but tie
    after the 9-decimal rounding, so the order is the byte order of chunk_id (COLLATE "C"). The chunk_id columns
    carry a linguistic ICU collation, as a glibc/ICU database default would (it puts "…a" before "…B"; the dev
    postgres:16-alpine default en_US.utf8 is byte order under musl, which would hide a missing COLLATE "C").
    The tables are session TEMP tables, which shadow chunks / postings for this connection only (the shared dev
    index is never touched; search_terms' rollback drops them)."""
    import math

    from graphrag.retrieval import lexical
    from graphrag.stores import pg
    bm = config.load("data")["bm25"]
    k1, b = float(bm["k1"]), float(bm["b"])
    m = 10 ** 9                                   # |c| of the two chunks: m and m + 1 (only |c| differs)
    hi, lo = "BOSHQA-m3tiea#1", "BOSHQA-m3tieB#1"  # hi: the shorter chunk (higher raw score)
    idf = math.log(1 + (2 - 2 + 0.5) / (2 + 0.5))
    avgdl = m + 0.5

    def raw(n):
        return idf * (k1 + 1) / (1 + k1 * (1 - b + b * n / avgdl))

    assert 0 < raw(m) - raw(m + 1) < 1e-9 and round(raw(m), 9) == round(raw(m + 1), 9)
    assert sorted([hi, lo]) == [lo, hi]           # byte order: "B" (0x42) < "a" (0x61)
    conn = pg.connect()
    try:
        with conn.cursor() as cur:
            cur.execute('CREATE TEMP TABLE chunks (chunk_id text COLLATE "und-x-icu" PRIMARY KEY, n_terms integer)')
            cur.execute('CREATE TEMP TABLE postings (term text, chunk_id text COLLATE "und-x-icu", tf integer)')
            cur.execute('SELECT %s::text COLLATE "und-x-icu" < %s::text', (hi, lo))
            assert cur.fetchone() == (True,)                 # the column collation alone orders hi first
            cur.execute("INSERT INTO chunks VALUES (%s, %s), (%s, %s)", (hi, m, lo, m + 1))
            cur.execute("INSERT INTO postings VALUES ('zzqxm3tie', %s, 1), ('zzqxm3tie', %s, 1)", (hi, lo))
        out = lexical.search_terms(conn, ["zzqxm3tie"], k=5)
        assert [c for c, _ in out] == [lo, hi]
        assert out[0][1] == out[1][1] == round(raw(m), 9)
    finally:
        conn.rollback()
        conn.close()
