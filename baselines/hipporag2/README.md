# HippoRAG 2

- Paper: *From RAG to Memory: Non-Parametric Continual Learning for Large Language Models* (HippoRAG 2), Gutiérrez et al., 2025.
- Code: https://github.com/OSU-NLP-Group/HippoRAG (we used the PyPI release `hipporag==2.0.0a4`).

Run as in the paper: OpenIE with the official prompts (verbatim, English), the official recognition-memory filter, Personalized PageRank with default parameters.
Deviations: `gpt-4.1-mini` instead of the paper's LLM, `bge-m3` embedder (via a small sentence-transformers adapter, `_runners/bge_m3.py`), our corpus, top-12.
Runner: `_runners/openie_mac.py` (extraction), `_runners/run_method.py hipporag ...` (index + retrieve).
