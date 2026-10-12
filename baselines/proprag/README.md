# PropRAG

- Paper: *PropRAG: Guiding Retrieval with Beam Search over Proposition Paths*, Wang & Han, EMNLP 2025 — https://arxiv.org/abs/2504.18070
- Code: https://github.com/ReLink-Inc/PropRAG (commit `3ec1034`).

Run as in the paper: proposition extraction with the official prompts (verbatim), the default two-stage beam search, default parameters.
Deviations: `gpt-4.1-mini` through the OpenAI API instead of OpenRouter, `bge-m3` embedder, our corpus, top-12.
Runner: `_runners/openie_mac.py`, `_runners/run_method.py proprag ...`.
