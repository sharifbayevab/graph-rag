# R1-Searcher

- Paper: *R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning*, Song et al., 2025.
- Code: https://github.com/RUCAIBox/R1-Searcher
- Checkpoint: `XXsongLALA/Qwen-2.5-7B-base-RAG-RL` (released by the authors), no fine-tuning.

Run as in the paper: the official `v0` prompt, stop tokens, temperature 0, 5 documents per search, up to 9 searches, vLLM.
The search tool is the environment: `bge-m3` dense retrieval over our corpus (the paper used bge-large-en over English Wikipedia).
Retrieved fragments of all searches are deduplicated in order and cut at 12.
Runner: `_runners/r1searcher_prep.py`, `_runners/r1searcher_run.py`.
