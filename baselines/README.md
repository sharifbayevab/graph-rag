# Baselines

Every baseline runs as its paper and official code define it. Nothing was tuned on our data.
The only deviations are listed below and are the same for all methods:

- corpus: the 16,643 base chunks of our Uzbek legislation corpus (the dataset itself is not published yet);
- embedder: `BAAI/bge-m3` for every method (it is not a trained part of any of them);
- LLM for indexing (HippoRAG 2, PropRAG): `gpt-4.1-mini`, temperature 0;
- retrieval budget: top-12 fragments per question.

| method | paper | official code | folder |
|---|---|---|---|
| Naive RAG | — | dense top-k over bge-m3 | [naive_rag](naive_rag) |
| HippoRAG 2 | Gutiérrez et al., 2025 | https://github.com/OSU-NLP-Group/HippoRAG | [hipporag2](hipporag2) |
| PropRAG | Wang & Han, EMNLP 2025 ([arXiv:2504.18070](https://arxiv.org/abs/2504.18070)) | https://github.com/ReLink-Inc/PropRAG | [proprag](proprag) |
| R1-Searcher | Song et al., 2025 | https://github.com/RUCAIBox/R1-Searcher | [r1searcher](r1searcher) |

Our runner scripts (corpus export, adapters, scoring) are in [`_runners/`](_runners).
