# Naive RAG

Dense retrieval: the question and every chunk are embedded with `BAAI/bge-m3`, and the top-k chunks by cosine similarity form the context.
It is part of our evaluation code (the `dense` baseline in `scripts/m7_pilot8.py baselines`); no external repository.
