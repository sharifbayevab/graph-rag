"""Minimal BAAI/bge-m3 adapter for the embedding interface of HippoRAG 2.0.0a4 and PropRAG (both forked the same
BaseEmbeddingModel: batch_encode(texts, instruction=..., norm=...) -> np.ndarray). Mirrors HippoRAG's own later
TransformersEmbeddingModel (sentence-transformers, no query instruction: bge-m3 dense retrieval uses none, unlike
NV-Embed-v2's "Instruct: ...\\nQuery: " prefix). max_seq_length = global_config.embedding_max_seq_len, as the NV-Embed-v2
classes use it. Encoding batch size is fixed at 64 (throughput only)."""
from __future__ import annotations

import numpy as np

MODEL = "BAAI/bge-m3"
BATCH = 64


def make_adapter(base_cls):
    class BGEM3EmbeddingModel(base_cls):
        def __init__(self, global_config=None, embedding_model_name: str = MODEL) -> None:
            super().__init__(global_config=global_config)
            import torch
            from sentence_transformers import SentenceTransformer
            self.embedding_model_name = embedding_model_name
            self.model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
            self.model.max_seq_length = int(self.global_config.embedding_max_seq_len)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

        def batch_encode(self, texts, **kwargs) -> np.ndarray:
            if isinstance(texts, str):
                texts = [texts]
            norm = kwargs.get("norm", self.global_config.embedding_return_as_normalized)
            emb = self.model.encode(list(texts), batch_size=BATCH, normalize_embeddings=bool(norm),
                                    show_progress_bar=len(texts) > 10000, convert_to_numpy=True)
            return emb.astype(np.float32)

    return BGEM3EmbeddingModel
