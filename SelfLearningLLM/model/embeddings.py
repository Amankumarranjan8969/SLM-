"""model/embeddings.py

Token embedding table. Deliberately minimal: a single `nn.Embedding`. The
LM head is created and (optionally) tied to this embedding's weight in
model/llm.py, not here, since weight tying is a property of how the two
are wired together at the top level, not of the embedding module itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Llama-style small init; final scale is also influenced by the
        # residual-scaling init applied to output projections in
        # transformer.py, so this only needs to be a sane starting point.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight
