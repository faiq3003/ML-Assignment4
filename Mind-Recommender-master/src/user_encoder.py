from __future__ import annotations

import torch
from torch import nn

from news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    """Encode a user's clicked-news history into a user preference vector."""

    def __init__(
        self,
        news_vector_dim: int,
        num_heads: int = 15,
        attention_dim: int = 200,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=news_vector_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.additive_attention = AdditiveAttention(news_vector_dim, attention_dim)

    def forward(
        self,
        clicked_news_vectors: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history_mask is None:
            history_mask = clicked_news_vectors.abs().sum(dim=-1).gt(0)

        safe_mask = history_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        clicked_news_vectors = self.dropout(clicked_news_vectors)
        contextualized, _ = self.self_attention(
            clicked_news_vectors,
            clicked_news_vectors,
            clicked_news_vectors,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        contextualized = self.dropout(contextualized)
        return self.additive_attention(contextualized, mask=history_mask)
