from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Attention pooling with a learned query vector."""

    def __init__(self, input_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, attention_dim)
        self.query = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.query(torch.tanh(self.projection(inputs))).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=-1) * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            weights = torch.softmax(scores, dim=-1)

        return torch.bmm(weights.unsqueeze(1), inputs).squeeze(1)


class NewsEncoder(nn.Module):
    """Encode a tokenized news title into a dense news vector."""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        num_heads: int = 15,
        attention_dim: int = 200,
        dropout: float = 0.2,
        pad_idx: int = 0,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(),
            freeze=freeze_embeddings,
            padding_idx=pad_idx,
        )
        embedding_dim = embedding_matrix.shape[1]
        self.dropout = nn.Dropout(dropout)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.additive_attention = AdditiveAttention(embedding_dim, attention_dim)

    def forward(self, title_tokens: torch.Tensor) -> torch.Tensor:
        valid_mask = title_tokens.ne(self.pad_idx)
        safe_mask = valid_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True

        embedded = self.dropout(self.embedding(title_tokens))
        contextualized, _ = self.self_attention(
            embedded,
            embedded,
            embedded,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        contextualized = self.dropout(contextualized)
        return self.additive_attention(contextualized, mask=valid_mask)
