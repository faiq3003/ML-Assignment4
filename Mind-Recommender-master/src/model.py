from __future__ import annotations

import numpy as np
import torch
from torch import nn

from news_encoder import NewsEncoder
from user_encoder import UserEncoder


class NRMSModel(nn.Module):
    """Neural News Recommendation with Multi-Head Self-Attention."""

    def __init__(
        self,
        embedding_matrix: np.ndarray | torch.Tensor,
        num_heads: int = 15,
        attention_dim: int = 200,
        dropout: float = 0.2,
        pad_idx: int = 0,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(embedding_matrix, np.ndarray):
            embedding_matrix = torch.from_numpy(embedding_matrix)
        embedding_dim = int(embedding_matrix.shape[1])

        self.news_encoder = NewsEncoder(
            embedding_matrix=embedding_matrix,
            num_heads=num_heads,
            attention_dim=attention_dim,
            dropout=dropout,
            pad_idx=pad_idx,
            freeze_embeddings=freeze_embeddings,
        )
        self.user_encoder = UserEncoder(
            news_vector_dim=embedding_dim,
            num_heads=num_heads,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(
        self,
        history_titles: torch.Tensor,
        candidate_titles: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate relevance logits for each user in the batch."""
        batch_size, history_size, title_length = history_titles.shape
        candidate_count = candidate_titles.shape[1]

        history_flat = history_titles.reshape(batch_size * history_size, title_length)
        history_vectors = self.news_encoder(history_flat).reshape(batch_size, history_size, -1)
        history_mask = history_titles.sum(dim=-1).ne(0)
        user_vectors = self.user_encoder(history_vectors, history_mask=history_mask)

        candidate_flat = candidate_titles.reshape(batch_size * candidate_count, title_length)
        candidate_vectors = self.news_encoder(candidate_flat).reshape(batch_size, candidate_count, -1)

        return torch.bmm(candidate_vectors, user_vectors.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def predict(self, history_titles: torch.Tensor, candidate_titles: torch.Tensor) -> torch.Tensor:
        """Return sigmoid-normalized recommendation scores."""
        self.eval()
        return torch.sigmoid(self.forward(history_titles, candidate_titles))
