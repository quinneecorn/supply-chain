"""
BiLSTM classifier for supply-chain relation extraction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import DROPOUT, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, NUM_LAYERS


class BiLSTMClassifier(nn.Module):
    """
    Embedding -> BiLSTM -> masked mean pooling -> Linear -> logits.

    Input shapes (per batch):
        input_ids:      (batch, seq_len)
        attention_mask: (batch, seq_len)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_LAYERS,
        num_classes: int = NUM_CLASSES,
        dropout: float = DROPOUT,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # BiLSTM outputs hidden_dim * 2 per direction
        self.num_classes = num_classes
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        self._shape_debug_printed = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # --- shape checks for debugging ---
        assert input_ids.dim() == 2, f"input_ids must be (B, L), got {input_ids.shape}"
        assert attention_mask.shape == input_ids.shape, (
            f"attention_mask {attention_mask.shape} != input_ids {input_ids.shape}"
        )

        batch_size, seq_len = input_ids.shape
        if not self._shape_debug_printed:
            print(
                f"[model] shapes — batch={batch_size}, seq_len={seq_len}, "
                f"embed_dim={self.embed_dim}, hidden_dim={self.hidden_dim} "
                f"(printed once; assertions run every forward pass)"
            )
            self._shape_debug_printed = True

        embedded = self.embedding(input_ids)  # (B, L, embed_dim)
        assert embedded.shape == (batch_size, seq_len, self.embed_dim), (
            f"Unexpected embedding shape: {embedded.shape}"
        )

        # LSTM expects (B, L, input_size)
        lstm_out, _ = self.lstm(embedded)  # (B, L, hidden_dim * 2)
        assert lstm_out.shape == (batch_size, seq_len, self.hidden_dim * 2), (
            f"Unexpected LSTM output shape: {lstm_out.shape}"
        )

        # Masked mean pooling over valid tokens
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        summed = (lstm_out * mask).sum(dim=1)        # (B, hidden_dim * 2)
        lengths = mask.sum(dim=1).clamp(min=1.0)     # (B, 1)
        pooled = summed / lengths
        assert pooled.shape == (batch_size, self.hidden_dim * 2), (
            f"Unexpected pooled shape: {pooled.shape}"
        )

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (B, num_classes)
        assert logits.shape == (batch_size, self.num_classes), (
            f"Unexpected logits shape: {logits.shape}"
        )

        return logits