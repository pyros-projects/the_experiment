# file: cnn_lm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer that ensures no information leakage from future tokens
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=0,  # No padding, we'll handle it manually
        )

    def forward(self, x):
        # Add padding only on the left side (past tokens)
        padding = (self.kernel_size - 1, 0)
        x = F.pad(x, padding)
        return self.conv(x)


class CNNLanguageModel(nn.Module):
    def __init__(
        self, vocab_size, embed_dim=128, num_filters=128, kernel_size=3, seq_len=64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.kernel_size = kernel_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Causal CNN layers
        self.conv1 = CausalConv1d(
            in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size
        )
        self.conv2 = CausalConv1d(
            in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size
        )

        # Layer normalization and dropout for regularization
        self.layer_norm1 = nn.LayerNorm(num_filters)
        self.layer_norm2 = nn.LayerNorm(num_filters)
        self.dropout = nn.Dropout(0.1)

        # Output projection
        self.fc = nn.Linear(num_filters, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass with causal masking.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Get shape information
        batch_size, seq_len = input_ids.shape

        # Embedding
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        emb = emb.transpose(1, 2)  # (batch, embed_dim, seq_len)

        # First causal convolution block
        h = self.conv1(emb)  # (batch, num_filters, seq_len)
        h = h.transpose(1, 2)  # (batch, seq_len, num_filters)
        h = self.layer_norm1(h)  # Layer normalization
        h = h.transpose(1, 2)  # (batch, num_filters, seq_len)
        h = F.gelu(h)  # GELU activation
        h = self.dropout(h)  # Apply dropout

        # Second causal convolution block
        h = self.conv2(h)  # (batch, num_filters, seq_len)
        h = h.transpose(1, 2)  # (batch, seq_len, num_filters)
        h = self.layer_norm2(h)  # Layer normalization
        h = F.gelu(h)  # GELU activation
        h = self.dropout(h)  # Apply dropout

        # Project to vocabulary size
        logits = self.fc(h)  # (batch, seq_len, vocab_size)

        return logits

    def init_weights(self):
        """Initialize weights for better training."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(_init_weights)
