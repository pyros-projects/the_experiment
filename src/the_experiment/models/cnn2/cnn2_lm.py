# file: cnn_lm.py

import torch
import torch.nn as nn


class CNNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Embedding(
            512, embed_dim
        )  # Maximum sequence length of 512

        # Stack of 1D CNN layers
        cnn_layers = []
        in_channels = embed_dim
        for _ in range(num_layers):
            cnn_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=hidden_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = hidden_dim

        self.cnn_stack = nn.Sequential(*cnn_layers)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Create position indices and embeddings
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        pos_embeddings = self.pos_encoder(positions)

        # Word embeddings
        embeddings = self.embedding(x)

        # Combine word and position embeddings
        embeddings = embeddings + pos_embeddings

        # CNN expects [batch, channels, length]
        x = embeddings.transpose(1, 2)

        # Pass through CNN layers
        features = self.cnn_stack(x)

        # Convert back to [batch, length, channels]
        features = features.transpose(1, 2)

        # Project to vocabulary size
        logits = self.output_layer(features)

        return logits, None  # Return None for hidden state to match LSTM interface
