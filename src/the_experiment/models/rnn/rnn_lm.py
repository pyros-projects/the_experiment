# file: rnn_lm.py

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        """
        input_ids: (batch_size, seq_len)
        hidden: optional (num_layers, batch_size, hidden_dim)
        Returns logits: (batch_size, seq_len, vocab_size)
        and hidden state
        """
        emb = self.embedding(input_ids)  # (batch_size, seq_len, embed_dim)
        output, hidden = self.lstm(emb, hidden)  # (batch_size, seq_len, hidden_dim)
        logits = self.fc(output)  # (batch_size, seq_len, vocab_size)
        return logits, hidden
