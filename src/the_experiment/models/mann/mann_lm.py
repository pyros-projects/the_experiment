# file: mann_lm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryUnit(nn.Module):
    def __init__(self, memory_size, memory_vector_dim, hidden_dim):
        """
        Initialize the external memory unit.
        
        Args:
            memory_size: Number of memory slots
            memory_vector_dim: Dimension of each memory vector
            hidden_dim: Dimension of the controller hidden state
        """
        super().__init__()
        
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.hidden_dim = hidden_dim
        
        # Initialize memory matrix
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_vector_dim))
        nn.init.kaiming_uniform_(self.memory)
        
        # Read/write heads
        self.read_head = nn.Linear(hidden_dim, memory_vector_dim)
        self.write_head = nn.Linear(hidden_dim, memory_vector_dim)
        
        # Key strength (sharpness of focus)
        self.key_strength = nn.Linear(hidden_dim, 1)
        
        # Erase and add vectors for writing
        self.erase_vector = nn.Linear(hidden_dim, memory_vector_dim)
        self.add_vector = nn.Linear(hidden_dim, memory_vector_dim)
        
    def attention(self, query, key_strength=None):
        """Compute attention weights over memory."""
        # Compute similarity between query and memory
        similarity = F.cosine_similarity(
            query.unsqueeze(1).expand(-1, self.memory_size, -1),
            self.memory.unsqueeze(0),
            dim=2
        )
        
        if key_strength is not None:
            similarity = similarity * key_strength
            
        # Apply softmax to get attention weights
        weights = F.softmax(similarity, dim=1)
        return weights
        
    def read(self, controller_state):
        """Read from memory using attention."""
        # Generate read query
        read_query = self.read_head(controller_state)
        key_str = F.softplus(self.key_strength(controller_state))
        
        # Get attention weights
        read_weights = self.attention(read_query, key_str)
        
        # Read from memory
        read_vector = torch.bmm(
            read_weights.unsqueeze(1),
            self.memory.unsqueeze(0).expand(controller_state.size(0), -1, -1)
        ).squeeze(1)
        
        return read_vector
        
    def write(self, controller_state):
        """Write to memory using attention."""
        # Generate write query
        write_query = self.write_head(controller_state)
        write_weights = self.attention(write_query)
        
        # Generate erase and add vectors
        erase = torch.sigmoid(self.erase_vector(controller_state))
        add = torch.tanh(self.add_vector(controller_state))
        
        # Erase and add to memory
        erase_matrix = torch.bmm(
            write_weights.unsqueeze(-1),
            erase.unsqueeze(1)
        )
        add_matrix = torch.bmm(
            write_weights.unsqueeze(-1),
            add.unsqueeze(1)
        )
        
        # Update memory
        self.memory = self.memory * (1 - erase_matrix.mean(0)) + add_matrix.mean(0)
        
        return write_weights

class MANNLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        hidden_dim=256,
        memory_size=128,
        memory_vector_dim=64,
        num_layers=1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Controller (LSTM)
        self.controller = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # External memory
        self.memory = MemoryUnit(
            memory_size=memory_size,
            memory_vector_dim=memory_vector_dim,
            hidden_dim=hidden_dim
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim + memory_vector_dim, vocab_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim + memory_vector_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, hidden=None):
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            hidden: Initial hidden state for LSTM controller
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            hidden: Final hidden state of controller
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed input
        embedded = self.embedding(input_ids)
        
        # Run through controller
        controller_output, hidden = self.controller(embedded, hidden)
        
        # Process each timestep with memory
        memory_outputs = []
        for t in range(seq_len):
            # Read from memory
            read_vector = self.memory.read(controller_output[:, t])
            
            # Write to memory
            self.memory.write(controller_output[:, t])
            
            # Concatenate controller output with memory read
            combined = torch.cat([
                controller_output[:, t],
                read_vector
            ], dim=1)
            
            memory_outputs.append(combined)
        
        # Stack timesteps
        output = torch.stack(memory_outputs, dim=1)
        
        # Layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        # Project to vocabulary
        logits = self.output(output)
        
        return logits, hidden
    
    def init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)