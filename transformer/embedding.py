# Standard

# PIP
import torch
import torch.nn as nn

# Custom


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        max_len=5000,
    ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)

        x = self.pe[:x.size(0)]

        return x


class Embedding(nn.Module):
    def __init__(
        self,
        num_vocabs,
        d_model,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=num_vocabs,
            embedding_dim=d_model,
        )
        self.positional_embedding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.token_embedding(x) + self.positional_embedding(x)
        x = self.dropout(x)

        return x
