# Standard

# PIP
import torch.nn as nn

# Custom


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head

        self.value = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)

    def forward(self, v, k, q, mask):
        

        return out
