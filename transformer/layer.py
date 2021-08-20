# Standard

# PIP
import torch
import torch.nn as nn

# Custom


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, max_len, d_model)

        x = self.fc1(x)
        # x: (batch_size, max_len, d_ff)

        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        # x: (batch_size, max_len, d_model)

        return x


class LayerNormalization(nn.Module):
    def __init__(
        self,
        d_model=512,
        epsilon=1e-5,
    ):
        super().__init__()
        
        self.epsilon = epsilon

        # nn.Variable
        # tensor grad=True
        self.gamma = nn.Parameter(torch.Tensor(d_model))
        self.beta = nn.Parameter(torch.Tensor(d_model))

        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.gamma.data, 1)
        nn.init.constant_(self.beta.data, 0)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()

        y = (x - mean) / std
        y = y * self.gamma + self.beta
        return y