# Standard
import math

# PIP
import torch.nn as nn

# Custom


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, q, k, v):
        # q, k, v: (batch_size, seq_len, num_heads, d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # q, k, v: (batch_size, num_heads, seq_len, d_k)

        k = k.transpose(2, 3)
        # k: (batch_size, num_heads, d_k, seq_len)

        out = torch.matmul(q, k) / self.sqrt_d_k
        # out: (batch_size, num_heads, seq_len, seq_len)

        out = F.softmax(out, dim=-1)
        # out: (batch_size, num_heads, seq_len, seq_len)

        out = torch.matmul(out, v)
        # out: (batch_size, num_heads, seq_len, d_k)

        out = out.transpose(1, 2)
        # out: (batch_size, seq_len, num_heads, d_k)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.value = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(self.d_k)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, v, k, q):
        # v, k, q: (batch_size, seq_len, d_model)

        # Linear Projection
        v = self.value(v)
        k = self.key(k)
        q = self.query(q)

        num_batches = v.shape[0]
        v = v.view(num_batches, -1, self.num_heads, self.d_k)
        k = k.view(num_batches, -1, self.num_heads, self.d_k)
        q = q.view(num_batches, -1, self.num_heads, self.d_k)
        # out: (batch_size, seq_len, num_heads, d_k)

        out = self.attention(q, k, v)
        # out: (batch_size, seq_len, num_heads, d_k)

        # TODO view + contiguous -> reshape
        out = out.contiguous().view(num_batches, -1, self.d_model)
        # out: (batch_size, seq_len, d_model)

        out = self.fc(out)
        # out: (batch_size, seq_len, d_model)

        return out
