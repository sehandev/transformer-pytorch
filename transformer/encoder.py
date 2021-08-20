# Standard

# PIP
import torch.nn as nn

# Custom
from transformer.layer import FeedForward

class EncoderLayer(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # Multi-Head Attention
        self.self_attention = MultiHeadAttention()
        self.layer_norm1 = LayerNormalization(
            d_model=cfg.D_MODEL,
            epsilon=cfg.EPSILON,
        )

        # Feed Forward
        self.ff = FeedForward(
            d_model=cfg.D_MODEL,
            d_ff=cfg.D_FF,
            dropout=cfg.DROPOUT,
        )
        self.layer_norm2 = LayerNormalization(
            d_model=cfg.D_MODEL,
            epsilon=cfg.EPSILON,
        )

    def forward(self, x):
        residual = x
        x = self.self_attention(
            q=x,
            k=x,
            v=x,
        )
        x = x + residual
        x = self.layer_norm1(x)

        residual = x
        x = self.ff(x)
        x = x + residual
        x = self.layer_norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                cfg=cfg,
            )
            for _ in range(cfg.NUM_LAYERS)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)

        return x
