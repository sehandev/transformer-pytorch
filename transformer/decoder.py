# Standard

# PIP
import torch.nn as nn

# Custom

class DecoderLayer(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # Masked Multi-Head Attention
        self.masked_attention = MaskedMultiHeadAttention()
        self.layer_norm1 = LayerNormalization(
            d_model=cfg.D_MODEL,
            epsilon=cfg.EPSILON,
        )

        # Multi-Head Attention
        self.attention = MultiHeadAttention()
        self.layer_norm2 = LayerNormalization(
            d_model=cfg.D_MODEL,
            epsilon=cfg.EPSILON,
        )

        # Feed Forward
        self.ff = FeedForward(
            d_model=cfg.D_MODEL,
            d_ff=cfg.D_FF,
            dropout=cfg.DROPOUT,
        )
        self.layer_norm3 = LayerNormalization(
            d_model=cfg.D_MODEL,
            epsilon=cfg.EPSILON,
        )

    def forward(self, x, encoder_output):
        residual = x
        x = self.masked_attention(
            q=x,
            k=x,
            v=x,
        )
        x = x + residual
        x = self.layer_norm1(x)

        residual = x
        x = self.attention(
            q=encoder_output,
            k=encoder_output,
            v=x,
        )
        x = x + residual
        x = self.layer_norm2(x)

        residual = x
        x = self.ff(x)
        x = x + residual
        x = self.layer_norm3(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                cfg=cfg,
            )
            for _ in range(cfg.NUM_LAYERS)
        ])

    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)

        return x
