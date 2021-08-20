# Standard

# PIP
import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom
from embedding import Embedding


class CustomModel(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        # Input Embedding & Positional Encoding
        self.input_embedding = Embedding(
            num_vocabs=cfg.NUM_ENCODER_VOCABS,
            d_model=cfg.D_MODEL,
            max_len=cfg.MAX_LEN,
            dropout=cfg.DROPOUT,
        )

        # Output Embedding & Positional Encoding
        self.output_embedding = Embedding(
            num_vocabs=cfg.NUM_DECODER_VOCABS,
            d_model=cfg.D_MODEL,
            max_len=cfg.MAX_LEN,
            dropout=cfg.DROPOUT,
        )

        self.encoder = Encoder(
            cfg=cfg,
        )
        self.decoder = Decoder(
            cfg=cfg,
        )

        self.fc = nn.Linear(cfg.D_MODEL, cfg.NUM_DECODER_VOCABS)
        self.softmax = nn.Softmax(2)

    def forward(self, source, target):
        source = self.input_embedding(source)
        out = self.encoder(source)

        target = self.output_embedding(target)
        out = self.decoder(target, out)
        # out: (batch_size, max_len, d_model)

        out = self.fc(out)
        # out: (batch_size, max_len, num_decoder_vocabs)

        out = self.softmax(out)
        # out: (batch_size, max_len, num_decoder_vocabs)

        return out
