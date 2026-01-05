import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import TestEmbedding
import numpy as np
import random
from layers.Augmentation import get_augmentation


def compute_patch_num(seq_len, patch_len, stride):
    """Compute the exact number of patches after right padding and unfold."""
    L, P, S = seq_len, patch_len, stride
    if L < P:
        pad_right = P - L
    else:
        rem = (L - P) % S
        pad_right = 0 if rem == 0 else (S - rem)
    Lp = L + pad_right
    patch_num = (Lp - P) // S + 1
    return patch_num


class Model(nn.Module):
    """
    TestFormer
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.patch_len = configs.patch_len
        stride = configs.patch_len
        patch_type = configs.patch_type
        if patch_type == "multi-variate":
            patch_num = compute_patch_num(
                configs.seq_len,
                configs.patch_len,
                stride)
        elif patch_type == "uni-variate":
            patch_num = compute_patch_num(
                configs.seq_len,
                configs.patch_len,
                stride) * configs.enc_in
        elif patch_type == "whole-variate":
            patch_num = configs.enc_in
        else:
            raise ValueError("patch_type not recognized")

        augmentations = ["mask", "channel"]  # same as default in the paper
        # Embedding
        self.enc_embedding = TestEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            configs.patch_len,
            stride,
            augmentations,
            patch_type,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "supervised":
            self.projection = nn.Linear(
                configs.d_model * patch_num,
                configs.num_class
            )

    def supervised(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # (batch_size, patch_num * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised":
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        else:
            raise ValueError("Task name not recognized or not implemented within the TestFormer model")
