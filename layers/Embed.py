import copy
import math
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils import weight_norm

from layers.Augmentation import get_augmentation
from data_provider.uea import bandpass_filter_func


# 2-d relative coordinates for 19 channels. We define position from left to right, top to bottom.
# Note that channels T3, T4, T5, T6 in old system are the same channels as T7, T8, P7, P8 in new system.
# 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3/T7', 'C3', 'Cz', 'C4', 'T4/T8',
# 'T5/P7', 'P3', 'Pz', 'P4', 'T6/P8', 'O1', 'O2'

CHANNEL_RELATIVE_COORDINATES = {
    "Fp1": (2, 1), "Fp2": (4, 1),
    "F7": (1, 2), "F3": (2, 2), "Fz": (3, 2), "F4": (4, 2), "F8": (5, 2),
    "T3": (1, 3), "C3": (2, 3), "Cz": (3, 3), "C4": (4, 3), "T4": (5, 3),
    "T5": (1, 4), "P3": (2, 4), "Pz": (3, 4), "P4": (4, 4), "T6": (5, 4),
    "O1": (2, 5), "O2": (4, 5),
}


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        sin_part = torch.sin(position * div_term)
        cos_part = torch.cos(position * div_term)

        pe[:, 0::2] = sin_part[:, :pe[:, 0::2].shape[1]]
        pe[:, 1::2] = cos_part[:, :pe[:, 1::2].shape[1]]

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class ChannelPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(ChannelPositionalEmbedding, self).__init__()
        if (d_model // 2) % 2 != 0:
            raise ValueError("d_model must be an even number for 2-D channel positional embedding.")
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, (d_model // 2)).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, (d_model // 2), 2).float() * -(math.log(10000.0) / (d_model // 2))
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        coordinates = torch.tensor(list(CHANNEL_RELATIVE_COORDINATES.values())).to(x.device)
        x_axis = self.pe[:, coordinates[:, 0].long()]
        y_axis = self.pe[:, coordinates[:, 1].long()]
        return torch.cat([x_axis, y_axis], dim=-1)


class TokenEmbedding(nn.Module):  # (batch_size, seq_len, enc_in)
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)  # c_in is seq_length here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_length)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)  # (batch_size, enc_in, d_model)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class ShallowNetEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout):
        super().__init__()

        self.shallow_net = nn.Sequential(
            nn.Conv2d(1, d_model, (1, 25), (1, 1)),
            nn.Conv2d(d_model, d_model, (c_in, 1), (1, 1)),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 4), (1, 2)),
            nn.Dropout(dropout),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, 1), stride=(1, 1)),
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1).unsqueeze(1)  # Shape becomes (B, 1, C, T)
        x = self.shallow_net(x)
        x = self.projection(x)
        # Rearrange the output to match the Transformer input format (B, patch_num, d_model)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class CrossChannelTokenEmbedding(nn.Module):  # (batch_size, 1, enc_in, seq_len)
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )

    def forward(self, x):
        x = self.tokenConv(x)
        return x  # (batch_size, d_model, 1, patch_num)


class UpDimensionChannelEmbedding(nn.Module):  # B x C x T
    def __init__(self, c_in, t_in, u_dim, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.u_dim = u_dim
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=u_dim,
            kernel_size=3,
            padding=padding,
            bias=False,
        )
        self.fc = nn.Linear(t_in, d_model)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)  # B x u_dim x T
        x = self.fc(x)  # B x u_dim x d_model
        return x


class MedformerEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len_list,
        stride_list,
        dropout,
        augmentation=["none"],
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.enc_in = enc_in
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]

        linear_layers = [
            CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding_t = PositionalEmbedding(d_model=d_model)
        self.position_embedding_c = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation])
        self.routers = nn.ParameterList(
            [nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for _ in self.patch_len_list])
        self.learnable_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in self.patch_len_list])

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)

        x_list = []
        for padding, value_embedding, router in zip(self.paddings, self.value_embeddings, self.routers):
            x_copy = x.clone()
            # per granularity augmentation
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x_new = self.augmentation[aug_idx](x_copy)
            # add positional embedding to tag each channel
            x_new = x_new + self.position_embedding_c(x_new)
            # temporal dimension
            x_new = padding(x_new).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new = value_embedding(x_new)  # (batch_size, d_model, 1, patch_num)
            x_new = x_new.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            # append router
            router = router.expand(x_new.size(0), 1, x_new.size(-1))  # (B,1,D)
            x_new = torch.cat([x_new, router], dim=1)  # (batch_size, patch_num+1, d_model)
            x_list.append(x_new)

        x = [
            x + cxt + self.position_embedding_t(x)
            for x, cxt in zip(x_list, self.learnable_embeddings)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        return x


class TestEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        seq_len,
        d_model,
        patch_len,
        stride,
        augmentation=["none"],
        patch_type="multi-variate",
    ):
        super().__init__()
        self.enc_in = enc_in
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.patch_type = patch_type

        if patch_type == "multi-variate":
            self.value_embedding = CrossChannelTokenEmbedding(
                c_in=enc_in,
                l_patch=patch_len,
                d_model=d_model,
                stride=stride,)
        elif patch_type == "uni-variate":
            # Uni-variate patch embedding
            self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
            nn.init.xavier_uniform_(self.value_embedding.weight)
        elif patch_type == "whole-variate":
            # Whole-variate patch embedding
            self.value_embedding = nn.Linear(seq_len, d_model)
            nn.init.xavier_uniform_(self.value_embedding.weight)
        # Positional encoding
        self.position_embedding = PositionalEmbedding(d_model)
        # Channel identity token (not dependent on seq_len)
        self.channel_token = nn.Parameter(torch.randn(1, enc_in, 1) * 0.02)
        # Data augmentation modules
        self.augmentation = nn.ModuleList([get_augmentation(aug) for aug in augmentation])

    def _pad_to_stride(self, x):
        """Pad the input so that unfolding covers the sequence evenly."""
        L = x.size(-1)
        if L < self.patch_len:
            pad_right = self.patch_len - L
        else:
            remainder = (L - self.patch_len) % self.stride
            pad_right = 0 if remainder == 0 else (self.stride - remainder)
        return F.pad(x, (0, pad_right), mode='replicate')

    def forward(self, x):
        """Forward pass
        Uni-variate: (B, seq_len, enc_in) -> (B, enc_in * patch_num, d_model).
        Multi-variate: (B, seq_len, enc_in) -> (B, patch_num, d_model).
        Whole-variate: (B, seq_len, enc_in) -> (B, enc_in, d_model)."""
        # Change to (B, C, L)
        x = x.permute(0, 2, 1).contiguous()

        # Apply augmentation only during training
        if self.training and len(self.augmentation) > 0:
            aug_idx = random.randint(0, len(self.augmentation) - 1)
            x = self.augmentation[aug_idx](x)

        # Add channel identity
        x = x + self.channel_token

        if self.patch_type == "uni-variate":
            # Dynamic padding
            x = self._pad_to_stride(x)
            # Unfold patches: (B, C, N, patch_len)
            x = x.unfold(-1, self.patch_len, self.stride)
            B, C, N, _ = x.shape
            # Linear projection: ((B*C), N, D)
            x = rearrange(x, 'b c n l -> (b c) n l')
            x = self.value_embedding(x)
            x = x + self.position_embedding(x)
            # Merge channels: (B, C*N, D)
            x = rearrange(x, '(b c) n d -> b (c n) d', b=B, c=C)
        elif self.patch_type == "whole-variate":
            x = self.value_embedding(x)
        elif self.patch_type == "multi-variate":
            x = self._pad_to_stride(x).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x = self.value_embedding(x)  # (batch_size, d_model, 1, patch_num)
            x = x.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)

        return x


class MultiResolutionData(nn.Module):
    def __init__(self, enc_in, resolution_list, stride_list):
        super().__init__()
        self.paddings = nn.ModuleList([nn.ReplicationPad1d((0, stride)) for stride in stride_list])

        self.multi_res = nn.ModuleList([
            nn.Conv1d(
                in_channels=enc_in,
                out_channels=enc_in,
                kernel_size=res,
                stride=res,
                padding=0,
                padding_mode='circular')
            for res in resolution_list
        ])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_list = []
        for l in range(len(self.multi_res)):
            out = self.paddings[l](x)
            out = self.multi_res[l](out)
            x_list.append(out)
        return x_list


class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model, res_len, augmentation=["none"]):
        super().__init__()
        self.d_model = d_model
        self.embeddings = nn.ModuleList([
            nn.Linear(int(res/2)+1, int(self.d_model/2)+1).to(torch.cfloat)
            for res in res_len
        ])

        self.augmentation = nn.ModuleList(
            [get_augmentation(aug) for aug in augmentation]
        )

    def forward(self, x_list):
        x_out = []
        for l in range(len(x_list)):
            x = torch.fft.rfft(x_list[l], dim=-1)
            out = self.embeddings[l](x)
            out = torch.fft.irfft(out, dim=-1, n=self.d_model)

            aug_idx = random.randint(0, len(self.augmentation) - 1)
            out = self.augmentation[aug_idx](out)
            x_out.append(out)

        return x_out


