import torch
import torch.nn as nn
from models.TransBTS_optimized.IntmdSequential import IntermediateSequential


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv3d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)

        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, C, H, W, D = x.shape
        q, k, v = (layer(tensor).view(B, self.head_dim, self.num_heads, -1) for layer, tensor in
                   zip((self.q, self.k, self.v), (x, x, x)))

        attn = torch.einsum('bchn,bchm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum('bhnm,bchm->bchn', attn, v).contiguous()
        x = x.view(B, C, H, W, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        x = self.norm(x)
        return self.dropout(self.fn(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv3d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
