import torch
import torch.nn as nn

class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 4096, 512)) # 8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings


class MLPPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(MLPPositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(3, 256, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, embedding_dim, kernel_size=1, bias=True)
        )

    @staticmethod
    def get_position_tensor(h, w, d, device):
        h_range = 2 * torch.arange(h, device=device) / (h - 1) - 1
        w_range = 2 * torch.arange(w, device=device) / (w - 1) - 1
        d_range = 2 * torch.arange(d, device=device) / (d - 1) - 1

        h_range = h_range[:, None, None].expand(-1, w, d)
        w_range = w_range[None, :, None].expand(h, -1, d)
        d_range = d_range[None, None, :].expand(h, w, -1)

        return torch.stack([h_range, w_range, d_range], dim=0).unsqueeze(0)

    def forward(self, x):
        b, c, h, w, d = x.size()
        device = x.device

        pos = self.get_position_tensor(h, w, d, device)
        pos_encoding = self.mlp(pos)
        return x + pos_encoding
