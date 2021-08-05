import torch
import torch.nn as nn


class MLPPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(MLPPositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv3d(3, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
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
