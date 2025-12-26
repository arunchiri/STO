import torch
from torch import nn

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel-1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class TCNForecastModel(nn.Module):
    def __init__(self, meta):
        super().__init__()

        self.store_emb = nn.Embedding(meta["n_stores"], 16)
        self.cluster_emb = nn.Embedding(meta["n_clusters"], 8)
        self.family_emb = nn.Embedding(meta["n_families"], 32)

        self.tcn = nn.Sequential(
            TemporalBlock(meta["seq_features"], 64),
            TemporalBlock(64, 64),
            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(
            nn.Linear(64 + 56, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        # temporal: [B, T, F] → [B, F, T]
        x = batch["seq"].transpose(1, 2)
        tcn_out = self.tcn(x).squeeze(-1)

        emb = torch.cat([
            self.store_emb(batch["store"]),
            self.cluster_emb(batch["cluster"]),
            self.family_emb(batch["family"]),
        ], dim=1)

        return self.head(torch.cat([tcn_out, emb], dim=1)).squeeze(1)
