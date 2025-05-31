import torch
import torch.nn as nn

class DistancePredictor(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=256, num_bins=10):
        super(DistancePredictor, self).__init__()
        self.linear_i = nn.Linear(embed_dim, hidden_dim)
        self.linear_j = nn.Linear(embed_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins)  # Prédire une distribution sur num_bins
        )

    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape

        # Transformer chaque vecteur en un espace plus petit
        x_i = self.linear_i(x)       # [B, L, H]
        x_j = self.linear_j(x)       # [B, L, H]

        # Calculer toutes les paires (broadcast)
        x_i = x_i.unsqueeze(2)       # [B, L, 1, H]
        x_j = x_j.unsqueeze(1)       # [B, 1, L, H]

        pair = x_i * x_j             # [B, L, L, H]

        dist_logits = self.output(pair)  # [B, L, L, num_bins]
        return dist_logits