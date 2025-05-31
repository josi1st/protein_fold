import torch
import torch.nn as nn

class LightOmegaBlock(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=4, hidden_dim=512, dropout=0.2):
        super(LightOmegaBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        x_norm = self.norm1(x)
        attn_output1, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + attn_output1  # Residual connection

        x_norm = self.norm2(x)
        attn_output2, _ = self.attn2(x_norm, x_norm, x_norm)
        x = x + attn_output2  # Residual connection

        x_norm = self.norm3(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output  # Residual connection
        return x