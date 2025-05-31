import torch
import torch.nn as nn

class StructureModule(nn.Module):
    def __init__(self, embed_dim=1280, hidden_dim=512, dropout=0.1):
        super(StructureModule, self).__init__()
        
        # Réseau pour prédire les distances
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
        # Couches de normalisation pour la stabilité du modèle
        self.layernorm = nn.LayerNorm(embed_dim)
        
        # Pour prédire la carte de distances entre tous les résidus
        self.distance_layer = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        
        # Passer par un réseau fully connected
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layernorm(x)
        
        # Calculer les distances entre les résidus
        distances = self.distance_layer(x)  # shape: [batch, seq_len, 1]
        
        # Appliquer un dropout
        distances = self.dropout(distances)
        
        # Retourner la carte de distances
        return distances
