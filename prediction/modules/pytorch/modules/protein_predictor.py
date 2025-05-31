import torch
import torch.nn as nn
from src.models.embedding import EmbeddingLayer
from src.models.light_omega_block import LightOmegaBlock
from src.models.structure_module import StructureModule

class ProteinPredictor(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=4, hidden_dim=512, seq_len=50, dropout=0.1):
        super(ProteinPredictor, self).__init__()
        
        # Modules
        self.embedding_layer = EmbeddingLayer(embed_dim=embed_dim)
        self.light_omega_block = LightOmegaBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
        self.structure_module = StructureModule(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, sequence):
        # Phase 1: Embedding de la séquence
        embeddings = self.embedding_layer(sequence)
        
        # Phase 2: Passage à travers le LightOmegaBlock
        x = self.light_omega_block(embeddings)
        
        # Phase 3: Prédiction de la structure (carte de distances)
        distances = self.structure_module(x)
        
        return distances
