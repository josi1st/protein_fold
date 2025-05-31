# src/models/embedding.py

import torch
from transformers import AutoTokenizer, AutoModel

class EmbeddingLayer:
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device=None):
        """
        Initialise le modèle ESM2 pour encoder les séquences.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Utilisation de l'appareil : {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # mode évaluation

    def get_embeddings(self, sequence: str) -> torch.Tensor:
        """
        Retourne un tenseur de forme (L, D), où L = longueur de la séquence
        et D = dimension de l'embedding (par défaut 1280).
        """
        if not sequence.startswith(" "):
            sequence = " " + sequence  # ESM attend un espace en début de séquence

        inputs = self.tokenizer(sequence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # On récupère les embeddings du dernier layer
        token_embeddings = outputs.last_hidden_state[0]  # (L+2, D)
        # On ignore le token de CLS (0) et de EOS (-1)
        residue_embeddings = token_embeddings[1:-1]  # (L, D)

        return residue_embeddings
