import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import torch
import numpy as np
from data.pdnet_dataset import PDNetDataset
from modules.embedding import EmbeddingLayer
from modules.light_omega_block import LightOmegaBlock
from modules.distance_predictor import DistancePredictor
from torch.utils.data import DataLoader
from tqdm import tqdm

# === Configuration ===
test_list_file = "data/test.txt"
batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation de l'appareil : {device}")

# === Dataset ===
test_dataset = PDNetDataset(data_dir="data/pdnet", file_list="data/test.txt")
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === Modèles ===
embedder = EmbeddingLayer(device=device)  # Pas de load_state_dict ici
omega = LightOmegaBlock().to(device)
predictor = DistancePredictor().to(device)

# === Chargement des checkpoints (sauf embedder)
ckpt_dir = "checkpoints"
omega_path = os.path.join(ckpt_dir, "omega.pt")
pred_path = os.path.join(ckpt_dir, "predictor.pt")

if os.path.exists(omega_path) and os.path.exists(pred_path):
    omega.load_state_dict(torch.load(omega_path, map_location=device))
    predictor.load_state_dict(torch.load(pred_path, map_location=device))
    print("[INFO] Checkpoints chargés.")
else:
    print("[WARNING] Checkpoints manquants. L'évaluation peut être incohérente.")

# === Évaluation ===
total_loss = 0.0
num_samples = 0

criterion = torch.nn.L1Loss()  # MAE

embedder.model.eval()
omega.eval()
predictor.eval()

with torch.no_grad():
    for sequences, true_dists in tqdm(test_loader, desc="Évaluation"):
        for seq, true_dist in zip(sequences, true_dists):
            embeddings = embedder.get_embeddings(seq).unsqueeze(0).to(device)  # [1, L, D]
            x = omega(embeddings)  # [1, L, D]
            pred_dist = predictor(x).squeeze(0).cpu()  # [L, L]
            loss = criterion(pred_dist, true_dist)
            total_loss += loss.item()
            num_samples += 1

# === Résultat ===
avg_loss = total_loss / num_samples
print(f"\n✅ Erreur MAE moyenne sur le jeu de test : {avg_loss:.4f}")
