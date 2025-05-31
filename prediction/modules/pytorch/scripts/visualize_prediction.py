import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from modules.embedding import EmbeddingLayer
from modules.light_omega_block import LightOmegaBlock
from modules.distance_predictor import DistancePredictor

# Argument pour la séquence (optionnel)
parser = argparse.ArgumentParser(description="Visualize predicted distogram for a protein sequence.")
parser.add_argument('--sequence', type=str, default="MGDVEKGKKIFIMKCSQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGYSYTAANKNKGIIWG",
                    help="Protein sequence to predict distogram for")
args = parser.parse_args()

sequence = args.sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation de l'appareil : {device}")

# === Chargement du modèle entraîné ===
checkpoint_dir = "checkpoints"
embedder = EmbeddingLayer(device=device)
omega = LightOmegaBlock().to(device)
predictor = DistancePredictor().to(device)

# Charger les poids entraînés
embedder_ckpt = os.path.join(checkpoint_dir, "embedder.pth")
omega_ckpt = os.path.join(checkpoint_dir, "omega.pth")
predictor_ckpt = os.path.join(checkpoint_dir, "predictor.pth")

if os.path.exists(embedder_ckpt):
    embedder.model.load_state_dict(torch.load(embedder_ckpt))
else:
    print("[ERREUR] Checkpoint 'embedder.pth' non trouvé dans 'checkpoints'.")
    sys.exit(1)

if os.path.exists(omega_ckpt):
    omega.load_state_dict(torch.load(omega_ckpt))
else:
    print("[ERREUR] Checkpoint 'omega.pth' non trouvé dans 'checkpoints'.")
    sys.exit(1)

if os.path.exists(predictor_ckpt):
    predictor.load_state_dict(torch.load(predictor_ckpt))
else:
    print("[ERREUR] Checkpoint 'predictor.pth' non trouvé dans 'checkpoints'.")
    sys.exit(1)

print("[INFO] Modèles chargés avec succès.")

# Mettre les modèles en mode évaluation
embedder.model.eval()
omega.eval()
predictor.eval()

# === Inférence ===
with torch.no_grad():
    embeddings = embedder.get_embeddings(sequence)  # [L, D]
    embeddings = embeddings.unsqueeze(0).to(device)  # [1, L, D]
    x = omega(embeddings)  # [1, L, D]
    pred_dist = predictor(x)  # [1, L, L, num_bins]

    # Si discrétisé, prendre l'indice du bin le plus probable
    if pred_dist.dim() == 4:  # [1, L, L, num_bins]
        pred_dist = torch.argmax(pred_dist, dim=-1)  # [1, L, L]
    distogram = pred_dist.squeeze(0).cpu().numpy()  # [L, L]

# === Visualisation ===
plt.figure(figsize=(6, 5))
plt.imshow(distogram, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Bin de distance (indice)')
plt.title(f"Prédiction du distogramme\nSéquence: {sequence[:20]}...")
plt.xlabel("Résidu i")
plt.ylabel("Résidu j")
plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, "predicted_distogram.png"))
print(f"[INFO] Distogramme sauvegardé dans '{checkpoint_dir}/predicted_distogram.png'.")
plt.close()