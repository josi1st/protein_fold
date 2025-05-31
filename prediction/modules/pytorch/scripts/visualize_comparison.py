import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.pdnet_dataset import PDNetDataset
from modules.embedding import EmbeddingLayer
from modules.light_omega_block import LightOmegaBlock
from modules.distance_predictor import DistancePredictor

# Argument pour sélectionner l'échantillon
parser = argparse.ArgumentParser(description="Compare predicted and real distograms.")
parser.add_argument('--index', type=int, default=0, help="Index of the sample in the dataset")
args = parser.parse_args()

# === Configuration ===
data_dir = "data/pdnet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Utilisation de l'appareil : {device}")

# === Chargement du dataset ===
dataset = PDNetDataset(data_dir)
if args.index >= len(dataset):
    print(f"[ERREUR] L'index {args.index} est hors limites. Le dataset contient {len(dataset)} échantillons.")
    sys.exit(1)

sequence, true_dist = dataset[args.index]  # [sequence str, distogramme torch.tensor [L, L, num_bins] ou [L, L]]
print(f"[DEBUG] Longueur de la séquence : {len(sequence)}")
print(f"[DEBUG] Forme de true_dist : {true_dist.shape}")

# Vérifier si la séquence est vide
if not sequence or len(sequence) == 0:
    print(f"[ERREUR] La séquence à l'index {args.index} est vide.")
    sys.exit(1)

# === Chargement du modèle ===
checkpoint_dir = "checkpoints"
embedder = EmbeddingLayer(device=device)
omega = LightOmegaBlock().to(device)
predictor = DistancePredictor().to(device)

# Charger les poids entraînés
embedder_ckpt = os.path.join(checkpoint_dir, "embedder.pth")
omega_ckpt = os.path.join(checkpoint_dir, "omega.pth")
predictor_ckpt = os.path.join(checkpoint_dir, "predictor.pth")

if all(os.path.exists(p) for p in [embedder_ckpt, omega_ckpt, predictor_ckpt]):
    embedder.model.load_state_dict(torch.load(embedder_ckpt, map_location=device))
    omega.load_state_dict(torch.load(omega_ckpt, map_location=device))
    predictor.load_state_dict(torch.load(predictor_ckpt, map_location=device))
    print("[INFO] Modèles chargés depuis les checkpoints.")
else:
    print("[ERREUR] Au moins un fichier de checkpoint est manquant dans 'checkpoints'.")
    sys.exit(1)

# Mettre les modèles en mode évaluation
embedder.model.eval()
omega.eval()
predictor.eval()

# === Inférence ===
with torch.no_grad():
    embeddings = embedder.get_embeddings(sequence).unsqueeze(0).to(device)  # [1, L, D]
    print(f"[DEBUG] Forme des embeddings : {embeddings.shape}")
    x = omega(embeddings)  # [1, L, D]
    print(f"[DEBUG] Forme après omega : {x.shape}")
    pred_dist = predictor(x)  # [1, L, L, num_bins]
    print(f"[DEBUG] Forme de pred_dist : {pred_dist.shape}")

    # Ajuster les dimensions si nécessaire
    if pred_dist.dim() == 4:  # [1, L, L, num_bins]
        pred_dist = torch.argmax(pred_dist, dim=-1)  # [1, L, L]
    pred_dist = pred_dist.squeeze(0).cpu()  # [L, L]
    print(f"[DEBUG] Forme de pred_dist après argmax : {pred_dist.shape}")

    if true_dist.dim() == 3:  # [L, L, num_bins]
        true_dist = torch.argmax(true_dist, dim=-1)  # [L, L]
    true_dist = true_dist.cpu()
    print(f"[DEBUG] Forme de true_dist après ajustement : {true_dist.shape}")

    # Ajuster les dimensions avec min_len pour éviter les erreurs
    min_len = min(pred_dist.shape[0], true_dist.shape[0])
    if min_len == 0:
        print("[ERREUR] Dimensions nulles après ajustement.")
        sys.exit(1)
    pred_dist = pred_dist[:min_len, :min_len]
    true_dist = true_dist[:min_len, :min_len]

# === Calcul de la KL-Divergence ===
kl_div = None
if pred_dist.dim() == 2 and true_dist.dim() == 2 and pred_dist.shape == true_dist.shape:
    if pred_dist.shape[0] > 0 and pred_dist.shape[1] > 0:
        kl_div = 0
        for i in range(pred_dist.shape[0]):
            for j in range(pred_dist.shape[1]):
                kl_div += entropy([true_dist[i, j].item() + 1e-10], [pred_dist[i, j].item() + 1e-10])
        kl_div /= (pred_dist.shape[0] * pred_dist.shape[1])
        print(f"KL-Divergence moyenne : {kl_div:.4f}")
    else:
        print("[AVERTISSEMENT] Les dimensions de pred_dist sont nulles, KL-Divergence non calculée.")
else:
    print("[AVERTISSEMENT] Les dimensions de pred_dist et true_dist ne correspondent pas, KL-Divergence non calculée.")

# === Affichage comparatif ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im0 = axs[0].imshow(pred_dist.numpy(), cmap='viridis', interpolation='nearest')
axs[0].set_title("Distogramme prédit")
axs[0].set_xlabel("Résidu i")
axs[0].set_ylabel("Résidu j")
plt.colorbar(im0, ax=axs[0], label="Indice de bin")

im1 = axs[1].imshow(true_dist.numpy(), cmap='viridis', interpolation='nearest')
axs[1].set_title("Distogramme réel")
axs[1].set_xlabel("Résidu i")
axs[1].set_ylabel("Résidu j")
plt.colorbar(im1, ax=axs[1], label="Indice de bin")

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, f"comparison_distogram_index_{args.index}.png"))
print(f"[INFO] Comparaison sauvegardée dans '{checkpoint_dir}/comparison_distogram_index_{args.index}.png'.")
plt.close()

# === Calcul de l'erreur absolue ===
abs_diff = torch.abs(pred_dist - true_dist)

# === Affichage de l'erreur absolue ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im2 = axs[0].imshow(abs_diff.numpy(), cmap='hot', interpolation='nearest')
axs[0].set_title("Erreur absolue (|prédit - réel|)")
axs[0].set_xlabel("Résidu i")
axs[0].set_ylabel("Résidu j")
plt.colorbar(im2, ax=axs[0], label="Erreur")

axs[1].hist(abs_diff.flatten().numpy(), bins=50, color='orange')
axs[1].set_title("Distribution des erreurs absolues")
axs[1].set_xlabel("Erreur |d_ij_prédit - d_ij_réel|")
axs[1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig(os.path.join(checkpoint_dir, f"error_distogram_index_{args.index}.png"))
print(f"[INFO] Erreur sauvegardée dans '{checkpoint_dir}/error_distogram_index_{args.index}.png'.")
plt.close()