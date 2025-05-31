import sys
import os
import torch
import csv
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Chemins
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data.pdnet_dataset import PDNetDataset
from modules.embedding import EmbeddingLayer
from modules.light_omega_block import LightOmegaBlock
from modules.distance_predictor import DistancePredictor

# Fonction de padding personnalisée
def collate_fn(batch):
    sequences, distograms = zip(*batch)
    max_seq_len = max(len(seq) for seq in sequences)
    padded_distograms = []
    padded_sequences = []
    for seq, distogram in batch:
        curr_len1, curr_len2, num_bins = distogram.shape
        pad1 = max_seq_len - curr_len1
        pad2 = max_seq_len - curr_len2
        padded_distogram = torch.nn.functional.pad(distogram, (0, 0, 0, pad2, 0, pad1))
        padded_distograms.append(padded_distogram)
        padded_sequences.append(seq)
    padded_distograms = torch.stack(padded_distograms)
    return padded_sequences, padded_distograms

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(f"[INFO] Utilisation de l'appareil : {device}")
data_dir = "data/pdnet"
batch_size = 1  # Gardé à 1 pour éviter les erreurs de mémoire
num_epochs = 20
lr = 5e-5
save_path = "checkpoints"
os.makedirs(save_path, exist_ok=True)

# Fichier CSV pour journaliser les pertes
csv_path = os.path.join(save_path, "loss_log.csv")
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["epoch", "train_loss", "val_loss"])

# === Chargement des datasets ===
train_list = "data/train.txt"
val_list = "data/test.txt"
train_dataset = PDNetDataset(data_dir, train_list)
val_dataset = PDNetDataset(data_dir, val_list)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# === Modules du modèle ===
embedder = EmbeddingLayer(device=device)
omega = LightOmegaBlock().to(device)
predictor = DistancePredictor().to(device)

# Charger les checkpoints existants avec gestion d'erreurs
omega_ckpt = os.path.join(save_path, "omega.pt")
predictor_ckpt = os.path.join(save_path, "predictor.pt")
embedder_ckpt = os.path.join(save_path, "embedder.pth")
try:
    if os.path.exists(omega_ckpt):
        omega.load_state_dict(torch.load(omega_ckpt, map_location=device))
    if os.path.exists(predictor_ckpt):
        predictor.load_state_dict(torch.load(predictor_ckpt, map_location=device))
    if os.path.exists(embedder_ckpt):
        embedder.model.load_state_dict(torch.load(embedder_ckpt))
    print("[INFO] Checkpoints chargés si disponibles.")
except RuntimeError as e:
    print(f"[WARN] Échec du chargement des checkpoints : {e}. Entraînement à partir de zéro.")

# === Fine-tuning : désactiver les gradients pour le modèle ESM sauf les 2 derniers blocs ===
for name, param in embedder.model.named_parameters():
    if not (name.startswith("encoder.layer.31") or name.startswith("encoder.layer.32")):
        param.requires_grad = False

# === Fonction de perte et optimiseur ===
params_to_optimize = list(filter(lambda p: p.requires_grad, embedder.model.parameters()))
params_to_optimize += list(omega.parameters()) + list(predictor.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# === Entraînement avec Early Stopping ===
loss_history = []
val_loss_history = []
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0
    omega.train()
    predictor.train()
    embedder.model.train()

    for sequences, true_distogram in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        true_distogram = true_distogram.to(device)
        max_len = true_distogram.shape[1]  # Longueur après padding

        # Générer embeddings et padder à max_len
        embeddings_list = []
        for seq in sequences:
            embed = embedder.get_embeddings(seq)
            curr_len = embed.shape[0]
            if curr_len < max_len:
                padding = (0, 0, 0, max_len - curr_len)
                embed = torch.nn.functional.pad(embed, padding, value=0.0)
            elif curr_len > max_len:
                embed = embed[:max_len, :]
            embeddings_list.append(embed)
        embeddings = torch.stack(embeddings_list).to(device)

        # Vérification des valeurs aberrantes
        if torch.any(torch.isnan(embeddings)) or torch.any(torch.isinf(embeddings)):
            print("[WARN] NaN ou Inf détecté dans les embeddings, échantillon ignoré.")
            continue

        # Appel direct sans checkpoint_sequential pour l'instant
        x = omega(embeddings)
        pred_dist = predictor(x)

        if torch.any(torch.isnan(pred_dist)) or torch.any(torch.isinf(pred_dist)):
            print("[WARN] NaN ou Inf détecté dans les prédictions, échantillon ignoré.")
            continue

        # Ajustement dimensions (sans troncage)
        pred_dist = pred_dist.permute(0, 3, 1, 2)
        true_distogram_labels = torch.argmax(true_distogram, dim=-1)
        loss = criterion(pred_dist, true_distogram_labels)

        if torch.isnan(loss):
            print("[ERREUR] La perte est NaN. Arrêt de l'entraînement.")
            exit()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=5.0)
        optimizer.step()
        torch.cuda.empty_cache()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[EPOCH {epoch+1}] Train Loss: {avg_loss:.6f}")
    loss_history.append(avg_loss)

    # Validation
    val_loss = 0.0
    omega.eval()
    predictor.eval()
    embedder.model.eval()
    with torch.no_grad():
        for sequences, true_distogram in val_dataloader:
            true_distogram = true_distogram.to(device)
            max_len = true_distogram.shape[1]

            embeddings_list = []
            for seq in sequences:
                embed = embedder.get_embeddings(seq)
                curr_len = embed.shape[0]
                if curr_len < max_len:
                    padding = (0, 0, 0, max_len - curr_len)
                    embed = torch.nn.functional.pad(embed, padding, value=0.0)
                elif curr_len > max_len:
                    embed = embed[:max_len, :]
                embeddings_list.append(embed)
            embeddings = torch.stack(embeddings_list).to(device)

            x = omega(embeddings)
            pred_dist = predictor(x)

            pred_dist = pred_dist.permute(0, 3, 1, 2)
            true_distogram_labels = torch.argmax(true_distogram, dim=-1)
            loss = criterion(pred_dist, true_distogram_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"[EPOCH {epoch+1}] Validation Loss: {avg_val_loss:.6f}")
    val_loss_history.append(avg_val_loss)

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(omega.state_dict(), omega_ckpt)
        torch.save(predictor.state_dict(), predictor_ckpt)
        torch.save(embedder.model.state_dict(), embedder_ckpt)
        print(f"[INFO] Meilleur modèle sauvegardé à l'époque {epoch+1}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    scheduler.step(avg_val_loss)

    # Journalisation CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, avg_loss, avg_val_loss])

# Sauvegarde finale
torch.save(omega.state_dict(), os.path.join(save_path, "omega.pth"))
torch.save(predictor.state_dict(), os.path.join(save_path, "predictor.pth"))
torch.save(embedder.model.state_dict(), os.path.join(save_path, "embedder.pth"))
print(f"[INFO] Modèles sauvegardés dans le dossier '{save_path}'")

# Tracer les courbes d'apprentissage
plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss')
plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
plt.xlabel("Époque")
plt.ylabel("Perte moyenne")
plt.title("Courbe d'apprentissage")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(save_path, "learning_curve.png"))
print("[INFO] Courbe d'apprentissage sauvegardée.")