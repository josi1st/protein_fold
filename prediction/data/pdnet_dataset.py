import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PDNetDataset(Dataset):
    def __init__(self, data_dir, file_list=None):
        self.samples = []
        max_length = 1500  # Nouvelle limite temporaire

        if file_list is not None and os.path.isfile(file_list):
            print(f"[INFO] Chargement depuis la liste : {file_list}")
            with open(file_list, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]

            for fname in filenames:
                path = os.path.join(data_dir, fname)
                if os.path.exists(path):
                    try:
                        data = np.load(path)
                        if "sequence" in data and "dist_matrix" in data:
                            sequence_length = len(str(data["sequence"]))
                            if sequence_length <= max_length:  # Filtrer les séquences trop longues
                                self.samples.append(path)
                            else:
                                print(f"[INFO] Séquence trop longue ({sequence_length}), ignorée : {fname}")
                        else:
                            print(f"[WARN] Champs manquants dans : {fname}")
                    except Exception as e:
                        print(f"[ERREUR] Lecture échouée pour {fname} : {e}")
                else:
                    print(f"[ERREUR] Fichier introuvable : {path}")

        else:
            print(f"[INFO] Chargement direct depuis le dossier : {data_dir}")
            for fname in os.listdir(data_dir):
                if fname.endswith(".npz"):
                    path = os.path.join(data_dir, fname)
                    try:
                        data = np.load(path)
                        if "sequence" in data and "dist_matrix" in data:
                            sequence_length = len(str(data["sequence"]))
                            if sequence_length <= max_length:  # Filtrer les séquences trop longues
                                self.samples.append(path)
                            else:
                                print(f"[INFO] Séquence trop longue ({sequence_length}), ignorée : {fname}")
                    except Exception as e:
                        print(f"[ERREUR] Lecture échouée pour {fname} : {e}")

        print(f"[INFO] Total des échantillons chargés : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)
        sequence = str(data["sequence"])
        dist_matrix = torch.tensor(data["dist_matrix"], dtype=torch.float32)

        if dist_matrix.dim() == 3 and dist_matrix.shape[-1] == 10:
            dist_matrix_binned = dist_matrix
        else:
            dist_matrix = torch.exp(dist_matrix) - 1
            if torch.any(torch.isinf(dist_matrix)) or torch.any(torch.isnan(dist_matrix)):
                print(f"[WARN] Valeurs infinies ou NaN détectées dans {path}. Remplacement par 0.")
                dist_matrix = torch.where(torch.isinf(dist_matrix) | torch.isnan(dist_matrix), torch.tensor(0.0), dist_matrix)
            max_distance = 100.0
            dist_matrix = torch.clamp(dist_matrix, min=0.0, max=max_distance)

            bins = torch.linspace(0, max_distance, 11)[:-1]
            dist_matrix_binned = torch.bucketize(dist_matrix, bins, right=True).long()
            if torch.any(dist_matrix_binned >= 10):
                print(f"[WARN] Indices > 9 détectés dans {path}. Clipping à 9.")
                dist_matrix_binned = torch.clamp(dist_matrix_binned, max=9)
            dist_matrix_binned = torch.nn.functional.one_hot(dist_matrix_binned, num_classes=10).float()

        return sequence, dist_matrix_binned