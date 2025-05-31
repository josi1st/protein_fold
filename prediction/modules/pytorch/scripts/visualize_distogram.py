import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Charger un fichier .npz spécifique
sample_file = "data/pdnet/1ubq.npz"  # à adapter selon ton fichier
data = np.load(sample_file)

# Extraire la carte de distances
true_dist = data["dist_matrix"]

# Afficher
plt.figure(figsize=(6, 5))
plt.imshow(true_dist, cmap='viridis')
plt.colorbar(label='Distance (Å)')
plt.title("Carte de distances réelle - 1ubq")
plt.xlabel("Résidus")
plt.ylabel("Résidus")
plt.tight_layout()
plt.show()
