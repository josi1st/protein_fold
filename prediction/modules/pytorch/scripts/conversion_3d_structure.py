import sys
import os
import torch
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom

# Ajouter les chemins pour accéder aux modules dans src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importer les classes nécessaires
from modules.embedding import EmbeddingLayer
from modules.light_omega_block import LightOmegaBlock
from modules.distance_predictor import DistancePredictor


def distogram_to_distances(distogram):
    bin_edges = np.linspace(0, 100, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Redimensionner bin_centers pour correspondre à la forme de distogram
    seq_len = distogram.shape[1]  # Taille de la séquence
    bin_centers_expanded = bin_centers[:, np.newaxis, np.newaxis]  # Forme (10, 1, 1)
    distogram = distogram.softmax(dim=-1).squeeze().detach().cpu().numpy()
    # Multiplication avec broadcasting
    distances = np.sum(distogram * bin_centers_expanded, axis=0)  # Somme sur l'axe des bins (0)
    return distances

def distance_to_coordinates(distance_matrix):
    L = distance_matrix.shape[0]
    H = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * H @ (distance_matrix ** 2) @ H
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    top_evecs = np.sqrt(np.maximum(eigvals[:3], 0))
    coords = eigvecs[:, :3] @ np.diag(top_evecs)
    return coords

def coords_to_pdb(coords, sequence, output_file="predicted_structure.pdb"):
    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    for i, (res_name, coord) in enumerate(zip(sequence, coords)):
        res = Residue.Residue((" ", i + 1, " "), res_name, " ")
        atom = Atom.Atom("CA", coord, 0.0, 1.0, " ", "CA", i + 1, element="C")
        res.add(atom)
        chain.add(res)
    model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

# Charger le modèle et prédire
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder = EmbeddingLayer(device=device)
omega = LightOmegaBlock().to(device)
predictor = DistancePredictor().to(device)
omega.load_state_dict(torch.load("checkpoints/omega.pth"))
predictor.load_state_dict(torch.load("checkpoints/predictor.pth"))
omega.eval()
predictor.eval()

# Exemple : utiliser la séquence correspondante au distogramme
sequence = "MKTFFVAVLTLAFASASSSVNQKAAQKAAKDVAAWTLKAAAGGNVVTVTVS"  # À ajuster à 200 résidus
embeddings = embedder.get_embeddings(sequence).unsqueeze(0).to(device)
x = omega(embeddings)
pred_distogram = predictor(x).permute(0, 3, 1, 2)
distance_map = distogram_to_distances(pred_distogram)
coords = distance_to_coordinates(distance_map)
coords_to_pdb(coords, sequence, "predicted_structure.pdb")
print("Structure 3D sauvegardée dans predicted_structure.pdb")