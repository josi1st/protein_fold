import torch
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
import os
from django.conf import settings

# Importer les modules depuis le dossier modules/
from ..modules.embedding import EmbeddingLayer
from ..modules.light_omega_block import LightOmegaBlock
from ..modules.distance_predictor import DistancePredictor

# Fonctions de conversion
def distogram_to_distances(distogram):
    bin_edges = np.linspace(0, 100, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    seq_len = distogram.shape[1]
    bin_centers_expanded = bin_centers[:, np.newaxis, np.newaxis]
    distogram = distogram.softmax(dim=-1).squeeze().detach().cpu().numpy()
    return np.sum(distogram * bin_centers_expanded, axis=0)

def distance_to_coordinates(distance_matrix):
    L = distance_matrix.shape[0]
    H = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * H @ (distance_matrix ** 2) @ H
    eigvals, eigvecs = eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    top_evecs = np.sqrt(np.maximum(eigvals[:3], 0))
    return eigvecs[:, :3] @ np.diag(top_evecs)

def coords_to_pdb(coords, sequence, output_file):
    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")
    for i, (res_name, coord) in enumerate(zip(sequence, coords)):
        res = Residue.Residue((" ", i + 1, " "), res_name, " ")
        atom = Atom.Atom("CA", coord, 0.0, 1.00, " ", "CA", i + 1, element="C")
        res.add(atom)
        chain.add(res)
    model.add(chain)
    structure.add(model)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
    return output_file

def predict_structure(sequence):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = EmbeddingLayer(device=device)
    omega = LightOmegaBlock().to(device)
    predictor = DistancePredictor().to(device)

    # Charger les checkpoints avec map_location pour gérer CPU/GPU
    omega_checkpoint = os.path.join(settings.BASE_DIR, 'prediction', 'checkpoints', 'omega.pth')
    predictor_checkpoint = os.path.join(settings.BASE_DIR, 'prediction', 'checkpoints', 'predictor.pth')
    omega.load_state_dict(torch.load(omega_checkpoint, map_location=device))
    predictor.load_state_dict(torch.load(predictor_checkpoint, map_location=device))
    omega.eval()
    predictor.eval()

    # Ajuster la longueur de la séquence
    if len(sequence) < 50:
        sequence += "X" * (50 - len(sequence))

    # Prédire
    embeddings = embedder.get_embeddings(sequence).unsqueeze(0).to(device)
    x = omega(embeddings)
    pred_distogram = predictor(x).permute(0, 3, 1, 2)
    distance_map = distogram_to_distances(pred_distogram)
    coords = distance_to_coordinates(distance_map)

    # Générer le fichier .pdb
    output_filename = f"predictions/prediction_{hash(sequence)}.pdb"
    output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
    coords_to_pdb(coords, sequence, output_path)
    return output_filename