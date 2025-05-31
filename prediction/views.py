from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from .models import ProteinPrediction, ModelInstallationPack, DownloadLog
from django.contrib.auth.models import User
from django.conf import settings
import os
from .modules.pytorch.scripts.predict_structure import predict_structure
import zipfile

def is_admin(user):
    """Vérifie si l'utilisateur est administrateur."""
    return user.is_staff

def home(request):
    """Affiche la page d'accueil."""
    return render(request, 'prediction/home.html')

def register(request):
    """Gère l'inscription dun nouvel utilisateur."""
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        if User.objects.filter(username=username).exists():
            messages.error(request, "Nom d'utilisateur déjà pris.")
            return redirect('prediction:register')
        user = User.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        messages.success(request, "Inscription réussie !")
        return redirect('prediction:home')
    return render(request, 'prediction/register.html')

def user_login(request):
    """Gère la connexion d'un utilisateur."""
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Connexion réussie !")
            return redirect('prediction:home')
        messages.error(request, "Nom d'utilisateur ou mot de passe incorrect.")
    return render(request, 'prediction/login.html')

@login_required
def user_logout(request):
    """Gère la déconnexion d'un utilisateur."""
    logout(request)
    messages.success(request, "Déconnexion réussie !")
    return redirect('prediction:home')

@login_required
def submit_sequence(request):
    if request.method == 'POST':
        sequence = request.POST['sequence'].strip().upper()
        valid_residues = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(residue in valid_residues for residue in sequence):
            messages.error(request, "Séquence invalide. Utilisez uniquement A,C,D,...,Y.")
            return redirect('prediction:submit')
        prediction = ProteinPrediction.objects.create(user=request.user, sequence=sequence)
        try:
            output_filename = predict_structure(sequence)
            prediction.pdb_file.name = output_filename
            prediction.status = 'COMPLETED'
        except Exception as e:
            prediction.status = 'FAILED'
            messages.error(request, f"Erreur lors de la prédiction : {str(e)}")
        prediction.save()
        messages.success(request, "Prédiction terminée !" if prediction.status == 'COMPLETED' else "Prédiction échouée.")
        return redirect('prediction:history')
    return render(request, 'prediction/submit.html')

@login_required
def prediction_history(request):
    """Affiche l'historique des prédictions de l'utilisateur."""
    predictions = ProteinPrediction.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'prediction/history.html', {'predictions': predictions})

@login_required
def visualize_structure(request, prediction_id):
    """Affiche la visualisation 3D d'une prédiction."""
    prediction = get_object_or_404(ProteinPrediction, id=prediction_id, user=request.user)
    if not prediction.pdb_file:
        messages.error(request, "Aucun fichier PDB disponible.")
        return redirect('prediction:history')
    return render(request, 'prediction/visualize.html', {'prediction': prediction})

@login_required
def download_packs(request):
    """Affiche les packs disponibles pour téléchargement."""
    packs = ModelInstallationPack.objects.all()
    if request.method == 'POST':  # Enregistrement du téléchargement
        pack_id = request.POST.get('pack_id')
        pack = get_object_or_404(ModelInstallationPack, id=pack_id)
        DownloadLog.objects.create(user_id_fk=request.user, model_installation_pack_id_fk=pack)
        messages.success(request, f"Pack {pack.version} téléchargé !")
        return redirect('prediction:packs')
    return render(request, 'prediction/packs.html', {'packs': packs})

@login_required
@user_passes_test(is_admin)
def admin_manage(request):
    """Gère la gestion administrateur (utilisateurs et prédictions)."""
    users = User.objects.all()
    predictions = ProteinPrediction.objects.all()
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'delete_user':
            user_id = request.POST.get('user_id')
            User.objects.filter(id=user_id).delete()
            messages.success(request, "Utilisateur supprimé.")
        elif action == 'delete_prediction':
            prediction_id = request.POST.get('prediction_id')
            ProteinPrediction.objects.filter(id=prediction_id).delete()
            messages.success(request, "Prédiction supprimée.")
        return redirect('prediction:admin_manage')
    return render(request, 'prediction/admin_manage.html', {'users': users, 'predictions': predictions})


@login_required
@user_passes_test(is_admin)
def upload_model_pack(request):
    """Permet à l'administrateur de téléverser un pack de modèle."""
    if request.method == 'POST':
        version = request.POST.get('version')
        pack_type = request.POST.get('pack_type')
        file = request.FILES.get('file')
        description = request.POST.get('description')

        # Vérifications
        if not version or not pack_type or not file or not description:
            messages.error(request, "Tous les champs sont obligatoires !")
            return redirect('prediction:upload_model_pack')

        if not file.name.endswith('.zip'):
            messages.error(request, "Le fichier doit être un fichier .zip !")
            return redirect('prediction:upload_model_pack')

        try:
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_contents = zip_ref.namelist()
                required_files = ['omega.pth', 'predictor.pth']  # Fichiers requis dans le .zip
                if not all(file in zip_contents for file in required_files):
                    messages.error(request, "Le fichier .zip doit contenir omega.pth et predictor.pth !")
                    return redirect('prediction:upload_model_pack')
        except zipfile.BadZipFile:
            messages.error(request, "Fichier .zip invalide ou corrompu !")
            return redirect('prediction:upload_model_pack')

        # Sauvegarde du pack
        pack = ModelInstallationPack(
            version=version,
            pack_type=pack_type,
            file=file,
            description=description
        )
        pack.sauvegarder()  # Utilise la méthode personnalisée
        messages.success(request, f"Pack v{version} téléversé avec succès !")
        return redirect('prediction:admin_packs')

    return render(request, 'prediction/admin/upload_model_pack.html')

@login_required
@user_passes_test(is_admin)
def admin_packs(request):
    """Affiche et gère les packs disponibles pour l'administrateur."""
    packs = ModelInstallationPack.objects.all()
    return render(request, 'prediction/admin/packs.html', {'packs': packs})

@login_required
@user_passes_test(is_admin)
def admin_delete_pack(request, pack_id):
    """Supprime un pack de modèle."""
    pack = get_object_or_404(ModelInstallationPack, id=pack_id)
    pack.supprimer()  # Utilise la méthode personnalisée
    messages.success(request, f"Pack {pack.version} supprimé avec succès !")
    return redirect('prediction:admin_packs')