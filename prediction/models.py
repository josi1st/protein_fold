from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

# Modèle pour les utilisateurs (hérité de User de Django)
class User(models.Model):
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=150, unique=True)
    password = models.CharField(max_length=128)  # Haché par Django
    email = models.EmailField(unique=True)
    date_inscription = models.DateTimeField(auto_now_add=True)

    def set_password(self, raw_password):
        """Définit le mot de passe en le hachant."""
        self.password = self.set_password(raw_password)

    def check_password(self, raw_password):
        """Vérifie si le mot de passe correspond au hachage."""
        return self.check_password(raw_password)

    def __str__(self):
        return self.username

    class Meta:
        # Utilisation du modèle User de Django à la place
        managed = False
        db_table = 'auth_user'

# Modèle pour les prédictions de protéines
class ProteinPrediction(models.Model):
    id = models.AutoField(primary_key=True)
    sequence = models.TextField()
    pdb_file = models.FileField(upload_to='predictions/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    status = models.CharField(max_length=10, choices=[('PENDING', 'En cours'), ('COMPLETED', 'Terminé'), ('FAILED', 'Échoué')], default='PENDING')

    def save(self, *args, **kwargs):
        """Enregistre la prédiction et valide la séquence."""
        if not self.pk and not hasattr(self, 'length'):  # Calculer length seulement à la création
            self.length = len(self.sequence)
        super().save(*args, **kwargs)

    def clean(self):
        """Valide que la séquence contient uniquement des acides aminés valides."""
        valid_residues = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(residue in valid_residues for residue in self.sequence):
            raise ValidationError("La séquence doit contenir uniquement des acides aminés valides (A,C,D,...,Y).")

    def __str__(self):
        return f"Prédiction {self.id} par {self.user.username}"

# Modèle pour les packs d'installation
class ModelInstallationPack(models.Model):
    id = models.AutoField(primary_key=True)
    version = models.CharField(max_length=10)
    pack_type = models.CharField(max_length=20, choices=[('LOCAL', 'Local (PC)'), ('MICROCONTROLLER', 'Microcontrôleur (Raspberry Pi)')])
    file = models.FileField(upload_to='install_packs/')
    date_mise_a_jour = models.DateTimeField(auto_now=True)
    description = models.TextField()

    def sauvegarder(self):
        """Enregistre le pack dans la base de données."""
        self.save()

    def supprimer(self):
        """Supprime le pack (pour l'administrateur)."""
        self.delete()

    def __str__(self):
        return f"Pack {self.pack_type} v{self.version}"

# Modèle pour enregistrer les téléchargements
class DownloadLog(models.Model):
    id = models.AutoField(primary_key=True)
    user_id_fk = models.ForeignKey(User, on_delete=models.CASCADE, related_name='downloads')
    model_installation_pack_id_fk = models.ForeignKey(ModelInstallationPack, on_delete=models.CASCADE, related_name='downloads')
    downloaded_at = models.DateTimeField(auto_now_add=True)

    def sauvegarder(self):
        """Enregistre l'événement de téléchargement."""
        self.save()

    def __str__(self):
        return f"Téléchargement {self.id} par {self.user_id_fk.username}"

# Modèle pour les sessions (géré par Django)
class Session(models.Model):
    cle_session = models.CharField(max_length=40, primary_key=True)
    user_id_fk = models.ForeignKey(User, on_delete=models.CASCADE)
    date_expiration = models.DateTimeField()

    def get_session_key(self):
        """Décode les données de session (simplifié)."""
        from django.contrib.sessions.models import Session as DjangoSession
        session = DjangoSession.objects.get(session_key=self.cle_session)
        return session.get_decoded()

    class Meta:
        # Utilisation du modèle Session de Django à la place
        managed = False
        db_table = 'django_session'