"""Chemins du dépôt. Le dossier version_finale/ réutilise le modèle du groupe là où il est
dans le dépôt, sans le recopier.
    fil_rouge_AG/                     chaîne génétique (Killian) + modèle du groupe
    notebook_recuit/notebook_recuit/  chaîne recuit (Lucie)
    IAS_val_gab_tabou/ordonnancement/ chaîne tabou (Gabriel)
Données : variable d'environnement BLOC_DONNEES, sinon le fichier du dossier recuit."""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]          # version_finale/
DEPOT = RACINE.parent
AG = DEPOT / "fil_rouge_AG"
RECUIT = DEPOT / "notebook_recuit" / "notebook_recuit"
TABOU = DEPOT / "IAS_val_gab_tabou" / "ordonnancement"
DONNEES = Path(os.environ.get("BLOC_DONNEES",
                              RECUIT / "donees_bloc_anonyme_pour_centrale_2026.xlsx"))
