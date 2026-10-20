"""Chemins du dépôt.
    etudes_origine/genetique/  étude génétique (Killian) + modèle du groupe (modele.py, analyse_couts.py…)
    etudes_origine/recuit/     étude recuit simulé (Lucie)
    etudes_origine/tabou/      étude recherche tabou (Gabriel)
Données : variable d'environnement BLOC_DONNEES, sinon donnees/donees_bloc_anonyme_pour_centrale_2026.xlsx
(le fichier n'est jamais versé dans le dépôt)."""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]          # chaine_finale/
DEPOT = RACINE.parent
AG = DEPOT / "etudes_origine" / "genetique"
RECUIT = DEPOT / "etudes_origine" / "recuit"
TABOU = DEPOT / "etudes_origine" / "tabou"
DONNEES = Path(os.environ.get("BLOC_DONNEES",
                              DEPOT / "donnees" / "donees_bloc_anonyme_pour_centrale_2026.xlsx"))
