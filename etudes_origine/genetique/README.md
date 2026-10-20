# Fil Rouge IA & Santé — méthode algorithme génétique

Analyse des gains en lits, en places ambulatoires et sur le planning quand on fait varier les
coefficients de la fonction coût `CoutTotal` (lits, places, remplissage, délai).

Chaîne testée pour chaque jeu de coefficients :

1. **Grille** : un AG construit une nouvelle grille de vacations à partir de janvier–avril 2022.
2. **Insertion** : les patients de 2022 sont insérés un par un, dans l'ordre des consultations
   (modèle V1, sans patients mobiles), et notés par `CoutTotal`.
3. **Journée** : l'AG journalier ordonne chaque journée dès qu'elle est figée.
4. **Mesure** sur mai–décembre 2022, hors apprentissage.

## Fichiers

| Fichier | Rôle |
|---|---|
| `modele.py` | Modèle commun du groupe : instance, solution, `CoutTotal`, urgences |
| `grille.py` | Grille actuelle des vacations (version 28) |
| `charger_historique.py` | Lecture de la base (TROS mesuré depuis l'entrée en salle calimed) |
| `genetique_jour.py` | AG journalier (ordre de passage) |
| `genetique_grille.py` | AG de construction de la grille |
| `analyse_couts.py` | Insertion patient par patient, journées, indicateurs, courbes |
| `etude_couts.py` | **Script principal** (version `.py` du notebook) |
| `rapport_html.py` + `rapport_gabarit.html` | Génération du rapport HTML |

## Utilisation

```bash
pip install -r requirements.txt
# placer donees_bloc_anonyme_pour_centrale_2026.xlsx dans le dossier (non versionné)
python etude_couts.py --rapide     # vérification, environ 1 min
python etude_couts.py              # analyse complète, environ 10 min
```

Options : `--donnees`, `--sortie` (par défaut `resultats/`), `--exclus SM` (praticiens exclus,
`--exclus` sans valeur pour n'exclure personne), `--sans-rapport`.

Sorties dans `resultats/` : tableaux CSV, figures PNG (`figures/`) et `Rapport_grille_AG.html`.

Le texte du rapport HTML (verdict, commentaires) est écrit pour l'exécution complète par défaut
(graine 0). Les graphiques et les tableaux, eux, sont toujours recalculés à partir des données.
