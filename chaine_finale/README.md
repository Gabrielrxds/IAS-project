# chaine_finale — la chaîne hybride du rapport

Protocole commun : patients réels de 2022, consultation 30 jours avant l'opération, durées apprises
sur 2019-2021, grilles et fantômes calés sur janvier-avril 2022, mesures sur mai-décembre 2022 ;
42 lits, 18 admissions ambulatoires par jour, réserve d'urgence de 5 % du temps, 2 lits, 2 admissions.

Chaque script ajoute lui-même `commun/`, `hybride/`, `hybride_sma/`, `aleas/` et `experiences/` au
chemin d'import, et réutilise le modèle commun du groupe dans `../etudes_origine/genetique/`.
Les chemins (données, études d'origine) sont dans `commun/chemins.py`.

| Dossier | Contenu |
|---|---|
| `commun/` | `protocole.py` (chargement, instance, simulateur, indicateurs, grilles anonymisées), `calendrier.py` (vacances, cible des lits, capacité), `politiques.py` (règles de date des études), `algos_grille.py` et `algos_journee.py` (tabou, recuit, génétique), `ml_prevision.py` (durées et nuits par apprentissage), `urgences_fssa.py` (urgences U1 / U2) |
| `hybride/` | `hybride_grille.py` (problème de grille), `chaine_hybride.py` (construction de la grille), `regle_hybride.py` (règle de date et patients fantômes), `lancer_hybride.py`, `tester_hybride.py` |
| `hybride_sma/` | `sma_aleas.py` (agents), `sma_pont.py` (branchement sur le plan du matin), `sma_matin.py` (plans du matin), `sma_v2.py`, `sma_v3.py` (versions du jour J), lancement et tests |
| `aleas/` | `gestion_aleas.py` (formulaire, catégories, réponses), `absences.py` (reprogrammation), `scenarios_aleas.py`, `experiences_aleas.py`, `experiences_absences.py`, `experiences_non_venues.py`, tests |
| `experiences/` | un script par résultat du rapport ; voir `../CORRESPONDANCE_RAPPORT.md` |
| `figures/` | `figures_rapport.py` : toutes les figures, lues dans `resultats/` |
| `grilles/` | `hybride.json` (retenue), `hybride_c*.json` (étude de la cible), `hybride_cycle4.json`, `hybride_variance.json`, grilles des études (`tabou`, `genetique`, `genetique_iso`, `recuit`), `statique_*.json` (trois algorithmes sur le même problème) |
| `resultats/` | résultats agrégés en JSON |

Ordre complet des calculs : `lancer_tout.sh`. Détail des aléas : `../docs/GESTION_ALEAS.md` et
`../docs/REPROGRAMMATION_ABSENCES.md`.
