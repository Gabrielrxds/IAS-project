# Du rapport au code

Pour chaque figure et tableau chiffré du rapport final (`rapport/rapport_final_v7.2.pdf`) : le script
qui calcule les résultats, le fichier où ils sont écrits, et la section de
`chaine_finale/figures/figures_rapport.py` qui trace la figure. Les chemins partent de `chaine_finale/`.
Les figures 1, 4, 6 et 13 sont des schémas dessinés dans le rapport (TikZ), sans calcul.

## 4. Le problème et les données

| Rapport | Calcul | Résultats | Figure |
|---|---|---|---|
| Fig. 2, lits occupés | `experiences/analyse_base.py` | `resultats/analyse_base.json` | tracée par le même script (`donnees_lits`) |
| Fig. 3, urgences | `experiences/analyse_base.py` | `resultats/analyse_base.json` | idem (`donnees_urgences`) |

## 5. Notre approche

| Rapport | Calcul | Résultats | Figure |
|---|---|---|---|
| Modèle mathématique (5.2) | `hybride/hybride_grille.py` (coût de la grille), `hybride/regle_hybride.py` (coût d'une date), `commun/algos_journee.py` (coût d'une journée) | — | voir `docs/MODELE_MATHEMATIQUE.md` |
| Fig. 5, temps en salle | `experiences/analyse_base.py` | `resultats/analyse_base.json` | `ml_durees` |
| Fig. 7, modèle de prévision | `commun/ml_prevision.py` | `resultats/ml_prevision.json` | `experiences/analyse_base.py` (`ml_resultats`) |
| Fig. 8, courbe vue par la règle de date | `experiences/qualite_fantomes.py` | `resultats/fantomes_v3.json` | `fantomes` (`fantomes_vue`) |
| Fig. 9, les trois cas | `experiences/fantomes_cas.py` | `resultats/fantomes_cas.json` | `fantomes_cas` |
| Fig. 10, erreur de la courbe | `experiences/qualite_fantomes.py` | `resultats/fantomes_v3.json` | `fantomes` (`fantomes_erreur`) |
| Calage des fantômes (α, κ) | `hybride/regle_hybride.py` (`calage_fantomes`) | `resultats/calage_fantomes.json` | — |

## 6. Résolution : des trois études à la chaîne hybride

| Rapport | Calcul | Résultats | Figure |
|---|---|---|---|
| Tab. 5, choix des trois études | `etudes_origine/` | — | — |
| Étude tabou (6.1.1) | `etudes_origine/tabou/vacations/planning_vacations.py`, grille : `experiences/grille_tabou.py` | `grilles/tabou.json` | — |
| Étude génétique, poids du coût (6.1.2) | `etudes_origine/genetique/etude_couts.py`, grille : `experiences/grille_genetique.py`, `grille_genetique_iso.py` | `grilles/genetique.json`, `genetique_iso.json` | — |
| Étude recuit, fonctions de coût (6.1.3) | `etudes_origine/recuit/etude_recuit.py`, `fonctions_cout.py`, grille : `experiences/grille_recuit.py` | `grilles/recuit.json` | — |
| Tab. 6, mouvements | `commun/algos_grille.py`, `commun/algos_journee.py` | — | — |
| Fig. 11, trois algorithmes sur la même grille | `experiences/etude_statique.py`, `experiences/evaluation_finale.py` | `resultats/etude_statique.json`, `final.json` | `couts` (`cout_grille`) |
| Fig. 12, réglage des paramètres | `experiences/reglage_parametres.py`, `experiences/reglage_lits.py` | `resultats/reglage_parametres.json`, `reglage_lits.json` | `reglage` |
| Tab. 7, valeurs clés sans / avec fantômes | `experiences/etudes_origine.py` (partie F), `experiences/fantomes_grilles10.py` | `resultats/etudes_origine.json`, `fantomes_grilles10.json` | `origine`, `chaines_origine` |
| Ordre de la journée (fin de 6.1) | `experiences/comparer_journee.py`, `experiences/gap_journee.py` | `resultats/journee_algos.json`, `gap_journee.json` | `journee` |
| Tab. 8, ce que la chaîne reprend | — | — | — |
| Fig. 14, construction de la grille hybride | `hybride/chaine_hybride.py` | `grilles/hybride.json` (méta-données) | `hybride_grille` |
| Algorithme 1, règle de date | `hybride/regle_hybride.py` (`RegleHybride`) | — | — |
| Niveau 3, journée | `commun/algos_journee.py` | — | — |
| Tab. 9, aléas du jour (SMA) | `hybride_sma/sma_aleas.py`, `hybride_sma/sma_pont.py` | — | — |
| Tab. 10-11, aléas connus à l'avance | `aleas/gestion_aleas.py`, `aleas/absences.py` | — | voir `docs/GESTION_ALEAS.md` |

## 7. Expérimentations et résultats

| Rapport | Calcul | Résultats | Figure |
|---|---|---|---|
| Fig. 15, chaînes complètes | `experiences/evaluation_finale.py`, `experiences/etudes_origine.py` (partie C) | `resultats/final.json`, `etudes_origine.json` | `chaines`, `chaines_origine` |
| Fig. 16, lits selon la nuit | `experiences/evaluation_finale.py` | `resultats/final.json` | `chaines` (`lits_semaine`) |
| Fig. 17, escalier | `experiences/recap_v3.py` | `resultats/recap_v3.json` | `escalier` |
| Fig. 18, fenêtre de tolérance | `experiences/evaluation_finale.py` (ablations) | `resultats/final.json` | `ablations` |
| Tab. 12, jour de l'opération | `hybride_sma/sma_matin.py`, `experiences/sma_experiences.py`, `experiences/recap_v3.py` | `resultats/sma.json`, `recap_v3.json` | — |
| Fig. 19, combinaisons | `experiences/recap_v3.py` | `resultats/recap_v3.json` | `combinaisons` |
| Fig. 20, cible | `experiences/etude_cibles.py` | `resultats/etude_cibles.json`, `grilles/hybride_c*.json` | `cible` |
| Tab. 13, grille régulière et plafond | `experiences/decomposition_vacances.py` | `resultats/vacances_v3.json` | — |
| Tab. 14, réserve pour les urgences | `hybride_sma/sma_matin.py` (réserves), `experiences/sma_experiences.py` | `resultats/sma.json` | — |
| Fig. 21, surcharge | `experiences/experiences_surcharge.py` | `resultats/surcharge.json` | `surcharge` |
| Fig. 22, aléas connus à l'avance | `aleas/experiences_aleas.py` | `resultats/aleas.json` (`aleas_avec_rattrapage.json` pour mémoire) | `aleas` |
| Limites, variabilité des graines | `experiences/stats_graines.py` | `resultats/stats_graines.json` | — |

## 8. Mise en œuvre

| Rapport | Où |
|---|---|
| 8.1, l'interface | à refaire à partir de `docs/CONTRATS_INTERFACE.md` |
| Tab. 15, normes | rapport seulement |

## 9. Conclusion

| Rapport | Résultats repris |
|---|---|
| Tab. 16, paramètres réglables | fenêtre et choix de la date : `final.json` ; cible : `etude_cibles.json` ; vacances : `vacances_v3.json` ; réserve : `sma.json` ; prévision des durées : `recap_v3.json` ; rattrapage : `aleas.json`, `aleas_avec_rattrapage.json` |

## Tracer toutes les figures

```bash
python3 chaine_finale/figures/figures_rapport.py chaine_finale/resultats/figures            # toutes
python3 chaine_finale/figures/figures_rapport.py chaine_finale/resultats/figures escalier   # une seule
```
