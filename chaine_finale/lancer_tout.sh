#!/bin/sh
# Rejoue toute l'étude du rapport, dans l'ordre des dépendances. Plusieurs heures sur 2 cœurs.
# À lancer depuis la racine du dépôt :   sh chaine_finale/lancer_tout.sh
# Données : export BLOC_DONNEES=/chemin/vers/donees_bloc_anonyme_pour_centrale_2026.xlsx
#           (ou fichier déposé dans donnees/). Sorties : chaine_finale/grilles, chaine_finale/resultats.
set -e
cd "$(dirname "$0")"
J0=2021-12-27; N=552            # horizon commun : jour_zero et nb_jours du protocole

# 0. Données et prévisions (partie 4 et 5.3 du rapport)
python3 experiences/analyse_base.py resultats/figures   # figures 2, 3, 5, 7 + analyse_base.json
python3 commun/ml_prevision.py                           # ml_prevision.json
python3 experiences/calage_consultation.py               # délai de consultation calé (sensibilité)

# 1. Une grille par étude d'origine, puis la grille hybride (parties 6.1 et 6.2)
python3 experiences/grille_tabou.py $J0 $N               # Gabriel : GRASP + tabou + path relinking (~8 min)
python3 experiences/grille_recuit.py $J0 $N 3            # Lucie   : recuit, grille A (~1 min)
python3 experiences/grille_genetique.py                  # Killian : AG, paramètres d'origine (~10 s)
python3 experiences/grille_genetique_iso.py              # Killian : AG, salles 2-5 (~10 s)
python3 hybride/chaine_hybride.py 0.75 1.5               # grille hybride (~2 min)

# 2. Les trois études et la chaîne hybride (parties 6 et 7.2)
python3 experiences/evaluation_finale.py                 # final.json : chaînes, ablations, coûts
python3 experiences/etudes_origine.py GDCF               # études d'origine, sans / avec fantômes
python3 experiences/fantomes_grilles10.py                # tableau des valeurs clés, 10 ordres
python3 experiences/etude_statique.py                    # trois algorithmes sur le même problème de grille
python3 experiences/etude_dynamique.py
python3 experiences/comparer_journee.py                  # journee_algos.json
python3 experiences/reglage_parametres.py                # réglage des paramètres
python3 experiences/reglage_lits.py

# 3. Patients fantômes (partie 5.4)
python3 experiences/qualite_fantomes.py                  # fantomes_v3.json : vue, suivi, erreur
python3 experiences/fantomes_cas.py                      # les trois cas
python3 experiences/dates_fantomes.py

# 4. Jour J, escalier, combinaisons (parties 7.3 et 7.4)
python3 hybride_sma/sma_matin.py reference
python3 hybride_sma/sma_matin.py hybride
for r in "0 0 0" "0.10 3 3" "0.15 4 4"; do
  python3 hybride_sma/sma_matin.py reference $r; python3 hybride_sma/sma_matin.py hybride $r
done
python3 experiences/gap_journee.py                       # optimum des journées par énumération
python3 experiences/journee_criteres.py
python3 experiences/sma_experiences.py                   # sma.json : jour J et réserves
python3 experiences/recap_v3.py                          # recap_v3.json : escalier et combinaisons

# 5. Sensibilité et aléas connus à l'avance (parties 7.5 et 7.6)
python3 experiences/etude_cibles.py
python3 experiences/decomposition_vacances.py
python3 experiences/experiences_surcharge.py
python3 aleas/experiences_aleas.py

# 6. Figures du rapport
python3 figures/figures_rapport.py resultats/figures
echo FIN
