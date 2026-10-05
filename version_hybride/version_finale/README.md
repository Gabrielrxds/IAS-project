# version_finale — chaîne hybride, et chaîne hybride + SMA

Deux versions à tester, sur le **protocole commun** du groupe (patients réels de 2022, consultation
30 jours avant, durées apprises sur 2019-2021, test de mai à décembre 2022, marges cumulées
quadratiquement, 42 lits, 18 admissions ambulatoires par jour, réserves d'urgence 5 % du TVO,
2 lits, 2 admissions).

| Dossier | Ce que c'est |
|---|---|
| `hybride/` | **Version « hybride »** : grille hybride + consultation avec patients fantômes + journée AG/tabou. Planning *statique* (sans aléa). |
| `hybride_sma/` | **Version « hybride + SMA »** : le plan du matin de la version hybride, que le système multi-agents de Kyllian fait vivre face aux urgences, annulations, lits fermés et durées réelles. |
| `commun/` | Protocole commun (données, indicateurs, simulateur des consultations, algorithmes de journée et de grille). |
| `grilles/hybride.json` | La grille hybride déjà calculée (vacations datées). |

Le code **réutilise le modèle du groupe là où il est dans le dépôt** (`fil_rouge_AG/` : `modele.py`,
`genetique_jour.py`, `genetique_grille.py`, `analyse_couts.py`, `grille.py`, `charger_historique.py`) :
`version_finale/` doit être à la racine du dépôt, à côté de `fil_rouge_AG/`.

## Installer

```bash
pip install -r version_finale/requirements.txt
```

Données : par défaut `notebook_recuit/notebook_recuit/donees_bloc_anonyme_pour_centrale_2026.xlsx`.
Pour un autre fichier :

```bash
export BLOC_DONNEES=/chemin/vers/donees_bloc_anonyme_pour_centrale_2026.xlsx
```

Toutes les commandes se lancent **depuis la racine du dépôt**.

## Version « hybride »

```bash
python3 version_finale/hybride/tester_hybride.py          # 6 tests, 1-2 min
python3 version_finale/hybride/lancer_hybride.py          # chaîne complète vs grille actuelle, ~2 min
python3 version_finale/hybride/lancer_hybride.py --graines 0 1 2 --sans-journee
python3 version_finale/hybride/lancer_hybride.py --delta 7 --fantomes aucun     # variantes de la règle
python3 version_finale/hybride/lancer_hybride.py --recalculer-grille            # refait la grille (~15-30 min)
```

| Fichier | Rôle |
|---|---|
| `hybride_grille.py` | Problème de grille à ressources égales (salles 2-5, budget d'heures actuel, quotas 75-150 %) |
| `chaine_hybride.py` | Construit la grille : recuits + croisement génétique + sélection par simulation sur l'apprentissage |
| `regle_hybride.py` | Règle de consultation : fenêtre de tolérance Δ + patients fantômes |
| `lancer_hybride.py` | Lance la chaîne et affiche les indicateurs → `resultats/hybride.json` |
| `tester_hybride.py` | Tests (ressources égales, aucun conflit, délai minimum, reproductibilité, journée, résultats attendus) |

## Version « hybride + SMA »

```bash
python3 version_finale/hybride_sma/tester_hybride_sma.py          # 6 tests, 2-4 min la 1re fois
python3 version_finale/hybride_sma/lancer_hybride_sma.py          # 5 tirages d'aléas, ~1 min
python3 version_finale/hybride_sma/lancer_hybride_sma.py --tirages 20 --urgences 2
python3 version_finale/hybride_sma/lancer_hybride_sma.py --reserves 0.10 3 3
python3 version_finale/hybride_sma/sma_experiences.py             # tout le rapport → resultats/sma.json
python3 version_finale/hybride_sma/figures_rapport_sma.py version_finale/resultats/figures_sma
```

| Fichier | Rôle |
|---|---|
| `sma_aleas.py` | SMA de Kyllian, **copie inchangée** de `SMA/sma_aleas.py` |
| `sma_pont.py` | Branche le SMA sur le protocole : marges quadratiques dans les agents salle, lits fermés visibles par les trois politiques, mesures |
| `sma_matin.py` | Plans du matin (grille actuelle ou hybride, réserves réglables), mis en cache |
| `lancer_hybride_sma.py` | Compare actuelle / hybride × statique / réoptimisation / SMA → `resultats/hybride_sma.json` |
| `tester_hybride_sma.py` | Tests (projection P90 = modèle, lits fermés, invariants des 3 politiques, plan intact, reproductibilité, SMA > statique) |
| `sma_experiences.py`, `figures_rapport_sma.py`, `tableau_sma.py` | Toutes les simulations et figures du rapport « hybride + SMA » |

## Résultats de référence (rapports du groupe)

| | Actuelle · statique | Actuelle · SMA | Hybride · statique | Hybride · SMA |
|---|---|---|---|---|
| Urgences opérées le jour même | 50 % | 54 % | 44 % | 54 % |
| Urgences sans place | 38 % | 33 % | 43 % | 34 % |
| Dépassement du bloc (min/jour) | 16,4 | 5,0 | 15,9 | 6,2 |
| Variance des lits | 16,6 | 17,0 | 7,1 | 7,6 |
| Variance des admissions | 14,6 | 14,8 | 8,9 | 9,1 |

(1,5 urgence par jour, 20 tirages, mai-décembre 2022.)

`cache/` et `resultats/` sont créés à l'exécution et ignorés par git.
