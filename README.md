# Gestion conjointe du bloc opératoire, des lits et de l'ambulatoire

Projet « IA et santé », Centrale Lille 2026. Approche hybride par métaheuristiques et patients fantômes.
Le rapport final est dans [`rapport/rapport_final_v7.2.pdf`](rapport/rapport_final_v7.2.pdf).

L'outil coordonne trois décisions, prises à trois moments :

| Niveau | Décision | Méthode retenue | Code |
|---|---|---|---|
| 1. Grille | quel chirurgien occupe chaque demi-journée de chaque salle (cycle paire / impaire) | recuits + croisement génétique + sélection par simulation | `chaine_finale/hybride/chaine_hybride.py` |
| 2. Date | la date proposée au patient en consultation | énumération dans une fenêtre de 14 jours, patients fantômes | `chaine_finale/hybride/regle_hybride.py` |
| 3. Journée | l'ordre de passage dans chaque vacation, la veille | meilleur ordre simple + recherche tabou | `chaine_finale/commun/algos_journee.py` |
| Jour J | urgences, retards, annulations, pannes | système multi-agents | `chaine_finale/hybride_sma/` |
| Aléas connus à l'avance | absences, salles, matériel, lits, annulations | moteur de reprogrammation | `chaine_finale/aleas/` |

## Organisation du dépôt

```
IAS-project/
├── chaine_finale/          la chaîne hybride, celle du rapport
│   ├── commun/             protocole (données, simulateur, indicateurs), calendrier et cible,
│   │                       algorithmes de grille et de journée, prévision par apprentissage
│   ├── hybride/            grille hybride, règle de date avec patients fantômes, lancement, tests
│   ├── hybride_sma/        système multi-agents du jour J, branché sur le plan du matin
│   ├── aleas/              aléas connus à l'avance : formulaire, reloger / décaler / raccourcir /
│   │                       fermer, reprogrammation, comblement
│   ├── experiences/        un script par résultat du rapport (écrit dans resultats/)
│   ├── figures/            figures du rapport, lues dans resultats/ (aucune valeur recopiée)
│   ├── grilles/            grilles calculées (chirurgiens anonymisés C1..C18)
│   ├── resultats/          résultats agrégés (JSON) : aucune donnée de patient
│   └── lancer_tout.sh      rejoue toute l'étude dans l'ordre
├── etudes_origine/         les trois études menées en parallèle (partie 6.1 du rapport)
│   ├── tabou/              Gabriel : GRASP, tabou à oscillation stratégique, path relinking
│   ├── genetique/          Killian : algorithme génétique ; contient aussi le modèle commun du
│   │                       groupe (modele.py, analyse_couts.py…) que la chaîne finale réutilise
│   ├── recuit/             Lucie : recuit simulé sur la grille actuelle
│   └── sma/                Kyllian : système multi-agents d'origine
├── docs/                   modèle mathématique, gestion des aléas, contrats pour l'interface
├── donnees/                emplacement du fichier de l'hôpital (jamais versé)
└── rapport/                rapport final (PDF)
```

Fonctions à appeler depuis une interface :
[`docs/CONTRATS_INTERFACE.md`](docs/CONTRATS_INTERFACE.md).

## Installer

```bash
pip install -r requirements.txt
```

Les données de l'hôpital ne sont **pas** dans le dépôt. Déposer le fichier
`donees_bloc_anonyme_pour_centrale_2026.xlsx` dans `donnees/`, ou indiquer son chemin :

```bash
export BLOC_DONNEES=/chemin/vers/donees_bloc_anonyme_pour_centrale_2026.xlsx
```

Au premier lancement, les données préparées sont mises en cache dans `chaine_finale/cache/` (ignoré par git).

## Lancer

Toutes les commandes se lancent depuis la racine du dépôt.

```bash
python3 chaine_finale/hybride/tester_hybride.py       # 6 tests de la chaîne, ~10 s
python3 chaine_finale/aleas/tester_absences.py        # 7 tests de la reprogrammation, ~15 s
python3 chaine_finale/aleas/tester_aleas.py           # 12 tests du moteur d'aléas
python3 chaine_finale/hybride_sma/tester_hybride_sma.py  # 6 tests du système multi-agents, ~1 min

python3 chaine_finale/hybride/lancer_hybride.py       # chaîne hybride contre la grille actuelle
python3 chaine_finale/figures/figures_rapport.py chaine_finale/resultats/figures   # toutes les figures
sh chaine_finale/lancer_tout.sh                       # toute l'étude (plusieurs heures)
```

## Confidentialité

- Aucune donnée de l'hôpital, aucun fichier `.xlsx`, `.csv` de la base ni cache `.pkl` dans le dépôt
  (voir `.gitignore`).
- Les chirurgiens sont désignés C1 à C18 dans les grilles, les résultats et les figures. Le code
  traduit ces codes à la lecture (`commun/protocole.py`, `lire_grille` / `sauver_grille`).
- Les modèles qui datent un groupe de patients à la fois par un tabou (« fixation progressive »)
  ont été écartés et ne sont pas dans le dépôt.
