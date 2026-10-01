# Contexte du projet — à coller en tête d'un prompt

> **Usage.** Ce document résume l'état du projet pour un assistant (ou un nouveau membre).
> Collez-le en entier au début d'une conversation, puis posez la question. Il décrit **ce
> qui existe, les décisions prises et pourquoi, les chiffres de référence, et ce qu'il ne
> faut pas casser**. Le détail du code est dans `ordonnancement/README.md`.

---

## 1. Le problème

Projet « IA et Santé » (Centrale, 2026) : **optimiser l'ordonnancement d'un bloc
opératoire** à partir de données réelles anonymisées.

- **Bloc** : salles 2 à 5, jours ouvrés. Les praticiens (18, désignés par un code de deux
  lettres : CL, SM, DE…) disposent de **vacations** (plages salle × jour) selon une grille
  cyclique de 4 semaines.
- **Données** :
  - `donees bloc anonyme pour centrale 2026.xlsx` (14 649 interventions, 2019–2022 ;
    version nettoyée par Lucie : `bdd_nettoyee.xlsx` / `bdd_nettoyee-2.csv`, séparateur `;`) ;
  - `Vacations Opératoires anonymisées.xlsx` (la grille réelle, recopiée en dur dans
    `ordonnancement/coeur/donnees.py` → `GRILLE`).
  - Les fichiers bruts ne sont **pas** versionnés (dépôt public).
- **Protocole d'évaluation** (commun à tous les résultats ci-dessous) : on **apprend sur
  2019–2021**, on **rejoue l'année 2022** (consultations tirées uniformément sur 364 jours,
  horizon 65 semaines, graine 0) et on mesure sur le **régime établi, semaines 5 à 50**.
- **Indicateurs ANAP** :
  - TVO = temps de vacation offert ;
  - TROS = entrée → sortie de salle ;
  - TIS = temps inter-salle ;
  - creux = TVO − ΣTROS − ΣTIS.
- **Ressources aval** :
  - un patient ambulatoire (0 nuit, 54 % des cas) occupe une **place**, libérée le soir ;
  - un patient hospitalisé occupe un **lit** par nuit ;
  - nuits = `Date Sortie − Date Inter`. **Ne jamais utiliser la colonne « Durée séjour »**,
    fausse pour ~1 900 séjours.

## 2. Qui a fait quoi dans le dépôt

| Fichier / dossier | Auteur | Contenu |
|---|---|---|
| `modele.py` (racine) | Gabriel | Structures de données v1 (`Instance`, `Solution`, `PlanningJour`). Identique à `ordonnancement/coeur/modele.py`. |
| `modele_fn_cout.py` | Killian | `modele.py` + fonctions de coût (places, lits, remplissage, délai). |
| `modele_fn_cout_1rdv.py` | Killian | idem + **urgences** (`SEMI_URGENCE`, `URGENCE_JOUR`, créneaux `MED_URGENCE`), délai max 6 mois (règle du prof), nuits calculées par les dates avec `nuits_avant`. |
| `bdd_nettoyee*.{xlsx,csv}` | Lucie | Base nettoyée. |
| `Guide_planning_bloc_operatoire.pdf` | Solène | Guide de 29 p. d'un module `planning.py` vérificateur : 5 classes, 8 règles dures, balayage, passage à CP-SAT. |
| `ordonnancement/` | Gabriel | Moteur complet : choix du jour, tabou de la journée, planning des vacations, simulation annuelle. |

**À harmoniser** (point ouvert) : `ordonnancement/` compte les nuits depuis `Date Inter`
(92 % des patients entrent le jour même), alors que Killian ajoute les `nuits_avant` depuis
`Date Entrée`. Les deux versions de `Patient` divergent : il faut en retenir une.

## 3. Architecture : décomposition hiérarchique en 3 étages

```
consultation ─► 1. durée estimée ─► 2. JOUR (vacation) ─► 3. HEURE + SALLE + place
               estimation.py       propositions.py        tabou.py (tabou local, la veille)
                                    simulation.py rejoue une année entière
en amont : la GRILLE des vacations (qui opère quand) ─► vacations/
```

### 3.1 Étape 1 — la durée (`estimation.py`)

- La durée proposée d_p est la médiane du praticien sur l'acte (CCAM), et la marge de
  risque μ_p vient du P90.
- Les marges sont **mutualisées** dans une vacation (facteur κ = 0,45, ≈ 1/√n).
- Le chirurgien valide (`valideur_auto`).

### 3.2 Étape 2 — choisir le JOUR (« ancien régime », modèle retenu)

Les patients sont placés **un par un, à la consultation**. La date ne bouge plus ensuite.

- On propose **2 dates** (« au plus tôt » et « recommandée ») ; le chirurgien prend la plus
  proche (`choix_plus_proche`).
- La **fenêtre** demandée (`FENETRES` : 7 / 21 / … jours) sert de **borne basse**.
- **Règle semaine creuse.** Si la densité de la semaine à venir est < 50 %
  (`SEUIL_DENSITE_PROCHE`, `JOURS_PROCHE = 7`), le délai minimum est levé : on propose au
  plus tôt.
- **Fermeture** d'une journée : à un remplissage de 85 %, ou la veille.
- **Objectif**, comparé lexicographiquement après le nombre de patients sans date :

  f(x) = w_r · (1/|M|) Σ_m Var_{v∈V_m}(τ_v)  +  w_l · Var_j(L_j) / L_ref²

  avec τ_v le taux de remplissage de la vacation v, et L_j le nombre de lits occupés le
  jour j. L'évaluation est **incrémentale** (sommes maintenues ; testé à 1e-16 près).
- **Pourquoi pas de métaheuristique ici.** Un patient a environ 50 vacations candidates :
  l'énumération exacte est instantanée.
- **Résultat sur les poids.**
  - Seul le rapport w_l/w_r compte.
  - Les résultats forment un **plateau** de 16 à 16 384 : 3 071 datés, délai 123 j, dans
    l'expérience dédiée.
  - Seul w_r = 0 les change (3 456 datés, 47 j), car le terme lits est nul pour les
    ambulatoires.

### 3.3 Étape 3 — la journée (`tabou.py`, `TabouLocal`)

- **Variable** : l'ordre des patients dans chaque vacation de la journée, donc leurs
  heures.
- **Coût** : violations d'abord (lexicographique), puis

  g(σ) = b1·creux/T + b2·Var(τ)_salles + b3·max A(t) + b4·Var_t(A)

  où A(t) est l'occupation des places.
- **Mouvements** : échange, insertion, report, reprise ; tenure 7 ; aspiration.
- Les **numéros de place** sont attribués par coloriage d'intervalles, ce qui est
  **optimal** une fois les heures fixées.

### 3.4 Le planning des vacations (`ordonnancement/vacations/`)

**(a) Réaffectation de la grille existante** — `grille_tabou.py`, `TabouGrille`.

- Les blocs sont des salle × jour × plage × semaine du cycle.
- **Contraintes** :
  - G1 : pas de chevauchement ;
  - G2 : un praticien n'opère que ses jours de semaine habituels ;
  - G3 : au moins un bloc par praticien ;
  - G4 : budget de changements K.
- **Mouvements** : transfert ou échange.
- **Demande** prévue en « enveloppe haute » (on se trompe exprès vers le haut).
- **Coût** : adéquation (manque au carré au-dessus de ρ* = 0,85, excès sous ρ_min = 0,5,
  pondéré par γ = 0,1) + w_lits · CV² des lits prévus.
- Sortie : `resultats/grille_proposee.csv`.

**(b) Reconstruction de zéro par demi-journées** — `planning_vacations.py`.

- **Créneaux** : salles 2–5, matin 8 h–13 h (5 h), après-midi 13 h 30–17 h 30 (4 h), cycle
  de 4 semaines, soit 138 créneaux (626 h). Budget de 594 h, comme la grille actuelle.
- **Besoin hebdomadaire** par praticien estimé sur l'historique : durée estimée + marge +
  TIS.
- **Profil de lits π_m** : lits occupés les jours suivants par heure opérée, propre à
  chaque chirurgien selon ses durées de séjour.
- **Coût** = adéquation + w_lits = 20 · CV²(lits prévus sur 28 j) + w_reg = 0,3 ·
  irrégularité.
- **Métaheuristique hybride** :
  1. GRASP (6 départs) ;
  2. tabou à **oscillation stratégique** (λ adaptatif sur le dépassement de budget) ;
  3. **path relinking** entre élites, dédoublonnées modulo les échanges de salle ;
  4. sélection **simheuristique** : chaque élite est rejouée sur un an, et on garde la
     plus faible variance des lits sans plus de 5 patients sans date en plus.
- **Variante par scénarios** : 40 tirages log-normaux, volatilité σ² = a + b/volume
  bornée.
- Sortie : `resultats/planning_reconstruit.csv` et `resultats/resultats_planning.json`.

## 4. Chiffres de référence (2022 rejoué, semaines 5–50)

| Configuration | Datés | Sans date | Délai méd. / P90 | Lits moy. | Lits méd. | **Var. lits** | Max lits |
|---|---|---|---|---|---|---|---|
| **Hôpital réel** (données) | — | — | — | 10,0 | 10,0 | **26,4** | 24 |
| Ancien régime, grille actuelle | 3 598 | 42 | 24 j / 58 j | 9,47 | 10,0 | **14,9** | 19 |
| Grille réaffectée (tabou, `enveloppe_K8`) | — | 3 | — / 49 j | — | — | — | — |
| **Grille reconstruite, volume actuel** | 3 598 | 42 | 23 j / 62 j | 9,38 | 9,5 | **9,15** (−39 %) | 17 |
| idem sans terme lits (contrôle) | 3 598 | 42 | 21 j / 64 j | 9,36 | 9,0 | 14,7 | — |
| Reconstruite, besoin prévu (enveloppe) | 3 493 | 147 | 24 j / 76 j | 9,13 | 9,0 | 9,77 | — |
| Reconstruite, scénarios | 3 417 | 223 | 17 j / 49 j | 9,02 | 9,0 | 10,9 | — |
| Reconstruite, oracle (besoin 2022 connu) | 3 636 | 4 | 21 j / 49 j | 9,58 | 10,0 | 9,94 | — |

Pour la grille reconstruite à volume actuel, comparée à la grille actuelle :

- part des nuits-lits produites jeudi + vendredi : 33 % → 53 % (elles tombent sur le
  week-end, quand les lits sont libres) ;
- remplissage des vacations : 69,7 % → 67,9 % ;
- patients par semaine : 63,3 → 63,7.

## 5. Leçons (à ne pas réapprendre)

1. **Le terme lits fait le gain** : sans lui, à budget égal, la variance reste à 14,7.
   C'est la position des opérations longues dans la semaine qui lisse les lits.
2. **La prévision de la demande est le goulot.**
   - Les volumes dérivent d'une année sur l'autre (h/semaine) : SM 12,4 → 6,7 ;
     CT 7,2 → 10,8 ; LR 1,3 → 3,8 ; GA 1,2 → 2,8.
   - Reconstruire de zéro **amplifie** l'erreur de prévision : 147 à 223 patients sans
     date, contre 4 avec l'oracle.
   - Règle retenue : **garder le volume actuel de chaque praticien et ne reconstruire que
     la répartition** dans la semaine.
   - Pour de petites réaffectations, l'enveloppe haute marche bien.
3. **Seul le rapport w_l/w_r compte**, et les résultats sont insensibles sur une large
   plage. Il est inutile de chercher des poids fins.
4. **39,5 % de creux** dans les vacations : c'est la grille qui ne suit pas la demande,
   pas l'ordonnancement des patients.

## 6. Ce qu'il reste à faire (par rendement décroissant)

1. **Prévoir la demande** par praticien (tendance, ruptures, arrivées et départs), puis
   reconstruire la grille avec cette prévision.
2. **Urgences** : intégrer le travail de Killian (créneaux `MED_URGENCE`, semi-urgences)
   dans `ordonnancement/`, et chiffrer le coût de la réserve de TVO.
3. **Harmoniser `Patient`** entre les deux versions (§2), et lire directement
   `bdd_nettoyee-2.csv`.
4. **Fermetures saisonnières** : calendrier réel d'ouverture dans l'`Instance`.
5. **Stochastique** : tirer les durées réelles et mesurer combien de journées débordent,
   pour valider la marge de risque.
6. **Vérificateur indépendant** : brancher les 8 règles dures du guide de Solène comme
   juge des plannings produits.

## 7. Conventions — ce qu'il ne faut pas casser

- Dans la recherche, les **durées sont en minutes entières** et les **jours en indices
  entiers**. Les `datetime` ne servent qu'aux frontières (lecture, affichage).
- L'**état de décision vit dans `Solution`**, jamais dans `Patient`, pour que la copie soit
  bon marché.
- Toute évaluation **incrémentale** doit rester égale au recalcul complet :
  `python3 ordonnancement/tests/tests.py` le vérifie (10 contrôles, ~1 min).
- **Invariant** : après le flux, tout patient a une date **ou** est explicitement hors
  horizon. Pas de liste d'attente cachée.
- **Comparaisons** : même protocole que le §1 (2022, graine 0, semaines 5–50).
  Comparer moyenne, médiane, **variance** et max des lits, le nombre de patients sans date,
  et le délai médian / P90.
- **Exclu du dépôt** : les modèles qui datent un **groupe** de patients à la fois par un
  tabou (fixation progressive, nuit de tabou sur un lot de patients). Ils ont été explorés
  puis écartés ; ne pas les réintroduire sans en discuter.

## 8. Lancer

```bash
cd ordonnancement
python3 tests/tests.py                                           # correction
python3 experiences/demo.py --donnees chemin/vers/donnees.xlsx    # chaîne complète, ~1 min
python3 experiences/experience_planning.py --donnees X actuelle volume   # ~10 min par variante
python3 experiences/experience_grille.py --donnees X actuelle enveloppe_K8   # variantes : voir VARIANTES
```

Le fichier de données peut aussi être passé par `BLOC_DONNEES=...`, ou déposé dans
`ordonnancement/donnees/`, qui est ignoré par git. Dépendances : Python ≥ 3.10, pandas,
numpy, openpyxl.
