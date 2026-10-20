# Scénarios de la soutenance : surcharge, aléas, échelle, Monte-Carlo

Script : `chaine_finale/experiences/scenarios_soutenance.py` · figures : `chaine_finale/figures/figures_scenarios.py`
· résultats : `chaine_finale/resultats/scenarios_*.json` · figures : `chaine_finale/resultats/figures_scenarios/`.

Quatre configurations, mêmes aléas pour les quatre dans un même tirage :

| | plan (grille + date + journée) | jour J |
|---|---|---|
| référence · sans agents | grille actuelle, première date libre, ordre glouton | plan suivi, urgences par règle de capacité |
| référence · avec agents | idem | système multi-agents |
| chaîne hybride · sans agents | grille hybride, règle de date + fantômes, nuits par apprentissage, journée tabou | plan suivi |
| chaîne hybride · avec agents | idem | système multi-agents (configuration retenue) |

Mai–décembre 2022, lits mesurés avec les séjours réels. Bandes et barres : ± un écart-type entre tirages.

## Point de départ (`base`, 10 tirages)

| | variance lits | pic | urgences à temps | attente urgences | dépassement | reportés | ambu. gardés la nuit |
|---|---|---|---|---|---|---|---|
| référence · sans agents | 13,9 | 19,7 | 91,8 % | 115 min | 11,4 min/j | 22,5 | 9,0 |
| référence · avec agents | 13,8 | 19,3 | 96,2 % | 57 min | 6,4 min/j | 32,0 | 7,3 |
| hybride · sans agents | 6,3 | 17,0 | 86,5 % | 137 min | 9,6 min/j | 26,8 | 13,4 |
| hybride · avec agents | 6,3 | 17,0 | 96,6 % | 75 min | 4,6 min/j | 42,8 | 9,9 |

Ces valeurs redonnent `resultats/recap_v3.json` (code actuel), mais **pas** le tableau 12 du rapport v7.2
(86,6 / 99,5 %, variance 5,1 / 5,2, attente 116 / 32 min, reportés 20 / 31, ambulatoires 20 / 12),
qui vient d'une version antérieure des résultats.

## Surcharge (`surcharge.png`) : +0 à +30 % de patients, heures de bloc constantes

- Sans date : référence 15 → 179, hybride 0 → 23. Délai pour 90 % : 114 → 177 j contre 64 → 131 j.
- La chaîne hybride opère plus de patients (75 / semaine contre 71 à +30 %).
- Sans agents, la chaîne hybride sert de moins en moins bien les urgences (86 → 72 % dans le délai) :
  ses vacations sont plus pleines. Avec agents, elle reste à 97 → 93 %.
- Variance des lits : hybride 6,3 → 9,8, référence 13,9 → 11,4 (la référence baisse parce qu'elle laisse
  des patients sans date).

## Aléas du jour (`aleas_global.png`, `aleas_types.png`)

Tous les aléas multipliés par 0 à 3 (urgences, annulations, chirurgiens absents, pannes, lits fermés) :

- Urgences dans le délai, hybride : 88 → 78 % sans agents, 97 → 93 % avec agents.
- Dépassement, hybride : 3,1 → 21,4 min/j sans agents, 1,8 → 8,6 avec agents.
- Prix des agents : plus de reports (hybride à ×3 : 93 contre 70).
- Variance des lits presque insensible (hybride 6,4 → 8,1, toujours sous la référence à 14).

Un aléa à la fois :
- annulations : elles libèrent des créneaux et améliorent les urgences, surtout sans agents ;
- chirurgien absent : principal moteur des reports (220 à 290 patients reportés à 10 % par chirurgien
  et par jour, contre 24 à 45 à 1 %) ;
- panne de salle (0 à 50 % des jours) : sans agents le dépassement de la chaîne hybride passe de 9,7 à
  22 min/j, avec agents de 4,2 à 7,2 min/j, au prix de reports (42 → 62) ;
- afflux d'urgences ×3 : 70 % (hybride) et 77 % (référence) dans le délai sans agents, 89–90 % avec ;
- lits fermés (1 à 3 lits, de 0 à 100 % des jours) : aucun indicateur ne bouge ; un essai avec 1 à 30 lits
  fermés chaque jour n'ajoute que 10 nuits en déficit (le programmé n'occupe pas plus de 17 lits sur 42).
  Non tracé (données dans `scenarios_aleas.json`).

L'ascenseur en panne n'est pas modélisé en tant que tel ; dans le moteur, il se déclare comme
« salle indisponible » (panne de salle) ou dans la catégorie « autre ».

## Petite / grande échelle (`echelle.png`)

Petite échelle : 9 chirurgiens sur 18 (50 % des patients, 48 % des heures de la grille actuelle),
21 lits, 9 admissions ambulatoires, réserves 1 / 1, urgences de ces chirurgiens seulement. Grille hybride
reconstruite par la même méthode, sur 2 salles, ou sur 4 (variante).

- Le lissage des lits tient : irrégularité (écart-type / moyenne) de 32–33 % (2 salles) ou 37–38 %
  (4 salles) contre 44–45 % pour la référence ; 26 % contre 41 % à grande échelle.
- Le gain sur les délais disparaît (délai pour 90 % ≈ 125–130 j des deux côtés) ; sur 2 salles, la grille
  hybride laisse 8 patients sans date (0 pour la référence et pour la variante 4 salles).
- Les urgences souffrent plus : avec 9 chirurgiens, peu opèrent des urgences. Sans agents, hybride 49 %
  (2 salles) ou 55 % (4 salles) contre 80 % pour la référence ; avec agents 82 % / 87 % contre 89 %.

## Monte-Carlo (`mc_distributions.png`, `mc_convergence.png`), 100 tirages

Chaque tirage refait l'ordre des consultations, tire les durées (médiane de l'acte + écart d'un patient
du même acte), les séjours des hospitalisés (nuits d'un patient hospitalisé du même acte) et les aléas
du jour (urgences rééchantillonnées). Moyenne [intervalle à 95 % des tirages] :

| | variance lits | urgences à temps | dépassement (min/j) | reportés |
|---|---|---|---|---|
| référence · sans agents | 13,8 [11,6–16,0] | 89,9 % [83,7–93,9] | 8,6 [5,0–13,4] | 20 [4–38] |
| référence · avec agents | 14,0 [12,1–16,2] | 95,6 % [91,6–98,4] | 3,8 [1,6–6,1] | 29 [7–52] |
| hybride · sans agents | 8,3 [6,9–9,7] | 86,0 % [80,5–92,3] | 8,5 [4,7–12,1] | 22 [7–40] |
| hybride · avec agents | 8,6 [7,0–10,4] | 96,0 % [92,3–99,2] | 4,1 [2,0–6,4] | 31 [11–57] |

Tirage par tirage, la chaîne hybride avec agents fait mieux que la référence sans agents sur la variance
des lits (100 / 100), le délai pour 90 % (100 / 100), le dépassement (100 / 100) et les urgences (99 / 100),
mais reporte davantage (84 / 100). Les séjours tirés au hasard dans l'acte font monter la variance
(6,3 → 8,6) : c'est une borne pessimiste de l'incertitude sur le séjour.

## Remarque sur les paramètres d'aléas

`GenerateurV3` utilise : annulations 4 %, chirurgien absent 1 %, panne de salle 5 % des jours ouvrés pendant
1 à 3 h, 1 à 3 lits fermés 15 % des jours. Le rapport (section 7.3) écrit « une salle indisponible 2 à 4 heures
un jour ouvré sur dix » et « 2 à 4 lits fermés un jour sur vingt ».
