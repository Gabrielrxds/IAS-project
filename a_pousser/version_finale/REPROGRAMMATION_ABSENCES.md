# Absences des chirurgiens, patients non venus et reprogrammation

Complément de `SPEC_INTERFACE_MEDECIN.md` (parcours D, §2.4 et §4.3 ; parcours E, §2.5). Ce document décrit
l'algorithme de reprogrammation et ce qu'il donne en simulation sur les patients réels de 2022
(grille hybride, période de test mai–décembre).

Code : `version_finale/hybride/absences.py` (algorithme), `experiences_absences.py` (simulations),
`tester_absences.py` (tests).

---

## 1. Ce qui se passe quand un chirurgien déclare une absence

Un chirurgien déclare, le jour `t`, une absence du jour `début` au jour `fin`.

1. **Fermer** ses vacations de la période : capacité nulle. Les patients fantômes qu'elles
   portaient disparaissent, et la salle apparaît « libérée » dans le planning des blocs.
2. **Libérer** les patients déjà programmés dans ces vacations.
3. **Rattrapage** (option) : ouvrir au chirurgien des **demi-journées de salle libres** (aucune
   vacation, ni d'urgence) entre `t + 2` et `fin + 21` jours, aux heures où il est présent et
   n'opère pas déjà. On en ouvre juste assez pour couvrir les minutes des patients déplacés, sans
   dépasser le temps de bloc perdu. Dans l'interface, chaque ouverture est d'abord **proposée au
   cadre de bloc**, qui la valide.
4. **Reprogrammer** chaque patient, en commençant par celui dont la date initiale est la plus
   proche (il a le moins de marge) :
   - même chirurgien ;
   - nouvelle date ≥ `t + 2` (le temps de prévenir le patient), ≥ son délai minimum, ≤ son délai
     maximum (6 mois après la consultation) ;
   - **congé** (déclaré ≥ 8 jours avant) : journées non figées seulement ; **maladie** (préavis
     court) : aussi les journées figées s'il reste de la place ;
   - choix du jour, deux variantes :
     - « premier » : le premier créneau réalisable ;
     - « hybride » : le même score qu'à la consultation (hausse de la variance des lits attendus,
       patients fantômes compris), sur 14 jours après le premier jour réalisable. Deux jours sont
       proposés et le plus proche est retenu ; dans l'interface, c'est le médecin qui choisit.
   - aucun créneau avant le délai maximum : le patient est **« à arbitrer »**.

La date n'est jamais modifiée en dehors de ce cas, et toute reprogrammation est tracée
(ancienne date, nouvelle date, validée par, patient prévenu le).

---

## 2. Scénarios simulés

- **Congés « réels »** : les semaines de 2022 où un chirurgien avait des vacations dans la grille
  actuelle mais n'a opéré **aucun** patient. C'est une approximation des congés, formations et
  absences : 170 semaines-chirurgien, dont 40 en août. Elles sont déclarées 42, 14 ou 3 jours avant.
- **Maladie** : chaque journée opératoire est perdue avec une probabilité de 1 % ou 2 %. L'arrêt
  dure 1 à 5 jours et est déclaré 0 à 2 jours avant. 5 tirages.

## 3. Résultats (période de test, grille hybride)

| Scénario | Reprogrammés | À arbitrer | Décalage moyen (médian) | Décalés > 30 j | Rattrapages (heures) | Variance lits | Sans date | Délai médian |
|---|---|---|---|---|---|---|---|---|
| Sans absence | 0 | 0 | — | 0 | 0 | 7,2 | 9 | 31 j |
| Congés déclarés 42 j avant · hybride | 189 | 31 | 20 j (16) | 30 | 0 | 8,4 | 31 | 50 j |
| Congés déclarés 14 j avant · hybride | 275 | 42 | 24 j (20) | 64 | 0 | 9,4 | 37 | 50 j |
| Congés déclarés 3 j avant · hybride | 316 | 42 | 29 j (22) | 88 | 0 | 10,0 | 36 | 48 j |
| Congés 3 j avant · hybride + rattrapage | 296 | 13 | 12 j (9) | 15 | 145 (609 h) | 12,1 | 19 | 36 j |
| Maladie 1 % · premier créneau | 36 | 1 | 31 j (30) | 16 | 0 | 6,9 | 9 | 34 j |
| Maladie 1 % · hybride | 37 | 1 | 32 j (29) | 18 | 0 | 6,9 | 9 | 34 j |
| **Maladie 1 % · hybride + rattrapage** | 35 | **0** | **7 j (5)** | **1** | 20 (98 h) | 7,7 | 9 | 32 j |
| Maladie 2 % · hybride | 51 | 0,6 | 34 j (34) | 26 | 0 | 6,7 | 9,6 | 35 j |
| **Maladie 2 % · hybride + rattrapage** | 48 | **0** | **8 j (6)** | **2** | 35 (169 h) | 7,8 | 9 | 32 j |
| Congés 42 j + maladie 1 % · hybride + rattrapage | 176 | 20 | −4 j (2) | 18 | 106 (502 h) | 10,8 | 26 | 44 j |
| *Grille actuelle + 1er créneau · congés 42 j* | 177 | 78 | 23 j (14) | 17 | 0 | 19,3 | 67 | 51 j |

Les lignes « maladie » sont des moyennes sur 5 tirages.

## 4. Ce qu'il faut retenir

1. **Déclarer tôt** les congés divise presque par deux les reprogrammations : 189 patients pour
   un congé saisi 42 jours avant, contre 316 pour un congé saisi 3 jours avant. Les patients
   consultés après la déclaration ne sont jamais placés pendant l'absence. Il ne reste à déplacer
   que ceux qui avaient été vus avant. Dans l'interface, la page « Mes absences » doit donc être
   simple et rappelée régulièrement aux chirurgiens.
2. **Maladie : le rattrapage est indispensable.** Sans lui, un patient dont le chirurgien tombe
   malade attend en moyenne **un mois de plus** : les vacations suivantes du chirurgien sont
   déjà pleines. En ouvrant des demi-journées de salle libres (la grille hybride en laisse environ
   20 %), le décalage tombe à **5–6 jours médians**, et presque plus personne ne dépasse 30 jours
   ni n'est « à arbitrer ».
3. **Premier créneau ou score hybride pour reprogrammer** : presque aucune différence sur ces
   volumes. Le choix de la méthode compte peu ; ce sont le **préavis** et le **rattrapage** qui
   font la différence.
4. **Le modèle devient réaliste** : avec les congés, la simulation opère 186 patients en août,
   contre 270 sans absence et 140 dans la réalité. Une partie du lissage obtenu sans absence venait
   d'un été « plein » qui n'existe pas.
5. **Prix des absences** : environ 22 % des semaines-chirurgien fermées, donc moins de capacité.
   Les patients sans date passent de 9 à 31 et le délai médian de 31 à 50 jours. C'est le coût réel
   des absences, que le modèle sans absences ne voyait pas. Avec les mêmes congés, la grille
   actuelle fait pire : 67 sans date, 78 patients à arbitrer, variance des lits de 19,3.
6. **Le lissage tient** : la variance des lits reste entre 7 et 10 (19 avec la grille actuelle).
   Le rattrapage la remonte un peu, parce qu'il regroupe des patients sur quelques jours.

## 5. Patients non venus (parcours E)

**Règle** : le jour de son opération, le patient ne se présente pas. Sa place du jour est perdue
(le SMA peut la donner à une urgence). Il devient une **nouvelle demande vue ce jour-là** :
- délais minimum et maximum recomptés depuis ce jour ;
- même règle de date que tous les patients (règle hybride, deux jours proposés) ;
- mêlé aux consultations du jour, **sans priorité** ni rattrapage.

Code : `simuler_absences(..., taux_non_venue=...)` dans `absences.py`.

**Simulation** (grille hybride, mai–décembre 2022, 5 tirages par taux) :

| Taux de non-venue | Non-venues | Heures de bloc perdues | Nouvelle date : délai moyen (médian, P90) | Sans date (non venus) | Délai médian des autres patients | Variance lits |
|---|---|---|---|---|---|---|
| 0 % | 0 | 0 | — | 0 | 31 j | 6,9 |
| 2 % | 42 | 50 h | 43 j (35 j, 76 j) | 0 | 34 j | 6,9 |
| 5 % | 114 | 135 h | 45 j (38 j, 79 j) | 0,4 | 38 j | 6,3 |

- Un patient non venu retrouve une date environ **5 semaines plus tard** (médiane), comme un
  nouveau patient : c'est le prix de la règle « sans priorité ».
- La règle n'est pas neutre pour les autres patients : chaque non venu reprend une place dans les
  semaines suivantes. À 5 %, le délai médian des autres passe de 31 à 38 jours. Ils ne sont jamais
  déplacés, mais les nouvelles demandes trouvent des vacations plus pleines.
- Le lissage des lits n'est pas dégradé.
- Les heures perdues (50 à 135 h sur 8 mois) sont des places libres le jour même : c'est le SMA
  qui peut les récupérer pour des urgences.

## 6. Limites et suites

- Les « congés réels » sont déduits des semaines sans activité : une semaine creuse faute de
  patients compte comme une absence.
- Les vacations de rattrapage sont acceptées automatiquement dans la simulation ; dans la réalité,
  le cadre les valide (personnel d'anesthésie et d'infirmerie disponible ?).
- Le chirurgien ne change jamais. Proposer un confrère de même spécialité réduirait encore les
  décalages ; c'est hors périmètre pour l'instant.
- L'ordre de passage des journées touchées n'est pas recalculé dans cette simulation (c'est le rôle
  du SMA le jour J).
