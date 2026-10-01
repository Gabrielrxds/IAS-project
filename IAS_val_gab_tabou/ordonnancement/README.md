# `ordonnancement/` — moteur de consultation, tabou et planning des vacations

Travail de Gabriel. Vue d'ensemble, chiffres et décisions : **`../CONTEXTE_PROJET.md`**.

```bash
python3 tests/tests.py                                         # correction, ~1 min
python3 experiences/demo.py --donnees chemin/donnees.xlsx       # chaîne complète
python3 experiences/banc_essai.py --donnees chemin/donnees.xlsx # banc d'essai, ~3 min
python3 experiences/experience_planning.py --donnees X actuelle volume   # planning reconstruit
python3 experiences/experience_grille.py --donnees X actuelle enveloppe_K8 # grille réaffectée
```
Le fichier de données (non versionné) se donne par `--donnees`, par `BLOC_DONNEES`, ou se
dépose dans `donnees/`. Les scripts importent `chemins.py`, qui rend `coeur/` et
`vacations/` importables.

| dossier / fichier | rôle |
|---|---|
| `coeur/modele.py` | structures de données : `Instance`, `Solution`, `PlanningJour` |
| `coeur/estimation.py` | durées opératoires : statistiques + validation par le chirurgien |
| `coeur/propositions.py` | **le moteur de consultation : proposer deux dates** |
| `coeur/tabou.py` | tabou local (la journée) + tabou hors-ligne (l'étalon) |
| `coeur/simulation.py` | rejouer une année : consultations, fermeture des journées, tabou local |
| `coeur/donnees.py` | lecture de l'Excel, grille de vacations, instance de test |
| `vacations/grille_tabou.py` | tabou de réaffectation de la grille existante |
| `vacations/planning_vacations.py` | planning de zéro par demi-journées (GRASP, tabou, path relinking) |
| `experiences/` | démo, banc d'essai, expériences grille et planning |
| `tests/tests.py` | contrôles de correction |
| `resultats/` | programme opératoire 2022, grille proposée, planning reconstruit, résultats JSON |
| `rapport/rapport_bloc.ipynb` | rapport (code masqué, sections 2, 3 et 5) |

> Les sections ci-dessous sont la documentation technique du moteur (étapes 1 à 3), rédigée
> avant le travail sur les vacations ; leurs chiffres de la section 5 viennent d'un premier
> protocole (1er semestre 2022). Les chiffres de référence actuels sont dans
> `CONTEXTE_PROJET.md` §4.

---

## 1. Le processus modélisé

```
   Le patient est vu en consultation
     └─> on estime la durée de son acte, le chirurgien la valide   estimation.py
     └─> on lui propose DEUX dates                                 propositions.py
     └─> le chirurgien en choisit une, devant le patient
     └─> cette date ne bouge plus jamais

   La veille de l'opération
     └─> on fixe l'ordre de la journée, donc les heures,           tabou.py
         donc les places et les lits
```

**Il n'y a pas de liste d'attente.** Un patient sort de consultation avec une
date, ou avec un « pas de date avant tel mois, on vous reconvoque » — jamais
avec un « on vous rappellera ». Dans le code, `Solution.hors_horizon` porte le
second cas, et l'invariant est vérifié par `tests.py` : tout patient a une date
ou est explicitement hors horizon, personne n'attend dans un tiroir.

---

## 2. Où le tabou sert, et où il ne sert pas

C'est le point le plus important du dossier, et c'est un réflexe à garder :
**avant de sortir une métaheuristique, compter la taille de l'espace de
recherche.**

### À la consultation : pas de tabou, et c'est mieux ainsi

Les dates déjà données sont figées et on insère UN patient. L'espace des
solutions, c'est l'ensemble des vacations où ce patient peut aller : celles de
SON chirurgien (il n'en change jamais), postérieures à sa consultation. Sur six
mois, une vingtaine. On les évalue toutes et on trie.

Une énumération exhaustive est **exacte** — elle trouve l'optimum, pas une
approximation — et **instantanée** : les 1 998 consultations du semestre sont
traitées en 0,3 seconde. Un tabou ferait moins bien et plus lentement. Le mettre
là serait de l'habillage.

Ce qui est non trivial dans `propositions.py`, ce n'est pas la recherche, ce
sont deux autres choses :

1. **Le critère de classement.** On ne classe pas les dates par « la salle est
   libre », mais par la dégradation qu'elles infligent au planning entier — la
   fonction objectif à trois plans, évaluée en delta. Une date qui remplit bien
   une vacation mais crée un pic de lits trois jours plus tard sera mal classée.
2. **Ce qu'on montre au chirurgien.** Un classement sans justification est
   inutilisable en consultation. Chaque proposition porte son taux de
   remplissage après insertion, l'occupation des lits et des places ce jour-là,
   et des alertes explicites.

### Dans la journée : un vrai tabou

`TabouLocal` décide la séquence des patients dans chaque vacation — donc les
heures, donc le pic de places et la forme de leur courbe d'occupation. Le
voisinage est combinatoire, le problème NP-difficile. Sur vos données, il fait
tomber le pic de places de 5,80 à 3,92 en moyenne, et sa dispersion temporelle
de 1,88 à 1,22.

### Hors ligne : un tabou pour mesurer

`TabouGlobal` calcule le meilleur planning atteignable **avec le recul** : tous
les patients connus d'avance, tout déplaçable. Ce planning n'est pas réalisable,
et ce n'est pas le but. Il sert d'**étalon** : l'écart avec le flux réel mesure
le prix de la contrainte « une date annoncée ne bouge plus » (section 5).

Le même moteur sert à la **replanification exceptionnelle** — congé imprévu,
fermeture de salle. Là on a le droit de déplacer des patients, et
`TabouGlobal.figer_avant(sol, aujourdhui + 30)` sanctuarise ceux dont la
convocation est déjà partie.

---

## 3. Les deux fonctions objectif

### Objectif n°1 — choisir le JOUR (consultation + tabou hors-ligne)

**Deux termes, deux poids, c'est tout.**

```
f = w_remplissage · moyenne sur les praticiens de Var( taux de leurs vacations )
  + w_lits        · Var( occupation des lits, jour par jour ) / ref_lits²
```

avec `Charge(v) = Σ d_p + Σ m_p + TIS·(n_v − 1)` et `taux(v) = Charge(v) / TVO_v`.
Défauts : `w_remplissage = 1`, `w_lits = 1`.

**Remplissage.** On ne cherche pas à remplir au maximum, on cherche à ce que
les journées d'un même praticien se *ressemblent*. Un patient ira donc
spontanément dans la vacation la moins remplie de son chirurgien, dans la
fenêtre demandée. On évite ainsi les journées à 95 %, qui débordent au moindre
aléa, et les journées à 30 %, qui mobilisent une équipe pour rien. Les
vacations vides comptent dans la variance, avec un taux de zéro : laisser une
journée à l'abandon coûte.

Ce choix a un effet mesuré et important : en rejouant les plannings avec les
durées réellement observées, **la part de vacations qui débordent tombe de
5,7 % à 2,9 %** par rapport au critère précédent. Lisser le remplissage est
plus sûr que le maximiser, et ce n'est pas un raisonnement, c'est une mesure.

**Lits.** Variance de l'occupation jour par jour, sur tous les jours de
l'horizon, week-ends compris. Divisée par le carré de l'occupation moyenne
cible, ce qui en fait un carré de coefficient de variation — sans dimension,
donc comparable au premier terme.

**Il n'y a plus de terme de délai.** Il est remplacé par la fenêtre que le
chirurgien choisit (§ suivant). Un poids de délai est une abstraction qu'il
faut calibrer, défendre, et qui applique le même arbitrage à tout le monde ;
une fenêtre est une décision clinique prise patient par patient par la
personne qui sait si le cas peut attendre.

**Les capacités ne sont pas dans l'objectif** — durée de vacation, lits,
places sont des contraintes dures. Un mouvement qui en viole une est écarté,
une date qui en viole une n'est pas proposée.

**Le nombre de patients programmés n'est pas dans l'objectif**, et ce point
mérite une explication parce que la première version le mettait. Avec un
poids, il écrasait tout : sur cette instance, le terme de report valait 5,5
contre 0,08 de remplissage et 0,11 de lits, soit 96 % de l'objectif. La
recherche ne faisait plus qu'une chose — caser des patients — et les deux
critères qu'on lui demandait d'optimiser étaient du bruit. Baisser le poids
aurait produit l'inverse : abandonner des patients pour gagner trois
décimales de lissage.

C'est le signe qu'il ne s'agit pas d'un compromis mais d'un **ordre de
priorité**. On compare donc `(patients sans date, coût)` lexicographiquement :
entre deux plannings, celui qui opère plus de monde gagne toujours ; à nombre
égal, le mieux lissé gagne. Aucun taux de change à inventer, et un poids de
moins. C'est le même principe que pour les violations du tabou local.

**Les deux poids servent** (fenêtre trois semaines, flux complet) :

| `w_remplissage` | `w_lits` | dates | sd remplissage | sd lits |
|---:|---:|---:|---:|---:|
| 1 | 0 | 1 428 | 0,284 | **4,63** |
| 1 | 1 | 1 426 | 0,296 | 4,10 |
| 0 | 1 | 1 443 | **0,333** | 4,13 |

### Les trois délais minimums

`FENETRES` dans `propositions.py` : une semaine (7 j), trois semaines (21 j),
six semaines (45 j). Le chirurgien annonce **« pas avant »** — bilan,
consultation d'anesthésie, délai de réflexion.

**C'est une borne inférieure, pas un intervalle.** Au-delà du délai minimum, la
recherche va jusqu'au bout de l'horizon. Un patient ressort donc toujours de
consultation avec une date, sauf si l'horizon entier est saturé pour son
praticien. Sur vos données, sur 431 patients sans date, **27 seulement** sont
au bord de l'horizon : les 404 autres sont les praticiens structurellement
sur-souscrits du § 6 (CL, SM, MT). Ce n'est pas un défaut de l'algorithme,
c'est le problème de répartition des vacations qui remonte à la surface.

**Le délai minimum est une donnée d'entrée du patient**
(`Patient.fenetre_jours`), au même titre que la durée estimée. Les deux moteurs
le respectent automatiquement : le moteur de consultation comme le tabou
hors-ligne.

En simulation il faut bien le fabriquer. `fenetre_uniforme(graine)` tire **une
chance sur trois** pour chacun des trois — pas parce que c'est la répartition
réelle, qu'on ne connaît pas, mais parce que c'est l'hypothèse la plus neutre :
elle n'est calée sur aucun fichier, donc le même protocole se rejoue à
l'identique sur une autre base de données. Le tirage dépend de la graine *et*
de l'identifiant du patient, pas de l'ordre de parcours : la fenêtre d'un
patient donné ne change pas si on ajoute ou retire d'autres patients. Sans
cela, comparer deux jeux de données ferait varier deux choses à la fois.

Sensibilité — la même population si tous les praticiens imposaient le même
délai minimum :

| délai minimum | dates | reports | délai médian | sd remplissage | sd lits | remplissage |
|---|---:|---:|---:|---:|---:|---:|
| pas avant 1 semaine | 1 616 | 382 | 71 j | 0,242 | **3,61** | 66,2 % |
| pas avant 3 semaines | 1 512 | 486 | 80 j | 0,293 | 4,24 | 62,1 % |
| pas avant 6 semaines | 1 316 | 682 | 99 j | 0,351 | 4,87 | 54,0 % |

Un délai minimum long ne fait pas que retarder : il retire des vacations du
champ de recherche, donc il dégrade *aussi* le lissage. Les deux effets vont
dans le même sens, il n'y a pas d'arbitrage à ce niveau — l'arbitrage est
ailleurs, dans le choix du chirurgien entre les deux dates.

### Les deux dates sont volontairement contrastées

C'est le point de conception le plus important du moteur, et il vient d'une
mesure.

Tant que la fenêtre bornait la recherche des deux côtés, proposer les deux
meilleures dates suffisait. Depuis qu'elle n'est qu'un délai minimum, **plus
rien ne tire les dates vers l'avant** : le lissage des taux de remplissage
pousse à répartir sur tout l'horizon, et les deux « meilleures » dates se
retrouvent toutes deux à quatre mois. Le chirurgien n'a alors aucun levier —
prendre la plus proche des deux faisait gagner *un* jour sur soixante-quatorze.

On construit donc deux propositions de **nature différente** :

- **au plus tôt** : la première date possible après le délai minimum ;
- **recommandée** : celle qui dégrade le moins le planning.

| le chirurgien prend… | dates | reports | délai médian | sd remplissage | sd lits |
|---|---:|---:|---:|---:|---:|
| toujours la recommandée | 1 567 | 431 | 72 j | 0,243 | **3,67** |
| toujours la plus proche | 1 778 | 196 | **49 j** | 0,269 | 4,91 |
| un mélange des deux | 1 738 | 243 | 50 j | **0,240** | 3,84 |

Le choix devient un vrai arbitrage. Celui qui prend toujours au plus tôt opère
211 patients de plus et divise le délai médian par 1,5, mais dégrade le lissage
des lits ; celui qui suit toujours la recommandation fait l'inverse. **Le
mélange des deux — c'est-à-dire la réalité d'un bloc — donne le meilleur
compromis des trois**, et le meilleur lissage du remplissage tout court.

**Un point à surveiller.** Sans terme de délai dans l'objectif, la date
« recommandée » peut tomber très loin : jusqu'à cinq mois sur certains
patients. Aucun chirurgien ne la retiendra, ce qui n'est pas grave, mais ce
n'est pas une recommandation crédible non plus. Si cela vous gêne, la parade
est de ne chercher la recommandation que parmi les N prochaines vacations du
praticien — au prix d'un paramètre N à calibrer.

### Objectif n°2 — organiser la JOURNÉE (tabou local)

**Deux familles : creux et places, lissés dans le temps.**

```
g = b_creux      · part du TVO de la journée laissée inoccupée
  + b_creux_var  · dispersion du remplissage entre les vacations du jour
  + b_pic_places · pic de places ambulatoires simultanées
  + b_var_places · variance TEMPORELLE de l'occupation des places
```

Défauts : `1`, `2`, `0,10`, `0,05`. Non normalisé, volontairement : les termes
sont déjà commensurables et on ne compare jamais deux journées entre elles,
seulement deux séquences d'une même journée.

Deux poids par famille, parce qu'il y a deux questions distinctes à chaque
fois. Pour le creux : combien de temps est perdu, et est-il réparti également
entre les salles ? Une journée où une salle finit à 12h et l'autre à 18h n'est
pas équivalente à deux salles qui finissent à 15h. Pour les places : le **pic**
dimensionne — c'est le nombre de places à équiper — et la **variance
temporelle** lisse — à pic égal, une journée régulière demande moins de
personnel de surveillance qu'une journée avec un bouchon à 11h.

La variance des places est **pondérée par la durée**, pas calculée sur la liste
des niveaux. Un profil qui vaut 10 pendant une minute puis 1 pendant huit
heures doit être jugé presque plat ; `moments_temporels` intègre sur le temps,
et `tests.py` vérifie ce cas précis.

**Ce qui n'est pas dans la fonction de coût.** Le dépassement de vacation, la
sortie d'un ambulatoire après fermeture de l'UCA et le dépassement de la
capacité en places sont des **violations**, pas des coûts. Elles sont comparées
*avant* le coût, lexicographiquement : entre deux séquences, celle qui viole le
moins gagne toujours. Ça évite d'inventer un taux de change entre « un patient
qui rentre chez lui à 20h30 » et « douze minutes de creux » — conversion qui
n'a aucun sens, et poids qu'on ne saurait pas défendre.

**Ce que l'enchaînement au plus tôt implique, mesuré.** Vous avez choisi
d'interdire l'insertion de temps mort : chaque patient entre en salle dès que
le précédent en sort, plus le TIS. C'est le bon choix d'exploitation — une
salle qui attend coûte cher — mais il faut en connaître la conséquence. Sur 25
journées :

| | creux moyen | pic places | sd places |
|---|---:|---:|---:|
| séquence initiale, sans tabou | 518 min | 5,80 | 1,88 |
| tabou, creux seul | 506 min | 5,20 | 1,65 |
| tabou, places seules | 511 min | 3,92 | **1,22** |
| **tabou, les deux (défaut)** | **508 min** | **3,92** | **1,22** |

Le creux ne peut bouger que de 2 % : sans décalage volontaire, il ne dépend de
la séquence que par un seul canal — le TIS réduit entre deux actes identiques,
donc le regroupement des actes similaires. Les places, elles, gagnent 32 % sur
le pic et 35 % sur la dispersion. Les poids par défaut atteignent quasiment
l'optimum des deux familles en même temps, il n'y a donc rien à arbitrer ici.

Si vous vouliez rendre les deux termes réellement antagonistes, il faudrait
autoriser les décalages — et accepter de faire attendre une salle.

---

## 4. Ce qui a changé par rapport à vos classes

**L'état de décision ne vit plus dans le patient.** Trois objets séparés :

- `Instance` — le problème, figé : patients, médecins, vacations, capacités.
- `Solution` — `patient -> vacation`, c'est-à-dire un JOUR (et une salle,
  puisque la vacation la porte). Pas d'heures.
- `PlanningJour` — la séquence, donc les heures, les numéros de place et de lit.

Une métaheuristique teste des millions d'états. Si l'état vit dans les objets
métier, chaque essai exige une copie profonde et la moindre incohérence se
propage. Ici, copier une solution revient à copier des dictionnaires d'entiers,
et l'évaluation d'un mouvement est incrémentale — O(nombre de nuits du patient)
au lieu de O(nombre de patients). `tests.py` vérifie que le delta annoncé égale
la variation réellement constatée ; c'est le contrôle qui compte, parce qu'une
évaluation incrémentale fausse produit un algorithme qui converge très bien vers
la mauvaise chose.

Autres changements :

- **SSPI supprimée.** Dès la sortie de salle, le patient occupe une place
  (ambulatoire) ou un lit (au moins une nuit).
- **La salle n'est plus un attribut du patient.** Elle est portée par la
  vacation : un patient placé dans une autre vacation de son chirurgien change
  de salle mécaniquement. Le lien patient–salle n'est pas immuable, le lien
  patient–chirurgien l'est (`Instance.vacations_possibles`).
- **Durées en minutes entières, jours en index entiers.** Les `datetime` ne
  servent qu'à l'affichage : 30 à 50 fois plus lents en arithmétique.
- **La marge forfaitaire de 30 min a disparu.** Chaque patient porte sa propre
  marge de risque, `P90 − médiane` de son acte. Une vacation d'actes
  prévisibles se remplit davantage ; une vacation d'actes variables se protège
  toute seule.
- **Deux capacités pour l'ambulatoire** : `capacite_places_jour` (admissions par
  jour, contrôlée à la consultation, où les heures sont inconnues) et
  `capacite_places` (pic simultané, contrôlé par le tabou local). Une place
  tourne dans la journée ; les confondre interdirait des journées parfaitement
  réalisables.

---

## 5. Résultats sur vos données (1er semestre 2022 rejoué)

1 998 patients réels, 552 vacations, 42 lits, 18 admissions ambulatoires/jour.

Fenêtres tirées uniformément, poids par défaut. Le tabou hors-ligne respecte
les mêmes fenêtres que le flux, donc la comparaison isole bien la myopie :

Délais minimums tirés uniformément, chirurgien suivant la recommandation. Le
tabou hors-ligne respecte les mêmes délais minimums que le flux, donc la
comparaison isole bien la myopie :

| | réalisé | flux de consultations | tabou hors-ligne |
|---|---:|---:|---:|
| patients programmés | — | 1 599 | **1 633** |
| écart-type de l'occupation des lits | 4,88 | **3,67** | 3,57 |
| pic de lits | 23 | 17 | **15** |
| dispersion du remplissage | — | 0,243 | 0,24 |

**Le prix de la myopie est de 34 patients sur 1 998, et 0,1 d'écart-type sur le
lissage.** Le tabou hors-ligne, qui voit tout le monde d'avance et peut tout
déplacer, ne fait guère mieux que le flux au fil de l'eau. La contrainte « une
date annoncée ne bouge plus » ne coûte donc pratiquement rien — ce n'est pas
elle qu'il faut relâcher.

Attention à la lecture : ce qui borne le résultat, ce n'est plus l'algorithme,
c'est la **capacité**. 399 patients ne reçoivent aucune date, dont 27 seulement
par effet de bord d'horizon — tout le reste vient des trois praticiens
sur-souscrits du § 6.

C'est le résultat à défendre : une fois le poids du délai correctement réglé,
donner deux dates en consultation sans jamais rien réorganiser produit un
planning presque aussi bon que celui qu'on obtiendrait en connaissant tous les
patients d'avance. **La contrainte « une date annoncée ne bouge plus » ne coûte
pratiquement rien ; ce n'est pas elle qu'il faut relâcher.**

Trois politiques de choix du chirurgien ont été simulées (suivre la
recommandation, prendre systématiquement la date la plus proche, choisir selon
la convenance du patient). L'écart entre elles est faible — écart-type des lits
2,80 / 2,99 / 2,81 — parce que les deux dates proposées sont déjà toutes deux
bonnes. **Le chirurgien peut donc choisir librement sans dégrader le planning**,
ce qui est exactement ce qu'on veut d'un outil d'aide à la décision.

### Ce que l'optimisation ne peut pas faire

Regardez la courbe hebdomadaire affichée par la démo. Après optimisation, la
variance résiduelle des lits est en grande partie l'effet week-end (12–13 lits
en semaine, 3–6 le dimanche). Aucun ordonnancement d'activité programmée ne
supprime ça : il n'y a pas de vacations le week-end. Le dire vous-même vaut
mieux que de laisser un jury le remarquer. Pour mesurer ce qui reste vraiment
optimisable, calculez la variance sur les jours ouvrés seulement.

---

## 6. Deux diagnostics à mettre dans le rapport

### La grille de vacations ne correspond pas à la demande

| praticien | patients | demandé (h) | offert (h) | taux |
|---|---:|---:|---:|---:|
| SM | 129 | 264 | 124 | **213 %** |
| MT | 180 | 267 | 221 | **121 %** |
| CL | 297 | 454 | 390 | **117 %** |
| … | | | | |
| MO | 89 | 146 | 247 | 59 % |
| JE | 15 | 30 | 98 | 30 % |
| FP | 0 | 0 | 52 | 0 % |

Trois praticiens demandent plus de temps de salle que la grille ne leur en
donne ; SM en demande deux fois plus. **Aucun algorithme d'ordonnancement ne
peut résoudre ça** : leurs patients seront reportés quoi qu'on fasse, et c'est
la principale source des 183 reports hors horizon. Pendant ce temps, MO, JE et
FP laissent des centaines d'heures inutilisées.

Le levier n'est pas l'ordonnancement, c'est la **répartition des vacations entre
praticiens** — un autre problème d'optimisation (*master surgical schedule*),
qu'on résout en déplaçant des plages, pas des patients. `diagnostic_praticiens()`
produit ce tableau ; c'est sans doute le résultat le plus actionnable du dossier.

### La colonne « Durée Séjour » du fichier est inutilisable

Sa corrélation avec la durée réellement observée est de −0,007. La première
ligne du fichier porte « 1 » pour un patient entré le 01/01 et sorti le 07/01,
et « 1 » aussi pour un vrai ambulatoire. 32 valeurs sont des numéros de série
Excel (jusqu'à 2 458 502) : une date tombée dans une colonne de durée.

Le séjour est donc recalculé par `Date Sortie − Date Inter`, et le résultat est
cliniquement cohérent — c'est le test qui compte : varices 0 nuit (100 % des
cas), canal carpien 0 nuit (97 %), prothèse de hanche 4 nuits, prothèse de genou
5 nuits, rachis lombaire complexe 3 nuits.

172 lignes ont par ailleurs un TROS nul ou négatif : écartées.

---

## 7. La grille de vacations

Votre Excel est en format visuel (cellules fusionnées, texte libre, un tableau
par jour). Elle est **retranscrite à la main** dans `donnees.GRILLE` : relisible
par le cadre de bloc, versionnable, insensible à une remise en forme du fichier.

Le cycle est de **4 semaines**, pas 2 : la grille distingue semaine paire et
impaire, mais plusieurs cases portent « 1 semaine impaire sur 2 », donc les
semaines impaires alternent elles-mêmes entre deux configurations.

Trois points **à faire valider par le cadre de bloc** :

1. **Jeudi, salle 3 (SR)** : la case porte « Stop 15h30 » mais le TVO calculé du
   fichier vaut 9,5 h, ce qui correspondrait à 8h–17h30. J'ai retenu 15h30.
2. **Mercredi, salle 5** : le TVO du fichier (7,25 h) ne correspond ni à la
   configuration paire (MT 8h–13h + urgences) ni à l'impaire (JE 8h–15h30).
3. **Lundi, salles 3 et 4** : j'ai attribué (GA / MT) et (TDO-BS + DS) à la
   première variante de semaine impaire, (GHREA / DEVOS) et (RL / MT) à la
   seconde. C'est la seule lecture qui évite que MT soit dans deux salles en même
   temps, mais elle mérite confirmation.

La **salle 1 est fermée** dans la version 28 : aucune plage ne lui est
attribuée. Ajouter des motifs avec `salle=1` permet de chiffrer ce
qu'apporterait sa réouverture — une bonne expérience pour le rapport, d'autant
que trois praticiens sont saturés.

`GHREA` et `GA` sont le même praticien (code GA = GHREA dans l'historique) ;
l'alias est géré. `URGENCES`, `LIBRE`, `DEVOS`, `DS`, `TDO/BS` et `DN` sont des
plages réservées : elles occupent de la salle mais ne reçoivent pas de patients
électifs (`RESERVES`).

Le fichier de vacations contient enfin une information **non modélisée** qui
ferait un bon prolongement : le nombre d'IBODE présents par plage horaire
(colonnes A et B — 8 à 9 le matin, 4 à 5 après 15h30). C'est une contrainte
cumulative sur le nombre de salles ouvrables simultanément, de même nature que
les lits, et elle explique probablement pourquoi certaines salles ferment à
15h30.

---

## 8. Comment tester

```
python3 tests/tests.py              # niveau 1 : correction, ~10 s
python3 experiences/banc_essai.py   # niveaux 2 à 4 : validité, bout en bout, robustesse, ~3 min
```

### La pyramide

| niveau | question | où |
|---|---|---|
| 1. Correction | le code fait-il ce qu'on croit ? | `tests.py` |
| 2. Validité statistique | triche-t-on avec les données ? | `validation_temporelle` |
| 3. Bout en bout | le planning survit-il à la réalité ? | `execution_reelle` |
| 4. Robustesse | que se passe-t-il quand on pousse ? | `escalade_charge`, `stabilite_graines` |
| 5. Qualité | est-ce mieux que l'existant ? | `demo.py` §5-6 |

Un test de correction ne dit **rien** de la qualité des plannings : il dit
seulement que l'algorithme optimise bien la chose qu'on lui a demandé
d'optimiser. Les deux questions sont distinctes et se testent séparément.

### Niveau 2 — l'estimateur voit-il le futur ?

`demo.py` calibre l'estimateur de durées sur tout l'historique 2019-2022 puis
rejoue le 1er semestre 2022 : les interventions testées servent aussi à
calculer les statistiques qui les prédisent. C'est une **fuite de données**.
La forme correcte est la validation *temporelle* — calibrer sur le passé,
tester sur le futur — jamais une validation croisée aléatoire, qui
entraînerait sur décembre pour prédire janvier.

| calibration | biais | MAE | P90 de l'erreur | couverture de la marge |
|---|---:|---:|---:|---:|
| avec fuite (2019-2022) | −2,2 min | 15,7 min | 35 min | 75,9 % |
| honnête (2019-2021) | −1,5 min | 16,3 min | 35 min | 77,6 % |

La fuite est bénigne (+0,6 min de MAE) : les statistiques par acte sont
stables d'une année sur l'autre, ce qui est rassurant en soi. Utilisez quand
même la calibration honnête dans le rapport, c'est la seule défendable.

### Niveau 3 — le test décisif

On construit le planning avec les durées **estimées**, puis on le rejoue avec
les durées **réellement observées** dans l'historique. C'est le seul test qui
réponde à « est-ce que ça marche » ; rejouer avec les durées estimées ne
testerait que la cohérence de l'algorithme avec lui-même.

| politique de durée | patients | vacations débordantes | minutes de débordement | creux réel |
|---|---:|---:|---:|---:|
| **médiane + marge P90** | 1 567 | **3,5 %** | 1 036 min | 1 410 h |
| médiane, sans marge | 1 713 | 12,2 % | 3 013 min | 1 230 h |
| P75, sans marge | 1 574 | 5,1 % | 905 min | 1 398 h |

**La marge de risque fait passer les vacations en débordement de 12,2 % à
3,5 %**, au prix de 180 heures de salle inutilisées et de 146 patients
programmés en moins. C'est l'arbitrage central du dossier, et il se tranche sur
ce tableau, pas au jugé.

Un détail qui valide la conception : la marge ne couvre que **78 % des
patients** pris un par un, mais seulement **3,5 % des vacations** débordent.
L'écart vient de la compensation — dans une vacation, les interventions plus
longues que prévu croisent les plus courtes. C'est exactement ce que suppose
`facteur_mutualisation = 0,45`, et cette ligne le confirme sur données réelles.

### Niveau 4 — escalade de charge

C'est votre idée : on réinscrit des patients tirés du fichier, avec un nouvel
identifiant et une date de consultation retirée au hasard (sinon on duplique
aussi le motif d'arrivée et on teste un flux artificiellement régulier).

| charge | patients | dates données | reports | délai médian | remplissage | sd lits | débordements réels |
|---|---:|---:|---:|---:|---:|---:|---:|
| ×1,0 | 1 998 | 1 567 | 431 | 74 j | 64,8 % | 3,63 | 3,5 % |
| ×1,1 | 2 198 | 1 598 | 600 | 76 j | 66,3 % | 3,70 | 3,9 % |
| ×1,2 | 2 398 | 1 650 | 748 | 76 j | 68,4 % | 3,71 | 5,7 % |
| ×1,3 | 2 597 | 1 669 | 928 | 78 j | 69,1 % | 3,76 | 4,9 % |
| ×1,5 | 2 997 | 1 716 | 1 281 | 78 j | 70,9 % | 3,78 | 5,3 % |

Ce qui compte n'est pas le point de rupture — on le connaît, c'est la
capacité — mais la **forme** de la dégradation. Ici elle est progressive et le
taux de report joue son rôle de soupape : le délai s'allonge et les reports
montent bien avant que les vacations ne débordent. Une dégradation brutale
signalerait un effet de seuil à comprendre avant toute mise en service.

### Niveau 4 — stabilité stochastique

Le tabou est stochastique (échantillonnage du voisinage, durée tabou tirée au
sort, perturbations). **Un résultat obtenu avec une seule graine n'est pas un
résultat, c'est un tirage.** Sur 5 graines × 3 000 itérations : coefficient de
variation de 1,6 % sur le coût, 1,9 % sur l'écart-type des lits.

Règle à appliquer partout dans le rapport : toute valeur issue du tabou s'écrit
moyenne ± écart-type sur au moins 5 graines. Si l'écart entre graines est du
même ordre que l'écart entre deux configurations comparées, la comparaison ne
veut rien dire.

Le moteur de consultation, lui, est **déterministe** — et c'est un avantage
pratique : un chirurgien qui reconsulte le système pour le même patient doit
revoir les mêmes dates.

### Non-régression

`REFERENCE` dans `banc_essai.py` fige quatre chiffres avec une tolérance,
mesurés sur la fenêtre de six semaines — celle qui laisse l'algorithme
s'exprimer ; mesurer sur une semaine reviendrait à tester une contrainte, pas
un algorithme. Une
modification innocente — un poids, un seuil, une inégalité large devenue
stricte — peut dégrader silencieusement la qualité des plannings sans casser
aucun test de correction. La tolérance n'est pas de la complaisance : elle est
calibrée sur le bruit stochastique mesuré ci-dessus. Régénérez ces valeurs
volontairement quand vous changez le modèle, ne les ajustez jamais pour faire
passer un test.

### Ce qui manque encore

1. **Stochastique sur les durées.** Ici on rejoue *la* durée observée. Un vrai
   test de robustesse tire la durée dans sa distribution (log-normale ajustée
   sur l'acte), relance 200 fois et compte la part de journées qui tiennent.
   On obtient une probabilité de dépassement au lieu d'un chiffre unique.
2. **Urgences.** Le modèle ne gère que le programmé. Injecter un flux de
   Poisson calibré sur les plages URGENCES de la grille et compter les
   déprogrammations est le test qui rapprocherait le plus la simulation d'un
   vrai bloc.
3. **Absences et fermetures.** Supprimer 5 % des vacations au hasard et
   mesurer le coût de la replanification avec `TabouGlobal.figer_avant` — le
   seul usage du tabou global en exploitation, et il n'est pas encore éprouvé.

---

## 9. Pour aller plus loin

- **La répartition des vacations** (section 6). C'est le levier à plus fort
  rendement, et il est inexploré. Formulation : affecter des plages
  salle × demi-journée aux praticiens pour minimiser la sur/sous-charge, sous
  contrainte de disponibilité des praticiens et de nombre d'IBODE. Là, l'espace
  de recherche est énorme — **c'est là qu'une métaheuristique se justifie
  vraiment**.
- **Balayer les autres poids** comme on l'a fait pour le délai, un à la fois, et
  tracer le front (remplissage en abscisse, écart-type des lits en ordonnée).
  On obtient une approximation du front de Pareto ; le point à retenir dessus est
  une décision de gestion, pas d'algorithme.
- **Calibrer `facteur_mutualisation`** (0,45 par défaut). Sommer les marges
  individuelles suppose que tous les actes dérapent en même temps : très
  pessimiste. La variance d'une somme de n aléas indépendants croît en n, son
  écart-type en √n — d'où un facteur de l'ordre de 1/√(patients par vacation).
  À calibrer sur vos dépassements réellement observés.
- **Comparer médiane et P75** pour la durée proposée (`valideur_quantile`) : le
  P75 réduit les dépassements mais crée mécaniquement du creux. Cet arbitrage se
  mesure.
- **Horizon glissant.** L'horizon est fixe à 6 mois ; les 183 reports viennent
  en grande partie du bord (4 % de reports pour les consultations du 1er mois,
  85 % pour celles du 6e). En exploitation l'horizon glisse et ces patients sont
  reprogrammés au mois suivant. Pour une mesure propre, n'évaluez que les
  consultations des trois premiers mois.
- **Stochastique.** Tout ici est déterministe. L'étape suivante est de tirer les
  durées réelles dans leur distribution et de mesurer combien de plannings
  tiennent — c'est ce qui valide vraiment la marge de risque.
