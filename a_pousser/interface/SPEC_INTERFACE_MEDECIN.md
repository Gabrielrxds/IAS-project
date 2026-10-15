# Interface de programmation pour le chirurgien — spécification et instructions Claude Code

> **À qui s'adresse ce document.** À Claude Code (ou à un membre du groupe) qui construit
> l'interface. Il décrit **ce qu'il faut construire**, **comment l'organiser** et **dans quel
> ordre**, avec des critères d'acceptation pour chaque étape.
>
> **Règle n°1 : aucun modèle dans cette version.** Ni le modèle de durée (ML), ni l'algorithme
> de dates (tabou, recuit, génétique, hybride) ne sont branchés : ils sont encore en test.
> L'interface appelle des **interfaces de code** (§4) derrière lesquelles on met pour l'instant
> des **bouchons** clairement signalés à l'écran. Brancher un vrai modèle plus tard ne doit
> demander aucune modification de l'interface.

---

## 1. Objectif

Un outil web que le chirurgien utilise pour :

1. **Programmer une intervention** pendant la consultation : il ouvre le dossier du patient, voit
   la durée estimée de l'intervention, donne son avis sur cette durée, puis l'algorithme lui
   propose **deux jours** et il en choisit un. La date est alors définitive.
2. **Voir son planning et celui des blocs** : ses vacations, ses patients, le remplissage de chaque
   vacation ; et l'occupation des salles 2 à 5.
3. **Saisir la durée réelle** après l'opération, ce qui alimente l'apprentissage du modèle de
   durée **de ce chirurgien**.
4. **Déclarer ses absences** (congé, formation, maladie). Ses vacations de la période sont
   fermées et les patients déjà programmés sont **reprogrammés** : le système propose une nouvelle
   date pour chacun, le chirurgien (ou le secrétariat) valide.
5. **Déclarer qu'un patient n'est pas venu** le jour de son opération : il est reprogrammé
   **sans priorité**, comme une nouvelle demande vue ce jour-là.
6. **Déclarer tout autre aléa** avec un formulaire unique (motif, raison, durée estimée) :
   - médecin indisponible quelques heures ;
   - salle indisponible ;
   - matériel manquant ;
   - lits fermés ;
   - annulation par le patient ;
   - patient non apte.

   Le système relogge, décale ou ferme les vacations touchées, reprogramme les patients et propose
   les places libérées aux patients en attente (§2.6, `GESTION_ALEAS.md`).

Le processus suit le paradigme du projet : **un seul rendez-vous**, la date est donnée à la
consultation et ne bouge plus — **sauf** si le chirurgien déclare une absence (§2.4) ou si le
patient ne vient pas (§2.5) ou si un autre aléa est déclaré (§2.6). Toutes ces exceptions sont
tracées et validées par un humain.

---

## 2. Les six parcours

### 2.1 Parcours A — Prise de rendez-vous (pendant la consultation)

```
[1] Dossier patient ──► [2] Durée estimée + avis du chirurgien ──► [3] Deux dates proposées ──► [4] Confirmation
```

**Étape 1 — Dossier patient**
- Rechercher un patient existant (nom, date de naissance, identifiant) ou en créer un.
- Champs du patient :
  - identité : nom, prénom, date de naissance (→ **âge** calculé), sexe ;
  - **diagnostic principal** (code CIM-10, saisie avec autocomplétion sur une table de codes) ;
  - **acte chirurgical** (code CCAM, autocomplétion sur le code et le libellé) ;
  - type de prise en charge : ambulatoire / hospitalisation, et nombre de nuits prévues si
    hospitalisation ;
  - délai minimum avant l'opération (par défaut 7 jours) et délai maximum (par défaut 6 mois) ;
  - urgence : programmé / semi-urgence (affiché, pas encore géré par l'algorithme) ;
  - commentaire libre.
- Les trois paramètres du modèle de durée sont **âge, diagnostic principal, acte** : ils sont
  obligatoires pour passer à l'étape 2.

**Étape 2 — Durée estimée et avis du chirurgien**
- Afficher la durée estimée renvoyée par l'estimateur (§4.1) : valeur centrale en minutes,
  intervalle bas–haut si disponible, nombre d'exemples sur lesquels l'estimation repose, version
  du modèle. **Tant que c'est le bouchon**, afficher un bandeau « Estimation factice — modèle non
  branché ».
- Le chirurgien répond à la question **« Selon vous, cette intervention prendra : plus de temps /
  autant / moins de temps ? »** (trois boutons, réponse obligatoire).
- Champ facultatif : sa propre estimation en minutes.
- On enregistre l'estimation affichée **et** l'avis. La durée utilisée pour réserver la vacation
  vaut, dans cette version : l'estimation du modèle, ou celle du chirurgien s'il l'a saisie. La
  règle de combinaison est configurable (§4.1) et sera décidée après les tests.

**Étape 3 — Deux dates proposées**
- L'interface appelle le moteur de propositions (§4.2) avec le patient, la durée retenue, le
  chirurgien et la date du jour.
- Afficher **exactement deux cartes**, chacune pour **un jour différent**, avec :
  - la date (jour de la semaine en toutes lettres) et le délai en jours depuis aujourd'hui ;
  - la vacation : salle, horaire ;
  - le remplissage de la vacation avant / après ajout du patient (barre) ;
  - pour un hospitalisé : les lits occupés prévus sur ses nuits ;
  - une étiquette fournie par le moteur (ex. « au plus tôt », « recommandée ») et un court motif.
- Si le moteur ne trouve rien : message clair (« aucune vacation disponible dans les 6 mois »),
  pas de date, patient enregistré comme « sans date ».
- **Tant que c'est le bouchon**, bandeau « Propositions provisoires — algorithme non branché ».

**Étape 4 — Confirmation**
- Le chirurgien clique sur une des deux cartes, puis sur « Confirmer ». L'intervention passe au
  statut **programmée** ; la date est **verrouillée** (pas de bouton de modification dans cette
  version).
- Écran récapitulatif : patient, acte, date, salle, horaire de la vacation, durée réservée.
- On enregistre aussi **laquelle des deux propositions** a été choisie (rang 1 ou 2) : c'est une
  donnée utile pour évaluer l'algorithme plus tard.

### 2.2 Parcours B — Plannings

**Planning du chirurgien**
- Vue semaine (par défaut) et vue mois ; navigation précédent / suivant / aujourd'hui.
- Chaque vacation du chirurgien : salle, horaire, **barre de remplissage** (minutes réservées /
  TVO), liste de ses patients (nom, acte, durée réservée, ambulatoire ou nombre de nuits).
- Code couleur du statut des interventions : programmée, réalisée (durée saisie), à saisir
  (opération passée sans durée réelle).
- Clic sur un patient → fiche de l'intervention.

**Planning des blocs**
- Grille **salles (1 à 5) × jours** de la semaine (la salle 1, vide dans la grille hybride, est la
  **salle de repli** pour les relogements), matin / après-midi, avec le chirurgien de
  chaque vacation (couleur par chirurgien), les créneaux d'urgence et les créneaux libres.
- Remplissage de chaque vacation (barre ou pourcentage).
- Sous la grille : **courbe des lits occupés** prévus chaque nuit de la semaine (calculée à partir
  des interventions programmées et des nuits prévues), et nombre d'admissions ambulatoires par
  jour.
- Sélecteur de semaine. La grille de vacations est un **cycle de 4 semaines** (paire, impaire A,
  paire, impaire B) déroulé sur le calendrier ; les jours fériés sont fermés.

### 2.3 Parcours C — Durée réelle et apprentissage

**Saisie post-opératoire**
- Une liste « Interventions à compléter » : interventions passées sans durée réelle, triées par
  date.
- Pour chacune, saisir l'**heure d'entrée en salle et l'heure de sortie de salle** (la durée
  réelle est calculée : c'est le **TROS**, la grandeur des données historiques). Champ « durée
  directe en minutes » en alternative. Validation : durée entre 5 min et 12 h.
- À la validation : l'intervention passe au statut **réalisée** et un **exemple
  d'apprentissage** est créé pour le modèle de ce chirurgien :
  `(âge, diagnostic, acte) → durée réelle`, avec en plus l'estimation affichée et l'avis du
  chirurgien (pour mesurer plus tard si son avis améliore l'estimation).

**Suivi du modèle (par chirurgien)**
- Nombre d'exemples (historiques importés + saisis dans l'outil), date du dernier entraînement,
  version.
- Erreur de l'estimation sur les interventions réalisées dans l'outil : erreur absolue moyenne
  (min), part des interventions qui ont dépassé l'estimation.
- **Qualité de l'avis du chirurgien** : quand il a dit « plus », la durée réelle était-elle
  effectivement plus longue ? Tableau 3 × 3 (avis × résultat réel : plus long / ±10 % / plus
  court).
- Bouton **« Réentraîner »** : appelle `entrainer()` (§4.1). Avec le bouchon, il répond « modèle
  non branché » et rien ne change. Prévoir aussi le réentraînement automatique tous les
  N nouveaux exemples (N configurable, désactivé par défaut).

### 2.4 Parcours D — Absences et reprogrammation

```
[1] Déclarer l'absence ──► [2] Impact (vacations fermées, patients touchés) ──► [3] Propositions de nouvelles dates ──► [4] Validation + avis aux patients
```

**Étape 1 — Déclarer une absence**
- Page « Mes absences » (calendrier) : sélectionner une **période** (date de début, date de fin,
  matin / après-midi / journée), un **motif** : congé, formation, maladie, autre ; commentaire libre.
- Une absence peut être déclarée par le chirurgien lui-même ou par le secrétariat / le cadre de
  bloc (champ « déclaré par »).
- Avertissement si la période contient des journées **déjà figées** (J−7) ou passées.
- Les absences connues à l'avance (congés) doivent être saisies **le plus tôt possible** : le
  système bloque alors ces jours pour les nouvelles prises de rendez-vous et il n'y a presque rien
  à reprogrammer (§2.4, encadré « pourquoi déclarer tôt »).

**Étape 2 — Impact**
- Liste des **vacations fermées** (date, salle, horaire) et des **patients touchés** (nom, acte,
  date initiale, ambulatoire / nuits, délai maximum restant).
- Les vacations fermées passent au statut `fermee` : elles n'apparaissent plus dans les
  propositions de dates. Dans le planning des blocs, la salle apparaît **« libérée »** (hachurée)
  pour que le cadre puisse la proposer à un autre chirurgien.

**Étape 3 — Propositions de nouvelles dates**
- Pour **chaque patient touché**, le moteur de reprogrammation (§4.3) renvoie **deux jours**
  (comme au parcours A), même chirurgien, ou « à arbitrer » s'il ne trouve rien avant le délai
  maximum du patient.
- Ordre de traitement : **date initiale la plus proche d'abord** (c'est le patient qui a le
  moins de marge).
- Option **« Ouvrir des vacations de rattrapage »** (cochée par défaut pour une maladie) : le
  moteur peut proposer une demi-journée de salle **libre** dans les 3 semaines suivant le retour,
  dans la limite du temps de bloc perdu. Elle n'est créée qu'après validation par le cadre de bloc
  (statut `rattrapage_proposee` → `rattrapage_validee`).
- Affichage par patient : date initiale → date proposée (rang 1, rang 2), décalage en jours,
  étiquette (« rattrapage », « au plus tôt », « recommandée »).

**Étape 4 — Validation et information des patients**
- Bouton « Valider » par patient et « Tout valider ». À la validation, l'intervention change de
  vacation, son ancienne date est conservée dans l'historique (`reprogrammation`) et le statut
  passe à `programmee` avec un indicateur « reprogrammée ».
- Liste « Patients à prévenir » (téléphone / courrier) avec la nouvelle date ; case « prévenu le ».
- Les patients « à arbitrer » restent visibles dans une liste dédiée jusqu'à une décision manuelle.

**Règles**
- Le chirurgien ne change jamais (règle du projet) ; pas de nouvelle date à moins de 2 jours de
  la validation ni après le délai maximum du patient.
- Congé (déclaré ≥ 8 jours avant) : uniquement des journées non figées. Maladie (préavis court) :
  aussi les journées figées de la semaine s'il reste de la place.
- Une absence peut être **annulée** tant que ses reprogrammations ne sont pas validées ; après,
  les vacations sont rouvertes mais les patients déjà prévenus ne bougent plus.

> **Pourquoi déclarer tôt (simulation sur 2022, grille hybride).** Les patients sont vus 30 jours
> avant leur opération. Un congé déclaré 42 jours avant touche 189 patients déjà datés ; déclaré
> 3 jours avant, 316. Voir `REPROGRAMMATION_ABSENCES.md`.

### 2.5 Parcours E — Patient non venu

```
[1] « Non venu » sur l'intervention du jour ──► [2] Nouvelle demande (sans priorité) ──► [3] Deux dates proposées ──► [4] Confirmation
```

- **Où** : dans le planning du chirurgien et dans la liste des interventions du jour, un bouton
  **« Patient non venu »** sur toute intervention dont la date est aujourd'hui (ou passée, si
  elle n'a pas de durée réelle saisie).
- **Saisie** : motif (absent sans prévenir, annulation le jour même par le patient,
  contre-indication découverte à l'admission, autre), commentaire.
- **Effet immédiat** : l'intervention passe au statut `non_venu`. Sa place du jour est perdue ;
  la vacation affiche « place libérée » (le bloc peut la donner à une urgence).
- **Règle de reprogrammation (simple)** : le patient est traité **comme une nouvelle demande vue
  en consultation aujourd'hui** :
  - délai minimum et délai maximum recomptés **à partir d'aujourd'hui** ;
  - **aucune priorité** : même moteur de propositions que le parcours A (§4.2), mêmes contrôles,
    pas de vacation de rattrapage, pas de passage devant les autres patients ;
  - le chirurgien (ou le secrétariat) choisit l'un des deux jours proposés et confirme ; une
    **nouvelle intervention** est créée, liée à l'ancienne (historique conservé).
- La durée estimée et l'avis du chirurgien de la première demande sont repris par défaut
  (modifiables).
- Liste **« À reprogrammer — non venus »** pour ceux qui n'ont pas encore reçu de nouvelle date.
- Le nombre de non-venues du patient est affiché ; à partir de la 2ᵉ, simple **alerte** au
  chirurgien (aucune règle automatique).


### 2.6 Parcours F — Déclarer un aléa (formulaire unique)

```
[1] Déclarer (quoi, motif, raison, début, durée estimée) ──► [2] Impact ──► [3] Actions proposées ──► [4] Validation + patients à prévenir
                                                            └──► en cours : « Prolonger » / « Terminer maintenant »
```

Les parcours D (absence) et E (non venu) sont des cas particuliers de ce formulaire ; leurs écrans
restent accessibles directement.

**Étape 1 — Déclarer.** Bouton « Signaler un aléa » dans la barre du haut. Formulaire en trois temps :

1. **Quoi :**
   - un médecin ;
   - une salle ;
   - un matériel (choix des types d'intervention concernés) ;
   - des lits (nombre) ;
   - un patient (recherche).
2. **Motif** (liste propre à chaque famille, cf. tableau ci-dessous), et **raison** en texte libre.
3. **Quand :**
   - début (date et heure) ;
   - **durée estimée** : champ numérique + unité (min / h / j / sem) ;
   - ou date de fin ;
   - pour un patient non apte, **« apte dans »** (durée estimée) ;
   - pour une annulation, **définitive** ou **report à partir du** (date).

| Famille | Motifs | Qui déclare |
|---|---|---|
| Médecin | congé, maladie, formation, urgence dans un autre service, retard, personnel, autre | chirurgien, secrétariat, cadre |
| Salle | panne de matériel, stérilisation, maintenance, dégât des locaux, personnel manquant, autre | cadre de bloc |
| Matériel | ancillaire indisponible, rupture d'implant, colonne d'endoscopie, autre | cadre de bloc, pharmacie |
| Lits | épidémie, travaux, manque de personnel, autre | cadre de l'unité d'hospitalisation |
| Annulation (patient) | choix du patient, empêchement, décès, transfert, autre | secrétariat |
| Non apte (patient) | infection, bilan incomplet, traitement à arrêter, décompensation, autre | chirurgien, anesthésiste |

**Étape 2 — Impact.** Liste de ce que l'aléa touche :
- vacations avec l'action prévue : **relogée** (salle → salle), **décalée** (nouvel horaire),
  **raccourcie**, **fermée** ;
- patients à déplacer ;
- nuits en **déficit de lits**, si on ne peut rien déplacer (patients déjà admis) ;
- places libérées à proposer (comblement).

**Étape 3 — Actions proposées** (moteur §4.4) :
- **relogements et décalages** : à valider par le cadre de bloc, car il faut une équipe ;
- **patients déplacés** : deux jours proposés, ou « à arbitrer » (même écran que le parcours D) ;
- **comblement** : pour chaque place libérée ≥ 2 jours à l'avance, une liste ordonnée de patients
  du même chirurgien qui pourraient avancer (le plus ancien demandeur d'abord). La secrétaire les
  appelle dans l'ordre, coche « accepte » ou « refuse », et la place est donnée au premier qui
  accepte ;
- **rattrapage** (option) : demi-journées libres proposées au cadre, comme au parcours D.

**Étape 4 — Validation.**
- « Valider » par ligne, ou « Tout valider ».
- Liste « Patients à prévenir », avec la nouvelle date et une case « prévenu le ».
- Liste « À arbitrer ».

**Aléa en cours.** Il apparaît dans un bandeau (planning des blocs et planning du chirurgien) avec
deux boutons :
- **« Prolonger »** : nouvelle durée estimée ; crée un aléa lié qui couvre la suite.
- **« Terminer maintenant »** : rouvre les vacations encore à venir, à J+2 au plus tôt, et lance un
  comblement sur les places rouvertes. Les patients déjà déplacés et prévenus ne reviennent pas.

**Règles** (détail dans `GESTION_ALEAS.md` §3) :
- le chirurgien ne change jamais ;
- pas de nouvelle date à moins de 2 jours ;
- délai maximum respecté, sinon « à arbitrer » ;
- aléa déclaré ≥ 8 jours avant son début : seulement des journées non figées ;
- une vacation raccourcie à moins de 60 min est fermée ;
- un **patient non apte** est replacé **en priorité** à partir de sa date d'aptitude, avec son délai
  maximum prolongé d'autant ;
- un **report** demandé par le patient est une nouvelle demande **sans priorité**, à partir de la
  date choisie ;
- une **annulation définitive** clôt l'intervention (statut `annulee`) ;
- le comblement ne propose que des patients du même chirurgien, à une date plus tardive et non
  figée.

> **Pourquoi c'est utile (simulation sur 2022, tous aléas ensemble).** Avec relogement, rattrapage
> et comblement, le nombre de patients « à arbitrer » passe de 45 à 22 et le décalage moyen de 34 à
> 13 jours ; 46 annulations sur 56 sont comblées. Voir `GESTION_ALEAS.md` §5.

---

## 3. Architecture recommandée

| Couche | Choix | Pourquoi |
|---|---|---|
| Back-end | **Python 3.11 + FastAPI** | Le code du groupe (modèle commun, algorithmes) est en Python : on le branchera directement. |
| Base de données | **SQLite** via **SQLModel** (ou SQLAlchemy) | Un fichier, rien à installer ; suffisant pour un prototype. |
| Front-end | **React + Vite + TypeScript** | Composants pour le parcours en étapes et les plannings. Plannings dessinés en **CSS Grid**, sans bibliothèque de calendrier lourde. |
| Graphiques | une petite bibliothèque (ex. Recharts) ou SVG à la main | Courbe des lits, barres de remplissage. |
| Tests | **pytest** (back), **Vitest** + Testing Library (front) | |

> Alternative plus rapide mais moins souple : une seule application **Streamlit**. À éviter si
> l'on veut un vrai parcours en étapes et des plannings cliquables.

### Arborescence

```
interface/
├── SPEC_INTERFACE_MEDECIN.md      ← ce document
├── CLAUDE.md                      ← consignes pour Claude Code (cf. §7)
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI, routes
│   │   ├── db.py                  # moteur SQLite, session
│   │   ├── modeles_db.py          # tables (§5)
│   │   ├── schemas.py             # entrées / sorties de l'API
│   │   ├── services/
│   │   │   ├── planning.py        # déroulage de la grille, remplissage, lits prévus
│   │   │   ├── rendezvous.py      # parcours A : création, verrouillage
│   │   │   └── apprentissage.py   # parcours C : exemples, métriques, déclenchement
│   │   ├── estimation/
│   │   │   ├── interface.py       # Protocol EstimateurDuree (§4.1)
│   │   │   └── bouchon.py         # EstimateurBouchon
│   │   ├── propositions/
│   │   │   ├── interface.py       # Protocol MoteurPropositions (§4.2)
│   │   │   └── bouchon.py         # MoteurBouchon
│   │   ├── reprogrammation/
│   │   │   ├── interface.py       # Protocol MoteurReprogrammation (§4.3)
│   │   │   └── bouchon.py         # ReprogrammationBouchon
│   │   └── aleas/
│   │       ├── interface.py       # Protocol MoteurAleas (§4.4)
│   │       └── bouchon.py         # AleasBouchon
│   ├── scripts/
│   │   ├── importer_historique.py # Excel anonymisé -> tables de référence + exemples
│   │   └── importer_grille.py     # grille V28 -> vacations du cycle
│   └── tests/
└── frontend/
    └── src/
        ├── pages/ (RendezVous, PlanningMedecin, PlanningBlocs, ACompleter, SuiviModele, Absences, NonVenus,
        │           Aleas, Comblement, AArbitrer)
        ├── components/ (EtapesRdv, CarteProposition, GrilleBlocs, BarreRemplissage, CourbeLits, BandeauBouchon)
        └── api/ (client typé)
```

---

## 4. Les interfaces de modèle (le point le plus important)

L'interface ne connaît **que** ces quatre contrats. Un modèle est choisi par configuration
(`ESTIMATEUR=bouchon`, `MOTEUR=bouchon`, `REPROGRAMMATION=bouchon`, `ALEAS=bouchon`), jamais en dur
dans les routes.

### 4.1 Estimateur de durée (un modèle par chirurgien)

```python
class Estimation(BaseModel):
    minutes: int                 # valeur centrale
    bas: int | None              # intervalle (ex. P10) si disponible
    haut: int | None             # ex. P90
    n_exemples: int              # exemples du chirurgien utilisés
    version: str                 # ex. "bouchon-0", "gbm-2026-11-02"
    source: str                  # "bouchon" | "modele"

class RapportEntrainement(BaseModel):
    medecin_id: int
    version: str
    n_exemples: int
    erreur_abs_moyenne: float | None
    message: str

class EstimateurDuree(Protocol):
    def predire(self, medecin_id: int, age: int, diagnostic: str, acte: str) -> Estimation: ...
    def entrainer(self, medecin_id: int, exemples: list[ExempleApprentissage]) -> RapportEntrainement: ...
```

**Bouchon (`EstimateurBouchon`)** :
- `predire` renvoie une valeur fixe configurable (par défaut 60 min), `bas = haut = None`,
  `source = "bouchon"`. **Pas de statistique ni d'apprentissage.**
- `entrainer` ne fait rien et renvoie `message = "modèle non branché"`.

**Combinaison avec l'avis du chirurgien** (fonction séparée, configurable) : dans cette version,
`duree_reservee = estimation_chirurgien si saisie, sinon estimation.minutes`. L'avis
« plus / autant / moins » est **enregistré mais n'ajuste rien** ; son usage sera décidé après les
tests.

### 4.2 Moteur de propositions de dates

```python
class Proposition(BaseModel):
    jour: date
    vacation_id: int
    salle: int
    debut: time
    fin: time
    remplissage_avant: float     # 0..1
    remplissage_apres: float
    lits_prevus: list[int]       # lits occupés prévus sur les nuits du patient
    etiquette: str               # ex. "au plus tôt", "recommandée"
    motif: str                   # phrase courte affichée au chirurgien

class MoteurPropositions(Protocol):
    def proposer(self, intervention: DemandeIntervention, aujourd_hui: date) -> list[Proposition]:
        """0 ou 2 propositions, sur deux JOURS distincts, réalisables
        (capacité de la vacation, lits, places ambulatoires, délai min/max)."""
```

**Bouchon (`MoteurBouchon`)** : les deux **premiers jours** où une vacation du chirurgien peut
accueillir la durée réservée (TVO − minutes déjà réservées − 15 min de temps inter-salle par
patient déjà présent), à partir de `aujourd'hui + délai minimum`. Étiquettes « 1re date
disponible » et « 2e date disponible ». Aucune optimisation.

**Contrôles de faisabilité** (dans un service commun, utilisés par tous les moteurs) :
- la vacation est celle du chirurgien et n'est pas un créneau d'urgence ;
- la charge après ajout reste sous le TVO ;
- la date est dans `[aujourd'hui + délai min, aujourd'hui + délai max]` et n'est ni un week-end
  ni un jour férié ;
- capacité de lits (42) et d'admissions ambulatoires par jour (18) non dépassées.

### 4.3 Moteur de reprogrammation (absences)

```python
class Absence(BaseModel):
    medecin_id: int
    debut: date
    fin: date                      # inclus
    demi_journee: Literal["matin", "apres_midi", "journee"]
    motif: Literal["conge", "formation", "maladie", "autre"]
    declaree_le: date

class ImpactAbsence(BaseModel):
    vacations_fermees: list[int]
    interventions_touchees: list[int]       # triées par date initiale

class VacationRattrapage(BaseModel):
    jour: date
    salle: int
    debut: time
    fin: time
    minutes_programmables: int

class PropositionReprogrammation(BaseModel):
    intervention_id: int
    date_initiale: date
    propositions: list[Proposition]         # 0 ou 2, mêmes champs qu'au §4.2
    rattrapages: list[VacationRattrapage]   # demi-journées libres proposées (peut être vide)
    statut: Literal["proposee", "a_arbitrer"]

class MoteurReprogrammation(Protocol):
    def impact(self, absence: Absence) -> ImpactAbsence: ...
    def reprogrammer(self, absence: Absence, aujourd_hui: date,
                     rattrapage: bool) -> list[PropositionReprogrammation]: ...
```

**Bouchon (`ReprogrammationBouchon`)** : ferme les vacations, puis, pour chaque intervention
touchée (date initiale croissante), les **deux premiers jours** réalisables du même chirurgien à
partir de `aujourd'hui + 2` (mêmes contrôles de faisabilité qu'au §4.2, vacations fermées
exclues). Pas de rattrapage. Bandeau « Reprogrammation provisoire — algorithme non branché ».

**Implémentation de référence (à brancher plus tard)** : `version_finale/hybride/absences.py`
(`Reprogrammateur`) : même score que la règle hybride (variance des lits attendus avec patients
fantômes), et ouverture de vacations de rattrapage dans les demi-journées de salle libres.


### 4.4 Moteur des aléas

```python
Famille = Literal["medecin", "salle", "materiel", "lits", "annulation", "non_apte", "non_venu"]

class AleaSaisi(BaseModel):
    famille: Famille
    motif: str                          # liste fermée par famille (GESTION_ALEAS.md §2)
    raison: str = ""
    declare_le: datetime
    debut: datetime | None              # ressources
    duree_estimee: str | None           # « 90 min », « 3 h », « 2 j » ; ou :
    fin: datetime | None
    medecin_id: int | None = None
    salle_id: int | None = None
    types_intervention: list[str] = []  # matériel
    nb_lits: int | None = None
    intervention_id: int | None = None  # patient
    report_a_partir: date | None = None # annulation avec report
    apte_a_partir: date | None = None   # non apte
    parent_id: int | None = None        # prolongation

class ActionVacation(BaseModel):
    vacation_id: int
    action: Literal["relogee", "decalee", "raccourcie", "fermee", "rouverte"]
    avant: VacationVue                  # salle, début, fin
    apres: VacationVue | None

class PlaceACombler(BaseModel):
    vacation_id: int
    jour: date
    minutes_libres: int
    candidats: list[int]                # interventions, dans l'ordre d'appel

class ImpactAlea(BaseModel):
    vacations: list[ActionVacation]
    interventions_a_deplacer: list[int]
    nuits_en_deficit: list[date]
    places_a_combler: list[PlaceACombler]

class MoteurAleas(Protocol):
    def impact(self, alea: AleaSaisi) -> ImpactAlea: ...
    def proposer(self, alea: AleaSaisi, aujourd_hui: date,
                 rattrapage: bool) -> list[PropositionReprogrammation]: ...   # même type qu'au §4.3
    def lever(self, alea_id: int, aujourd_hui: date) -> ImpactAlea: ...      # « Terminer maintenant »
```

**Bouchon (`AleasBouchon`)** :
- **Médecin** : ferme les vacations touchées (pas de décalage).
- **Salle** : reloge dans la première salle libre de 1 à 5 sur la même plage horaire, sinon ferme.
- **Matériel** : déplace les interventions des types concernés.
- **Lits** : signale seulement les nuits en déficit.
- **Patients déplacés** : les deux premiers jours réalisables, comme au §4.3.
- **Comblement** : candidats du même chirurgien triés par date de consultation.
- Bandeau « Gestion des aléas provisoire — algorithme non branché ».

**Implémentation de référence (à brancher plus tard)** : `version_finale/aleas/gestion_aleas.py`
(`GestionnaireAleas`). Elle hérite du `Reprogrammateur` du §4.3 et ajoute :
- le relogement ;
- le décalage ;
- le raccourcissement ;
- le filtre matériel ;
- la fermeture de lits ;
- le comblement ;
- la levée.

`tester_aleas.py` donne les invariants à reprendre dans les tests de l'interface.

---

## 5. Données

### 5.1 Tables

| Table | Champs principaux |
|---|---|
| `medecin` | id, code (ex. « CL »), nom affiché, spécialité, couleur |
| `salle` | id, numéro (2-5), actif |
| `creneau_grille` | id, semaine_cycle (0-3), jour_semaine (0-4), salle_id, debut, fin, medecin_id \| null, type (programme / urgence / libre) |
| `vacation` | id, date, salle_id, medecin_id, debut, fin, tvo_min — déroulée depuis `creneau_grille` |
| `patient` | id, nom, prenom, date_naissance, sexe, commentaire |
| `ref_diagnostic` | code CIM-10, libellé |
| `ref_acte` | code CCAM, libellé |
| `intervention` | id, patient_id, medecin_id, diagnostic, acte, age_a_la_consultation, ambulatoire, nuits_prevues, delai_min, delai_max, urgence, date_consultation, statut (brouillon / programmee / sans_date / realisee / annulee / non_venu), vacation_id \| null, duree_reservee_min, intervention_precedente_id \| null |
| `estimation` | id, intervention_id, minutes, bas, haut, version, source, avis_medecin (plus / egal / moins), estimation_medecin_min \| null, horodatage |
| `proposition_faite` | id, intervention_id, rang (1-2), jour, vacation_id, etiquette, moteur_version, choisie (bool) |
| `realisation` | intervention_id, heure_entree_salle, heure_sortie_salle, duree_reelle_min, saisie_par, horodatage |
| `exemple_apprentissage` | id, medecin_id, age, diagnostic, acte, duree_reelle_min, origine (historique / outil), intervention_id \| null |
| `version_modele` | id, medecin_id, version, date, n_exemples, erreur_abs_moyenne, actif |
| `absence` | id, medecin_id, debut, fin, demi_journee, motif, commentaire, declaree_le, declaree_par, statut (active / annulee) |
| `vacation` (ajouts) | statut (ouverte / fermee / rattrapage_proposee / rattrapage_validee), absence_id \| null, origine (grille / rattrapage) |
| `non_venue` | id, intervention_id, jour, motif, commentaire, declaree_par, horodatage, nouvelle_intervention_id \| null |
| `reprogrammation` | id, intervention_id, absence_id, vacation_initiale_id, date_initiale, vacation_nouvelle_id \| null, date_nouvelle \| null, rang_choisi, statut (proposee / validee / a_arbitrer), validee_par, validee_le, patient_prevenu_le \| null |
| `alea` | id, famille, motif, raison, declare_le, declare_par, debut, fin, duree_estimee (texte saisi), medecin_id \| null, salle_id \| null, types_intervention (json) \| null, nb_lits \| null, intervention_id \| null, report_a_partir \| null, apte_a_partir \| null, parent_id \| null, statut (en_cours / termine / annule), termine_le \| null |
| `action_alea` | id, alea_id, action (fermee / raccourcie / decalee / relogee / rouverte / liberee / reprogramme / a_arbitrer / avance / annule / nouvelle_demande / deficit_lits / rattrapage), vacation_id \| null, intervention_id \| null, avant (json), apres (json), statut (proposee / validee / refusee), validee_par, validee_le |
| `appel_comblement` | id, alea_id, vacation_id, intervention_id, rang, reponse (en_attente / accepte / refuse / injoignable), appele_le, appele_par |
| `vacation` (ajouts) | statut (+ relogee / decalee / raccourcie), alea_id \| null, salle_origine_id \| null, debut_origine, fin_origine |

**Règles** :
- une intervention `programmee` a une `vacation_id` et ne peut plus en changer (verrou côté API),
  **sauf** par une `reprogrammation` validée liée à une `absence` ;
- aucune intervention ne peut être programmée dans une vacation `fermee` ;
- une intervention programmée peut aussi changer de vacation par une `action_alea` validée
  (reprogramme, avance), et seulement ainsi en dehors des parcours D et E ;
- une vacation `relogee`, `decalee` ou `raccourcie` garde sa forme d'origine (`*_origine`), ce qui
  permet de la rouvrir si l'aléa se termine plus tôt ;
- « non venu » n'est possible que le jour de l'intervention ou après, et seulement si aucune
  durée réelle n'est saisie ; la nouvelle intervention a `date_consultation` = jour de la
  déclaration ;
- une seule `estimation` et deux `proposition_faite` par intervention programmée ;
- `realisation` n'est possible que si la date de la vacation est passée.

### 5.2 Import des données du projet

- **Historique** (`donees bloc anonyme pour centrale 2026.xlsx`) :
  - `ref_diagnostic` ← colonne « CIM Diag Pr. » (617 codes, toujours renseignée) ;
  - `ref_acte` ← « CCAM 1 » comme **code** de l'acte (570 codes). Le libellé affiché est le texte
    « Interv Type » le plus fréquent pour ce code ; ne pas utiliser « Interv Type » comme code :
    c'est du texte libre (1 433 variantes) ;
  - `medecin` ← colonne « Praticien » ;
  - `exemple_apprentissage` (origine = historique) avec :
    - âge = Date Inter − Date Naissance, en années (date de naissance renseignée sur toutes les
      lignes) ;
    - durée = **heure de sortie de salle − heure d'entrée en salle (calimed)**. Ne pas prendre
      la colonne d'entrée en SSPI, qui a un intitulé voisin.

  Les données sont anonymisées : **aucun vrai patient n'est créé**. Pour la démonstration,
  générer des patients fictifs marqués « démo ».
- **Grille** : reprendre la transcription de la version 28 de `fil_rouge_AG/grille.py` (cycle de
  4 semaines, créneaux URGENCES et LIBRE inclus), puis dérouler les vacations sur 12 mois, sans
  les jours fériés.

### 5.3 Confidentialité

Prototype local, **données anonymisées ou fictives uniquement**. Aucune donnée réelle de patient
tant que l'outil n'est pas hébergé sur une infrastructure agréée (HDS) et validé par l'hôpital.
Pas d'envoi de données vers des services externes.

---

## 6. API (première version)

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/medecins` | liste |
| GET | `/references/diagnostics?q=` , `/references/actes?q=` | autocomplétion |
| GET/POST | `/patients`, `/patients/{id}` | recherche, création |
| POST | `/interventions` | crée une intervention en brouillon (étape 1) |
| POST | `/interventions/{id}/estimation` | appelle l'estimateur, renvoie `Estimation` |
| PUT | `/interventions/{id}/avis` | enregistre avis + estimation du chirurgien, fixe la durée réservée |
| POST | `/interventions/{id}/propositions` | appelle le moteur, renvoie 0 ou 2 `Proposition` |
| POST | `/interventions/{id}/confirmer` | `{rang}` → programme et verrouille |
| GET | `/planning/medecin/{id}?du=&au=` | vacations + interventions + remplissage |
| GET | `/planning/blocs?semaine=` | grille salles × jours + lits prévus + ambulatoires |
| GET | `/interventions/a-completer?medecin_id=` | opérations passées sans durée réelle |
| POST | `/interventions/{id}/realisation` | heures d'entrée/sortie → exemple d'apprentissage |
| GET | `/modeles/{medecin_id}` | suivi : exemples, erreurs, tableau avis × réel |
| POST | `/modeles/{medecin_id}/entrainer` | appelle `entrainer()` |
| GET/POST | `/medecins/{id}/absences` | liste, déclaration d'une absence |
| GET | `/absences/{id}/impact` | vacations fermées, interventions touchées |
| POST | `/absences/{id}/reprogrammer` | `{rattrapage: bool}` → appelle le moteur §4.3, renvoie les propositions |
| POST | `/reprogrammations/{id}/valider` | `{rang}` → déplace l'intervention, historise l'ancienne date |
| POST | `/absences/{id}/valider-tout` | valide le rang 1 de toutes les propositions |
| PUT | `/reprogrammations/{id}/prevenu` | date à laquelle le patient a été prévenu |
| DELETE | `/absences/{id}` | annule l'absence (tant que rien n'est validé) |
| POST | `/interventions/{id}/non-venu` | `{motif, commentaire}` → statut `non_venu`, crée la nouvelle demande (brouillon) |
| GET | `/interventions/a-reprogrammer?medecin_id=` | non venus sans nouvelle date |
| POST | `/aleas` | `AleaSaisi` → crée l'aléa, renvoie `ImpactAlea` |
| GET | `/aleas?statut=en_cours` | aléas en cours (bandeaux) |
| POST | `/aleas/{id}/proposer` | `{rattrapage: bool}` → propositions par intervention (§4.3) |
| POST | `/actions-alea/{id}/valider` | valide un relogement, un décalage, une reprogrammation (`{rang}`) |
| POST | `/aleas/{id}/valider-tout` | valide toutes les actions proposées |
| POST | `/aleas/{id}/prolonger` | `{duree_estimee \| fin}` → crée un aléa lié |
| POST | `/aleas/{id}/terminer` | « Terminer maintenant » → rouvre, renvoie les places à combler |
| GET | `/comblement?medecin_id=` | places libérées et candidats dans l'ordre d'appel |
| POST | `/appels-comblement/{id}` | `{reponse}` → si « accepte », avance l'intervention |
| GET | `/interventions/a-arbitrer` | tous les patients sans date proposée, quel que soit l'aléa |
| GET | `/config` | estimateur et moteurs actifs (pour afficher les bandeaux « bouchon ») |

---

## 7. Instructions pour Claude Code

### 7.1 Préparer le dépôt

1. Créer `interface/` à la racine du dépôt `IAS-project` et y placer ce fichier.
2. Créer `interface/CLAUDE.md` avec ce contenu :

```markdown
# Interface chirurgien — consignes
- Spécification : SPEC_INTERFACE_MEDECIN.md (source de vérité). La relire avant chaque étape.
- AUCUN modèle de ML ni algorithme d'optimisation : uniquement les bouchons de §4.
  Les routes n'importent jamais un modèle directement, seulement les interfaces de
  app/estimation/interface.py et app/propositions/interface.py.
- Données : anonymisées ou fictives uniquement. Ne jamais commiter de fichier .xlsx.
- Durées en minutes entières ; la durée réelle est le TROS (sortie − entrée de salle).
- Une intervention programmée ne change plus de date, sauf par une reprogrammation validée après
  une absence (parcours D), une non-venue (parcours E, sans priorité) ou un aléa déclaré
  (parcours F, y compris un comblement accepté par le patient).
- Gestion des aléas : GESTION_ALEAS.md (catalogue et règles). Le moteur de référence est
  version_finale/aleas/gestion_aleas.py ; dans cette version, seul AleasBouchon est branché.
- Chaque étape se termine par : tests qui passent + une démonstration à l'écran.
- Langue de l'interface : français.
```

3. Lancer Claude Code dans `interface/`.

### 7.2 Prompts, étape par étape

Donner **une étape à la fois** et vérifier ses critères avant de passer à la suivante.

**Étape 1 — Squelette**
> Lis SPEC_INTERFACE_MEDECIN.md et CLAUDE.md. Crée le squelette back-end (FastAPI, SQLite, SQLModel)
> et front-end (React + Vite + TypeScript) selon l'arborescence du §3. Ajoute un script `make dev`
> (ou deux commandes documentées dans un README) qui lance les deux. Une page d'accueil liste
> les médecins depuis l'API.

✅ `pytest` passe ; la page d'accueil affiche une liste (même vide) venue du back-end.

**Étape 2 — Données et import**
> Implémente les tables du §5.1 et les scripts d'import du §5.2 (historique Excel passé en argument,
> grille V28). Génère 30 patients fictifs marqués « démo ». Écris des tests : nombre de vacations
> déroulées sur une semaine connue, aucun créneau le week-end ni un jour férié, durée historique =
> sortie − entrée de salle.

✅ Après import : 18 médecins, des vacations sur 12 mois, des exemples d'apprentissage par médecin.

**Étape 3 — Interfaces de modèle et bouchons**
> Implémente §4.1 et §4.2 : Protocols, EstimateurBouchon, MoteurBouchon, service de faisabilité,
> sélection par configuration. Tests : le moteur renvoie 0 ou 2 propositions sur deux jours
> distincts, toutes réalisables ; jamais une vacation d'un autre médecin ni un créneau d'urgence.

✅ Tests verts ; `GET /config` indique « bouchon » pour les deux.

**Étape 4 — Parcours A (prise de rendez-vous)**
> Construis le parcours en 4 étapes du §2.1 (front + routes du §6). Bandeaux « bouchon » visibles.
> L'avis plus/autant/moins est obligatoire. La confirmation verrouille la date. Test de bout en
> bout : créer un patient fictif, obtenir deux propositions, confirmer, vérifier que l'intervention
> apparaît dans le planning et qu'une seconde confirmation est refusée.

✅ Un rendez-vous complet se fait en moins d'une minute à la souris.

**Étape 5 — Parcours B (plannings)**
> Construis le planning du chirurgien (semaine/mois) et le planning des blocs (salles × jours,
> remplissage, courbe des lits prévus, ambulatoires par jour) du §2.2, en CSS Grid.

✅ Un rendez-vous confirmé à l'étape 4 apparaît dans les deux vues avec le bon remplissage.

**Étape 6 — Parcours C (durée réelle, suivi du modèle)**
> Construis la liste « à compléter », la saisie des heures d'entrée et de sortie de salle, la création
> de l'exemple d'apprentissage, et la page de suivi du §2.3 (erreur moyenne, tableau avis × réel,
> bouton Réentraîner relié au bouchon). Pour tester, ajoute une commande qui avance l'horloge de
> l'application (date simulée) afin de rendre des opérations « passées ».

✅ Saisir une durée réelle fait apparaître un exemple « outil » et met à jour le tableau avis × réel.

**Étape 7 — Parcours D (absences et reprogrammation)**
> Implémente le §2.4 et le §4.3 : page « Mes absences » (calendrier, motif), tables `absence` et
> `reprogrammation`, statut des vacations, `ReprogrammationBouchon`, écran d'impact, écran de
> propositions par patient (deux jours, étiquettes), validation unitaire et « tout valider »,
> liste des patients à prévenir et des patients à arbitrer. Dans le planning des blocs, afficher
> les vacations fermées hachurées « libérée ». Tests :
> 1. déclarer une absence ferme exactement les vacations du chirurgien sur la période ;
> 2. aucune proposition (parcours A ou D) ne tombe dans une vacation fermée ;
> 3. chaque intervention touchée reçoit 2 jours distincts du même chirurgien, ≥ aujourd'hui + 2 et
>    ≤ son délai maximum, ou le statut « à arbitrer » ;
> 4. une reprogrammation validée déplace l'intervention et conserve l'ancienne date dans
>    l'historique ; une intervention non touchée ne bouge jamais ;
> 5. annuler une absence non validée rouvre les vacations.

✅ Déclarer une semaine de congé pour un chirurgien qui a 5 patients cette semaine-là produit
5 propositions ; « Tout valider » les place et ils apparaissent dans le planning aux nouvelles dates.

**Étape 8 — Parcours E (patient non venu)**
> Implémente le §2.5 : bouton « Patient non venu » sur les interventions du jour (planning du
> chirurgien et liste du jour), table `non_venue`, statut `non_venu`, création d'une nouvelle
> intervention liée (date de consultation = aujourd'hui, durée et avis repris), liste « À
> reprogrammer — non venus », puis les étapes 3–4 du parcours A pour lui donner une date.
> Tests :
> 1. « non venu » est refusé avant le jour de l'intervention et après saisie d'une durée réelle ;
> 2. la place libérée est visible dans le planning des blocs ;
> 3. les deux propositions respectent les délais recomptés depuis aujourd'hui ;
> 4. aucune vacation de rattrapage n'est proposée, et aucune autre intervention ne bouge ;
> 5. l'historique relie l'ancienne et la nouvelle intervention.

✅ Déclarer un patient non venu puis le reprogrammer prend moins d'une minute ; il apparaît à sa
nouvelle date et l'ancienne intervention reste visible avec le statut « non venu ».

**Étape 9 — Parcours F (aléas)**
> Lis GESTION_ALEAS.md. Implémente le §2.6 et le §4.4 :
> - le formulaire unique « Signaler un aléa » (familles, motifs, raison, durée estimée en
>   min / h / j / sem ou date de fin, champs propres à chaque famille) ;
> - les tables `alea`, `action_alea` et `appel_comblement`, et les ajouts à `vacation` ;
> - `AleasBouchon` ;
> - l'écran d'impact, l'écran d'actions (relogements et décalages à valider par le cadre, patients à
>   déplacer avec deux jours proposés, à arbitrer) et la page « Comblement » (appels dans l'ordre,
>   accepte / refuse) ;
> - le bandeau des aléas en cours, avec « Prolonger » et « Terminer maintenant » ;
> - dans le planning des blocs, les vacations relogées, décalées, raccourcies ou fermées avec une
>   icône et l'horaire d'origine au survol.
>
> Ajoute une fonction `lire_duree("1 h 30") -> 90`. Tests :
> 1. une salle en panne reloge chaque vacation touchée dans une salle libre de 1 à 5 sur la même
>    plage, sinon la ferme ; jamais deux vacations actives qui se chevauchent dans une salle ;
> 2. un médecin indisponible 3 h : ses vacations touchées ne chevauchent plus la période ;
> 3. matériel manquant : aucune intervention des types concernés sur la période, les autres ne
>    bougent pas ;
> 4. lits fermés : les nuits sans déficit signalé respectent la capacité réduite moins la
>    réserve ;
> 5. annulation : définitive → statut `annulee` ; report → nouvelle intervention à partir de la
>    date choisie, sans priorité ; la place apparaît dans « Comblement » si elle est à J+2 ou
>    plus ;
> 6. non apte : propositions ≥ date d'aptitude, délai maximum prolongé ;
> 7. comblement : un « accepte » avance l'intervention dans la place, même chirurgien, date plus
>    tôt ; un « refuse » passe au suivant ;
> 8. « Terminer maintenant » rouvre les vacations à J+2 ou plus avec leur forme d'origine ;
> 9. aucun patient déplacé à moins de 2 jours, ni après son délai maximum, ni chez un autre
>    chirurgien.

✅ Signaler « salle 3 en panne, 2 j » montre les vacations relogées (ou fermées) et leurs patients ;
« Tout valider » met le planning des blocs à jour. Signaler une annulation fait apparaître la place
dans « Comblement » ; « accepte » sur le premier candidat l'y place.

**Étape 10 — Finitions**
> Gestion des erreurs (aucune proposition, saisie invalide), états vides, accessibilité clavier,
> responsive tablette, README d'utilisation avec captures.

### 7.3 Brancher les modèles plus tard (hors périmètre de cette version)

- **Durée** : écrire une classe qui respecte `EstimateurDuree` (ex. un modèle par chirurgien sur
  âge, diagnostic, acte, entraîné sur `exemple_apprentissage`), puis la sélectionner par la
  configuration. Rien d'autre ne change.
- **Dates** : écrire un adaptateur `MoteurPropositions` autour de l'algorithme retenu par le
  groupe (ex. la règle de `version_finale/hybride/regle_hybride.py`). Il doit rendre deux jours
  distincts et réalisables.
- **Reprogrammation** : adaptateur `MoteurReprogrammation` autour de
  `version_finale/hybride/absences.py` (`Reprogrammateur`, option `rattrapage`).
- **Aléas** : adaptateur `MoteurAleas` autour de `version_finale/aleas/gestion_aleas.py`
  (`GestionnaireAleas.declarer_alea`, `Levee`). Les objets `IndispoMedecin`, `IndispoSalle`,
  `IndispoMateriel`, `FermetureLits`, `AnnulationPatient`, `PatientNonApte` et `NonVenue`
  correspondent un à un aux familles de `AleaSaisi`. Le journal `Action` alimente `action_alea`.
- Les tables `estimation`, `proposition_faite` et `realisation` permettront alors de comparer les
  modèles sur les décisions réelles des chirurgiens.

---

## 8. Hors périmètre (pour l'instant)

- Authentification réelle et gestion des droits (un simple sélecteur « je suis le Dr X » suffit
  pour le prototype).
- Modification d'une date programmée **en dehors** d'un aléa déclaré (parcours D, E, F).
- Affectation d'une salle libérée à un autre chirurgien (le cadre le fait à la main pour l'instant).
- Remplacement du chirurgien absent par un confrère (le patient garde son chirurgien).
- Ordonnancement de la journée (heures de passage), urgences, dépassements et retards du jour J
  (rôle du SMA, `version_finale/hybride_sma/`), secrétariat, anesthésie.
- Connexion au dossier patient informatisé de l'hôpital.
