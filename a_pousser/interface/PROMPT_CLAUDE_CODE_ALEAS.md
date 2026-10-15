# Prompt Claude Code — gestion des aléas dans l'interface

À coller dans Claude Code, lancé dans `interface/`, une fois les étapes 1 à 6 de la spec faites
(squelette, données, bouchons, rendez-vous, plannings, durée réelle). Les fichiers
`SPEC_INTERFACE_MEDECIN.md` et `GESTION_ALEAS.md` doivent être dans `interface/`.

Le prompt est découpé en **4 temps**. Donne-les **un par un** et vérifie les critères ✅ avant de
passer au suivant.

---

## Temps 1 — Modèle de données et moteur bouchon

```text
Lis SPEC_INTERFACE_MEDECIN.md (§2.4, §2.5, §2.6, §4.3, §4.4, §5.1, §6) et GESTION_ALEAS.md
(catalogue §2, règles §3). Ce sont les sources de vérité ; en cas de doute, relis-les plutôt que
d'inventer une règle.

Objectif : la couche données et un moteur d'aléas BOUCHON, sans écran.

1. Tables (SQLModel) :
   - absence, reprogrammation, non_venue (si absentes) ;
   - alea, action_alea, appel_comblement, et les ajouts à vacation (statut relogee / decalee /
     raccourcie / fermee, alea_id, salle_origine_id, debut_origine, fin_origine).
   Ajoute une migration ou un script de recréation de la base de démo.
2. app/aleas/interface.py : AleaSaisi, ActionVacation, PlaceACombler, ImpactAlea, Protocol
   MoteurAleas (impact, proposer, lever), exactement comme au §4.4. Les motifs sont des listes
   fermées par famille (GESTION_ALEAS.md §2) ; refuse un motif inconnu (erreur 422).
3. app/aleas/duree.py : lire_duree("90 min" | "1 h 30" | "3h" | "2 j" | "1 sem") -> minutes, et
   periode(debut, duree_estimee | fin) -> (debut, fin). Une durée en jours couvre des journées
   entières.
4. app/aleas/bouchon.py : AleasBouchon, en respectant les règles de GESTION_ALEAS.md §3 :
   - medecin : ferme les vacations qui chevauchent la période (pas de décalage dans le bouchon) ;
   - salle : pour chaque vacation touchée, la reloge dans la première salle de 1 à 5 libre sur
     toute sa plage et non indisponible ; sinon la ferme ;
   - materiel : libère seulement les interventions des types concernés dans la période, et les
     exclut des propositions tant que l'aléa est en cours ;
   - lits : baisse la capacité des nuits ; si les lits programmés dépassent capacité − réserve,
     libère les hospitalisés programmés les plus récents (date ≥ aujourd'hui + 2, jamais un patient
     déjà admis) ; sinon signale la nuit en déficit ;
   - annulation : définitive → statut annulee ; report → nouvelle intervention en brouillon avec
     date_consultation = report_a_partir, sans priorité ;
   - non_apte : propositions ≥ apte_a_partir, délai maximum prolongé de la durée d'inaptitude ;
   - non_venu : comme le parcours E ;
   - propositions : les deux premiers jours réalisables du même chirurgien, ≥ aujourd'hui + 2,
     ≤ délai maximum, sinon "a_arbitrer" ; aléa déclaré ≥ 8 jours avant son début → seulement
     des journées non figées (J+7) ; date initiale la plus proche traitée d'abord ;
   - comblement : pour chaque place libérée à J+2 ou plus, les interventions du même chirurgien
     programmées plus tard (jour non figé), triées par date de consultation ;
   - lever : rouvre les vacations modifiées dont la date est ≥ aujourd'hui + 2 et qu'aucun autre
     aléa en cours ne touche, en reprenant leur forme d'origine, puis renvoie les places à combler.
   Chaque modification produit une ligne action_alea (statut proposee).
5. Sélection par configuration : ALEAS=bouchon ; GET /config l'affiche.
6. Tests pytest (une base de démo avec 2 semaines de vacations et 30 patients fictifs suffit) :
   les 9 tests listés à l'étape 9 du §7.2 de la spec, plus lire_duree.
Ne crée aucun écran dans ce temps. Ne branche aucun algorithme du dossier version_finale.
```

✅ `pytest` vert ; `GET /config` affiche `ALEAS: bouchon`.

---

## Temps 2 — Routes

```text
Implémente les routes du §6 pour les aléas : POST /aleas, GET /aleas?statut=en_cours,
POST /aleas/{id}/proposer, POST /actions-alea/{id}/valider, POST /aleas/{id}/valider-tout,
POST /aleas/{id}/prolonger, POST /aleas/{id}/terminer, GET /comblement?medecin_id=,
POST /appels-comblement/{id}, GET /interventions/a-arbitrer.

Règles à faire respecter par l'API (verrous) :
- une intervention programmée ne change de vacation que par une action_alea validée, une
  reprogrammation validée (parcours D) ou un non-venu (parcours E) ;
- un relogement ou un décalage n'est appliqué au planning qu'après validation (rôle « cadre ») ;
- un appel de comblement « accepte » avance l'intervention dans la place si elle est encore
  libre et réalisable, sinon répond 409 et passe au candidat suivant ;
- prolonger crée un nouvel aléa avec parent_id ; terminer met statut = termine et termine_le ;
- tout est historisé (action_alea.avant / apres), rien n'est supprimé.
Tests d'API pour chaque route, y compris les refus (motif inconnu, date passée, place déjà prise,
intervention d'un autre chirurgien).
```

✅ Les tests d'API passent ; un scénario « salle 3 en panne 2 j » via `curl` renvoie l'impact,
puis `valider-tout` modifie les vacations.

---

## Temps 3 — Écrans

```text
Construis le parcours F (§2.6) côté front :
1. Bouton « Signaler un aléa » dans la barre du haut → formulaire en 3 temps (Quoi / Motif et
   raison / Quand). La durée estimée se saisit avec un champ numérique + une unité
   (min, h, j, sem) ou une date de fin. Champs propres à chaque famille : types d'intervention
   (matériel), nombre de lits, patient (recherche), « définitive / report à partir du »
   (annulation), « apte dans » (non apte).
2. Écran Impact : vacations avec leur action (relogée S2→S1, décalée 8h-13h→10h-15h,
   raccourcie, fermée), interventions à déplacer, nuits en déficit, places à combler.
3. Écran Actions : relogements et décalages (bouton « Valider » réservé au cadre), patients à
   déplacer (deux jours proposés, comme au parcours D, ou « à arbitrer »), « Tout valider »,
   liste « Patients à prévenir ».
4. Page Comblement : par place libérée, la liste ordonnée des candidats avec téléphone,
   boutons « Accepte / Refuse / Injoignable ».
5. Bandeau « Aléas en cours » dans les deux plannings, avec « Prolonger » et
   « Terminer maintenant ».
6. Planning des blocs : salle 1 affichée (salle de repli) ; icônes pour les vacations relogées,
   décalées, raccourcies, fermées ; horaire et salle d'origine au survol.
7. Page « À arbitrer » commune à tous les aléas.
Bandeau « Gestion des aléas provisoire — algorithme non branché » tant que ALEAS=bouchon.
Tests Vitest des composants principaux (formulaire, validation de la durée, liste de
comblement).
```

✅ À la souris, en moins de 2 minutes :
- signaler « salle 3 en panne, 2 j » puis tout valider ;
- signaler l'annulation d'un patient, puis l'accepter pour le premier candidat en comblement ;
- terminer l'aléa salle plus tôt et voir les vacations rouvertes.

---

## Temps 4 — Brancher le vrai moteur (quand le groupe le décide)

```text
Écris app/aleas/adaptateur_groupe.py, une classe MoteurAleasGroupe qui respecte MoteurAleas
en s'appuyant sur version_finale/aleas/gestion_aleas.py :
- construis l'Instance et la Solution du modèle du groupe à partir de la base (vacations,
  interventions programmées, capacités) ;
- traduis AleaSaisi vers IndispoMedecin / IndispoSalle / IndispoMateriel / FermetureLits /
  AnnulationPatient / PatientNonApte / NonVenue (minutes depuis le jour 0 = début de l'horizon) ;
- appelle GestionnaireAleas.declarer_alea, puis traduis gestionnaire.actions en action_alea
  (statut proposee) ; « Terminer maintenant » = Levee ;
- options : rattrapage (case à cocher), comblement (toujours), p_accepte inutilisé (ce sont les
  vrais appels).
Ajoute ALEAS=groupe dans la configuration. Rejoue les 9 tests du Temps 1 avec ALEAS=groupe ;
ils doivent passer sans modification.
```

✅ Les mêmes tests passent avec `ALEAS=bouchon` et `ALEAS=groupe`.
