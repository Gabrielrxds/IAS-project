# Gestion des aléas du planning opératoire

Ce document fixe **la liste des aléas gérés**, la **règle** appliquée à chacun et ce qu'ils coûtent
en simulation sur les patients de 2022. Il complète `SPEC_INTERFACE_MEDECIN.md` (parcours F, §4.4)
et `REPROGRAMMATION_ABSENCES.md`.

Le code de référence est dans `version_finale/aleas/` :

| Fichier | Rôle |
|---|---|
| `gestion_aleas.py` | Moteur : aléas, gestionnaire, simulation. |
| `scenarios_aleas.py` | Générateurs d'aléas, avec les fréquences réunies dans `FREQUENCES`. |
| `experiences_aleas.py` | Tableau du §5. |
| `tester_aleas.py` | 12 tests, tous verts. |

---

## 1. Le principe : une seule saisie pour tous les aléas

Tout aléa se déclare avec **le même formulaire** :

| Champ | Contenu |
|---|---|
| Quoi | Médecin, salle, matériel, lits ou patient. |
| Motif | Liste fermée, propre à chaque famille (§2). |
| Raison | Texte libre (« appelé en réanimation », « table d'opération en panne »…). |
| Début | Date et heure. |
| Durée estimée | Saisie par le déclarant (« 90 min », « 3 h », « 2 j », « 1 sem »), ou une date de fin. |

L'estimation peut être fausse, d'où deux boutons sur tout aléa en cours :
- **« Prolonger »** : un nouvel aléa couvre la suite, il est relié au premier.
- **« Terminer maintenant »** (levée) : ce qui est encore à venir est rouvert, et les places
  rouvertes sont proposées aux patients en attente (comblement).

Toutes les modifications sont tracées dans un journal (`Action`) : qui, quoi, ancienne et nouvelle
date, validé par, patient prévenu le.

## 2. Catalogue

### 2.1 Les aléas gérés

| # | Aléa | Motifs | Ce que fait le moteur | Statut |
|---|---|---|---|---|
| 1 | **Congé** du médecin | congé, formation | Ferme ses vacations sur la période. Reprogramme ses patients, uniquement sur des journées non figées. | Déjà fait, repris ici |
| 2 | **Médecin absent** (maladie) | maladie | Idem, et autorise aussi les journées figées s'il reste de la place. Rattrapage en option. | Déjà fait, repris ici |
| 3 | **Médecin indisponible quelques heures**, motif et durée estimée saisis | urgence dans un autre service, retard, personnel, autre | **Décale** la vacation si la salle est libre après (ou avant), sinon la **raccourcit**. Les patients qui ne tiennent plus sont reprogrammés. | **Nouveau** |
| 4 | **Salle indisponible** (problème matériel) | panne de matériel, stérilisation, maintenance, dégât des locaux, personnel manquant | **Reloge** chaque vacation touchée dans une salle libre, même jour et même horaire. Sinon décale, raccourcit ou ferme. | **Nouveau** |
| 5 | **Matériel manquant** pour un type d'acte | ancillaire, rupture d'implant, colonne d'endoscopie | Déplace **seulement** les interventions de ce type hors de la période. Aucune nouvelle intervention de ce type n'y est programmée. | **Nouveau** |
| 6 | **Fermeture de lits** | épidémie, travaux, manque de personnel | Baisse la capacité de lits sur les nuits concernées. En cas de dépassement, déplace les hospitalisés programmés les plus récents, jamais un patient déjà admis. Sinon, signale un déficit. | **Nouveau** |
| 7 | **Annulation par le patient** (à l'avance) | choix du patient, empêchement, décès, transfert | Annulation définitive ou **report** à partir d'une date (nouvelle demande, **sans priorité**). La place est **comblée**. | **Nouveau** |
| 8 | **Patient non apte**, avec durée estimée avant d'être apte | infection, bilan incomplet, traitement à arrêter, décompensation | Reprogrammé **avec priorité** à partir de la date d'aptitude. Délai maximum prolongé d'autant. La place est comblée. | **Nouveau** |
| 9 | **Patient non venu** le jour même | absent sans prévenir, annulation le jour même, contre-indication à l'admission | Nouvelle demande vue ce jour-là, **sans priorité**. La place du jour est perdue (le SMA peut la donner à une urgence). | Déjà fait, repris ici |
| 10 | **Durée estimée fausse** | — | Prolongation (nouvel aléa), ou levée qui rouvre les vacations à venir et les comble. | **Nouveau** |
| 11 | **Comblement** des places libérées | — | Une place libérée ≥ 2 jours à l'avance est proposée aux patients du **même chirurgien** opérés plus tard, le plus ancien demandeur d'abord. | **Nouveau** |

### 2.2 Ce qui n'est pas ici, et pourquoi

Le jour J, les aléas suivants sont gérés par le **SMA** (`hybride_sma/`), qui réordonne la journée en
temps réel :
- urgences ;
- dépassements de durée ;
- retards dans la journée ;
- annulations de dernière minute ;
- lits indisponibles le jour même.

Hors périmètre pour l'instant :
- **remplacement par un confrère** de même spécialité. La règle du projet est que le chirurgien ne
  change jamais ;
- **anesthésiste absent** : il se déclare comme une salle indisponible (motif « personnel
  manquant ») ;
- **grève**.

## 3. Règles communes

1. **Même chirurgien**, toujours.
2. **Pas de nouvelle date à moins de 2 jours**, le temps de prévenir le patient
   (`PREAVIS_PATIENT`). Le délai maximum (6 mois après la consultation) est respecté ; sinon le
   patient passe « à arbitrer », dans une liste dédiée, en attendant une décision humaine.
3. **Ordre de traitement :** la date initiale la plus proche d'abord. C'est le patient qui a le
   moins de marge.
4. **Préavis :**
   - un aléa déclaré **≥ 8 jours** avant son début ne touche que des journées non figées (prévu,
     comme un congé) ;
   - en deçà, on peut aussi remplir les journées figées s'il reste de la place (imprévu, comme une
     maladie).
5. **Choix du jour :** même score que la règle hybride (variance des lits attendus, patients
   fantômes compris). Deux jours sont proposés et le plus proche est retenu ; dans l'interface,
   c'est le médecin ou la secrétaire qui choisit.
6. **Vacation raccourcie :**
   - on garde le plus grand morceau restant s'il dure **≥ 60 min**, sinon on ferme ;
   - on le **prolonge** pour récupérer les minutes perdues si la salle et le chirurgien sont libres,
     entre 7 h 30 et 18 h 30 ;
   - si des patients ne tiennent plus, on sort d'abord les plus longs : cela libère le plus de
     minutes en déplaçant le moins de patients.
7. **Relogement :** même jour, même horaire, salle libre sur toute la plage et non indisponible.
   La salle 1, inutilisée par la grille hybride, est la réserve naturelle.
8. **Priorité :**
   - le patient non apte est replacé tout de suite, avant les consultations du jour ;
   - l'annulation avec report et le non venu repassent par la consultation, sans priorité.
9. **Comblement :**
   - seulement pour une place libérée ≥ 2 jours à l'avance ;
   - patients du même chirurgien dont la date est plus tardive et non figée ;
   - au plus 8 appels ;
   - en simulation, chaque patient accepte avec une probabilité de 50 %.

## 4. Fréquences utilisées en simulation (hypothèses)

La base 2022 ne contient **ni pannes, ni fermetures de lits, ni annulations**. Seuls les congés en
sont déduits (les semaines sans activité). Les autres fréquences sont des hypothèses de travail,
réunies dans `FREQUENCES` et `TauxPatients`, et sont à remplacer dès que l'hôpital fournit ses
chiffres.

| Aléa | Fréquence | Durée estimée | Préavis | Durée réelle |
|---|---|---|---|---|
| Congés | Semaines sans activité de 2022 (170 semaines-chirurgien) | 1 semaine | 42 j | = estimée |
| Maladie | 1 % des journées opératoires | 1 à 5 j | 0 à 2 j | = estimée |
| Médecin quelques heures | 2 % des journées opératoires | 1 à 4 h | 0 à 1 j | × 0,6 à 1,6 |
| Panne de salle | 0,5 % des salles-jours | 2 à 4 h (70 %) ou 1 à 3 j | 0 | × 0,5 à 2 |
| Maintenance | 2 par salle et par an | 1 à 2 j | 21 j | = estimée |
| Matériel manquant | 1 par mois, type d'acte fréquent | 1 à 5 j | 0 à 3 j | = estimée |
| Fermeture de lits | 0,5 par mois, 2 à 6 lits | 7 à 21 nuits | 2 à 10 j | = estimée |
| Annulation patient | 3 % des patients (30 % définitives) | report à 14–45 j | 3 à 20 j | — |
| Patient non apte | 1 % | apte dans 7 à 30 j | 0 à 7 j | — |
| Non venu | 2 % | — | jour même | — |

## 5. Résultats

Grille hybride, patients de 2022, mesures sur mai–décembre, moyenne de 5 graines. **0 conflit**
dans tous les scénarios.

| Scénario | Patients touchés | Reprogrammés | À arbitrer | Décalage moyen | Avancés (comblement) | Relogées | Sans date | Délai médian | Variance lits |
|---|---|---|---|---|---|---|---|---|---|
| Référence sans aléa | 0 | 0 | 0 | — | 0 | 0 | 9 | 31 j | 6,9 |
| Congés + maladie 1 %, sans rattrapage | 163 | 171 | 31 | 25 j | 0 | 0 | 34 | 51 j | 9,0 |
| Congés + maladie 1 %, **avec rattrapage** | 152 | 151 | 20 | **8 j** | 0 | 0 | 26 | 48 j | 10,4 |
| Médecin indisponible quelques heures (2 %) | 14 | 12 | 1,6 | 26 j | 0 | 0 | 10,6 | 32 j | 6,9 |
| Salles, **relogement salles 1–5** | 2 | 2 | 0 | 7 j | 0 | 16 | 9,0 | 31 j | 6,9 |
| Salles, relogement salles 2–5 | 27 | 27 | 0,6 | 28 j | 0 | 4 | 9,8 | 34 j | 6,9 |
| Salles, sans relogement | 32 | 31 | 1,2 | 30 j | 0 | 0 | 10,4 | 34 j | 6,9 |
| Matériel manquant (1/mois) | 9 | 9 | 0,4 | 34 j | 0 | 0 | 9,2 | 32 j | 7,0 |
| Lits fermés (2–6 lits) | 0 | 0 | 0 | — | 0 | 0 | 9,0 | 31 j | 6,9 |
| Lits fermés, stress 32 lits × 3 semaines | 60 | 60 | 0,2 | 22 j | 0 | 0 | 9,2 | 34 j | 7,3 |
| Annulations 3 %, sans comblement | 57 | 0 | 0 | — | 0 | 0 | 9,2 | 30 j | 6,6 |
| Annulations 3 %, **avec comblement** | 56 | 0 | 0 | — | **46** | 0 | 8,4 | 30 j | 6,8 |
| Non aptes 1 %, sans comblement | 24 | 23 | 0,6 | 33 j | 0 | 0 | 9,6 | 33 j | 7,0 |
| Non aptes 1 %, avec comblement | 21 | 21 | 0,4 | 29 j | 14 | 0 | 9,6 | 32 j | 7,2 |
| Non venus 2 % | 44 | 0 | 0 | — | 0 | 0 | 10,4 | 34 j | 6,9 |
| **Tout ensemble**, gestion minimale | 350 | 264 | 45 | 34 j | 0 | 0 | 44 | 61 j | 9,3 |
| **Tout ensemble**, gestion complète | 293 | 205 | **22** | **13 j** | **78** | 13 | **27** | **51 j** | 10,4 |

« Sans date » exclut les annulations définitives (environ 20). « Gestion minimale » : ni relogement,
ni rattrapage, ni comblement. « Gestion complète » : les trois.

## 6. Ce qu'il faut retenir

1. **La salle 1 suffit à absorber les pannes.**
   - En relogeant dans les salles 1 à 5, 16 vacations sont déplacées et seulement 2 patients
     décalés.
   - Sans la salle 1, il y en a 27 ; sans relogement, 32, avec environ un mois de décalage chacun.
   - Il faut donc garder la salle 1 comme **salle de repli** plutôt que de l'attribuer.
2. **Gérer tous les aléas ensemble divise par deux les dégâts.** Avec relogement, rattrapage et
   comblement :
   - patients à arbitrer : 45 → 22 ;
   - décalage moyen : 34 → 13 jours ;
   - délai médian : 61 → 51 jours ;
   - patients sans date : 44 → 27.
3. **Le comblement récupère les annulations :** 46 patients avancés pour 56 annulations, sans
   dégrader le lissage des lits.
4. **Médecin indisponible quelques heures :**
   - la moitié des vacations touchées sont **décalées** plutôt que raccourcies, ce qui protège
     leurs patients ;
   - mais un patient qui ne tient plus attend environ 4 semaines : les vacations suivantes du
     chirurgien sont pleines ;
   - piste : autoriser le rattrapage aussi pour ces aléas courts.
5. **Les lits ne sont pas une contrainte dans notre modèle :**
   - le planning hybride en occupe au plus environ 16 sur 42 ;
   - fermer 25 lits pendant 3 semaines ne déplace personne ; en fermer 32 en déplace 60 ;
   - c'est la conséquence des 42 lits du protocole, pas une propriété de l'hôpital, qui a culminé
     à 30 lits.
6. **Matériel et non aptes :**
   - peu de patients touchés ;
   - mais environ un mois de décalage, puisque les vacations du chirurgien sont pleines.

## 7. Limites

- Les fréquences du §4 sont des hypothèses, sauf les congés.
- Le comblement suppose que la moitié des patients appelés acceptent d'avancer. Dans la réalité,
  c'est la secrétaire qui appelle.
- Le personnel (IBODE, anesthésiste) n'est pas modélisé : un relogement suppose qu'une équipe est
  disponible. Le cadre de bloc valide chaque relogement dans l'interface.
- L'ordre de passage des journées touchées n'est pas recalculé ici : c'est le rôle du SMA le jour J.
