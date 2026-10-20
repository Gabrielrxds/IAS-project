# Contrats pour l'interface

L'interface (site web : chirurgien, secrétariat, cadre de bloc) n'est pas dans ce dépôt ; elle sera
refaite à partir d'ici. Ce document liste ce qu'elle doit appeler dans `chaine_finale/`, sans
modifier le code du groupe : une couche d'adaptation traduit les requêtes du site en appels ci-dessous.
Les spécifications fonctionnelles d'origine sont dans le rapport, partie 8.1.

## Principe

- Le code travaille en **jours entiers depuis `don.jour_zero`** et en **minutes depuis minuit**.
- Les objets centraux viennent du modèle commun (`etudes_origine/genetique/modele.py`) :
  `Instance` (patients, vacations, capacités), `Solution` (affectation patient → vacation),
  `Vacation`, `Patient`.
- `commun/protocole.py` fait le lien avec les données : `charger()` → `(df, don)` ;
  `instance(don, lignes_de_grille, Reglages())` → `Instance` ; `mesurer(don, sol)` → indicateurs.
- Les chirurgiens sont affichés **C1..C18** (`protocole.codes_anonymes()`), jamais avec le code de la
  base ni un nom.
- Une décision n'est jamais appliquée sans validation humaine : les fonctions *proposent*, le site
  fait confirmer.

## Les six briques

| Brique | Ce que fait le site | Appel | Entrée | Sortie |
|---|---|---|---|---|
| 1. Estimer la durée et le séjour | à la saisie du patient en consultation | `commun/ml_prevision.py` : `apprendre(df, don)` une fois, puis `predictions()` ou `don_prevu(don, mode)` | âge, sexe, diagnostic, acte prévu, chirurgien, nombre d'actes | durée médiane, marge (P90 − médiane), loi du nombre de nuits, entrée la veille |
| 2. Proposer deux dates | après l'avis du chirurgien | `hybride/regle_hybride.py` : `RegleHybride(don, delta=14, fantomes="previsions")`, méthodes `preparer(sol, t)` puis `consulter(sol, pid, t, figes)` | état du planning `sol`, patient `pid`, jour `t`, journées figées | vacation choisie ; les deux meilleures sont celles de plus petit coût Φ (éq. 5 du rapport) |
| 3. Vérifier la faisabilité | avant de confirmer un rendez-vous | contrainte de capacité d'une vacation, marges cumulées en racine des carrés : `etudes_origine/genetique/modele.py` (`Solution`) et `commun/protocole.py` (`Reglages`) | vacation, patients | vrai / faux, temps restant |
| 4. Ordonner la journée | la veille (J−1) | `commun/algos_journee.py` : `ALGOS["tabou"](inst, jour, sequences, Poids(), 1000, 0, graines=True)` | journée close, patients par vacation | ordre de passage, heures, places ambulatoires |
| 5. Déclarer un aléa connu à l'avance | bouton « Signaler un aléa » | `aleas/gestion_aleas.py` : `GestionnaireAleas(...).declarer_alea(sol, alea, t, figes)` avec un objet `IndispoMedecin`, `IndispoSalle`, `IndispoMateriel`, `FermetureLits`, `AnnulationPatient`, `PatientNonApte` ou `NonVenue` ; `periode(jour, heure, duree)` lit une durée saisie (« 90 min », « 3 h », « 2 j ») | catégorie, ressource, début, durée, motif | liste d'`Action` : vacation relogée / décalée / raccourcie / fermée, patients reprogrammés (deux dates), places à combler, patients à arbitrer |
| 6. Simuler une journée au bloc | page « Scénarios » | `hybride_sma/sma_pont.py` : `simuler(matin, generateur, politique, optimiseur, avec_plans_b, jours)` ; plan du matin : `hybride_sma/sma_matin.py` : `charger_ou_construire("hybride")` | plan du matin, générateur d'aléas (`sma_aleas.GenerateurAleas`) | déroulé de la journée, urgences placées, dépassements, reports |

Et pour le cadre de bloc :

| Besoin | Appel |
|---|---|
| Recalculer la grille des vacations | `hybride/chaine_hybride.py` : `main(part_min=0.75, part_max=1.5, nom="hybride", cible=(0.70, 0.65), cycle=2)` → `grilles/hybride.json` |
| Lire une grille | `protocole.lire_grille(nom)["vacations"]` : lignes `[date ISO, salle, début, fin, code]` |
| Indicateurs (lits, admissions, délais) | `protocole.mesurer(don, sol)` |

## Réglages exposés au service

Ce sont les paramètres du tableau 16 du rapport ; le site peut les proposer en réglages avancés.

| Paramètre | Où | Valeur retenue |
|---|---|---|
| Fenêtre de tolérance | `RegleHybride(delta=…)` | 14 jours |
| Cible week-end / vacances | `commun/calendrier.py` (`F_WE`, `F_VAC`) ou `RegleHybride(cible=(…, …))` | 0,70 / 0,65 |
| Capacité en vacances | `commun/calendrier.py` (`CAPA_VAC` × `LITS_ARMES`) | 0,70 × 17 = 12 lits |
| Réserve pour les urgences | `Reglages(tampon_urgence=…, reserve_lits=…, reserve_places=…)` | 5 %, 2 lits, 2 admissions |
| Volume de chaque chirurgien | `chaine_hybride.main(part_min, part_max)` | 75 % à 150 % |
| Retard accepté le jour J | `hybride_sma/sma_v3.py` (`LIMITE_RETARD`) | 60 minutes |
| Rattrapage d'une absence | `GestionnaireAleas(..., rattrapage=False)` | non |
