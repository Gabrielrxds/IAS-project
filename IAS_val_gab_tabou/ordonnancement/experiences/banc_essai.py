r"""
banc_essai.py — Comment tester ce système.

    python3 banc_essai.py [chemin_du_fichier.xlsx]

LA PYRAMIDE DES TESTS, DU PLUS BAS AU PLUS HAUT
------------------------------------------------
Niveau 1 — CORRECTION        `tests.py`
    Des invariants qui doivent être vrais, toujours, sinon le code est faux.
    Rapides, déterministes, ils tournent à chaque modification.
      « le delta incrémental égale-t-il le recalcul complet ? »
      « deux patients peuvent-ils se retrouver dans la même salle au même
        moment ? »
    Un test de correction ne dit RIEN de la qualité des plannings. Il dit
    seulement que l'algorithme optimise bien la chose qu'on lui a demandé
    d'optimiser.

Niveau 2 — VALIDITÉ STATISTIQUE     `validation_temporelle` (ici)
    Est-ce qu'on ne triche pas avec les données ? C'est le test qu'on oublie
    le plus souvent et celui qui invalide le plus de résultats.

Niveau 3 — BOUT EN BOUT      `execution_reelle` (ici)
    On construit le planning avec les durées ESTIMÉES, puis on le rejoue avec
    les durées RÉELLEMENT observées. Combien de vacations débordent ?
    C'est le seul test qui réponde vraiment à « est-ce que ça marche ».

Niveau 4 — ROBUSTESSE        `escalade_charge`, `stabilite_graines` (ici)
    Que se passe-t-il quand on augmente la charge ? Quand on change la graine
    aléatoire ? Un algorithme qui s'effondre à +10 % de charge, ou dont le
    résultat dépend fortement du hasard, n'est pas utilisable.

Niveau 5 — QUALITÉ           `demo.py` sections 5 et 6
    Comparaison au planning réel et à l'étalon hors-ligne.

C'est le niveau 3 que vous décriviez : « j'inscris des patients du fichier et
je fais tourner les algos pour voir si ça marche ». La nuance importante,
c'est qu'il faut les rejouer avec leurs VRAIES durées, sinon on ne teste que
la cohérence de l'algorithme avec lui-même.
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from dataclasses import dataclass, replace

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import chemins  # rend coeur/ et vacations/ importables
from donnees import charger_historique, construire_estimateur, construire_instance
from estimation import valideur_auto, valideur_quantile
from modele import Instance, Patient, Solution
from propositions import choix_meilleur, simuler_consultations
from propositions import FENETRES
from tabou import (Evaluateur, ParametresTabou, Poids, TabouGlobal,
                   solution_initiale)

FICHIER = chemins.fichier_donnees()


def titre(t: str) -> None:
    print("\n" + "═" * 78)
    print(t)
    print("═" * 78)


# ===========================================================================
# NIVEAU 2 — Validité statistique : ne pas tricher avec les données
# ===========================================================================


def erreur_estimation(inst: Instance) -> dict:
    """Écart entre la durée estimée et la durée réellement observée.

    Trois choses à regarder, et elles ne disent pas la même chose :

      BIAIS   moyenne de (estimé - réel). S'il est positif, on surestime
              systématiquement : les vacations finissent tôt, on perd du temps
              de salle. S'il est négatif, on déborde tous les jours. C'est le
              défaut le plus grave parce qu'il s'accumule : dix interventions
              sous-estimées de 8 minutes, c'est 80 minutes de retard le soir.

      MAE     erreur absolue moyenne. Elle ne s'accumule pas — les erreurs se
              compensent — mais elle dit à quel point une journée est
              imprévisible.

      COUVERTURE  part des cas où réel <= estimé + marge. C'est LA mesure qui
              valide le dimensionnement de la marge de risque. Si la marge
              vaut P90 - médiane, on attend environ 90 %.
    """
    ecarts, absolus, couverts, n = [], [], 0, 0
    for p in inst.patients.values():
        if p.duree_reelle <= 0:
            continue
        n += 1
        e = p.duree_op - p.duree_reelle
        ecarts.append(e)
        absolus.append(abs(e))
        if p.duree_reelle <= p.duree_op + p.marge_perso:
            couverts += 1
    if not n:
        return {}
    return {
        "n": n,
        "biais_min": statistics.fmean(ecarts),
        "mae_min": statistics.fmean(absolus),
        "mediane_erreur_abs": statistics.median(absolus),
        "p90_erreur_abs": sorted(absolus)[int(0.9 * (n - 1))],
        "couverture_marge": couverts / n,
    }


def validation_temporelle(chemin: str) -> None:
    """Le test qu'on oublie : l'estimateur voit-il le futur ?

    Dans `demo.py`, l'estimateur est calibré sur TOUT l'historique 2019-2022,
    puis on rejoue le 1er semestre 2022. Les interventions de 2022 servent
    donc à la fois à calculer les statistiques et à être prédites : c'est une
    FUITE DE DONNÉES (*data leakage*). Le résultat est trop beau, et il ne dit
    rien de ce qui se passera en 2023.

    La forme correcte : calibrer sur le passé, tester sur le futur. C'est une
    validation temporelle, la seule honnête quand les données ont une date —
    on ne fait pas de validation croisée aléatoire sur des séries temporelles,
    sinon on entraîne sur décembre pour prédire janvier.

    Si l'écart entre les deux colonnes est faible, la fuite était bénigne et
    les résultats de `demo.py` tiennent. Si elle est forte, il faut refaire
    toutes les mesures avec l'estimateur honnête.
    """
    titre("NIVEAU 2 — VALIDITÉ STATISTIQUE : l'estimateur voit-il le futur ?")
    df = charger_historique(chemin)

    avant_2022 = df[df["date_inter"] < "2022-01-01"]
    print(f"  Calibration honnête  : {len(avant_2022)} interventions 2019-2021")
    print(f"  Calibration en fuite : {len(df)} interventions 2019-2022")
    print(f"  Test                 : 1er semestre 2022\n")

    resultats = {}
    for nom, source in (("avec fuite (2019-2022)", df),
                        ("honnête (2019-2021)", avant_2022)):
        est = construire_estimateur(source)
        inst, _ = construire_instance(df, est, nb_semaines=26)
        inst.capacite_places_jour = 18
        resultats[nom] = (erreur_estimation(inst), inst)

    print(f"  {'calibration':24} {'biais':>9} {'MAE':>8} {'méd.':>8} "
          f"{'P90':>8} {'couverture':>12}")
    for nom, (e, _) in resultats.items():
        print(f"  {nom:24} {e['biais_min']:+8.1f}m {e['mae_min']:7.1f}m "
              f"{e['mediane_erreur_abs']:7.1f}m {e['p90_erreur_abs']:7.1f}m "
              f"{e['couverture_marge']:11.1%}")

    e_fuite = resultats["avec fuite (2019-2022)"][0]
    e_vrai = resultats["honnête (2019-2021)"][0]
    degradation = e_vrai["mae_min"] - e_fuite["mae_min"]
    print(f"\n  Dégradation de la MAE en supprimant la fuite : {degradation:+.1f} min")
    if abs(degradation) < 1.0:
        print("  Fuite bénigne : les statistiques par acte sont stables d'une année")
        print("  sur l'autre, ce qui est rassurant en soi. Les mesures de demo.py")
        print("  tiennent. Utilisez quand même la calibration honnête dans le")
        print("  rapport — c'est la seule défendable devant un jury.")
    else:
        print("  Fuite significative : toutes les mesures doivent être refaites")
        print("  avec l'estimateur calibré sur le passé seulement.")


# ===========================================================================
# NIVEAU 3 — Bout en bout : le planning survit-il à la réalité ?
# ===========================================================================


@dataclass
class ResultatExecution:
    vacations_utilisees: int
    vacations_debordantes: int
    minutes_debordement: int
    debordement_max: int
    patients: int
    minutes_creux_reel: int

    @property
    def part_debordantes(self) -> float:
        return self.vacations_debordantes / max(1, self.vacations_utilisees)


def execution_reelle(inst: Instance, sol: Solution) -> ResultatExecution:
    """Rejoue le planning avec les durées RÉELLEMENT observées.

    C'est le test décisif. On a construit le planning avec des durées
    estimées ; on remplace chaque durée par celle qu'a vraiment duré
    l'intervention dans l'historique, et on regarde ce qui casse.

    Une vacation « déborde » si TROS réel cumulé + TIS dépasse le TVO. En
    vrai bloc, cela signifie : du personnel en heures supplémentaires, ou une
    intervention déprogrammée en fin de journée.

    Note méthodologique : le planning n'est pas celui qui a réellement eu
    lieu, donc les durées réelles sont réaffectées à d'autres journées que
    les leurs. C'est légitime — la durée d'une intervention dépend de l'acte
    et du chirurgien, pas du jour où elle est faite.
    """
    utilisees = debordantes = 0
    total_debordement = 0
    pire = 0
    creux = 0
    patients = 0

    for vid, v in inst.vacations.items():
        pids = [pid for pid, x in sol.affectation.items() if x == vid]
        if not pids:
            continue
        utilisees += 1
        patients += len(pids)
        reel = sum(inst.patients[p].duree_reelle or inst.patients[p].duree_op
                   for p in pids) + inst.tis * (len(pids) - 1)
        depassement = max(0, reel - v.tvo)
        creux += max(0, v.tvo - reel)
        if depassement:
            debordantes += 1
            total_debordement += depassement
            pire = max(pire, depassement)

    return ResultatExecution(utilisees, debordantes, total_debordement, pire,
                             patients, creux)


def comparer_politiques_duree(chemin: str) -> None:
    """Trois façons d'estimer la durée, jugées sur ce qui se passe le jour J.

    L'arbitrage est toujours le même et il est fondamental :
      estimer court  -> on remplit bien les vacations, et on déborde le soir
      estimer long   -> on ne déborde jamais, et on gaspille du temps de salle
    La question n'a pas de réponse théorique, elle se tranche en mesurant.
    """
    titre("NIVEAU 3 — BOUT EN BOUT : planifier avec l'estimé, exécuter avec le réel")
    df = charger_historique(chemin)
    estimateur = construire_estimateur(df[df["date_inter"] < "2022-01-01"])

    variantes = {
        "médiane + marge P90": dict(valideur=valideur_auto, marge=True),
        "médiane, sans marge": dict(valideur=valideur_auto, marge=False),
        "P75, sans marge": dict(valideur=valideur_quantile(0.75), marge=False),
    }

    print(f"  {'politique de durée':24} {'patients':>9} {'vacations':>10} "
          f"{'débordantes':>12} {'min. déb.':>10} {'pire':>7} {'creux réel':>11}")
    for nom, cfg in variantes.items():
        inst, _ = construire_instance(df, estimateur, nb_semaines=26,
                                      valideur=cfg["valideur"])
        inst.capacite_places_jour = 18
        if not cfg["marge"]:
            for p in inst.patients.values():
                p.marge_perso = 0
        sol, journal = simuler_consultations(inst, Poids(), choix=choix_meilleur,
                                         fenetre=FENETRE_REFERENCE)
        r = execution_reelle(inst, sol)
        print(f"  {nom:24} {r.patients:9d} {r.vacations_utilisees:10d} "
              f"{r.part_debordantes:11.1%} {r.minutes_debordement:9d}m "
              f"{r.debordement_max:6d}m {r.minutes_creux_reel / 60:10.0f}h")

    print("\n  Lecture : la marge de risque achète de la sécurité le soir au prix")
    print("  de temps de salle inutilisé. Le P75 fait la même chose autrement,")
    print("  en gonflant la durée de chaque patient plutôt qu'en réservant un")
    print("  tampon. Choisir entre les deux est une décision de cadre de bloc,")
    print("  mais elle doit se prendre sur ce tableau, pas au jugé.")


# ===========================================================================
# NIVEAU 4 — Robustesse
# ===========================================================================


def cloner_instance(inst: Instance, patients: dict[int, Patient]) -> Instance:
    """Nouvelle instance, mêmes vacations et mêmes capacités, autres patients."""
    neuve = Instance(
        capacite_lits=inst.capacite_lits, capacite_places=inst.capacite_places,
        capacite_places_jour=inst.capacite_places_jour, tis=inst.tis,
        tis_meme_acte=inst.tis_meme_acte, fermeture_uca=inst.fermeture_uca,
        surveillance_ambu=inst.surveillance_ambu,
        heure_sortie_hospit=inst.heure_sortie_hospit,
        jour_zero=inst.jour_zero, nb_jours=inst.nb_jours)
    neuve.vacations = dict(inst.vacations)
    neuve.medecins = dict(inst.medecins)
    neuve.patients = patients
    return neuve.indexer()


def dupliquer_patients(inst: Instance, facteur: float, graine: int = 0) -> Instance:
    """Réinscrit des patients tirés de la liste existante, pour monter en charge.

    C'est exactement votre idée : reprendre des patients déjà décrits dans le
    fichier et les réinscrire. Deux précautions :

      - on leur donne un NOUVEL identifiant, sinon ils écrasent l'original ;
      - on leur retire leur date de consultation d'origine et on en tire une
        au hasard, sinon on duplique aussi le motif d'arrivée et on teste un
        flux artificiellement régulier.

    Le clone garde en revanche son chirurgien, son acte, sa durée réelle et sa
    durée de séjour : c'est le même type de patient, pas un patient inventé.
    """
    alea = random.Random(graine)
    originaux = list(inst.patients.values())
    patients = {p.id: p for p in originaux}
    a_ajouter = int(round((facteur - 1.0) * len(originaux)))
    prochain = max(patients) + 1 if patients else 0

    for _ in range(max(0, a_ajouter)):
        modele = alea.choice(originaux)
        # `replace` d'un dataclass : copie champ à champ, y compris avec
        # slots=True (où `__dict__` n'existe pas).
        clone = replace(modele, id=prochain,
                        jour_demande=alea.randrange(0, max(1, inst.nb_jours - 30)))
        # nouvelle fenêtre aussi : le clone est un nouveau passage en
        # consultation, pas une copie carbone du précédent.
        clone.fenetre_jours = FENETRES[alea.choice(list(FENETRES))]
        patients[prochain] = clone
        prochain += 1
    return cloner_instance(inst, patients)


def escalade_charge(inst: Instance, facteurs=(1.0, 1.1, 1.2, 1.3, 1.5)) -> None:
    """Jusqu'où le bloc tient-il ?

    On augmente la charge et on regarde COMMENT le système se dégrade. Ce qui
    importe n'est pas le point de rupture — on le connaît, c'est la capacité —
    mais la FORME de la dégradation :

      - une dégradation progressive (le délai s'allonge, les reports montent
        doucement) est le signe d'un algorithme sain qui arbitre ;
      - une dégradation brutale (tout va bien puis tout casse) signale un
        effet de seuil qu'il faut comprendre avant de mettre en service.

    À surveiller particulièrement : le délai et le taux de report doivent
    monter AVANT que les dépassements de vacation n'apparaissent. Si les
    vacations débordent alors qu'on pouvait encore reporter, c'est que la
    pénalité de dépassement est mal calibrée.
    """
    titre("NIVEAU 4 — ESCALADE DE CHARGE : on réinscrit des patients du fichier")
    print(f"  {'charge':>7} {'patients':>9} {'dates':>7} {'reports':>8} "
          f"{'délai méd.':>11} {'remplissage':>12} {'sd lits':>9} {'déb. réel':>10}")
    for f in facteurs:
        gonflee = dupliquer_patients(inst, f, graine=0)
        t = time.time()
        sol, journal = simuler_consultations(gonflee, Poids(), choix=choix_meilleur,
                                             fenetre=FENETRE_REFERENCE)
        r = journal.resume()
        ind = sol.indicateurs()
        ex = execution_reelle(gonflee, sol)
        print(f"  x{f:<6.2f} {len(gonflee.patients):9d} {r['patients_dates']:7d} "
              f"{r['patients_reportes_hors_horizon']:8d} {r['delai_median_j']:11.0f} "
              f"{ind['taux_remplissage_moyen']:11.1%} {ind['lits_ecart_type']:9.2f} "
              f"{ex.part_debordantes:9.1%}")
    print("\n  Le taux de report est la soupape : c'est lui qui absorbe la")
    print("  surcharge, et c'est le comportement voulu — mieux vaut dire non à")
    print("  la consultation que promettre une date intenable.")


def stabilite_graines(inst: Instance, graines=(0, 1, 2, 3, 4),
                      iterations: int = 3000) -> None:
    """Le tabou dépend-il trop du hasard ?

    Une métaheuristique est stochastique : échantillonnage du voisinage, durée
    tabou tirée au sort, perturbations de diversification. Un résultat obtenu
    avec une seule graine n'est pas un résultat — c'est un tirage.

    Règle à appliquer partout dans le rapport : toute valeur issue du tabou
    s'écrit moyenne ± écart-type sur au moins 5 graines. Si l'écart-type entre
    graines est du même ordre que l'écart entre deux configurations qu'on
    compare, la comparaison ne veut rien dire et il faut plus d'itérations.

    Le moteur de consultation, lui, est DÉTERMINISTE : même entrée, même
    sortie. C'est un avantage pratique considérable — un chirurgien qui
    reconsulte le système pour le même patient doit revoir les mêmes dates.
    """
    titre("NIVEAU 4 — STABILITÉ DU TABOU HORS-LIGNE SUR PLUSIEURS GRAINES")
    mesures = {"cout": [], "sd_lits": [], "sd_places": [], "remplissage": [],
               "programmes": []}
    for g in graines:
        poids = Poids()
        ev = Evaluateur(inst, poids)
        depart = solution_initiale(inst, ev)
        tg = TabouGlobal(inst, poids, ParametresTabou(
            iterations=iterations, graine=g, verbeux=False))
        sol = tg.resoudre(depart)
        ind = sol.indicateurs()
        mesures["cout"].append(ev.cout(sol))
        mesures["sd_lits"].append(ind["lits_ecart_type"])
        mesures["sd_places"].append(ind["places_ecart_type"])
        mesures["remplissage"].append(ind["taux_remplissage_moyen"])
        mesures["programmes"].append(ind["patients_programmes"])

    print(f"  {len(graines)} graines, {iterations} itérations chacune\n")
    print(f"  {'mesure':16} {'moyenne':>10} {'écart-type':>12} {'min':>10} "
          f"{'max':>10} {'CV':>8}")
    for nom, vals in mesures.items():
        m = statistics.fmean(vals)
        e = statistics.pstdev(vals)
        cv = e / m if m else 0
        print(f"  {nom:16} {m:10.4f} {e:12.4f} {min(vals):10.4f} "
              f"{max(vals):10.4f} {cv:7.1%}")
    cv_cout = statistics.pstdev(mesures["cout"]) / statistics.fmean(mesures["cout"])
    print()
    if cv_cout < 0.02:
        print(f"  Coefficient de variation du coût : {cv_cout:.1%}. Le tabou est")
        print("  stable, une seule graine suffirait — mais publiez quand même")
        print("  moyenne et écart-type.")
    else:
        print(f"  Coefficient de variation du coût : {cv_cout:.1%}. C'est trop :")
        print("  augmentez les itérations ou la diversification avant de comparer")
        print("  quoi que ce soit.")


# ===========================================================================
# NIVEAU 1 bis — Non-régression
# ===========================================================================


FENETRE_REFERENCE = None
"""Fenêtre utilisée par le banc d'essai : celle de chaque patient.

`None` veut dire « chacun garde la sienne ». En simulation elles sont tirées
uniformément entre les trois, sans rien caler sur le fichier source — c'est ce
qui permet de rejouer exactement le même protocole sur une autre base de
données. Mettre une chaîne ici forcerait la même fenêtre pour tout le monde,
ce qui ne sert qu'à répondre à une question de sensibilité.
"""

REFERENCE = {
    # Mesures obtenues sur le 1er semestre 2022, poids par défaut, choix =
    # recommandation. À REGÉNÉRER volontairement quand vous changez le modèle,
    # jamais à ajuster pour faire passer un test.
    "patients_dates": (1567, 40),          # (valeur attendue, tolérance)
    "lits_ecart_type": (3.63, 0.25),
    "remplissage_ecart_type": (0.246, 0.02),
    "taux_remplissage_moyen": (0.648, 0.02),
}


def non_regression(inst: Instance) -> bool:
    """Le garde-fou le plus rentable de tout le projet.

    Une modification innocente — un poids, un seuil, une inégalité large
    devenue stricte — peut dégrader silencieusement la qualité des plannings
    sans casser aucun test de correction. On fige donc les chiffres de
    référence et on vérifie qu'ils ne bougent pas.

    La tolérance n'est pas de la complaisance : elle absorbe le bruit
    stochastique mesuré par `stabilite_graines`. Une tolérance plus étroite
    que ce bruit produirait des échecs aléatoires, et on finirait par ne plus
    lire le test.
    """
    titre("NON-RÉGRESSION")
    sol, journal = simuler_consultations(inst, Poids(), choix=choix_meilleur,
                                         fenetre=FENETRE_REFERENCE)
    obtenu = {**journal.resume(), **sol.indicateurs()}
    ok = True
    for cle, (attendu, tol) in REFERENCE.items():
        val = obtenu[cle]
        passe = abs(val - attendu) <= tol
        ok &= passe
        print(f"  {'OK ' if passe else 'ÉCHEC'} {cle:26} attendu {attendu:>8} "
              f"± {tol:<6} obtenu {val:8.3f}")
    print("\n  " + ("Aucune régression." if ok else
                    "RÉGRESSION : une mesure a bougé au-delà du bruit stochastique."))
    return ok


# ===========================================================================


def main(chemin: str) -> None:
    validation_temporelle(chemin)
    comparer_politiques_duree(chemin)

    df = charger_historique(chemin)
    estimateur = construire_estimateur(df[df["date_inter"] < "2022-01-01"])
    inst, _ = construire_instance(df, estimateur, nb_semaines=26)
    inst.capacite_places_jour = 18

    escalade_charge(inst)
    stabilite_graines(inst)
    non_regression(inst)

    titre("CE QU'IL RESTE À TESTER")
    print("""
  Trois tests qui manquent encore, par ordre de valeur :

  1. STOCHASTIQUE SUR LES DURÉES. Ici on rejoue LA durée réelle observée.
     Un vrai test de robustesse tire la durée dans sa distribution (log-normale
     ajustée sur l'historique de l'acte), refait tourner 200 fois, et compte
     la part de journées qui tiennent. On obtient une probabilité de
     dépassement, pas un chiffre unique — c'est ce qui permet de dire « cette
     vacation tient dans 92 % des cas ».

  2. URGENCES. Le modèle ne gère que le programmé. Injecter un flux d'urgences
     (Poisson, taux calibré sur les plages URGENCES de la grille) et mesurer
     combien de patients programmés sont déprogrammés est LE test qui
     rapprocherait la simulation de la réalité d'un bloc.

  3. ABSENCES ET FERMETURES. Supprimer au hasard 5 % des vacations
     (congé, panne, fermeture de salle) et mesurer le coût de la
     replanification avec `TabouGlobal.figer_avant`. C'est le seul usage où le
     tabou global sert en exploitation, et il n'est pas encore éprouvé.
""")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FICHIER)  # ou --donnees X
