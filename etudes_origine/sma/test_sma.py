r"""
test_sma.py — Tests du SMA sur la base commune (V1 corrigée), données 2022.

    python test_sma.py donees_bloc_anonyme_pour_centrale_2026.xlsx

Vérifie :
  1. invariants de chaque journée, pour les trois politiques : chaque
     programmé opéré une fois (ou annulé), jamais chez un autre chirurgien,
     jamais un autre jour, jamais un programmé dans un créneau URGENCES, pas
     de chevauchement en salle, Solution de travail à jour ;
  2. la Solution de travail reste cohérente pour modele.py
     (`controle_coherence`) et n'a aucun conflit de règle « urgences »
     (U1 programmé en créneau URGENCES, U3 programmé dans le tampon,
     U4/U5 programmé dans les réserves) ni G1/G5/G7 ;
  3. la Solution et l'Instance d'origine ne sont PAS modifiées ;
  4. reproductibilité ;
  5. optimiseurs interchangeables : AG journalier, un « collègue » qui renvoie
     un PlanningJour, un autre un tuple, un optimiseur FAUX (perd un patient)
     et un qui PLANTE — la simulation continue, les échecs sont comptés ;
  6. plans B issus de plusieurs méthodes.
"""

import sys
import time

from demo_sma import construire_instance, construire_solution
from modele import deriver_creneaux
from sma_aleas import (POLITIQUES, GenerateurAleas, ParametresSMA, adapter,
                       construire_plans_b, optimiseur_ag_journalier,
                       optimiseur_glouton, preparer_plans, resumer,
                       sequences_du_jour, simuler_periode)

REGLES_INTERDITES = {"G1", "G5", "G7", "U1", "U3", "U4", "U5"}


def instantane(sol):
    inst = sol.inst
    return (dict(sol.affectation), list(sol.lits_jour), list(sol.places_jour),
            set(inst.patients), set(sol.hors_horizon),
            {v: (x.debut, x.fin, x.med_id) for v, x in inst.vacations.items()})


def verifier_periode(J, work):
    for j in J:
        j.controle_coherence()
    work.controle_coherence()
    mauvais = [c for c in work.verifier() if c.regle in REGLES_INTERDITES]
    assert not mauvais, [str(c) for c in mauvais[:3]]


def main(chemin):
    inst, _ = construire_instance(chemin)
    sol = construire_solution(inst)
    avant = instantane(sol)
    jours = [j for j in inst.jours_ouvres
             if inst.date_du_jour(j).year == 2022 and sequences_du_jour(sol, j)][:60]
    gen = GenerateurAleas(urgences_par_jour=2.0, p_annulation=0.05, p_lits=0.3, graine=3)

    print(f"1-2. invariants sur {len(jours)} jours consécutifs, trois politiques")
    res = {}
    for pol in POLITIQUES:
        b, J, work = simuler_periode(sol, jours, gen, ParametresSMA(politique=pol),
                                     garder_journees=True)
        verifier_periode(J, work)
        res[pol] = resumer(b)
        r = res[pol]
        print(f"      {pol:<15} OK — urgences : {int(r['urgences_operees_jour_meme'])} le jour "
              f"même, {int(r['urgences_operees_lendemain'])} le lendemain, "
              f"{int(r['urgences_echec'])} échecs ; {int(r['depassement_min'])} min de dépassement")

    print("3. Solution et Instance d'origine intactes :", end=" ")
    assert instantane(sol) == avant
    print("OK")

    print("4. reproductibilité :", end=" ")
    b2, _, _ = simuler_periode(sol, jours, gen, ParametresSMA(politique="sma"))
    assert resumer(b2) == res["sma"]
    print("OK")

    print(f"5. optimiseurs interchangeables ({len(jours)} jours, politique réoptimisation)")
    jours20 = jours[:20]
    collegue_planning = adapter(lambda i, j, s: deriver_creneaux(i, j, optimiseur_glouton(i, j, s)),
                                "collègue (renvoie un PlanningJour)")
    collegue_tuple = adapter(lambda i, j, s: (optimiseur_glouton(i, j, s), {"stats": 0}),
                             "collègue (renvoie un tuple)")

    def faux(i, j, s):
        s = {v: list(x) for v, x in s.items()}
        for v in s:
            if s[v]:
                s[v].pop()                 # perd un patient
                break
        return s

    def plante(i, j, s):
        raise RuntimeError("bug du collègue")

    optimiseurs = {"glouton": optimiseur_glouton,
                   "AG journalier": optimiseur_ag_journalier(temps_max=0.3),
                   "PlanningJour": collegue_planning, "tuple": collegue_tuple,
                   "faux": adapter(faux, "faux"), "plante": adapter(plante, "plante")}
    for nom, opt in optimiseurs.items():
        t0 = time.perf_counter()
        b, J, work = simuler_periode(sol, jours, gen, ParametresSMA(politique="reoptimisation"),
                                     optimiseur=opt, garder_journees=True)
        verifier_periode(J, work)
        r = resumer(b)
        print(f"      {nom:<14} {int(r['reordonnancements']):>3} réordonnancements, "
              f"{int(r['echecs_optimiseur']):>3} échecs, {time.perf_counter() - t0:.1f} s")
        if nom in ("faux", "plante"):
            assert r["echecs_optimiseur"] > 0 and r["reordonnancements"] == 0
        else:          # sortie valide : acceptée quand elle améliore le plan
            assert r["echecs_optimiseur"] == 0 and r["reordonnancements"] > 0

    print("6. plans du matin par l'AG, plans B de deux méthodes, politique SMA")
    plans = preparer_plans(sol, jours20, optimiseur_ag_journalier(temps_max=0.5))
    pb = {j: construire_plans_b(inst, j, plans[j],
                                {"glouton": optimiseur_glouton, "PlanningJour": collegue_planning})
          for j in jours20}
    b, J, work = simuler_periode(sol, jours20, gen, ParametresSMA(politique="sma"),
                                 optimiseur=optimiseur_ag_journalier(temps_max=0.3),
                                 plans=plans, plans_b=pb, garder_journees=True)
    verifier_periode(J, work)
    print(f"      OK — {sum(len(p) for p in pb.values())} plans B sur {len(jours20)} jours")
    assert instantane(sol) == avant
    print("\nTous les tests passent.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "donees_bloc_anonyme_pour_centrale_2026.xlsx")
