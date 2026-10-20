"""Réglage des paramètres des trois métaheuristiques jugé sur les LITS (rapport V3, 6.2).

Comme reglage_parametres.json (un paramètre à la fois, les autres à leur valeur de départ,
budget 15 000 évaluations, 4 départs aléatoires), mais chaque grille obtenue est déroulée puis
rejouée sur la période d'apprentissage (janvier-avril 2022, mesure février-avril), chaque patient
recevant la première date libre. On retient la valeur qui donne la plus faible variance des lits
des nuits de semaine (critère n°1), sans patient sans date de plus.
Sortie : resultats/reglage_lits.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import time
from multiprocessing import Pool

import numpy as np

from protocole import charger, instance, vacations_vers_neutre, mesurer_profils, Reglages, RACINE
from etude_statique import probleme
from genetique_grille import PoidsGrille
from algos_grille import ALGOS
from politiques import simuler, PremierCreneau
from chaine_hybride import don_apprentissage

W = PoidsGrille(lits=4.0)
BUDGET = 15000
GRAINES = (0, 1, 2, 3)
PLANS = [("tabou", "k", [5, 15, 30, 60], 30), ("tabou", "tenure", [2, 5, 15, 40], 15),
         ("recuit", "T0_facteur", [0.1, 0.3, 1.0, 3.0, 10.0], 1.0),
         ("recuit", "Tmin_facteur", [0.0005, 0.005, 0.05], 0.005),
         ("génétique", "population", [10, 20, 40, 80], 40),
         ("génétique", "p_mutation", [0.3, 0.7, 1.0], 0.7),
         ("génétique", "descente", [0, 1500, 4500], 1500)]


def un(args):
    algo, param, v, g = args
    don, reg, prob = probleme()
    s = ALGOS[algo](prob, W, BUDGET, g, **{param: v})
    dapp = don_apprentissage(don)
    vacs = prob.derouler(np.array(s.X), don.jour_zero, don.nb_jours, don.feries)
    inst = instance(dapp, vacations_vers_neutre(dapp, vacs), reg)
    sol = simuler(inst, PremierCreneau(), graine=0)
    m = mesurer_profils(dapp, sol.lits_jour, sol.places_jour)
    t0, t1 = dapp.test
    coh = [p for p, j in dapp.jour_reel.items() if t0 <= j <= t1]
    sans = sum(sol.affectation[p] is None for p in coh)
    d = [inst.vacations[sol.affectation[p]].jour - inst.patients[p].jour_demande for p in coh
         if sol.affectation[p] is not None]
    return dict(algo=algo, param=param, valeur=v, graine=g, cout=float(s.cout), crit1=float(m["lits_crit1"]),
                pic=float(m["lits_crit1_max"]), places=float(m["places_ecart_type"]), sans_date=int(sans),
                delai_p90=float(np.percentile(d, 90)), secondes=s.secondes)


if __name__ == "__main__":
    t = time.time()
    taches = [(a, p, v, g) for a, p, vals, _ in PLANS for v in vals for g in GRAINES]
    with Pool(2) as P:
        R = P.map(un, taches, chunksize=1)
    for a, p, vals, _ in PLANS:
        for v in vals:
            rs = [r for r in R if r["algo"] == a and r["param"] == p and r["valeur"] == v]
            print(a, p, v, "coût %.3f lits %.2f pic %.1f sans %.1f" % tuple(
                np.mean([r[k] for r in rs]) for k in ("cout", "crit1", "pic", "sans_date")), flush=True)
    (RACINE / "resultats" / "reglage_lits.json").write_text(json.dumps(dict(
        plans=[dict(algo=a, param=p, valeurs=v, depart=d) for a, p, v, d in PLANS], runs=R,
        secondes=time.time() - t), default=float))
    print(f"fin ({time.time() - t:.0f} s)")
