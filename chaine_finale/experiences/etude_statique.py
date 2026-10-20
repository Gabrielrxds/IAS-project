r"""
etude_statique.py — Approche STATIQUE : construire la grille des vacations avec chacune des trois
métaheuristiques (tabou, recuit simulé, algorithme génétique), sur le même problème et avec le même
budget d'évaluations.

Problème (celui de la grille hybride) : cycle de 2 semaines (paire / impaire), au plus une journée
d'écart entre les deux semaines d'un chirurgien, autant d'heures de bloc qu'aujourd'hui, volume de
chaque chirurgien entre 75 % et 150 % de l'actuel, coût F = 4 écart des lits à la cible + admissions
+ temps perdu + délai + salles (chaque terme rapporté à la grille actuelle).

A. Optimisation : 3 algorithmes x budgets x 5 graines, départ aléatoire ; coût, temps, convergence.
B. Ce que vaut chaque grille sur les patients de 2022 : la grille est déroulée sur l'année, puis les
   patients consultent et reçoivent la première date libre (règle neutre, la même pour tous, pour
   juger la grille seule). Indicateurs du protocole commun, mai-décembre 2022.
Sortie : resultats/etude_statique.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from protocole import charger, instance, mesurer, vacations_vers_neutre, grille_actuelle_neutre, \
    lire_grille, sauver_grille, Reglages, RACINE
from analyse_couts import Reglages as R0  # noqa: F401
from genetique_grille import PoidsGrille
from hybride_grille import probleme_hybride
from algos_grille import ALGOS
from politiques import simuler, PremierCreneau
import calendrier

BUDGETS = [2000, 5000, 15000, 40000]
GRAINES = range(5)
W = PoidsGrille(lits=4.0)
K = ["lits_crit1", "lits_semaine_variance", "lits_semaine_max", "lits_ecart_cible", "lits_ratio_weekend",
     "lits_ratio_vacances", "lits_variance", "places_ecart_type", "places_max", "sans_date",
     "delai_median", "delai_p90", "occupation_reelle_pct", "heures_offertes_semaine"]


def probleme():
    _, don = charger()
    reg = Reglages()
    reg.cycle_semaines = 2
    return don, reg, probleme_hybride(don, reg, part_min=0.75, part_max=1.5, f_we=calendrier.F_WE)


def optimiser(args):
    algo, budget, g = args
    don, reg, prob = probleme()
    s = ALGOS[algo](prob, W, budget, g)
    t = prob.termes_individu(s.X)
    return dict(algo=algo, budget=budget, graine=g, cout=float(s.cout), evaluations=s.evaluations,
                secondes=s.secondes, trace=s.trace, X=s.X.tolist(),
                termes={k: float(v / prob.ref[k]) for k, v in t.items() if k in prob.ref},
                ecart_semaines=float(t.get("ecart_semaines", 0.0)))


def simuler_grille(args):
    nom, X = args
    don, reg, prob = probleme()
    if X is None:
        lignes = grille_actuelle_neutre(don) if nom == "actuelle" else lire_grille(nom)["vacations"]
    else:
        vacs = prob.derouler(np.array(X), don.jour_zero, don.nb_jours, don.feries)
        lignes = vacations_vers_neutre(don, vacs)
    sol = simuler(instance(don, lignes, reg), PremierCreneau(), graine=0)
    m = mesurer(don, sol)
    return nom, {k: m[k] for k in K}


if __name__ == "__main__":
    BUDGETS = [int(b) for b in sys.argv[1:]] or BUDGETS
    t0 = time.time()
    taches = [(a, b, g) for b in BUDGETS for a in ALGOS for g in GRAINES]
    with Pool() as P:
        R = P.map(optimiser, taches, chunksize=1)
    for b in BUDGETS:
        for a in ALGOS:
            c = [r["cout"] for r in R if r["algo"] == a and r["budget"] == b]
            print(f"B={b:6d} {a:10} coût {np.mean(c):.3f} ± {np.std(c):.3f}  "
                  f"temps {np.mean([r['secondes'] for r in R if r['algo'] == a and r['budget'] == b]):.0f} s",
                  flush=True)
    # B : chaque grille du plus grand budget, rejouée sur 2022
    bmax = max(BUDGETS)
    # grille retenue de chaque méthode (plus faible coût au plus grand budget) : chaîne mono-méthode
    don, reg, prob = probleme()
    for a in ALGOS:
        r = min((r for r in R if r["algo"] == a and r["budget"] == bmax), key=lambda r: r["cout"])
        vacs = prob.derouler(np.array(r["X"]), don.jour_zero, don.nb_jours, don.feries)
        sauver_grille("statique_" + a.replace("é", "e"), vacations_vers_neutre(don, vacs),
                      {"methode": a, "graine": r["graine"], "budget": bmax, "cout": r["cout"], "X": r["X"]})
    a_sim = [(f"{r['algo']}#{r['graine']}", r["X"]) for r in R if r["budget"] == bmax]
    a_sim += [("actuelle", None), ("hybride", None)]
    with Pool() as P:
        S = dict(P.map(simuler_grille, a_sim, chunksize=1))
    for k in ("actuelle", "hybride", *ALGOS):
        rs = [v for n, v in S.items() if n == k or n.startswith(k + "#")]
        print(k, {m: round(float(np.mean([r[m] for r in rs])), 2) for m in K[:6] + ["sans_date", "delai_median", "delai_p90"]})
    (RACINE / "resultats" / "etude_statique.json").write_text(json.dumps(
        dict(optimisation=R, simulation=S, budgets=BUDGETS, secondes=time.time() - t0), default=float))
    print(f"fin ({time.time() - t0:.0f} s)")
