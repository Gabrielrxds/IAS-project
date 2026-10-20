"""Dater les patients : énumération (un patient à la fois, meilleure date) et lots du jour
optimisés par tabou, recuit ou génétique, chacun SANS et AVEC la courbe des fantômes.
Grille actuelle, 3 ordres de consultation, budget 400 évaluations par lot.
Sortie : resultats/dates_fantomes.json"""
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

from protocole import charger, grille_actuelle_neutre, mesurer, RACINE
from etude_dynamique import dater_par_lots

K = ["lits_crit1", "lits_crit1_max", "lits_semaine_variance", "places_ecart_type", "sans_date",
     "delai_median", "delai_p90"]


def un(args):
    methode, fant, g = args
    _, don = charger()
    t = time.time()
    sol, st = dater_par_lots(don, grille_actuelle_neutre(don), methode, 400, g, fantomes=fant)
    m = mesurer(don, sol)
    return dict(methode=methode, fantomes=fant, graine=g, secondes=time.time() - t,
                temps_lot=st["temps_moyen_s"], **{k: m[k] for k in K})


if __name__ == "__main__":
    t = time.time()
    taches = [(m, f, g) for m in ("glouton", "tabou", "recuit", "génétique") for f in (False, True)
              for g in (0, 1, 2)]
    with Pool(2) as P:
        R = P.map(un, taches, chunksize=1)
    for m in ("glouton", "tabou", "recuit", "génétique"):
        for f in (False, True):
            rs = [r for r in R if r["methode"] == m and r["fantomes"] == f]
            print(m, "fantômes" if f else "sans", {k: round(float(np.mean([r[k] for r in rs])), 2) for k in K},
                  flush=True)
    (RACINE / "resultats" / "dates_fantomes.json").write_text(json.dumps(dict(runs=R, secondes=time.time() - t),
                                                                          default=float))
    print(f"fin ({time.time() - t:.0f} s)")
