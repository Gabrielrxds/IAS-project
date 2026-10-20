"""Grille de chaque étude + règle de date qui regarde les lits (fenêtre de 14 j), sans puis avec
patients fantômes, 10 ordres de consultation (rapport V6, tableau des études).
Sortie : resultats/fantomes_grilles10.json"""
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
import etudes_origine as E
from protocole import RACINE

if __name__ == "__main__":
    t = time.time()
    jobs = [("F", f"{g} | {r}", g, r, None, s) for g in ("actuelle", "tabou", "génétique", "recuit", "hybride")
            for r in ("hybride_sans", "hybride") for s in range(10)]
    with Pool(2) as P:
        res = P.map(E.run, jobs, chunksize=1)
    (RACINE / "resultats" / "fantomes_grilles10.json").write_text(json.dumps({"runs": res}, default=float))
    for nm in dict.fromkeys(r["nom"] for r in res):
        rs = [r for r in res if r["nom"] == nm]
        print(f"{nm:28s}", {k: (round(float(np.mean([r[k] for r in rs])), 2), round(float(np.std([r[k] for r in rs])), 2))
                             for k in ("lits_crit1", "lits_crit1_max", "places_ecart_type", "delai_median", "delai_p90",
                                       "sans_date", "occupation_reelle_pct")}, flush=True)
    print(f"fin ({time.time() - t:.0f} s)")
