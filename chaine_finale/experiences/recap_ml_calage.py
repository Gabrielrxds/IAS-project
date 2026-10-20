"""Relance, avec les fantômes recalés sur les durées prévues par apprentissage, les entrées
« fantômes + durées et nuits par apprentissage » de recap_v3.json (plan, matin, jour J)."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
from multiprocessing import Pool
import numpy as np
import recap_v3 as R
R.CALAGE_ML = True
from protocole import RACINE

if __name__ == "__main__":
    f = RACINE / "resultats" / "recap_v3_calage_ml.json"
    out = json.loads((RACINE / "resultats" / "recap_v3.json").read_text())
    e = [x for x in R.ESCALIER if x[3] == "previsions" and x[4] == "ml"]
    with Pool(2) as P:
        plans = P.map(R._plan, [(*x, g) for x in e for g in R.GRAINES_PLAN])
        R._matin(("previsions", "ml"))
        jour = P.map(R._jour, [("previsions", "ml", pol, g) for pol in ("statique", "sma") for g in R.GRAINES_J], chunksize=2)
    noms = {x[0] for x in e}
    out["plans"] = [r for r in out["plans"] if r["nom"] not in noms] + plans
    out["jour"] = [r for r in out["jour"] if not (r["fantomes"] == "previsions" and r["prevision"] == "ml")] + jour
    f.write_text(json.dumps(out, default=float))
    for n in noms:
        rs = [r for r in plans if r["nom"] == n]
        print(n, {k: round(float(np.mean([r[k] for r in rs])), 2) for k in ("lits_crit1", "lits_crit1_max", "delai_median", "delai_p90")})
    for pol in ("statique", "sma"):
        rs = [r for r in jour if r["politique"] == pol]
        print(pol, {k: round(float(np.mean([r[k] for r in rs])), 2) for k in ("lits_crit1", "lits_crit1_max", "urg_delai", "depassement_jour", "reportes")})
