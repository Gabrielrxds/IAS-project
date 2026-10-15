"""Patients non venus : reprogrammation sans priorité (nouvelle demande le jour de l'absence).
Grille hybride, test mai-déc. 2022, taux 0 / 2 / 5 %, 5 graines. Sortie : resultats/non_venues.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, statistics as st
import numpy as np
from protocole import charger, instance, lire_grille, mesurer, Reglages, RACINE
from regle_hybride import RegleHybride
from absences import simuler_absences
df, don = charger(); reg = Reglages(); t0, t1 = don.test
G = lire_grille("hybride")["vacations"]
out = []
for taux in (0.0, 0.02, 0.05):
    rs = []
    for g in range(5):
        inst = instance(don, G, reg)
        sol, RP = simuler_absences(inst, RegleHybride(don, delta=14, fantomes="uniformes"), [], "hybride", g, taux_non_venue=taux)
        m = mesurer(don, sol)
        NV = [n for n in RP.non_venues if t0 <= n.jour_initial <= t1]
        ids = {n.pid for n in NV}
        d = [n.jour_nouveau - n.jour_initial for n in NV if n.jour_nouveau is not None]
        # délais des AUTRES patients de la cohorte (non touchés)
        coh = [p for p, j in don.jour_reel.items() if t0 <= j <= t1 and p not in ids and sol.affectation.get(p) is not None]
        dl = [inst.vacations[sol.affectation[p]].jour - inst.patients[p].jour_demande for p in coh]
        rs.append(dict(non_venus=len(NV), sans_date_nv=sum(n.jour_nouveau is None for n in NV), recidives=sum(n.rang > 1 for n in NV),
                       re_delai_moyen=float(np.mean(d)) if d else 0.0, re_delai_median=float(np.median(d)) if d else 0.0,
                       re_delai_p90=float(np.percentile(d, 90)) if d else 0.0,
                       heures_perdues=sum(inst.patients[n.pid].duree_op for n in NV) / 60,
                       lits_variance=m["lits_variance"], lits_max=m["lits_max"], sans_date=m["sans_date"],
                       autres_delai_median=float(np.median(dl)), autres_delai_p90=float(np.percentile(dl, 90)), conflits=m["conflits"]))
    agg = {"taux": taux, **{k: st.mean(r[k] for r in rs) for k in rs[0]}}
    out.append(agg)
    print({k: round(v, 2) for k, v in agg.items()}, flush=True)
(RACINE / "resultats" / "non_venues.json").write_text(json.dumps(out))
