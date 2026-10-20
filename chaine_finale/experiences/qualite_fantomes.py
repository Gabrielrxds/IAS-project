"""Qualité de la prévision des lits par les patients fantômes (rapport V3, partie 5.2).

Chaque jour de consultation t, la règle hybride « voit » pour chaque nuit future k une courbe
E_t(k) = lits des patients déjà datés + fantômes programmés + fantômes d'urgence.
On la compare aux lits finalement occupés cette nuit-là (programmés + urgences réelles).

Variantes (grille hybride, graine 0) : aucun fantôme, fantômes V1 (remplissage figé),
fantômes V2 sans urgences, fantômes V2 avec urgences (retenus).
Sorties : resultats/fantomes_v3.json — RMSE et biais par horizon ; décomposition de la courbe
vue à une date ; évolution, pour quelques nuits, de ce qui est réservé et de ce qui est prévu.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys

import numpy as np

from protocole import charger, instance, lire_grille, mesurer, Reglages, RACINE, fond_urgences
from politiques import simuler
from regle_hybride import RegleHybride, RegleHybrideV2

HORIZONS = (1, 7, 14, 21, 28, 35, 42, 56)
VARIANTES = [("aucun", "Sans fantôme"), ("v1", "Fantômes V1"),
             ("previsions_sans_urgences", "Prévision, sans urgences"),
             ("previsions", "Prévision avec urgences (retenue)")]


class Espion:
    def __init__(self, r):
        self.r, self.snap = r, {}

    def preparer(self, sol, t):
        self.r.preparer(sol, t)
        E = self.r.E
        a, b = t + 1, min(len(E), t + 64)
        d = {"E": E[a:b].copy()}
        if isinstance(self.r, RegleHybrideV2):
            d.update(Er=self.r.Er[a:b].copy(), Eg=self.r.Eg[a:b].copy(), Eu=self.r.Eu[a:b].copy())
        self.snap[t] = (a, d)

    def consulter(self, sol, pid, t, f):
        return self.r.consulter(sol, pid, t, f)

    def apres(self, sol, vid, t, f):
        self.r.apres(sol, vid, t, f)


def main():
    _, don = charger()
    reg = Reglages()
    G = lire_grille("hybride")["vacations"]
    fl, _ = fond_urgences(don)
    t0, t1 = don.test
    out = {"horizons": HORIZONS, "variantes": []}
    for cle, nom in VARIANTES:
        S = Espion(RegleHybride(don, delta=14, fantomes=cle))
        sol = simuler(instance(don, G, reg), S, graine=0)
        m = mesurer(don, sol)
        n = min(len(fl), len(sol.lits_jour))
        final = np.asarray(sol.lits_jour, float)[:n] + fl[:n]
        rm, bi = {}, {}
        for h in HORIZONS:
            e = [d["E"][h - 1] - final[a + h - 1] for t, (a, d) in S.snap.items()
                 if t0 <= t <= t1 - h and len(d["E"]) >= h]
            rm[h], bi[h] = float(np.sqrt(np.mean(np.square(e)))), float(np.mean(e))
        v = dict(cle=cle, nom=nom, rmse=rm, biais=bi, lits_crit1=m["lits_crit1"],
                 lits_semaine_variance=m["lits_semaine_variance"], lits_max=m["lits_max"],
                 delai_median=m["delai_median"], sans_date=m["sans_date"])
        print(nom, {h: round(x, 2) for h, x in rm.items()}, "crit1", round(m["lits_crit1"], 2), flush=True)
        if cle == "previsions":
            # courbe vue un jour de consultation (début juin) et ce qui s'est réellement passé
            # jour de consultation de début septembre : six semaines sans vacances ni jour férié
            import datetime as _dt
            tv = next(t for t in sorted(S.snap) if don.date(t) >= _dt.date(2022, 9, 5) and don.date(t).weekday() == 0)
            a, d = S.snap[tv]
            n = min(len(d["E"]), 42)
            v["vue"] = dict(jour=tv, date=str(don.date(tv)), premiere_nuit=a,
                            dates=[str(don.date(a + i)) for i in range(n)],
                            Er=d["Er"][:n].tolist(), Eg=d["Eg"][:n].tolist(), Eu=d["Eu"][:n].tolist(),
                            final=final[a:a + n].tolist())
            # pour quelques nuits : réservé / fantômes au fil des jours de consultation
            nuits = [tv + 29, tv + 30, tv + 33]     # un mardi, un mercredi, un samedi
            v["suivi"] = []
            for k in nuits:
                lig = []
                for t in sorted(S.snap):
                    a2, d2 = S.snap[t]
                    i = k - a2
                    if 0 <= i < len(d2["E"]):
                        lig.append(dict(t=t, avance=k - t, Er=float(d2["Er"][i]), Eg=float(d2["Eg"][i]),
                                        Eu=float(d2["Eu"][i])))
                v["suivi"].append(dict(nuit=k, date=str(don.date(k)), final=float(final[k]),
                                       final_urgences=float(fl[k]), lignes=lig))
        out["variantes"].append(v)
    (RACINE / "resultats" / "fantomes_v3.json").write_text(json.dumps(out, default=float))


if __name__ == "__main__":
    main()
