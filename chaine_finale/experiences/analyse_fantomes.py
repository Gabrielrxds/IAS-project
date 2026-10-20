"""Patients fantômes : avec vs sans, sur le protocole commun (test mai-décembre 2022).
  A. grille hybride et grille actuelle, Δ = 14 j : sans fantômes / avec fantômes, 5 graines
  B. fenêtre Δ de 0 à 28 jours, sans / avec, 3 graines (grille hybride)
  C. mécanisme (graine 0, grille hybride, Δ = 14) : jour choisi par patient, courbe attendue à une date,
     qualité de prévision de la courbe attendue
Sortie : resultats/fantomes.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import statistics as st
import numpy as np
from protocole import charger, instance, grille_actuelle_neutre, lire_grille, mesurer, mesurer_reel, Reglages, RACINE
from politiques import simuler
from regle_hybride import RegleHybrideV1 as RegleHybride, classe

df, don = charger(); reg = Reglages()
GR = {"hybride": lire_grille("hybride")["vacations"], "actuelle": grille_actuelle_neutre(don)}
t0, t1 = don.test
CLES = ["lits_variance", "lits_ecart_type", "lits_p95", "lits_max", "lits_moyenne", "lits_variation_nuit",
        "places_ecart_type", "places_max", "sans_date", "delai_median", "delai_moyen", "delai_p90",
        "patients_par_semaine", "occupation_reelle_pct"]


class Espion(RegleHybride):
    """Règle hybride instrumentée : jour choisi vs première date possible, courbes attendues."""

    def __init__(self, *a, instants=(), horizons=(7, 14, 28, 42, 56), **k):
        super().__init__(*a, **k)
        self.choix_log, self.instants, self.horizons = [], set(instants), horizons
        self.snap, self.prev = {}, []

    def _fantomes(self):
        G = np.zeros_like(self.E)
        E = self.E
        self.E = G
        for vid in self.dans:
            self._poser_fantomes(vid, self.r[vid])
        self.E = E
        return G

    def preparer(self, sol, t):
        super().preparer(sol, t)
        if t0 - 7 <= t <= t1 - 60:
            G = self._fantomes() if self.mode != "aucun" else np.zeros_like(self.E)
            if t % 7 == 0:
                self.prev.append((t, {h: float(self.E[t + h]) for h in self.horizons},
                                  {h: float(self.E[t + h] - G[t + h]) for h in self.horizons}))
            if t in self.instants:
                self.snap[t] = {"E": self.E[t + 1:t + 94].tolist(), "G": G[t + 1:t + 94].tolist()}

    def consulter(self, sol, pid, t, figes):
        inst = self.inst
        j0 = None
        for vid in inst.vacations_possibles(pid):
            j = inst.vacations[vid].jour
            if j in figes:
                continue
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if ok:
                j0 = j
                break
        vid = super().consulter(sol, pid, t, figes)
        if vid is not None and j0 is not None:
            self.choix_log.append((pid, classe(inst.patients[pid]), inst.vacations[vid].jour - j0,
                                   inst.vacations[vid].jour, t))
        return vid


def lancer(grille, mode, delta=14, graine=0, espion=False, instants=()):
    R = (Espion(don, delta=delta, fantomes=mode, instants=instants) if espion
         else RegleHybride(don, delta=delta, fantomes=mode))
    sol = simuler(instance(don, GR[grille], reg), R, graine=graine)
    return sol, R


out = {"A": [], "B": [], "C": {}}
# ------------------------------------------------------------------ A
for g in ("hybride", "actuelle"):
    for mode in ("aucun", "uniformes"):
        ms = [mesurer(don, lancer(g, mode, 14, s)[0]) for s in range(5)]
        agg = {"grille": g, "fantomes": mode}
        for k in CLES:
            agg[k] = st.mean(m[k] for m in ms); agg[k + "_sd"] = st.pstdev(m[k] for m in ms)
        agg["lits_serie"] = ms[0]["lits_serie"]
        out["A"].append(agg)
        print("A", g, mode, {k: round(agg[k], 2) for k in ("lits_variance", "lits_max", "places_ecart_type", "sans_date", "delai_median", "delai_p90")}, flush=True)
# ------------------------------------------------------------------ B
for d in (0, 7, 14, 21, 28):
    for mode in ("aucun", "uniformes"):
        ms = [mesurer(don, lancer("hybride", mode, d, s)[0]) for s in range(3)]
        agg = {"delta": d, "fantomes": mode}
        for k in CLES:
            agg[k] = st.mean(m[k] for m in ms); agg[k + "_sd"] = st.pstdev(m[k] for m in ms)
        out["B"].append(agg)
        print("B", d, mode, round(agg["lits_variance"], 2), round(agg["delai_median"], 1), flush=True)
# ------------------------------------------------------------------ C
instants = [t0 + 35, t0 + 105]          # début juin, début août (environ)
for mode in ("aucun", "uniformes"):
    sol, R = lancer("hybride", mode, 14, 0, espion=True, instants=instants)
    L = np.array(sol.lits_jour, dtype=float)
    prev = []
    for t, e, r in R.prev:
        for h in e:
            if t + h < len(L) and t0 <= t + h <= t1:
                prev.append({"t": t, "h": h, "attendu": e[h], "reels_seuls": r[h], "final": float(L[t + h])})
    out["C"][mode] = {
        "choix": R.choix_log, "prev": prev,
        "snap": {str(t): {**v, "final": L[t + 1:t + 94].tolist(), "date": don.date(t).isoformat()}
                 for t, v in R.snap.items()},
        "lits": [int(x) for x in L[t0:t1 + 1]], "dates": [don.date(j).isoformat() for j in range(t0, t1 + 1)],
    }
    print("C", mode, len(R.choix_log), "choix", len(prev), "prévisions", flush=True)
out["reel"] = mesurer_reel(don)
(RACINE / "resultats" / "fantomes.json").write_text(json.dumps(out, default=float))
print("ok")
