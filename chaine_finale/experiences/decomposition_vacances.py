"""Grille régulière (paire / impaire) contre grille libre sur 4 semaines, avec et sans plafond de
lits en vacances (règle hybride, 3 ordres de consultation). Sortie : resultats/vacances_v3.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import protocole
from protocole import charger, instance, lire_grille, mesurer, Reglages, RACINE
from politiques import simuler
from regle_hybride import RegleHybride

K = ["lits_crit1", "lits_crit1_max", "lits_semaine_variance", "lits_semaine_max", "lits_ratio_weekend",
     "lits_ratio_vacances", "lits_variance", "sans_date", "delai_median", "delai_p90", "patients_par_semaine"]


def main():
    _, don = charger()
    out = {}
    for g in ("hybride_cycle4", "hybride"):
        for cap in (False, True):
            protocole.CAPACITE_VACANCES = cap
            r = []
            for s in (0, 1, 2):
                inst = instance(don, lire_grille(g)["vacations"], Reglages())
                m = mesurer(don, simuler(inst, RegleHybride(don, delta=14), graine=s))
                r.append({k: m[k] for k in K})
            avg = {k: sum(x[k] for x in r) / 3 for k in K}
            out[f"{g}|plafond={cap}"] = avg
            print(g, cap, " ".join(f"{k[:14]}={avg[k]:.2f}" for k in K), flush=True)
    protocole.CAPACITE_VACANCES = True
    (RACINE / "resultats" / "vacances_v3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
