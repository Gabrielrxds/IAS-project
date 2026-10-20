r"""
etude_cibles.py — Quelle cible de lits pour le week-end et les vacances ?
Pour chaque (f_we, f_vac) : grille hybride reconstruite avec la cible, règle de date avec la cible,
3 ordres de consultation. Mesures indépendantes de la cible : variance et pic des nuits de semaine
hors vacances (lissage), rapports week-end / semaine et vacances / semaine, délais.
Sortie : resultats/etude_cibles.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, statistics as st, sys
from multiprocessing import Pool
from protocole import charger, instance, lire_grille, mesurer, Reglages, RACINE
from politiques import simuler
from regle_hybride import RegleHybride
import chaine_hybride

CIBLES = [(1.0, 1.0), (0.85, 0.70), (0.80, 0.75), (0.75, 0.70), (0.70, 0.65), (0.65, 0.60)]
K = ["lits_crit1", "lits_crit1_max", "lits_variance", "lits_max", "lits_p95", "lits_semaine_variance", "lits_semaine_max", "lits_nuits_semaine_moy",
     "lits_ratio_weekend", "lits_ratio_vacances", "places_ecart_type", "sans_date", "delai_median", "delai_p90",
     "occupation_reelle_pct", "conflits"]


def nom(c):
    return "hybride_variance" if c == (1.0, 1.0) else f"hybride_c{int(c[0]*100)}_{int(c[1]*100)}"


def grille(c):
    # chaque couple a sa grille (cycle pair / impair) ; (1, 1) = variance seule, sans cible
    chaine_hybride.main(nom=nom(c), cible=None if c == (1.0, 1.0) else c)
    return c


def un(args):
    c, g = args
    _, don = charger()
    inst = instance(don, lire_grille(nom(c))["vacations"], Reglages())
    sol = simuler(inst, RegleHybride(don, delta=14, fantomes="uniformes", cible=None if c == (1.0, 1.0) else c), graine=g)
    m = mesurer(don, sol)
    return dict(cible=c, graine=g, **{k: m[k] for k in K})


if __name__ == "__main__":
    with Pool() as P:
        P.map(grille, CIBLES, chunksize=1)
        R = P.map(un, [(c, g) for c in CIBLES for g in (0, 1, 2)], chunksize=1)
    (RACINE / "resultats" / "etude_cibles.json").write_text(json.dumps(R, indent=1))
    print("cible | var tot | var sem | pic sem | WE/sem | vac/sem | sem moy | sans date | délai méd | délai 90 | pic")
    for c in CIBLES:
        rs = [r for r in R if tuple(r["cible"]) == c]
        m = {k: st.mean(r[k] for r in rs) for k in K}
        print(c, f"{m['lits_variance']:.2f} | {m['lits_semaine_variance']:.2f} | {m['lits_semaine_max']:.1f} | "
              f"{m['lits_ratio_weekend']:.2f} | {m['lits_ratio_vacances']:.2f} | {m['lits_nuits_semaine_moy']:.1f} | "
              f"{m['sans_date']:.0f} | {m['delai_median']:.0f} | {m['delai_p90']:.0f} | {m['lits_max']:.1f}")
