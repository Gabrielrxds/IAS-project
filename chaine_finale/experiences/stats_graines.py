"""Robustesse statistique : 10 graines (ordre des consultations d'un même jour), chaîne hybride
(sans journée) vs grille actuelle + premier créneau vs règle hybride sans fantômes.
Tests de Wilcoxon appariés (même graine). Sortie : resultats/stats_graines.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, numpy as np
from scipy.stats import wilcoxon
from protocole import charger, instance, lire_grille, grille_actuelle_neutre, mesurer, Reglages, RACINE
from politiques import simuler, PremierCreneau
from regle_hybride import RegleHybride
df, don = charger(); reg = Reglages()
G = {"hybride": lire_grille("hybride")["vacations"], "actuelle": grille_actuelle_neutre(don)}
CONF = {"reference": ("actuelle", lambda: PremierCreneau()),
        "hybride": ("hybride", lambda: RegleHybride(don, delta=14, fantomes="uniformes")),
        "hybride_sans_fantomes": ("hybride", lambda: RegleHybride(don, delta=14, fantomes="aucun"))}
K = ["lits_crit1", "lits_crit1_max", "lits_variance", "lits_semaine_variance", "lits_ecart_cible", "lits_semaine_max", "lits_ratio_weekend", "lits_max", "lits_p95", "places_ecart_type", "sans_date", "delai_median", "delai_p90", "occupation_reelle_pct"]
res = {c: {k: [] for k in K} for c in CONF}
for s in range(10):
    for c, (g, R) in CONF.items():
        m = mesurer(don, simuler(instance(don, G[g], reg), R(), graine=s))
        for k in K: res[c][k].append(m[k])
    print("graine", s, flush=True)
out = {"valeurs": res, "tests": {}}
for a, b in (("hybride", "reference"), ("hybride", "hybride_sans_fantomes")):
    for k in K:
        x, y = np.array(res[a][k]), np.array(res[b][k])
        d = x - y
        try:
            p = float(wilcoxon(x, y).pvalue) if np.any(d != 0) else 1.0
        except ValueError:
            p = float("nan")
        ic = (float(d.mean() - 2.262 * d.std(ddof=1) / np.sqrt(len(d))), float(d.mean() + 2.262 * d.std(ddof=1) / np.sqrt(len(d))))
        out["tests"][f"{a} vs {b} · {k}"] = {"moy_a": float(x.mean()), "sd_a": float(x.std(ddof=1)), "moy_b": float(y.mean()),
                                              "sd_b": float(y.std(ddof=1)), "diff": float(d.mean()), "ic95": ic, "p_wilcoxon": p}
        print(f"{a} vs {b} {k:22} {x.mean():8.2f} vs {y.mean():8.2f}  diff {d.mean():+.2f} IC95 [{ic[0]:+.2f},{ic[1]:+.2f}] p={p:.4f}")
(RACINE / "resultats" / "stats_graines.json").write_text(json.dumps(out, default=float))
