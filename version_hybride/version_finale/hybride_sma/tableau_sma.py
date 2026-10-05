"""Tableau de bord du rapport SMA (moyennes sur les graines) à partir de resultats/sma.json."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, numpy as np
from chemins import RACINE
d = json.load(open(RACINE / "resultats" / "sma.json"))
R = d["runs"]

def agg(runs):
    out = {}
    def m(f):
        v = [f(r) for r in runs]; return float(np.mean(v)), float(np.std(v))
    S = lambda k: (lambda r: r["resume"][k])
    for k in ["urgences_arrivees", "urgences_reprises", "urgences_operees_jour_meme", "urgences_operees_lendemain",
              "urgences_echec", "depassement_min", "salles_en_depassement", "conversions_nuit", "hebergements",
              "transferts", "reordonnancements", "patients_decales", "changements_salle", "annulations",
              "messages", "temps_optimiseur_s", "echecs_optimiseur", "surcapacite_places", "programmes"]:
        out[k] = m(S(k))
    out["attente"] = m(lambda r: r["resume"]["attente_urgence_moy"])
    out["pct_jour_meme"] = m(lambda r: 100 * r["resume"]["urgences_operees_jour_meme"] / max(1, r["resume"]["urgences_arrivees"]))
    out["pct_echec"] = m(lambda r: 100 * r["resume"]["urgences_echec"] / max(1, r["resume"]["urgences_arrivees"]))
    out["dep_jour"] = m(lambda r: r["resume"]["depassement_min"] / r["resume"]["jours"])
    out["lits_var"] = m(lambda r: np.var(r["lits"]))
    out["lits_moy"] = m(lambda r: np.mean(r["lits"]))
    out["lits_p95"] = m(lambda r: np.percentile(r["lits"], 95))
    out["lits_pic"] = m(lambda r: max(r["lits"]))
    out["n15"] = m(lambda r: sum(x >= 15 for x in r["lits"]))
    out["deficit"] = m(lambda r: r["nuits_deficit"])
    out["adm_var"] = m(lambda r: np.var(r["adm"]))
    out["adm_moy"] = m(lambda r: np.mean(r["adm"]))
    out["adm_pic"] = m(lambda r: max(r["adm"]))
    out["pic_places"] = m(lambda r: np.mean([b["pic_places"] for b in r["bilans"]]) if r.get("bilans") else np.nan)
    out["jours10"] = m(lambda r: sum(b["pic_places"] >= 10 for b in r["bilans"]) if r.get("bilans") else np.nan)
    out["opere_sem"] = m(lambda r: sum(s["opere"] for s in r["salles"]) / 60 / (245 / 7) if r.get("salles") else np.nan)
    out["occ"] = m(lambda r: 100 * sum(s["opere"] for s in r["salles"] if not s.get("urgence")) / sum(s["tvo"] for s in r["salles"] if not s.get("urgence")) if r.get("salles") else np.nan)
    out["n"] = len(runs)
    return out

def groupe(exp, chaine, pol, urg=None, durees="reelles", graine_none=False):
    rs = [r for r in R if r["exp"] == exp and r["chaine"] == chaine and r["politique"] == pol and r["durees"] == durees
          and (urg is None or r["urgences_par_jour"] == urg) and ((r["graine"] is None) == graine_none)]
    return agg(rs) if rs else None

if __name__ == "__main__":
    for ch in ("reference", "hybride"):
        for pol in ("statique", "reoptimisation", "sma"):
            g = groupe("A", ch, pol)
            print(f"\n== {ch} {pol} (n={g['n']})")
            print("  ".join(f"{k}={v[0]:.1f}±{v[1]:.1f}" for k, v in g.items() if k != "n"))
        for dur in ("estimees", "reelles"):
            g = groupe("A", ch, "sma", durees=dur, graine_none=True)
            print(f"\n== {ch} sans aléa ({dur})")
            print("  ".join(f"{k}={v[0]:.1f}" for k, v in g.items() if k != "n"))
