r"""
lancer_hybride_sma.py — Chaîne hybride (plan du matin) + SMA de Kyllian (journée), face aux aléas.

    python3 hybride_sma/lancer_hybride_sma.py                       # 5 tirages, 1,5 urgence / jour
    python3 hybride_sma/lancer_hybride_sma.py --tirages 20 --urgences 2
    python3 hybride_sma/lancer_hybride_sma.py --reserves 0.10 3 3   # tampon de TVO, lits, admissions
    python3 hybride_sma/lancer_hybride_sma.py --jours 40            # test rapide sur 40 jours

Compare, avec les MÊMES aléas :
  plans du matin   grille actuelle + premier créneau  |  chaîne hybride complète
  politiques       statique | réoptimisation | SMA
Aléas (générateur de Kyllian) : urgences (Poisson), annulations, lits fermés, durées réelles.
Le rapport complet (balayages, réserves, journée racontée) : hybride_sma/sma_experiences.py
Sortie : resultats/hybride_sma.json et un tableau à l'écran.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json

import numpy as np

import sma_pont as SP
import sma_aleas as S
from protocole import RACINE
from sma_matin import charger_ou_construire


def jours_de(m, n=None):
    inst = m["sol"].inst
    a, b = min(m["jours"]), max(m["jours"])
    j = [x for x in inst.jours_ouvres if a <= x <= b]
    return j[:n] if n else j


def mesures(r):
    R, n = r["resume"], max(1, r["resume"]["urgences_arrivees"])
    sal = [s for s in r["salles"] if not s.get("urgence")]
    return {"urgences arrivées": R["urgences_arrivees"],
            "% urgences opérées le jour même": 100 * R["urgences_operees_jour_meme"] / n,
            "% urgences opérées le jour ouvré suivant": 100 * R["urgences_operees_lendemain"] / n,
            "% urgences sans place": 100 * R["urgences_echec"] / n,
            "attente d'une urgence (min)": R["attente_urgence_moy"],
            "dépassement du bloc (min / jour)": R["depassement_min"] / R["jours"],
            "vacations > 30 min de dépassement": sum(s["depassement"] > 30 for s in sal),
            "temps opéré / TVO utilisé (%)": 100 * sum(s["opere"] for s in sal) / sum(s["tvo"] for s in sal),
            "programmés décalés > 30 min (%)": 100 * R["patients_decales"] / max(1, R["programmes"]),
            "variance des lits": float(np.var(r["lits"])), "pic de lits": max(r["lits"]),
            "nuits où les lits manquent": r["nuits_deficit"],
            "variance des admissions / jour": float(np.var(r["adm"])),
            "jours au-delà de 12 places": sum(b["pic_places"] > 12 for b in r["bilans"]),
            "ambulatoires gardés la nuit": R["conversions_nuit"],
            "réordonnancements": R["reordonnancements"], "messages / jour": R["messages"] / R["jours"]}


def lancer(tirages=5, urgences=1.5, reserves=None, n_jours=None, politiques=S.POLITIQUES):
    out = {}
    for chaine in ("reference", "hybride"):
        m = charger_ou_construire(chaine, reserves)
        opt = SP.optimiseur_tabou(300) if chaine == "hybride" else S.optimiseur_glouton
        jours = jours_de(m, n_jours)
        for pol in politiques:
            lignes = []
            for g in range(1, tirages + 1):
                gen = S.GenerateurAleas(urgences_par_jour=urgences, p_annulation=0.04, p_lits=0.15, graine=g)
                r, _ = SP.simuler(m, gen, pol, opt, chaine == "hybride", jours)
                lignes.append(mesures(r))
            out[f"{chaine} · {pol}"] = {k: float(np.mean([x[k] for x in lignes])) for k in lignes[0]}
            print(f"{chaine:9} {pol:15} {tirages} tirages, {len(jours)} jours", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybride + SMA face aux aléas")
    ap.add_argument("--tirages", type=int, default=5)
    ap.add_argument("--urgences", type=float, default=1.5, help="urgences par jour (moyenne)")
    ap.add_argument("--reserves", type=float, nargs=3, metavar=("TAMPON", "LITS", "PLACES"))
    ap.add_argument("--jours", type=int, default=None, help="limiter aux N premiers jours ouvrés")
    a = ap.parse_args()
    res = (a.reserves[0], int(a.reserves[1]), int(a.reserves[2])) if a.reserves else None
    out = lancer(a.tirages, a.urgences, res, a.jours)
    cols = list(out)
    print(f"\n{'indicateur':40}" + "".join(f"{c.replace('reference', 'actuelle')[:22]:>23}" for c in cols))
    for k in out[cols[0]]:
        print(f"{k:40}" + "".join(f"{out[c][k]:23.1f}" for c in cols))
    (RACINE / "resultats").mkdir(exist_ok=True)
    (RACINE / "resultats" / "hybride_sma.json").write_text(json.dumps(
        {"parametres": vars(a), "resultats": out}, indent=1, default=float))
    print("\nrésultats : resultats/hybride_sma.json")
