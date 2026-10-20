r"""
experiences_surcharge.py — Charge normale contre surcharge de patients programmés.

On ajoute 10 %, 20 % ou 30 % de patients à l'année 2022 : chaque patient ajouté copie un patient
réel tiré au hasard (même chirurgien, même acte, même séjour) et se présente en consultation 0 à
3 jours plus tôt ou plus tard que lui. Les heures de bloc ne changent pas. On compare la
référence (grille actuelle + premier créneau) et la chaîne hybride (dates seulement, sans
ordonnancement des journées : les indicateurs de lits, places et délais n'en dépendent pas).

Sortie : resultats/surcharge.json et un tableau à l'écran.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import copy
import dataclasses
import json
import random
import statistics as st
from multiprocessing import Pool

from protocole import charger, instance, grille_actuelle_neutre, lire_grille, mesurer, Reglages, RACINE
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride

NIVEAUX = (0.0, 0.10, 0.20, 0.30)
GRAINES = (0, 1, 2)
K = ["lits_crit1", "lits_variance", "lits_max", "lits_p95", "places_ecart_type", "places_max", "sans_date",
     "delai_median", "delai_p90", "occupation_reelle_pct", "patients_par_semaine", "patients_test", "conflits"]


def surcharger(don, taux, graine):
    """Copie de `don` avec taux × (patients de 2022) patients en plus."""
    if taux == 0:
        return don
    rng = random.Random(1000 + graine)
    d = copy.copy(don)
    d.patients = dict(don.patients)
    d.jour_reel = dict(don.jour_reel)
    base = sorted(don.jour_reel)
    nid = max(don.patients) + 1
    for _ in range(round(taux * len(base))):
        p0 = don.patients[rng.choice(base)]
        dec = rng.randint(-3, 3)
        j_dem = min(max(0, p0.jour_demande + dec), don.nb_jours - 1)
        p = dataclasses.replace(p0, id=nid, jour_demande=j_dem)
        d.patients[nid] = p
        d.jour_reel[nid] = min(max(0, don.jour_reel[p0.id] + dec), don.nb_jours - 1)
        nid += 1
    return d


def un(args):
    taux, chaine, g = args
    _, don = charger()
    d = surcharger(don, taux, g)
    reg = Reglages()
    if chaine == "reference":
        inst = instance(d, grille_actuelle_neutre(d), reg)
        regle = PremierCreneau()
    else:
        inst = instance(d, lire_grille("hybride")["vacations"], reg)
        regle = RegleHybride(d, delta=14, fantomes="uniformes")
    sol = simuler(inst, regle, graine=g)
    m = mesurer(d, sol)
    return dict(taux=taux, chaine=chaine, graine=g, **{k: m[k] for k in K})


def main():
    taches = [(t, c, g) for t in NIVEAUX for c in ("reference", "hybride") for g in GRAINES]
    with Pool() as P:
        R = P.map(un, taches, chunksize=1)
    (RACINE / "resultats").mkdir(exist_ok=True)
    (RACINE / "resultats" / "surcharge.json").write_text(json.dumps(R, indent=1))
    print("| Surcharge | Chaîne | Var. lits | Pic | Lits 95 % | É.-t. adm. | Sans date | Délai méd. | Délai 90 % | Occupation | Patients/sem. |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in NIVEAUX:
        for c in ("reference", "hybride"):
            rs = [r for r in R if r["taux"] == t and r["chaine"] == c]
            m = {k: st.mean(r[k] for r in rs) for k in K}
            print(f"| +{round(100 * t)} % | {c} | {m['lits_variance']:.1f} | {m['lits_max']:.1f} | {m['lits_p95']:.1f} | "
                  f"{m['places_ecart_type']:.2f} | {m['sans_date']:.0f} | {m['delai_median']:.0f} | {m['delai_p90']:.0f} | "
                  f"{m['occupation_reelle_pct']:.0f} % | {m['patients_par_semaine']:.1f} |")


if __name__ == "__main__":
    main()
