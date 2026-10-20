"""
etudes_origine.py — Les trois études d'origine (Gabriel : tabou, Killian : génétique, Lucie : recuit),
chacune avec SES choix de modèle, rejouées dans le simulateur actuel (rapport V5, 6.2 et 7.2).

  G  grille d'origine seule        grille de l'étude, chaque patient à la première date libre
  D  règle de date d'origine seule grille actuelle ; tabou : au plus tôt + meilleure vacation du jour,
                                   fermeture à 85 % ; génétique : 12 dates notées, 2 proposées, la plus
                                   proche retenue ; recuit (version en ligne) : première date libre
  C  chaîne complète d'origine     grille + règle de date + ordre de la journée de l'étude
                                   (référence et chaîne hybride avec les mêmes mesures)
  F  fantômes sur chaque grille    grille de l'étude + règle hybride (14 j), sans / avec fantômes

Trois ordres de consultation (graines 0, 1, 2). Lits comptés comme dans tout le rapport (crit1 :
variance des nuits de semaine, urgences comprises). Sortie : resultats/etudes_origine.json
    python3 etudes_origine.py [GDCF]
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from protocole import charger, instance, grille_actuelle_neutre, lire_grille, mesurer, Reglages, RACINE
from politiques import PremierCreneau, RegleTabou, RegleGenetique, simuler
from regle_hybride import RegleHybride

GRAINES = (0, 1, 2)
GRILLE = {"actuelle": None, "tabou": "tabou", "génétique": "genetique_iso", "recuit": "recuit", "hybride": "hybride"}
K = ["lits_crit1", "lits_crit1_max", "lits_ratio_weekend", "places_ecart_type", "sans_date", "delai_median",
     "delai_p90", "heures_offertes_semaine", "occupation_reelle_pct", "vacations_inutilisees_pct",
     "patients_par_semaine"]


def _lignes(don, g):
    return grille_actuelle_neutre(don) if GRILLE[g] is None else lire_grille(GRILLE[g])["vacations"]


def _regle(don, nom):
    if nom == "premier":
        return PremierCreneau()
    if nom == "tabou":
        return RegleTabou()
    if nom == "génétique":
        from analyse_couts import calibrer_cout
        _, _, ref = calibrer_cout(don, Reglages())
        return RegleGenetique(ref)
    if nom == "hybride":
        return RegleHybride(don, delta=14, fantomes="previsions")
    if nom == "hybride_sans":
        return RegleHybride(don, delta=14, fantomes="aucun")
    raise ValueError(nom)


def _journee(inst, sol, algo):
    """Ordre de chaque journée de test par l'algorithme de l'étude ; mesures avec les durées réelles."""
    import algos_journee as AJ
    from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
    t0, t1 = inst.don.test if hasattr(inst, "don") else (None, None)
    out = []
    for j in JOURS:
        seq = sequences_du_jour(sol, j)
        if not seq:
            continue
        seq = {v: sorted(s, key=lambda p: (inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        if algo:
            try:
                seq = AJ.ALGOS[algo](inst, j, seq, Poids(), 1000, 0, graines=True).sequences
            except JourneeInfaisable:
                pass
        out.append(AJ.metriques(inst, j, seq))
    return dict(hors_uca_reel=float(np.sum([m["hors_uca_reel"] for m in out])),
                depassement_reel_min=float(np.mean([m["depassement_reel_min"] for m in out])),
                pic_places=float(np.mean([m["pic_places"] for m in out])))


def run(job):
    partie, nom, grille, regle, algo, g = job
    _, don = charger()
    global JOURS
    t0, t1 = don.test
    JOURS = don.jours_ouvres(t0, t1)
    inst = instance(don, _lignes(don, grille), Reglages())
    R = _regle(don, regle)
    t = time.time()
    sol = simuler(inst, R, graine=g)
    sec = time.time() - t
    m = mesurer(don, sol)
    r = dict(partie=partie, nom=nom, grille=grille, regle=regle, graine=g, secondes=sec,
             secondes_par_patient=sec / max(1, m.get("patients_test", 1)), **{k: m[k] for k in K})
    if partie == "C":
        r.update(_journee(inst, sol, algo))
        r["lits_serie"] = list(map(float, m["lits_serie"]))
    return r


def jobs(parties):
    J = []
    if "G" in parties:
        for g in ("actuelle", "tabou", "génétique", "recuit"):
            J += [("G", g, g, "premier", None, s) for s in GRAINES]
    if "D" in parties:
        for r, nm in (("premier", "première date libre (recuit)"), ("tabou", "tabou"), ("génétique", "génétique")):
            J += [("D", nm, "actuelle", r, None, s) for s in GRAINES]
    if "C" in parties:
        for nm, g, r, a in (("référence", "actuelle", "premier", None), ("tabou", "tabou", "tabou", "tabou"),
                            ("génétique", "génétique", "génétique", "génétique"), ("recuit", "recuit", "premier", "recuit"),
                            ("hybride", "hybride", "hybride", "tabou")):
            J += [("C", nm, g, r, a, s) for s in GRAINES]
    if "F" in parties:
        for g in ("actuelle", "tabou", "génétique", "recuit", "hybride"):
            for r in ("hybride_sans", "hybride"):
                J += [("F", f"{g} | {r}", g, r, None, s) for s in GRAINES]
    return J


if __name__ == "__main__":
    parties = sys.argv[1] if len(sys.argv) > 1 else "GDCF"
    f = RACINE / "resultats" / "etudes_origine.json"
    old = json.loads(f.read_text()) if f.exists() else {"runs": []}
    t = time.time()
    with Pool(2) as P:
        res = P.map(run, jobs(parties), chunksize=1)
    runs = [r for r in old["runs"] if r["partie"] not in parties] + res
    for p in parties:
        for nm in dict.fromkeys(r["nom"] for r in runs if r["partie"] == p):
            rs = [r for r in runs if r["partie"] == p and r["nom"] == nm]
            print(p, f"{nm:32s}", {k: round(float(np.mean([r[k] for r in rs])), 2) for k in
                                   ("lits_crit1", "lits_crit1_max", "places_ecart_type", "sans_date", "delai_median",
                                    "delai_p90", "heures_offertes_semaine", "occupation_reelle_pct")
                                   + (("hors_uca_reel", "depassement_reel_min") if p == "C" else ())}, flush=True)
    f.write_text(json.dumps({"runs": runs, "secondes": time.time() - t}, default=float))
    print(f"fin ({time.time() - t:.0f} s)")
