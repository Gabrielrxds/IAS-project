r"""
chaine_hybride.py — La chaîne HYBRIDE complète.

GRILLE (hors ligne, figée un an)
  1. Problème « ressources égales » (hybride_grille.ProblemeHybride) : codage de la
     chaîne génétique, salles 2-5 et budget d'heures actuels, volumes de la règle B du
     recuit bornés (75 %-150 % du temps actuel), lits sur toutes les nuits.
  2. Candidates : recuits (meilleur algorithme mesuré à budget égal) avec plusieurs
     graines et deux poids du terme lits, puis enfants par CROISEMENT DE JOURNÉES des
     meilleures (opérateur génétique) repris par un recuit froid.
  3. Sélection SIMHEURISTIQUE (chaîne tabou) : chaque candidate est rejouée sur la
     période d'APPRENTISSAGE (janvier-avril, patients d'apprentissage seulement), avec
     la règle de consultation hybride ; on garde la plus faible variance des lits
     parmi celles qui ne laissent pas plus de patients sans date que la meilleure + 2.
CONSULTATION : regle_hybride.RegleHybride (fenêtre de tolérance + patients fantômes).
JOURNÉE      : graines heuristiques de l'AG (Killian) + tabou (Gabriel), même coût.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import copy
import json
import random
import sys
import time

import numpy as np

from protocole import (charger, instance, mesurer, sauver_grille, vacations_vers_neutre,
                       Reglages, RACINE, mesurer_profils)
from analyse_couts import Config
from genetique_grille import AGGrille, ParametresAGGrille, PoidsGrille
from hybride_grille import probleme_hybride
from algos_grille import recuit
from politiques import simuler
from regle_hybride import RegleHybride

DELTA, FANTOMES, CHOIX = 14, "uniformes", "plus_proche"


def don_apprentissage(don):
    """Copie de l'étude réduite aux patients d'apprentissage ; mesure de février à avril."""
    d = copy.copy(don)
    a0, a1 = don.app
    d.patients = {p: x for p, x in don.patients.items() if don.jour_reel[p] <= a1}
    d.jour_reel = {p: j for p, j in don.jour_reel.items() if j <= a1}
    debut_mesure = next(j for j in range(a0, a1) if don.date(j).month == 2)
    d.test = (debut_mesure, a1)
    return d


def candidates(prob, budget=15000, graines=4, poids_lits=(1.0, 4.0), enfants=4, verbeux=True):
    out = []
    for wl in poids_lits:
        w = PoidsGrille(lits=wl)
        for g in range(graines):
            s = recuit(prob, w, budget, g)
            out.append(dict(origine=f"recuit w_lits={wl} graine={g}", X=s.X, cout=s.cout, w=wl))
            if verbeux:
                print(f"   {out[-1]['origine']}: coût {s.cout:.4f}", flush=True)
    # enfants : croisement de journées entre les meilleures (même poids), recuit froid
    for wl in poids_lits:
        w = PoidsGrille(lits=wl)
        a = AGGrille(prob, w, ParametresAGGrille(graine=7))
        rng = random.Random(int(wl * 10))
        pool = sorted([c for c in out if c["w"] == wl], key=lambda c: c["cout"])
        for e in range(enfants):
            p1, p2 = pool[0]["X"], pool[1 + e % (len(pool) - 1)]["X"]
            x, _ = a.croiser(p1, p2)
            if not a.reparer(x):
                continue
            s = recuit(prob, w, budget // 3, 100 + e, T0_facteur=0.3, depart=x)
            out.append(dict(origine=f"croisement+recuit w_lits={wl} #{e}", X=s.X, cout=s.cout, w=wl))
            if verbeux:
                print(f"   {out[-1]['origine']}: coût {s.cout:.4f}", flush=True)
    return out


def evaluer_apprentissage(don_app, prob, X, reg):
    vacs = prob.derouler(X, don_app.jour_zero, don_app.nb_jours, don_app.feries)
    lignes = vacations_vers_neutre(don_app, vacs)
    inst = instance(don_app, lignes, reg)
    sol = simuler(inst, RegleHybride(don_app, delta=DELTA, fantomes=FANTOMES, choix=CHOIX))
    t0, t1 = don_app.test
    m = mesurer_profils(don_app, sol.lits_jour, sol.places_jour)
    coh = [p for p, j in don_app.jour_reel.items() if t0 <= j <= t1]
    m["sans_date"] = sum(sol.affectation[p] is None for p in coh)
    d = [inst.vacations[sol.affectation[p]].jour - inst.patients[p].jour_demande
         for p in coh if sol.affectation[p] is not None]
    m["delai_median"] = float(np.median(d))
    return m, lignes


def main(part_min=0.75, part_max=1.5, budget=15000, graines=4, nom="hybride"):
    t = time.time()
    df, don = charger(); reg = Reglages()
    prob = probleme_hybride(don, reg, part_min=part_min, part_max=part_max)
    print(f"budget {prob.budget} demi-journées / cycle ; quotas {dict(zip([p.code for p in prob.profils], prob.quotas.tolist()))}")
    cands = candidates(prob, budget, graines)
    dapp = don_apprentissage(don)
    for c in cands:
        m, _ = evaluer_apprentissage(dapp, prob, c["X"], reg)
        c.update(app_variance=m["lits_variance"], app_sans=m["sans_date"], app_delai=m["delai_median"])
        print(f"   {c['origine']:40} apprentissage : var lits {m['lits_variance']:.2f}, "
              f"{m['sans_date']} sans date, délai méd {m['delai_median']:.0f}", flush=True)
    mini = min(c["app_sans"] for c in cands)
    retenue = min((c for c in cands if c["app_sans"] <= mini + 2), key=lambda c: c["app_variance"])
    print("retenue :", retenue["origine"])
    vacs = prob.derouler(retenue["X"], don.jour_zero, don.nb_jours, don.feries)
    sauver_grille(nom, vacations_vers_neutre(don, vacs), {
        "methode": "hybride : recuit + croisement génétique + sélection simheuristique",
        "secondes": time.time() - t, "retenue": retenue["origine"], "X": retenue["X"].tolist(),
        "codes": [p.code for p in prob.profils], "quotas": prob.quotas.tolist(),
        "quotas_actuels": prob.quotas_actuels, "salles": prob.salles,
        "candidates": [{k: v for k, v in c.items() if k != "X"} for c in cands],
        "part_min": part_min, "part_max": part_max})
    print(f"grille hybride enregistrée ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    a = [float(x) for x in sys.argv[1:3]] if len(sys.argv) > 2 else [0.75, 1.5]
    main(*a, nom=sys.argv[3] if len(sys.argv) > 3 else "hybride")
