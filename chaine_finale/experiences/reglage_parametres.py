"""Réglage des paramètres des trois métaheuristiques sur le protocole commun (rapport V3, 6.2).

A. Grille : problème commun (coût F, budget 15 000 évaluations, celui de la chaîne hybride),
   départ aléatoire, 3 graines par réglage.
     tabou     : voisins tirés par itération k x durée d'interdiction (tenure)
     recuit    : température initiale (facteur du 1er décile des dégradations) x température finale
     génétique : taille de population x probabilité de mutation ; et sans descente finale
B. Journée : les 144 journées dont l'ordre optimal est connu (gap_journee.json), budget de 300
   évaluations, départ sans ordre simple ; part des journées à l'optimum et écart moyen.
Sortie : resultats/reglage_parametres.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import pickle
import time
from multiprocessing import Pool

import numpy as np

from protocole import RACINE
from etude_statique import probleme
from genetique_grille import PoidsGrille
from algos_grille import ALGOS as AG_GRILLE

W = PoidsGrille(lits=4.0)
BUDGET_GRILLE = 15000
GRAINES = (0, 1, 2)

GRILLE = (
    [("tabou", dict(k=k, tenure=t)) for k in (10, 30, 60) for t in (5, 15, 40)]
    + [("recuit", dict(T0_facteur=a, Tmin_facteur=b)) for a in (0.3, 1.0, 3.0) for b in (0.0005, 0.005, 0.05)]
    + [("génétique", dict(population=p, p_mutation=m)) for p in (20, 40, 80) for m in (0.3, 0.7, 1.0)]
    + [("génétique", dict(population=40, p_mutation=0.7, descente=0))]
)


def un_grille(args):
    algo, kw, g = args
    _, _, prob = probleme()
    s = AG_GRILLE[algo](prob, W, BUDGET_GRILLE, g, **kw)
    return dict(algo=algo, kw=kw, graine=g, cout=float(s.cout), secondes=s.secondes)


JOURNEE = (
    [("tabou", dict(k=k, tenure=t)) for k in (5, 20, 50) for t in (2, 7, 20)]
    + [("recuit", dict(T0_facteur=a, Tmin_facteur=b)) for a in (0.1, 0.5, 2.0) for b in (0.001, 0.01, 0.1)]
    + [("génétique", dict(population=p)) for p in (10, 20, 40, 80)]
)


def un_journee(args):
    algo, kw = args
    import algos_journee as AJ
    from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
    M = pickle.loads((RACINE / "cache" / "sma_matin_hybride.pkl").read_bytes())
    sol, inst = M["sol"], M["sol"].inst
    G = json.loads((RACINE / "resultats" / "gap_journee.json").read_text())
    ok, gaps = 0, []
    for l in G:
        seq = sequences_du_jour(sol, l["jour"])
        try:
            s = AJ.ALGOS[algo](inst, l["jour"], seq, Poids(), 300, 0, graines=False, **kw)
        except JourneeInfaisable:
            continue
        gap = 100 * (s.cout - l["optimum"]) / max(1e-9, l["optimum"])
        gaps.append(gap)
        ok += gap < 1e-6
    return dict(algo=algo, kw=kw, optimum=ok, journees=len(gaps), ecart_moyen=float(np.mean(gaps)))


if __name__ == "__main__":
    # la partie grille a été faite à part (resultats/reglage_parametres.json, un paramètre à la fois) ;
    # ici, la journée seulement
    t = time.time()
    with Pool(2) as P:
        B = P.map(un_journee, JOURNEE, chunksize=1)
    for r in B:
        print("journée", r["algo"], r["kw"], r["optimum"], "/", r["journees"], round(r["ecart_moyen"], 4), flush=True)
    (RACINE / "resultats" / "reglage_journee.json").write_text(
        json.dumps(dict(journee=B, secondes=time.time() - t), default=float))
    print(f"fin ({time.time() - t:.0f} s)")
