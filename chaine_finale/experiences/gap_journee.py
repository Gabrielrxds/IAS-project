"""Gap d'optimalité de l'étage « journée » : énumération exhaustive des ordres de passage sur les
journées où l'affectation patient -> vacation est imposée (un chirurgien = une vacation) et où le
nombre d'ordres possibles reste énumérable. Même coût J que les métaheuristiques.
Sortie : resultats/gap_journee.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import itertools, json, math, pickle, time
import numpy as np
from protocole import RACINE
from genetique_jour import Poids, ParametresAG, sequences_du_jour, JourneeInfaisable, REPORT
import algos_journee as AJ
M = pickle.loads((RACINE / "cache" / "sma_matin_hybride.pkl").read_bytes())
sol, inst = M["sol"], M["sol"].inst
LIM = 400_000
lignes = []
for j in M["jours"]:
    seq = sequences_du_jour(sol, j)
    if not seq: continue
    juge = AJ.Juge(inst, j, seq, poids=Poids(), params=ParametresAG(graine=0))
    try:
        juge.verifier_faisabilite()
    except JourneeInfaisable:
        continue
    if any(len([v for v in juge.choix[i] if v != REPORT]) > 1 for i in range(juge.n)):
        continue                                        # affectation libre : hors champ de l'énumération
    taille = math.prod(math.factorial(len(s)) for s in seq.values())
    if taille > LIM: continue
    t = time.perf_counter()
    vids = list(seq)
    idx = {p: i for i, p in enumerate(juge.pids)}
    seqi = {v: [idx[p] for p in seq[v]] for v in vids}
    best = math.inf
    for combo in itertools.product(*[itertools.permutations(seqi[v]) for v in vids]):
        files = {v: list(c) for v, c in zip(vids, combo)}
        for v in juge.vids:
            files.setdefault(v, [])
        c = juge._evaluer_files(files).cout
        best = min(best, c)
    t_ex = time.perf_counter() - t
    r = {"jour": j, "patients": juge.n, "vacations": len(vids), "ordres": taille, "optimum": best, "s_exact": t_ex}
    for a in AJ.ALGOS:
        for b in (100, 300, 1000):
            s = AJ.ALGOS[a](inst, j, seq, Poids(), b, 0, graines=True)
            r[f"{a}_{b}"] = s.cout; r[f"{a}_{b}_s"] = s.secondes
            r["depart"] = s.cout_depart
    lignes.append(r)
    print(j, juge.n, taille, round(best, 2), {a: round(r[f"{a}_1000"] - best, 3) for a in AJ.ALGOS}, round(t_ex, 1), "s", flush=True)
(RACINE / "resultats" / "gap_journee.json").write_text(json.dumps(lignes, default=float))
print(len(lignes), "journées énumérées")
for a in AJ.ALGOS:
    for b in (100, 300, 1000):
        g = [100 * (r[f"{a}_{b}"] - r["optimum"]) / r["optimum"] for r in lignes]
        print(a, b, "gap moyen %", round(np.mean(g), 4), "max %", round(max(g), 4), "optimum atteint", sum(x < 1e-6 for x in g), "/", len(g))
g = [100 * (r["depart"] - r["optimum"]) / r["optimum"] for r in lignes]
print("départ heuristique gap moyen %", round(np.mean(g), 3), "max", round(max(g), 3), "optimum", sum(x < 1e-6 for x in g))
