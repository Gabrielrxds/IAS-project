"""Étage JOURNÉE : AG / recuit / tabou sur les mêmes journées figées, même coût,
même budget d'évaluations distinctes, plusieurs budgets, avec et sans graines."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, sys, time, statistics as st
from protocole import charger, instance, grille_actuelle_neutre, Reglages, RACINE
from politiques import PremierCreneau, simuler
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
from algos_journee import ALGOS, metriques
df, don = charger(); reg = Reglages()
inst = instance(don, grille_actuelle_neutre(don), reg)
sol = simuler(inst, PremierCreneau())
jours = [j for j in don.jours_ouvres(*don.test) if sequences_du_jour(sol, j)][::5]
lignes = []
for graines in (False, True):
    for B in (100, 300, 1000, 3000):
        for j in jours:
            seq = sequences_du_jour(sol, j)
            seq = {v: sorted(s, key=lambda p: (inst.patients[p].jour_demande, p)) for v, s in seq.items()}
            n = sum(len(s) for s in seq.values())
            for a, f in ALGOS.items():
                for g in (0, 1):
                    try:
                        s = f(inst, j, seq, Poids(), B, g, graines=graines)
                    except JourneeInfaisable:
                        continue
                    m = metriques(inst, j, s.sequences)
                    lignes.append(dict(graines=graines, budget=B, jour=j, patients=n, algo=a, graine=g,
                                       cout=s.cout, depart=s.cout_depart, evals=s.evaluations,
                                       secondes=s.secondes, trace=(s.trace if B == 3000 and g == 0 else None), **m))
        sub = [l for l in lignes if l["graines"] == graines and l["budget"] == B]
        print(f"graines={graines} B={B}: " + " | ".join(
            f"{a} {st.mean(l['cout'] for l in sub if l['algo']==a):.1f} ({st.mean(l['secondes'] for l in sub if l['algo']==a):.2f}s)"
            for a in ALGOS) + f" | départ {st.mean(l['depart'] for l in sub):.1f}", flush=True)
(RACINE / "resultats").mkdir(exist_ok=True)
(RACINE / "resultats" / "journee_algos.json").write_text(json.dumps(lignes, default=str))
