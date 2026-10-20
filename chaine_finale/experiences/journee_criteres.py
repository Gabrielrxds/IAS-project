"""Ordre de la journée : ce que chaque méthode améliore, critère par critère (rapport V3, 6.2).
Mêmes journées que comparer_journee.py (grille actuelle, un jour ouvré de test sur cinq).
Lignes : ordre d'inscription, meilleur ordre simple (départs de l'AG), tabou, recuit, génétique
(1 000 évaluations, départ sans ordre simple). Mesures avec les durées estimées et réelles.
Sortie : resultats/journee_criteres.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import statistics as st

from protocole import charger, instance, grille_actuelle_neutre, Reglages, RACINE
from politiques import PremierCreneau, simuler
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable, ParametresAG
import algos_journee as AJ

df, don = charger()
inst = instance(don, grille_actuelle_neutre(don), Reglages())
sol = simuler(inst, PremierCreneau())
jours = [j for j in don.jours_ouvres(*don.test) if sequences_du_jour(sol, j)][::5]
L = []
for j in jours:
    seq = sequences_du_jour(sol, j)
    seq = {v: sorted(s, key=lambda p: (inst.patients[p].jour_demande, p)) for v, s in seq.items()}
    try:
        juge = AJ.Juge(inst, j, seq, poids=Poids(), params=ParametresAG(graine=0), graines=True)
        juge.verifier_faisabilite()
    except JourneeInfaisable:
        continue
    ins = juge._ind_depuis_ordre(list(range(juge.n)))
    juge.evaluer(ins)
    simple = min(juge.population_initiale(), key=lambda i: i.cout)
    lignes = [("inscription", ins.cout, juge.sequences(ins), 0.0), ("ordre simple", simple.cout, juge.sequences(simple), 0.0)]
    for a in ("tabou", "recuit", "génétique"):
        s = AJ.ALGOS[a](inst, j, seq, Poids(), 1000, 0, graines=False)
        lignes.append((a, s.cout, s.sequences, s.secondes))
    for nom, c, sq, sec in lignes:
        L.append(dict(jour=j, methode=nom, cout=c, secondes=sec, **AJ.metriques(inst, j, sq)))
out = {}
for nom in ("inscription", "ordre simple", "tabou", "recuit", "génétique"):
    rs = [r for r in L if r["methode"] == nom]
    out[nom] = {k: st.mean(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], (int, float)) and k != "jour"}
    print(nom, {k: round(v, 2) for k, v in out[nom].items()})
(RACINE / "resultats" / "journee_criteres.json").write_text(json.dumps(dict(moyennes=out, lignes=L), default=float))
