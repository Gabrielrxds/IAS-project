r"""
tester_hybride_sma.py — Tests du croisement hybride + SMA.

    python3 hybride_sma/tester_hybride_sma.py      # ou : pytest hybride_sma/tester_hybride_sma.py

  1. sans aléa ni retard, la fin P90 prévue par un agent salle = la charge du modèle
     (marges cumulées comme le protocole : √Σ marge²) ;
  2. un lit fermé baisse la capacité vue par la Solution, donc par les trois politiques ;
  3. invariants sur 30 jours, trois politiques : chaque programmé opéré une fois ou annulé,
     jamais un autre jour, jamais chez un autre chirurgien, pas de chevauchement en salle,
     Solution de travail à jour, aucun programmé dans un créneau URGENCES ni dans les réserves ;
  4. le plan du matin (Solution et instance d'origine) n'est jamais modifié ;
  5. même graine d'aléas => mêmes résultats ;
  6. résultat attendu : le SMA réduit le dépassement du bloc et les urgences sans place.
Durée : 2 à 4 minutes au premier lancement (construction des plans du matin), puis < 1 min.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from functools import lru_cache

import sma_pont as SP
import sma_aleas as S
from sma_matin import charger_ou_construire

REGLES_INTERDITES = {"G1", "G5", "G7", "U1", "U3", "U4", "U5"}


@lru_cache(None)
def matin():
    return charger_ou_construire("hybride")


def jours(n=30):
    m = matin()
    inst = m["sol"].inst
    a = min(m["jours"])
    return [j for j in inst.jours_ouvres if a <= j <= max(m["jours"])][:n]


def generateur(graine=3):
    return S.GenerateurAleas(urgences_par_jour=2.0, p_annulation=0.05, p_lits=0.3, graine=graine)


def instantane(sol):
    return (dict(sol.affectation), list(sol.lits_jour), list(sol.places_jour), set(sol.inst.patients),
            sol.inst.capacite_lits_nuit)


def test_projection_quadratique():
    m = matin()
    work = S.copie_de_travail(m["sol"])
    work.inst.capacite_lits_nuit = [work.inst.capacite_lits] * work.inst.nb_jours
    j = jours(5)[2]
    seq = S.plan_du_jour(work, j, m["plans"].get(j))
    J = S.Journee(work, j, seq, [], S.ParametresSMA(politique="sma"), S.optimiseur_glouton)
    verifiees = 0
    for vid, s in J.salles.items():
        if not s.file:
            continue
        fin90, _ = s.projeter(0, p90=True)
        attendu = s.v.debut + work.charge(vid)
        assert abs(fin90 - attendu) <= 1, (vid, fin90, attendu)
        verifiees += 1
    assert verifiees >= 2, verifiees


def test_lit_ferme_vu_par_la_solution():
    m = matin()
    work = S.copie_de_travail(m["sol"])
    inst = work.inst
    inst.capacite_lits_nuit = [inst.capacite_lits] * inst.nb_jours
    j = jours(5)[1]
    seq = S.plan_du_jour(work, j, m["plans"].get(j))
    alea = S.Alea(10 * 60, "lits", nb_lits=5, nuits=1)
    J = S.Journee(work, j, seq, [alea], S.ParametresSMA(politique="statique"), S.optimiseur_glouton)
    J.executer()
    assert inst.cap_lits(j) == inst.capacite_lits - 5
    assert m["sol"].inst.cap_lits(j) == m["sol"].inst.capacite_lits      # l'original ne bouge pas


def test_invariants_trois_politiques():
    m = matin()
    for pol in S.POLITIQUES:
        _, J = SP.simuler(m, generateur(), pol, SP.optimiseur_tabou(300), True, jours(), garder=set(jours()))
        for x in J.values():
            x.controle_coherence()
        work = next(iter(J.values())).sol
        work.controle_coherence()
        mauvais = [c for c in work.verifier() if c.regle in REGLES_INTERDITES]
        assert not mauvais, [str(c) for c in mauvais[:3]]


def test_plan_du_matin_intact():
    m = matin()
    avant = instantane(m["sol"])
    SP.simuler(m, generateur(), "sma", S.optimiseur_glouton, True, jours())
    assert instantane(m["sol"]) == avant


def test_reproductible():
    m = matin()
    a, _ = SP.simuler(m, generateur(), "sma", S.optimiseur_glouton, True, jours())
    b, _ = SP.simuler(m, generateur(), "sma", S.optimiseur_glouton, True, jours())
    for r in (a, b):
        r["resume"].pop("temps_optimiseur_s")
    assert a["resume"] == b["resume"] and a["lits"] == b["lits"]


def test_sma_meilleur_que_statique():
    m = matin()
    J = jours(80)
    res = {}
    for pol in ("statique", "sma"):
        rs = [SP.simuler(m, generateur(g), pol, SP.optimiseur_tabou(300), True, J)[0]["resume"] for g in (1, 2, 3)]
        res[pol] = (sum(r["depassement_min"] for r in rs), sum(r["urgences_echec"] for r in rs))
    assert res["sma"][0] < res["statique"][0], res
    assert res["sma"][1] <= res["statique"][1], res
    print(f"      80 jours, 3 tirages : dépassement {res['statique'][0]} -> {res['sma'][0]} min, "
          f"urgences sans place {res['statique'][1]} -> {res['sma'][1]}")


if __name__ == "__main__":
    t0 = time.time()
    tests = [(k, f) for k, f in dict(globals()).items() if k.startswith("test_")]
    echecs = 0
    for k, f in tests:
        t = time.time()
        try:
            f()
            print(f"OK    {k} ({time.time() - t:.0f} s)")
        except AssertionError as e:
            echecs += 1
            print(f"ÉCHEC {k} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests réussis en {time.time() - t0:.0f} s")
    _sys.exit(1 if echecs else 0)
