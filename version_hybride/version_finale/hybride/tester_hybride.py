r"""
tester_hybride.py — Tests de la chaîne hybride (sans aléa).

    python3 hybride/tester_hybride.py          # ou : pytest hybride/tester_hybride.py

  1. la grille hybride respecte les ressources actuelles (heures, salles 2 à 5) ;
  2. la simulation ne viole aucune règle du modèle (capacités, réserves, délai minimum...) ;
  3. chaque patient reçoit au plus une date, jamais avant consultation + 7 jours ;
  4. même graine => mêmes dates (reproductible) ;
  5. l'ordonnancement d'une journée garde les mêmes patients, chacun chez son chirurgien ;
  6. résultats attendus : lits et admissions plus réguliers qu'avec la grille actuelle,
     pas plus de patients sans date.
Durée : 1 à 2 minutes.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from functools import lru_cache

from protocole import (charger, instance, grille_actuelle_neutre, lire_grille, mesurer,
                       neutre_vers_vacations, Reglages)
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
import algos_journee as AJ


@lru_cache(None)
def donnees():
    return charger()[1]


@lru_cache(None)
def solution(nom, graine=0):
    don = donnees()
    if nom == "hybride":
        inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
        return simuler(inst, RegleHybride(don, delta=14, fantomes="uniformes"), graine=graine)
    inst = instance(don, grille_actuelle_neutre(don), Reglages())
    return simuler(inst, PremierCreneau(), graine=graine)


def heures_semaine(don, lignes):
    t0, t1 = don.test
    vacs = [v for v in neutre_vers_vacations(don, lignes) if t0 <= v.jour <= t1]
    return sum(v.fin - v.debut for v in vacs) / 60 / ((t1 - t0 + 1) / 7), {v.bloc_id for v in vacs}


def test_grille_ressources_egales():
    don = donnees()
    h_hyb, salles = heures_semaine(don, lire_grille("hybride")["vacations"])
    h_act, _ = heures_semaine(don, grille_actuelle_neutre(don))
    assert h_hyb <= 1.01 * h_act, (h_hyb, h_act)
    assert salles <= {2, 3, 4, 5}, salles


def test_aucun_conflit():
    conflits = solution("hybride").verifier()
    assert not conflits, [str(c) for c in conflits[:5]]


def test_une_date_et_delai_minimum():
    sol = solution("hybride")
    inst = sol.inst
    for pid, vid in sol.affectation.items():
        if vid is None:
            continue
        p = inst.patients[pid]
        assert inst.vacations[vid].jour >= p.jour_demande + 7, pid
        v = inst.vacations[vid]
        assert v.med_id == p.med_id and not v.urgence, pid


def test_reproductible():
    don = donnees()
    inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
    s2 = simuler(inst, RegleHybride(don, delta=14, fantomes="uniformes"), graine=0)
    assert s2.affectation == solution("hybride").affectation


def test_journee_memes_patients():
    sol = solution("hybride")
    inst, don = sol.inst, donnees()
    jours = [j for j in don.jours_ouvres(*don.test) if sequences_du_jour(sol, j)][:10]
    faites = 0
    for j in jours:
        seq = sequences_du_jour(sol, j)
        try:
            s = AJ.tabou(inst, j, seq, Poids(), 300, 0, graines=True).sequences
        except JourneeInfaisable:
            continue
        faites += 1
        assert sorted(p for x in s.values() for p in x) == sorted(p for x in seq.values() for p in x), j
        for vid, x in s.items():
            assert all(inst.patients[p].med_id == inst.vacations[vid].med_id for p in x), j
    assert faites >= 8, f"seulement {faites} journées ordonnées sur {len(jours)}"


def test_resultats_attendus():
    don = donnees()
    h = mesurer(don, solution("hybride"))
    r = mesurer(don, solution("reference"))
    assert h["lits_variance"] < 0.6 * r["lits_variance"], (h["lits_variance"], r["lits_variance"])
    assert h["places_ecart_type"] < r["places_ecart_type"], (h["places_ecart_type"], r["places_ecart_type"])
    assert h["sans_date"] <= r["sans_date"], (h["sans_date"], r["sans_date"])
    print(f"      lits : variance {h['lits_variance']:.1f} (actuelle {r['lits_variance']:.1f}) ; "
          f"admissions : écart-type {h['places_ecart_type']:.2f} ({r['places_ecart_type']:.2f}) ; "
          f"sans date {h['sans_date']} ({r['sans_date']})")


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
