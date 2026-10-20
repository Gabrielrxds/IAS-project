r"""
tester_absences.py — Tests de la reprogrammation après une absence de chirurgien.

    python3 hybride/tester_absences.py          # ou : pytest hybride/tester_absences.py

  1. les vacations du chirurgien sur la période sont fermées, et elles seules ;
  2. plus aucun patient n'est opéré dans une vacation fermée ;
  3. chaque patient déplacé est reprogrammé chez SON chirurgien, au plus tôt 2 jours après la
     déclaration et avant son délai maximum, ou marqué « à arbitrer » ;
  4. aucun conflit de règle du modèle (capacités, lits, réserves, délais) ;
  5. un patient non touché par une absence garde sa date ;
  6. le rattrapage n'ouvre que des demi-journées de salle libres, et réduit le décalage ;
  7. patient non venu : nouvelle date ≥ jour de non-venue + délai minimum, même chirurgien,
     aucun autre patient déplacé (sans priorité), aucun conflit.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from functools import lru_cache
import numpy as np
from protocole import charger, instance, lire_grille, Reglages
from regle_hybride import RegleHybride
from politiques import simuler
from absences import absences_reelles, absences_maladie, simuler_absences, PREAVIS_PATIENT


@lru_cache(None)
def donnees():
    return charger()[1]


@lru_cache(None)
def scenario(rattrapage=False):
    don = donnees()
    inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
    A = absences_reelles(don, 14) + absences_maladie(inst, 0.01, 0)
    sol, RP = simuler_absences(inst, RegleHybride(don, delta=14, fantomes="uniformes"), A, "hybride", 0,
                               rattrapage=rattrapage, feries=don.feries)
    return sol, RP, A


def test_vacations_fermees():
    sol, RP, A = scenario()
    inst = sol.inst
    for vid in RP.fermees:
        v = inst.vacations[vid]
        assert v.fin == v.debut, vid
        assert any(a.med_id == v.med_id and a.debut <= v.jour <= a.fin for a in A), vid


def test_aucun_patient_en_vacation_fermee():
    sol, RP, _ = scenario()
    assert not [p for p, v in sol.affectation.items() if v in RP.fermees]


def test_reprogrammations_valides():
    sol, RP, _ = scenario()
    inst = sol.inst
    for r in RP.journal:
        p = inst.patients[r.pid]
        if r.jour_nouveau is None:
            assert sol.affectation.get(r.pid) is None
            continue
        v = inst.vacations[sol.affectation[r.pid]] if sol.affectation.get(r.pid) is not None else None
        if v is None:          # re-déplacé puis non replacé par une absence ultérieure
            continue
        assert v.med_id == p.med_id
        assert r.jour_nouveau >= r.declaration + PREAVIS_PATIENT
        assert r.jour_nouveau <= inst.jour_max(r.pid)


def test_aucun_conflit():
    for ratt in (False, True):
        sol, _, _ = scenario(ratt)
        c = sol.verifier()
        assert not c, [str(x) for x in c[:3]]


def test_patients_non_touches_inchanges():
    don = donnees()
    sol_ref = simuler(instance(don, lire_grille("hybride")["vacations"], Reglages()),
                      RegleHybride(don, delta=14, fantomes="uniformes"), graine=0)
    inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
    sol, RP = simuler_absences(inst, RegleHybride(don, delta=14, fantomes="uniformes"), [], "hybride", 0)
    assert sol.affectation == sol_ref.affectation           # sans absence : identique à la chaîne hybride


def test_rattrapage():
    sol0, RP0, _ = scenario(False)
    sol1, RP1, _ = scenario(True)
    inst = sol1.inst
    for vid in RP1.rattrapages:
        v = inst.vacations[vid]
        autres = [inst.vacations[x] for x in inst.vacations_du_jour[v.jour] if x != vid and x not in RP1.fermees]
        assert not any(o.bloc_id == v.bloc_id and o.debut < v.fin and o.fin > v.debut for o in autres), vid
        assert not any(o.med_id == v.med_id and o.debut < v.fin and o.fin > v.debut and not o.urgence for o in autres), vid
    d0 = [r.jour_nouveau - r.jour_initial for r in RP0.journal if r.jour_nouveau is not None and r.declaration >= r.jour_initial - 7]
    d1 = [r.jour_nouveau - r.jour_initial for r in RP1.journal if r.jour_nouveau is not None and r.declaration >= r.jour_initial - 7]
    assert np.mean(d1) < np.mean(d0), (np.mean(d0), np.mean(d1))
    print(f"      absences à court préavis : décalage moyen {np.mean(d0):.1f} j sans rattrapage, {np.mean(d1):.1f} j avec")


def test_non_venus():
    don = donnees()
    inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
    sol, RP = simuler_absences(inst, RegleHybride(don, delta=14, fantomes="uniformes"), [], "hybride", 0,
                               taux_non_venue=0.05)
    assert RP.non_venues and not RP.journal                 # aucune reprogrammation « prioritaire »
    for n in RP.non_venues:
        p = inst.patients[n.pid]
        if n.jour_nouveau is None:
            continue
        assert n.jour_nouveau >= n.jour_initial + p.fenetre_jours, n
        if sol.affectation.get(n.pid) is not None:
            assert inst.vacations[sol.affectation[n.pid]].med_id == p.med_id
    c = sol.verifier()
    assert not c, [str(x) for x in c[:3]]
    print(f"      {len(RP.non_venues)} non-venues, nouvelle date médiane "
          f"{np.median([n.jour_nouveau - n.jour_initial for n in RP.non_venues if n.jour_nouveau is not None]):.0f} j après")


if __name__ == "__main__":
    t0 = time.time()
    tests = [(k, f) for k, f in dict(globals()).items() if k.startswith("test_")]
    echecs = 0
    for k, f in tests:
        t = time.time()
        try:
            f(); print(f"OK    {k} ({time.time() - t:.0f} s)")
        except AssertionError as e:
            echecs += 1; print(f"ÉCHEC {k} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests réussis en {time.time() - t0:.0f} s")
    _sys.exit(1 if echecs else 0)
