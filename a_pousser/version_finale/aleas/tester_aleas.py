r"""
tester_aleas.py — Tests de la gestion des aléas.

    python3 aleas/tester_aleas.py          # ou : pytest aleas/tester_aleas.py

Un scénario « tout ensemble » (générateurs) et un scénario CIBLÉ, construit sur un premier
passage sans aléa pour être sûr de toucher des patients : médecin indisponible 3 h, salle en panne
une journée, matériel manquant 5 jours, 30 lits fermés 10 nuits, absence de 10 jours levée au
bout de 3, et des aléas patients (annulations, non aptes, non venus).

  1. aucun conflit de règle (capacités, lits, réserves, délais) ;
  2. aucun patient dans une vacation fermée ou vide ;
  3. un patient reprogrammé garde son chirurgien, a une date ≥ déclaration + 2 jours (≥ date
     d'aptitude pour un non apte) et ≤ son délai maximum, sinon il est « à arbitrer » ;
  4. une vacation raccourcie reste dans sa plage d'origine et ne touche plus la période
     d'indisponibilité ; une vacation décalée non plus, reste dans 7h30-18h30 et ne chevauche
     ni une autre vacation de sa salle ni une autre vacation du chirurgien ;
  5. salle : plus aucune vacation active dans la salle pendant la panne ; une vacation relogée
     garde jour et horaires et ne chevauche aucune vacation de sa nouvelle salle ;
  6. matériel : plus aucune intervention du type concerné pendant la période ;
  7. lits : lits programmés ≤ capacité réduite − réserve, sauf déficit signalé ;
  8. comblement : un patient avancé garde son chirurgien et gagne des jours ;
  9. annulation définitive : sans date à la fin ; report : date ≥ report + délai minimum ;
 10. levée : les vacations encore à venir retrouvent leur forme d'origine ;
 11. non venu : nouvelle date ≥ jour de non-venue + délai minimum.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from collections import Counter
from functools import lru_cache

from protocole import charger, instance, lire_grille, Reglages
from regle_hybride import RegleHybride
from politiques import simuler
import gestion_aleas as GA
import scenarios_aleas as SC
from gestion_aleas import JOUR, PREAVIS_PATIENT

TAUX = GA.TauxPatients(annulation=0.05, non_apte=0.02, non_venue=0.03)


@lru_cache(None)
def donnees():
    return charger()[1]


def _inst():
    return instance(donnees(), lire_grille("hybride")["vacations"], Reglages())


def _regle():
    return RegleHybride(donnees(), delta=14, fantomes="uniformes")


@lru_cache(None)
def cibles():
    """Premier passage sans aléa : on y choisit des jours et des patients sûrs d'être touchés."""
    don = donnees()
    inst = _inst()
    sol = simuler(inst, _regle())
    t0, t1 = don.test
    occ = Counter()
    for p, v in sol.affectation.items():
        if v is not None and t0 + 30 <= inst.vacations[v].jour <= t1 - 30:
            occ[v] += 1
    v1, v2 = [v for v, _ in occ.most_common(2)]
    a, b = inst.vacations[v1], inst.vacations[v2]
    types = Counter(inst.patients[p].type_interv for p, v in sol.affectation.items() if v is not None
                    and b.jour <= inst.vacations[v].jour <= b.jour + 4).most_common(1)[0][0]
    m = Counter(x.med_id for x in inst.vacations.values() if not x.urgence and t0 + 60 <= x.jour <= t0 + 80).most_common(1)[0][0]
    return dict(v_med=(a.med_id, a.jour, a.debut), v_salle=(b.bloc_id, b.jour), type=types, med_long=m,
                j_long=t0 + 60, j_lits=t0 + 40)


def aleas_cibles():
    c = cibles()
    m, j, h = c["v_med"]
    s, js = c["v_salle"]
    jl = c["j_long"]
    longue = GA.IndispoMedecin(jl - 20, "formation", duree_estimee="10 j",
                               debut=jl * JOUR, fin=(jl + 10) * JOUR, med_id=c["med_long"])
    return [
        GA.IndispoMedecin(j - 1, "urgence_autre_service", raison="appelé en réanimation", duree_estimee="3 h",
                          debut=j * JOUR + h, fin=j * JOUR + h + 180, med_id=m),
        GA.IndispoSalle(js, "panne_materiel", raison="table d'opération", duree_estimee="1 j",
                        debut=js * JOUR, fin=(js + 1) * JOUR, salle=s),
        GA.IndispoSalle(js, "personnel_manquant", duree_estimee="1 j", debut=js * JOUR, fin=(js + 1) * JOUR, salle=1),
        GA.IndispoMateriel(js - 10, "ancillaire_indisponible", raison=c["type"], duree_estimee="5 j",
                           debut=js * JOUR, fin=(js + 5) * JOUR, types=frozenset({c["type"]})),
        GA.FermetureLits(c["j_lits"] - 5, "epidemie", duree_estimee="10 j",
                         debut=c["j_lits"] * JOUR, fin=(c["j_lits"] + 10) * JOUR, nb=30),
        longue,
        GA.Levee(jl + 2, alea=longue),
    ]


@lru_cache(None)
def scenario(nom="cible"):
    don = donnees()
    inst = _inst()
    if nom == "cible":
        A = aleas_cibles()
    else:
        A = (SC.medecin_absences(don, inst, 14, graine=1) + SC.medecin_temporaires(inst, graine=1)
             + SC.salles(inst, graine=1) + SC.materiel(inst, graine=1) + SC.lits(inst, graine=1))
    sol, G = GA.simuler_aleas(inst, _regle(), A, graine=1, feries=don.feries, rattrapage=True, taux=TAUX)
    return sol, G


def _tous(f):
    for nom in ("cible", "generateurs"):
        f(*scenario(nom))


# --------------------------------------------------------------------------------------------

def test_aucun_conflit():
    def f(sol, G):
        assert sol.verifier() == []
    _tous(f)


def test_vacations_fermees_vides():
    def f(sol, G):
        inst = sol.inst
        for p, v in sol.affectation.items():
            if v is not None:
                assert v not in G.fermees and inst.vacations[v].tvo > 0, (p, v)
    _tous(f)


def test_reprogrammations():
    def f(sol, G):
        inst = sol.inst
        for a in G.actions:
            if a.action != "reprogramme":
                continue
            p = inst.patients[a.pid]
            v = inst.vacations[a.vid]
            assert v.med_id == p.med_id
            assert a.jour_nouveau >= a.declaration + PREAVIS_PATIENT
            assert a.jour_nouveau <= inst.jour_max(a.pid)
            al = G.aleas[a.alea]
            if isinstance(al, GA.PatientNonApte):
                assert a.jour_nouveau >= al.apte_a_partir
        n = Counter(a.action for a in G.actions)
        assert n["reprogramme"] > 0
    _tous(f)


def test_vacations_raccourcies():
    sol, G = scenario("cible")
    inst = sol.inst
    vues = 0
    for a in G.actions:
        if a.action != "raccourcie":
            continue
        orig = G.modifs[a.alea][a.vid]
        v = inst.vacations[a.vid]
        assert orig.debut <= v.debut < v.fin <= orig.fin
        assert G.aleas[a.alea].touche(v.jour, v.debut, v.fin) is None
        vues += 1
    for a in G.actions:
        if a.action != "decalee":
            continue
        v = inst.vacations[a.vid]
        assert G.aleas[a.alea].touche(v.jour, v.debut, v.fin) is None
        assert GA.OUVERTURE <= v.debut < v.fin <= GA.FERMETURE
        for x in inst.vacations_du_jour[v.jour]:
            w = inst.vacations[x]
            if x != a.vid and x not in G.fermees and w.tvo > 0 and (w.bloc_id == v.bloc_id or w.med_id == v.med_id):
                assert not (w.debut < v.fin and w.fin > v.debut), (v, w)
        vues += 1
    acts = Counter(a.action for a in G.actions if a.alea == 0)
    assert vues > 0 or acts["fermee"] > 0, acts


def test_salle_en_panne():
    sol, G = scenario("cible")
    inst = sol.inst
    pannes = [a for a in G.aleas.values() if isinstance(a, GA.IndispoSalle)]
    for a in pannes:
        for j in range(max(a.declaration, a.jour_debut), a.jour_fin + 1):
            for vid in inst.vacations_du_jour.get(j, ()):
                v = inst.vacations[vid]
                if v.bloc_id == a.salle and vid not in G.fermees and v.tvo > 0:
                    assert a.touche(j, v.debut, v.fin) is None, (a, v)
    relogees = [a for a in G.actions if a.action == "relogee"]
    assert relogees, Counter(a.action for a in G.actions if a.famille == "salle")
    for r in relogees:
        if r.vid in G.fermees:          # relogée puis fermée par une 2e panne (salle 1 ici)
            continue
        v, orig = inst.vacations[r.vid], G.modifs[r.alea][r.vid]
        assert (v.jour, v.debut, v.fin, v.med_id) == (orig.jour, orig.debut, orig.fin, orig.med_id)
        for x in inst.vacations_du_jour[v.jour]:
            w = inst.vacations[x]
            if x != r.vid and w.bloc_id == v.bloc_id and x not in G.fermees and w.tvo > 0:
                assert not (w.debut < v.fin and w.fin > v.debut), (v, w)


def test_materiel():
    def f(sol, G):
        inst = sol.inst
        for a in G.aleas.values():
            if not isinstance(a, GA.IndispoMateriel):
                continue
            for p, x in sol.affectation.items():
                if x is None or inst.patients[p].type_interv not in a.types:
                    continue
                v = inst.vacations[x]
                if v.jour >= a.declaration:
                    assert a.touche(v.jour, v.debut, v.fin) is None, (p, v)
    _tous(f)
    sol, G = scenario("cible")
    assert any(a.action == "liberee" and a.famille == "materiel" for a in G.actions)


def test_lits():
    sol, G = scenario("cible")
    inst = sol.inst
    a = next(x for x in G.aleas.values() if isinstance(x, GA.FermetureLits))
    deficits = {x.jour_initial for x in G.actions if x.action == "deficit_lits"}
    for k in range(a.jour_debut, a.jour_fin + 1):
        if k in deficits:
            continue
        prog = sol.lits_jour[k] - sol.lits_urg_jour[k]
        assert prog <= inst.cap_lits(k) - inst.reserve_lits, (k, prog, inst.cap_lits(k))
    assert any(x.famille == "lits" and x.action == "liberee" for x in G.actions)


def test_comblement():
    def f(sol, G):
        inst = sol.inst
        av = [a for a in G.actions if a.action == "avance"]
        assert av
        for a in av:
            assert a.jour_nouveau < a.jour_initial
            assert a.jour_nouveau >= a.declaration + PREAVIS_PATIENT
            assert inst.vacations[a.vid].med_id == inst.patients[a.pid].med_id
    _tous(f)


def test_annulations():
    def f(sol, G):
        inst = sol.inst
        assert G.annules
        for p in G.annules:
            assert sol.affectation[p] is None and p not in sol.hors_horizon
        for a in G.actions:
            if a.famille == "annulation" and a.action == "nouvelle_demande" and sol.affectation[a.pid] is not None:
                al = G.aleas[a.alea]
                j = inst.vacations[sol.affectation[a.pid]].jour
                assert j >= al.report_a_partir + inst.patients[a.pid].fenetre_jours
    _tous(f)


def test_levee():
    sol, G = scenario("cible")
    inst = sol.inst
    rouv = [a for a in G.actions if a.action == "rouverte"]
    assert rouv
    for r in rouv:
        assert inst.vacations[r.vid] == G.modifs[r.alea][r.vid]
        assert r.vid not in G.fermees


def test_non_venus():
    def f(sol, G):
        inst = sol.inst
        nv = [a for a in G.actions if a.famille == "non_venu"]
        assert nv
        for a in nv:
            x = sol.affectation[a.pid]
            if x is not None and inst.patients[a.pid].jour_demande == a.declaration:
                assert inst.vacations[x].jour >= a.declaration + inst.patients[a.pid].fenetre_jours
    _tous(f)


def test_durees():
    assert GA.lire_duree("90 min") == 90
    assert GA.lire_duree("1 h 30") == 90
    assert GA.lire_duree("3h") == 180
    assert GA.lire_duree("2 j") == 2 * JOUR
    assert GA.periode(10, duree="2 j") == (10 * JOUR, 12 * JOUR)
    assert GA.periode(10, 9 * 60, "90 min") == (10 * JOUR + 540, 10 * JOUR + 630)
    try:
        GA.IndispoSalle(0, "inconnu")
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    import time
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    ok = 0
    for nom, f in tests:
        t = time.time()
        try:
            f()
            ok += 1
            print(f"OK    {nom} ({time.time() - t:.0f} s)")
        except AssertionError as e:
            print(f"ÉCHEC {nom} : {e}")
    print(f"{ok}/{len(tests)} tests réussis")
    sys_exit = 0 if ok == len(tests) else 1
    raise SystemExit(sys_exit)
