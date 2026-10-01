r"""
tests.py — Contrôles de correction. `python3 tests.py`

Ce fichier n'est pas décoratif. Dans une métaheuristique, les deux bugs qui
coûtent le plus cher sont invisibles : une évaluation incrémentale qui dérive
de l'évaluation complète, et un compteur maintenu à la main qui se désynchronise
du vrai état. L'algorithme continue de tourner, il converge même, mais il
optimise une fonction qui n'est pas la vôtre. Ces deux contrôles (n°2 et n°3)
sont donc les plus importants du fichier.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import chemins  # rend coeur/ et vacations/ importables
import random

from modele import (Instance, Patient, Solution, Vacation, colorier_intervalles,
                    moments_temporels, pic, profil_cumulatif)
from tabou import Evaluateur, Poids, TabouLocal

alea = random.Random(1)


def test_coloriage():
    """Le coloriage doit utiliser exactement `pic` couleurs, sans collision."""
    for _ in range(300):
        iv = [(i, d, d + alea.randint(1, 50))
              for i, d in enumerate(alea.randint(0, 100)
                                    for _ in range(alea.randint(1, 20)))]
        couleurs = colorier_intervalles(iv)
        assert max(couleurs.values()) + 1 == pic(profil_cumulatif(
            (d, f) for _, d, f in iv))
        for a in iv:
            for b in iv:
                if a[0] < b[0] and couleurs[a[0]] == couleurs[b[0]]:
                    assert not (a[1] < b[2] and b[1] < a[2])
    print("OK  coloriage optimal et sans collision (300 tirages)")


def test_variance_temporelle():
    """La variance d'un profil en escalier doit être PONDÉRÉE PAR LA DURÉE.

    Contre-exemple qui a motivé ce test : un profil qui vaut 10 pendant une
    minute puis 1 pendant huit heures. La variance « par événement » le
    jugerait très irrégulier ; la variance temporelle, correctement, le juge
    presque plat.
    """
    # profil plat à 3 : variance nulle, moyenne 3
    moy, var = moments_temporels([(0, 3), (600, 0)], 0, 600)
    assert abs(moy - 3) < 1e-9 and var < 1e-9

    # moitié à 0, moitié à 4 : moyenne 2, variance 4
    moy, var = moments_temporels([(300, 4), (600, 0)], 0, 600)
    assert abs(moy - 2) < 1e-9 and abs(var - 4) < 1e-9

    # pic très bref : la moyenne et la variance doivent rester faibles
    moy, var = moments_temporels([(0, 1), (100, 10), (101, 1), (600, 0)], 0, 600)
    assert moy < 1.1, moy
    assert var < 0.2, var
    print("OK  variance temporelle pondérée par la durée (3 cas)")


def instance_jouet() -> Instance:
    inst = Instance(nb_jours=40, capacite_lits=5, capacite_places=3,
                    capacite_places_jour=4)
    for v in range(20):
        inst.ajouter_vacation(Vacation(id=v, bloc_id=1 + v % 3, med_id=1 + v % 2,
                                       jour=v, debut=480, fin=780))
    for i in range(40):
        inst.ajouter_patient(Patient(id=i, med_id=1 + i % 2,
                                     duree_op=alea.randint(20, 120),
                                     marge_perso=alea.randint(5, 30),
                                     duree_sejour=alea.randint(1, 5),
                                     fenetre_jours=0,   # pas de délai minimum
                                     ambulatoire=(i % 3 == 0)))
    return inst.indexer()


def test_delta_incremental(inst):
    """LE contrôle décisif : delta annoncé == variation réellement constatée."""
    ev = Evaluateur(inst, Poids())
    sol = Solution(inst)
    ecart_max = 0.0
    for _ in range(4000):
        pid = alea.choice(list(inst.patients))
        vid = alea.choice(inst.vacations_possibles(pid) + [None])
        m = ev.evaluer_mouvement(sol, pid, vid)
        if m is None:
            continue
        avant = ev.cout(sol)
        sol.affecter(pid, vid)
        ecart_max = max(ecart_max, abs((ev.cout(sol) - avant) - m.delta))
    assert ecart_max < 1e-9, ecart_max
    print(f"OK  delta incrémental == recalcul complet "
          f"(écart max {ecart_max:.1e}, 4000 mouvements)")
    return sol


def test_compteurs(inst, sol):
    """Les profils maintenus à la main doivent survivre à un recomptage."""
    lits = [0] * inst.nb_jours
    places = [0] * inst.nb_jours
    for pid, vid in sol.affectation.items():
        if vid is None:
            continue
        p, v = inst.patients[pid], inst.vacations[vid]
        if p.ambulatoire:
            places[v.jour] += 1
        else:
            for j in range(v.jour, min(v.jour + p.nb_nuits, inst.nb_jours)):
                lits[j] += 1
    assert lits == sol.lits_jour and places == sol.places_jour
    print("OK  profils lits/places cohérents avec un recomptage direct")


def test_chirurgien(inst, sol):
    for pid, vid in sol.affectation.items():
        if vid is not None:
            assert inst.vacations[vid].med_id == inst.patients[pid].med_id
    print("OK  contrainte chirurgien respectée (le voisinage l'impose)")


def test_pas_de_chevauchement(inst):
    sol = Solution(inst)
    for pid in list(inst.patients)[:20]:
        cands = inst.vacations_possibles(pid)
        if cands:
            sol.affecter(pid, cands[0])
    local = TabouLocal(inst, iterations=30)
    jours = {inst.vacations[v].jour for v in sol.affectation.values() if v is not None}
    for j in jours:
        pj, _ = local.resoudre(sol, j)
        par_salle: dict[int, list] = {}
        for c in pj.creneaux.values():
            par_salle.setdefault(c.bloc_id, []).append((c.debut, c.fin))
        for iv in par_salle.values():
            iv.sort()
            for a, b in zip(iv, iv[1:]):
                assert a[1] <= b[0], (j, a, b)
    print("OK  tabou local : aucun chevauchement dans une même salle")




def test_propositions(inst):
    """Le moteur de consultation ne doit jamais proposer une date qui casse
    une capacité, ni deux créneaux du même jour."""
    from modele import Solution
    from propositions import MoteurPropositions, Proposition, Report

    moteur = MoteurPropositions(inst)
    proposees = 0
    for pid in inst.patients:
        res = moteur.proposer(pid, k=2)   # fenêtre du patient
        if isinstance(res, Report):
            moteur.enregistrer_report(res)
            continue
        assert len({p.jour for p in res}) == len(res), "deux dates le même jour"
        assert all(isinstance(p, Proposition) for p in res)
        # la date retenue ne doit créer aucun conflit
        avant = len(moteur.solution.verifier())
        moteur.appliquer(res[0])
        apres = len(moteur.solution.verifier())
        assert apres <= avant, (pid, avant, apres)
        proposees += 1
    assert not moteur.solution.verifier()
    assert proposees > 0, "aucune date posée : le test ne teste rien"
    print(f"OK  moteur de consultation : {proposees} dates posées, aucun conflit")


def test_grille_tabou():
    """Tabou de la grille : contraintes dures respectées, coût retourné == recalcul,
    budget de changements tenu."""
    import random, numpy as np
    from grille_tabou import (blocs_depuis_grille, Demande, TabouGrille,
                              ParametresGrille, CYCLE)
    blocs = blocs_depuis_grille()
    prat = sorted({b.proprio for b in blocs if b.proprio and not b.reserve})
    rnd = random.Random(3)
    D = {m: rnd.uniform(0, 60) for m in prat}
    pi = {m: np.array([rnd.random() * .1 for _ in range(CYCLE)]) for m in prat}
    for budget in (None, 6):
        tg = TabouGrille(blocs, Demande(D, pi), ParametresGrille(iterations=40, budget=budget))
        r = tg.resoudre()
        F, _ = tg.evaluer(r.proprio)
        assert abs(F - r.cout) < 1e-9, (F, r.cout)
        for b in blocs:
            if b.reserve:
                assert r.proprio[b.id] == b.proprio                     # réservés fixes
        par = {}
        for b in blocs:
            m = r.proprio[b.id]
            if m is None or b.reserve:
                continue
            assert b.jour in tg.jours[m]                                 # (G2)
            for c in par.get(m, []):
                assert not b.chevauche(c), (m, b, c)                     # (G1)
            par.setdefault(m, []).append(b)
        assert all(par.get(m) for m in prat)                             # (G3)
        if budget is not None:
            assert r.changements <= budget                               # (G4)
        assert r.cout <= tg.evaluer(tg.origine)[0] + 1e-9
    print("OK  tabou de la grille : contraintes G1-G4, coût exact, budget tenu")


def test_planning_zero():
    """Planning de vacations construit de zéro : évaluation incrémentale == recalcul,
    contraintes (simultanéité, 8 demi-journées/semaine, 1 créneau minimum, budget)."""
    import random, numpy as np
    from planning_vacations import (decouper, heures_actuelles, Besoin, Parametres,
                                    Planificateur, CYCLE)
    cr = decouper(); ha = heures_actuelles(); prat = sorted(ha)
    rnd = random.Random(5)
    B = Besoin({m: rnd.uniform(0, 15) for m in prat}, {},
               {m: np.array([rnd.random() * .1 for _ in range(CYCLE)]) for m in prat})
    P = Parametres(departs=1, iterations=15, elites=1, budget_heures=400)
    pl = Planificateur(cr, B, P); M = pl.M
    e = pl.grasp()
    for typ, chg in list(pl.voisinage(e))[:400:7]:              # delta == recalcul
        if not M.admissible(e, chg):
            continue
        F, h = M.evaluer(e, chg, 3.0)
        x = list(e["x"])
        for cid, a, b in chg:
            x[cid] = b
        e2 = M.etat(x)
        assert abs(F - M.cout(e2, 3.0)) < 1e-9 and abs(h - e2["heures"]) < 1e-9
    # cas qui a révélé un bogue : échange de deux créneaux à la même heure, puis tentative
    # de redonner cette heure à l'un des deux praticiens
    meme = [(i, j) for i in range(len(cr)) for j in range(len(cr))
            if i < j and cr[i].instant == cr[j].instant and e["x"][i] and e["x"][j]
            and e["x"][i] != e["x"][j]]
    if meme:
        i, j = meme[0]; a, b = e["x"][i], e["x"][j]
        M.appliquer(e, [(i, a, b), (j, b, a)])
        assert e["occ"][a][cr[i].instant] == 1 and e["occ"][b][cr[i].instant] == 1
        autre = next((k for k in range(len(cr)) if cr[k].instant == cr[i].instant
                      and e["x"][k] is None), None)
        if autre is not None:
            assert not M.admissible(e, [(autre, None, a)])       # a opère déjà à cette heure
    for _ in range(3):
        F, x = pl.tabou(e)
        e = M.etat(x)
        for m in prat:
            inst = [cr[i].instant for i in range(len(cr)) if x[i] == m]
            assert len(inst) == len(set(inst)), m
    e2 = M.etat(x)
    assert abs(F - M.cout(e2)) < 1e-9 and e2["heures"] <= P.budget_heures + 1e-9
    for m in prat:
        assert e2["n"][m] >= 1
        inst = [cr[i].instant for i in range(len(cr)) if x[i] == m]
        assert len(inst) == len(set(inst))
        for w in range(4):
            assert sum(1 for i in range(len(cr)) if x[i] == m and cr[i].semaine == w) <= 8
    print("OK  planning de zéro : coût incrémental exact, contraintes tenues")


def test_pas_de_liste_attente(inst):
    """Invariant du modèle : après le flux, tout patient a une date ou est
    explicitement hors horizon. Personne n'attend dans un tiroir."""
    from propositions import simuler_consultations

    sol, journal = simuler_consultations(inst)
    for pid in inst.patients:
        a_une_date = sol.affectation[pid] is not None
        assert a_une_date ^ (pid in sol.hors_horizon), pid
    assert journal.poses + journal.reports == len(inst.patients)
    assert journal.poses > 0, "personne n'est placé : le test ne teste rien"
    # deux dates toujours DISTINCTES quand il y a le choix
    from propositions import MoteurPropositions, Report
    m = MoteurPropositions(inst)
    for pid in list(inst.patients)[:15]:
        res = m.proposer(pid, k=2)
        if isinstance(res, Report):
            continue
        assert len({p.jour for p in res}) == len(res)
        if len(res) == 2:
            assert res[0].delai <= res[1].delai, "la date 1 doit être la plus proche"
        m.appliquer(res[0])
    print(f"OK  aucune liste d'attente : {journal.poses} dates, "
          f"{journal.reports} hors horizon, 0 en suspens")


if __name__ == "__main__":
    test_coloriage()
    test_variance_temporelle()
    inst = instance_jouet()
    sol = test_delta_incremental(inst)
    test_compteurs(inst, sol)
    test_chirurgien(inst, sol)
    test_pas_de_chevauchement(inst)
    test_propositions(instance_jouet())
    test_pas_de_liste_attente(instance_jouet())
    test_grille_tabou()
    test_planning_zero()
    print("\nTous les contrôles passent.")
