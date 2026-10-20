r"""
etude_dynamique.py — Approche DYNAMIQUE : insérer de nouveaux patients dans un planning existant
avec chacune des trois métaheuristiques (tabou, recuit simulé, algorithme génétique).

Le planning existant est celui de la grille actuelle : les patients déjà datés ne bougent jamais.
Chaque jour, les patients vus en consultation reçoivent une date. Le coût d'une date est la hausse
de l'écart des lits à la cible \mathcal E_f sur les 13 semaines suivantes (lits des patients déjà
datés, sans patients fantômes : c'est la brique que la chaîne hybride ajoutera). Les dates
candidates d'un patient sont les vacations de son chirurgien réalisables dans les 14 jours qui
suivent la première date possible (comme la règle hybride).

D1. Un patient à la fois. Énumération exacte (l'optimum) contre les trois métaheuristiques à petit
    budget, départ aléatoire : part des insertions qui trouvent l'optimum, temps.
D2. Plusieurs patients à la fois : tous les patients vus le même jour reçoivent leur date ensemble
    (ils se disputent les mêmes vacations et les mêmes nuits). Glouton (un par un, exact pour
    chacun) contre tabou, recuit et génétique partant du glouton, même budget d'évaluations par
    jour. Année 2022 rejouée ; indicateurs du protocole commun sur mai-décembre.
Sortie : resultats/etude_dynamique.json
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import math
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import calendrier
from protocole import charger, instance, mesurer, grille_actuelle_neutre, lire_grille, Reglages, RACINE
from modele import Solution

DELTA, H = 14, 93
LOT_MAX = 25
PENALITE = 1e3
K = ["lits_crit1", "lits_semaine_variance", "lits_semaine_max", "lits_ecart_cible", "lits_ratio_weekend",
     "lits_ratio_vacances", "lits_variance", "places_ecart_type", "sans_date", "delai_median",
     "delai_p90", "occupation_reelle_pct"]


# ---------------------------------------------------------------------------
# Le problème d'insertion d'un jour : n patients, une liste de candidates chacun
# ---------------------------------------------------------------------------

class Insertion:
    def __init__(self, sol, prof, pids, t, figes, attendu=None):
        self.sol, self.inst, self.pids = sol, sol.inst, list(pids)
        a, b = t + 1, min(len(prof), t + H)
        self.a, self.b = a, b
        if attendu is not None:        # courbe attendue (patients datés + fantômes), cf. regle_hybride
            E = np.array([attendu[k] if k < len(attendu) else 0 for k in range(a, b)], float)
        else:
            E = np.array([sol.lits_jour[k] if k < len(sol.lits_jour) else 0 for k in range(a, b)], float)
        f = prof[a:b]
        self.E, self.f = E, f
        self.S2, self.SEf, self.Sff, self.n = float(E @ E), float(E @ f), float(f @ f), b - a
        self.cands = [self._candidates(p, figes) for p in self.pids]
        self.cache = {}
        self.evaluations = 0

    def _candidates(self, pid, figes):
        sol, inst = self.sol, self.inst
        out, premier = [], None
        for vid in inst.vacations_possibles(pid):
            j = inst.vacations[vid].jour
            if j in figes:
                continue
            if premier is not None and j > premier + DELTA:
                break
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if not ok:
                continue
            if premier is None:
                premier = j
            nuits = [k - self.a for k in inst.nuits_occupees(pid, j) if self.a <= k < self.b]
            out.append((vid, j, nuits))
        return out

    def _ecart(self, d):
        dS2 = sum(2 * self.E[k] * x + x * x for k, x in d.items())
        dSEf = sum(self.f[k] * x for k, x in d.items())
        e0 = (self.S2 - self.SEf ** 2 / self.Sff) / self.n
        e1 = (self.S2 + dS2 - (self.SEf + dSEf) ** 2 / self.Sff) / self.n
        return e1 - e0

    def cout(self, a) -> float:
        """a[i] = indice de la candidate du patient i (-1 : sans date)."""
        key = tuple(a)
        if key in self.cache:
            return self.cache[key]
        self.evaluations += 1
        d = defaultdict(float)
        for i, c in enumerate(a):
            if c >= 0:
                for k in self.cands[i][c][2]:
                    d[k] += 1.0
        c = self._ecart(d) + PENALITE * self._conflits(a)
        self.cache[key] = c
        return c

    def _conflits(self, a) -> int:
        """Réalisabilité CONJOINTE : plusieurs nouveaux patients dans une même vacation, ou sur des
        nuits déjà presque pleines. Seuls ces patients-là sont posés dans la Solution et vérifiés
        par ses propres règles (aucune règle oubliée)."""
        inst, sol = self.inst, self.sol
        par_vac = defaultdict(list)
        compte = defaultdict(list)
        for i, c in enumerate(a):
            if c >= 0:
                par_vac[self.cands[i][c][0]].append(i)
                for k in self.cands[i][c][2]:
                    compte[k].append(i)
        suspects = set()
        for v, l in par_vac.items():
            if len(l) > 1:
                suspects.update(l)
        for k, l in compte.items():
            j = k + self.a
            if len(l) > 1 and sol.lits_jour[j] + len(l) > inst.cap_lits(j) - inst.reserve_lits:
                suspects.update(l)
        if not suspects:
            return 0
        pos = [(self.pids[i], self.cands[i][a[i]][0]) for i in suspects]
        for p, v in pos:
            sol.affecter(p, v)
        n = sum(sol.viole(p, v) for p, v in pos)
        for p, _ in pos:
            sol.affecter(p, None)
        return n

    def appliquer(self, a):
        for i, c in enumerate(a):
            if c >= 0:
                self.sol.affecter(self.pids[i], self.cands[i][c][0])


# ---------------------------------------------------------------------------
# Les méthodes : glouton exact patient par patient, tabou, recuit, génétique
# ---------------------------------------------------------------------------

def glouton(P: Insertion, rng, ordre=None):
    """Un patient après l'autre, chacun à sa meilleure candidate compte tenu des précédents
    (posés pour de vrai dans la Solution, puis retirés : on rend le vecteur d'affectation)."""
    sol = P.sol
    a = [-1] * len(P.pids)
    E, S2, SEf = P.E.copy(), P.S2, P.SEf
    poses = []
    for i in (ordre or range(len(P.pids))):
        pid = P.pids[i]
        best, bc = None, math.inf
        for c, (vid, j, nuits) in enumerate(P.cands[i]):
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if not ok:
                continue
            P.evaluations += 1
            dS2 = sum(2 * P.E[k] + 1 for k in nuits)
            dSEf = sum(P.f[k] for k in nuits)
            v = ((P.S2 + dS2 - (P.SEf + dSEf) ** 2 / P.Sff) - (P.S2 - P.SEf ** 2 / P.Sff)) / P.n
            if v < bc - 1e-12:
                best, bc = c, v
        if best is not None:
            a[i] = best
            vid, j, nuits = P.cands[i][best]
            sol.affecter(pid, vid)
            poses.append(pid)
            for k in nuits:
                P.S2 += 2 * P.E[k] + 1
                P.SEf += P.f[k]
                P.E[k] += 1
    for pid in poses:
        sol.affecter(pid, None)
    P.E, P.S2, P.SEf = E, S2, SEf
    return a


def _voisin(P, a, rng):
    choix = [i for i in range(len(a)) if len(P.cands[i]) > 1]
    if not choix:
        return None
    i = rng.choice(choix)
    c = rng.randrange(len(P.cands[i]) - 1)
    c = c + 1 if c >= a[i] else c
    b = list(a)
    b[i] = c
    return b, i


def tabou(P, a, budget, rng, k=8):
    cur, cc = list(a), P.cout(a)
    best, bc = list(cur), cc
    interdit, it = {}, 0
    while P.evaluations < budget and it < 20 * budget:
        it += 1
        cand = None
        for _ in range(k):
            v = _voisin(P, cur, rng)
            if v is None:
                return best
            b, i = v
            c = P.cout(b)
            if interdit.get((i, b[i]), 0) > it and c >= bc - 1e-12:
                continue
            if cand is None or c < cand[0]:
                cand = (c, b, i)
            if P.evaluations >= budget:
                break
        if cand is None:
            continue
        c, b, i = cand
        interdit[(i, cur[i])] = it + max(2, len(cur) // 3)
        cur, cc = b, c
        if c < bc - 1e-12:
            best, bc = list(b), c
    return best


def recuit(P, a, budget, rng):
    cur, cc = list(a), P.cout(a)
    best, bc = list(cur), cc
    ds = []
    for _ in range(10):
        v = _voisin(P, cur, rng)
        if v is None:
            return best
        d = P.cout(v[0]) - cc
        if d > 1e-12:
            ds.append(d)
    ds.sort()
    T = (ds[len(ds) // 10] if ds else 1e-3)
    Tmin = 0.005 * T
    reste = max(1, budget - P.evaluations)
    alpha = (Tmin / T) ** (1 / reste)
    garde = 0
    while P.evaluations < budget and garde < 20 * budget:
        garde += 1
        v = _voisin(P, cur, rng)
        if v is None:
            break
        c = P.cout(v[0])
        d = c - cc
        if d <= 0 or (d / T < 700 and rng.random() < math.exp(-d / T)):
            cur, cc = v[0], c
            if c < bc - 1e-12:
                best, bc = list(cur), c
        T *= alpha
    return best


def genetique(P, a, budget, rng, pop=10):
    n = len(a)
    if n == 0:
        return a

    def muter(x, taux):
        y = list(x)
        for i in range(n):
            if len(P.cands[i]) > 1 and rng.random() < taux:
                y[i] = rng.randrange(len(P.cands[i]))
        return y

    popu = [list(a)] + [muter(a, 0.3) for _ in range(pop - 1)]
    fit = [P.cout(x) for x in popu]
    garde = 0
    while P.evaluations < budget and garde < 20 * budget:
        garde += 1
        enfants = []
        for _ in range(pop - 1):
            p1 = min(rng.sample(range(pop), 2), key=lambda i: fit[i])
            p2 = min(rng.sample(range(pop), 2), key=lambda i: fit[i])
            e = [popu[p1][i] if rng.random() < 0.5 else popu[p2][i] for i in range(n)]
            enfants.append(muter(e, 1.0 / n))
        f_e = [P.cout(x) for x in enfants]
        elite = min(range(pop), key=lambda i: fit[i])
        popu, fit = [popu[elite]] + enfants, [fit[elite]] + f_e
    return popu[min(range(pop), key=lambda i: fit[i])]


METHODES = {"tabou": tabou, "recuit": recuit, "génétique": genetique}


# ---------------------------------------------------------------------------
# Rejeu de l'année : chaque jour, les patients vus en consultation reçoivent leur date
# ---------------------------------------------------------------------------

def rejouer(args):
    methode, budget, graine, grille = args
    _, don = charger()
    lignes = grille_actuelle_neutre(don) if grille == "actuelle" else lire_grille(grille)["vacations"]
    sol, st = dater_par_lots(don, lignes, methode, budget, graine)
    m = mesurer(don, sol)
    return dict(methode=methode, budget=budget, graine=graine, grille=grille, **st, **{k: m[k] for k in K})


def dater_par_lots(don, lignes, methode, budget, graine, reg=None, apres_figer=None, traces=None,
                   fantomes=False):
    """Rejoue l'année : chaque jour, le lot des patients vus en consultation reçoit ses dates avec
    la méthode donnée (« glouton », « tabou », « recuit », « génétique »). Renvoie (sol, stats).
    apres_figer(sol, j) est appelé quand la journée j est figée (sept jours avant).
    traces : liste qui reçoit, pour les lots de la période de test d'au moins 10 patients, la
    trace (évaluations, meilleur coût) de la recherche."""
    inst = instance(don, lignes, reg or Reglages())
    prof = calendrier.profil(inst.jour_zero, inst.nb_jours + 64)
    sol = Solution(inst)
    rng = random.Random(graine)
    par_jour = defaultdict(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    figes, gains, tailles, evals, temps = set(), [], [], [], []
    R = None
    if fantomes:                       # la même prévision que la chaîne hybride (fantômes + urgences)
        from regle_hybride import RegleHybride
        R = RegleHybride(don, delta=DELTA)
    for t in range(inst.nb_jours):
        pids = par_jour.get(t, [])
        rng.shuffle(pids)
        sol.jour_courant = t
        for q in range(0, len(pids), LOT_MAX):
            lot = pids[q:q + LOT_MAX]
            att = None
            if R is not None:
                R.preparer(sol, t)
                att = R.E
            P = Insertion(sol, prof, lot, t, figes, attendu=att)
            t0 = time.perf_counter()
            a0 = glouton(P, rng)
            c0 = P.cout(a0)
            a, c1 = a0, c0
            if methode != "glouton" and len(lot) > 1:
                P.evaluations = 0
                tr = None
                if traces is not None and t >= don.test[0] and len(lot) >= 10:
                    tr, cout0 = [(0, c0)], P.cout

                    def suivi(x, _c=cout0, _tr=tr):
                        v = _c(x)
                        if v < _tr[-1][1] - 1e-12:
                            _tr.append((P.evaluations, v))
                        return v
                    P.cout = suivi
                a = METHODES[methode](P, a0, budget, rng)
                if tr is not None:
                    P.cout = cout0
                    traces.append(dict(jour=t, patients=len(lot), trace=tr))
                c1 = P.cout(a)
                if c1 > c0:                         # on garde la meilleure des deux
                    a, c1 = a0, c0
            temps.append(time.perf_counter() - t0)
            P.appliquer(a)
            # réparation : un patient dont toutes les candidates ont été prises par le reste du lot
            # reçoit sa date seul, avec des candidates recalculées (comme à la consultation)
            for i, c in enumerate(a):
                if c < 0:
                    P1 = Insertion(sol, prof, [P.pids[i]], t, figes, attendu=att)
                    a1 = glouton(P1, rng)
                    if a1[0] >= 0:
                        P1.appliquer(a1)
                    else:
                        sol.hors_horizon.add(P.pids[i])
            if t >= don.test[0] - 30:
                gains.append(c0 - c1)
                tailles.append(len(lot))
                evals.append(P.evaluations)
        figes.add(t + 7)
        if apres_figer is not None:
            apres_figer(sol, t + 7)
    sol.jour_courant = None
    sol.definir_fenetre(0, inst.nb_jours - 1)
    conflits = len(sol.verifier())
    return sol, dict(conflits=conflits,
                     gain_moyen=float(np.mean(gains)), part_jours_ameliores=float(np.mean(np.array(gains) > 1e-9)),
                     patients_par_jour=float(np.mean(tailles)), temps_moyen_s=float(np.mean(temps)),
                     temps_max_s=float(np.max(temps)))


# ---------------------------------------------------------------------------
# D1 : un patient à la fois — l'optimum exact contre les métaheuristiques à petit budget
# ---------------------------------------------------------------------------

def un_patient(args):
    budget, graine = args
    _, don = charger()
    inst = instance(don, grille_actuelle_neutre(don), Reglages())
    prof = calendrier.profil(inst.jour_zero, inst.nb_jours + 64)
    sol = Solution(inst)
    rng = random.Random(graine)
    par_jour = defaultdict(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    figes = set()
    res = {m: [] for m in METHODES}
    ncand, t_exact = [], []
    for t in range(inst.nb_jours):
        pids = par_jour.get(t, [])
        rng.shuffle(pids)
        sol.jour_courant = t
        for pid in pids:
            P = Insertion(sol, prof, [pid], t, figes)
            n = len(P.cands[0])
            if n == 0:
                sol.hors_horizon.add(pid)
                continue
            t0 = time.perf_counter()
            couts = [P.cout([c]) for c in range(n)]
            t_exact.append(time.perf_counter() - t0)
            opt = min(couts)
            ncand.append(n)
            for m, f in METHODES.items():
                P.cache, P.evaluations = {}, 0
                depart = [rng.randrange(n)]
                a = f(P, depart, budget, rng) if n > 1 else depart
                res[m].append(abs(P.cout(a) - opt) < 1e-12)
            sol.affecter(pid, P.cands[0][int(np.argmin(couts))][0])
        figes.add(t + 7)
    return dict(budget=budget, graine=graine, candidats=ncand, temps_exact_ms=1000 * float(np.mean(t_exact)),
                optimum={m: float(np.mean(v)) for m, v in res.items()})


if __name__ == "__main__":
    t0 = time.time()
    with Pool() as P:
        D1 = P.map(un_patient, [(b, 0) for b in (3, 5, 10)])
    for d in D1:
        print("D1 budget", d["budget"], d["optimum"], "candidats médian",
              float(np.median(d["candidats"])), "max", max(d["candidats"]),
              "énumération %.2f ms" % d["temps_exact_ms"], flush=True)
    budgets = [int(b) for b in sys.argv[1:]] or [100, 400]
    taches = [("glouton", 0, g, "actuelle") for g in (0, 1, 2)]
    taches += [(m, b, g, "actuelle") for m in METHODES for b in budgets for g in (0, 1, 2)]
    with Pool() as P:
        D2 = P.map(rejouer, taches, chunksize=1)
    for m in ["glouton", *METHODES]:
        for b in ([0] if m == "glouton" else budgets):
            rs = [r for r in D2 if r["methode"] == m and r["budget"] == b]
            print("D2", m, b, {k: round(float(np.mean([r[k] for r in rs])), 3) for k in
                              ("lits_semaine_variance", "lits_ecart_cible", "lits_ratio_weekend",
                               "sans_date", "delai_median", "delai_p90", "gain_moyen",
                               "part_jours_ameliores", "temps_moyen_s", "conflits")}, flush=True)
    for d in D1:
        d["candidats_hist"] = np.bincount(d.pop("candidats")).tolist()
    (RACINE / "resultats" / "etude_dynamique.json").write_text(json.dumps(
        dict(un_patient=D1, lot=D2, secondes=time.time() - t0), default=float))
    print(f"fin ({time.time() - t0:.0f} s)")
