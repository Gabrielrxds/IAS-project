r"""
hybride_grille.py — Le niveau GRILLE de la méthode hybride.

1. LE PROBLÈME (ce que l'on garde de chaque méthode)
   - Codage en demi-journées X[semaine, jour, salle, demi] de la chaîne génétique
     (Killian) ; ses mutations conservent quotas et absence de conflit.
   - RESSOURCES ÉGALES à aujourd'hui : salles 2 à 5 seulement, et autant de
     demi-journées par cycle que la grille actuelle en occupe (la grille génétique
     d'origine ouvrait la salle 1 et offrait 33 % d'heures en plus).
   - Volume de chaque chirurgien : sa part de la demande récente (janvier-avril),
     appliquée au budget actuel, sans descendre sous 50 % de son temps actuel
     (règle « grille B » du recuit, Lucie) : on ne reconstruit pas l'offre totale
     sur une prévision (leçon de la chaîne tabou, Gabriel).
   - Lits sur TOUTES les nuits du cycle (chaînes tabou et recuit) : la fitness
     d'origine ne regardait que les nuits de semaine, d'où des grilles qui
     reportent les lits sur le week-end.
   - Mêmes autres termes que la chaîne génétique (places, temps perdu, délai).

2. L'ALGORITHME (ce que chaque métaheuristique fait le mieux, mesuré à budget égal)
   Phase 1  DIVERSIFIER : plusieurs recuits courts depuis des grilles aléatoires
            (le recuit est le meilleur et le plus régulier à petit budget).
   Phase 2  RECOMBINER : AG stationnaire sur ces élites ; croisement par JOURS
            entiers (une journée bien construite passe d'un parent à l'enfant),
            réparation des quotas, puis tabou court sur l'enfant (AG mémétique) ;
            l'enfant remplace le pire s'il est meilleur et nouveau.
   Phase 3  INTENSIFIER : tabou sur la meilleure grille avec le budget restant.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import datetime as dt
import math
import random
import time

import numpy as np

from protocole import AG  # noqa: F401
from genetique_grille import (AGGrille, ParametresAGGrille, PoidsGrille, ProblemeGrille,
                              DEMI, INTERDIT, VIDE)
from grille import creneaux_du
from algos_grille import Compteur, Sortie, recuit as recuit_x, tabou as tabou_x


SALLES_ACTUELLES = (2, 3, 4, 5)


def demi_journees_actuelles(codes: list[str], salles=SALLES_ACTUELLES, annee: int = 2022):
    """Demi-journées par cycle de 4 semaines occupées par chaque chirurgien dans la
    grille actuelle (une case est comptée si le créneau la recouvre)."""
    idx = {s: i for i, s in enumerate(salles)}
    occ = {}
    for w in range(1, 5):
        k = w % 4
        for d in range(5):
            for c in creneaux_du(dt.date.fromisocalendar(annee, w, d + 1)):
                if c.salle not in idx:
                    continue
                for h, (a, b) in enumerate(DEMI):
                    if c.debut < b and a < c.fin:
                        occ[(k, d, c.salle, h)] = c.code
    q = {m: 0 for m in codes}
    for code in occ.values():
        if code in q:
            q[code] += 1
    return q


def heures_programmees_actuelles(salles=SALLES_ACTUELLES, annee: int = 2022) -> float:
    """Heures de vacations programmées par cycle de 4 semaines dans la grille actuelle."""
    h = 0.0
    for w in range(1, 5):
        for d in range(5):
            for c in creneaux_du(dt.date.fromisocalendar(annee, w, d + 1)):
                if c.salle in salles:
                    h += (c.fin - c.debut) / 60
    return h


class ProblemeHybride(ProblemeGrille):
    """ProblemeGrille à ressources égales, lits sur toutes les nuits."""

    def __init__(self, profils, part_min: float = 0.5, part_max: float = 10.0,
                 heures_par_case: float = 4.9, **kw):
        kw.setdefault("salles", SALLES_ACTUELLES)
        super().__init__(profils, **kw)
        codes = [p.code for p in profils]
        actuel = demi_journees_actuelles(codes, self.salles)
        # budget en HEURES : autant d'heures de bloc programmé que la grille actuelle.
        # Une case vaut en moyenne `heures_par_case` heures une fois les journées
        # (8h-17h30) formées : 4,9 h mesuré sur les grilles produites.
        budget = int(round(heures_programmees_actuelles(self.salles) / heures_par_case))
        dem = np.array([p.minutes_semaine for p in profils], dtype=float)
        part = dem / dem.sum() if dem.sum() else np.ones(len(dem)) / len(dem)
        plancher = np.array([max(1, math.ceil(part_min * actuel[c])) for c in codes])
        plafond = np.array([max(1, math.floor(part_max * actuel[c])) for c in codes])
        q = np.minimum(np.maximum(plancher, part * budget), np.maximum(plancher, plafond))
        # ramener au budget (plus grands restes), sans passer sous le plancher
        for _ in range(50):
            libre = budget - plancher.sum()
            souple = np.maximum(0, q - plancher)
            q = plancher + (souple / souple.sum() * libre if souple.sum() else 0)
            if (q >= plancher - 1e-9).all():
                break
        if q.sum() < budget:
            marge = np.maximum(0, np.maximum(plancher, plafond) - q)
            q = q + marge / marge.sum() * (budget - q.sum()) if marge.sum() else q
        base = np.floor(q).astype(int)
        reste = budget - base.sum()
        for i in np.argsort(-(q - base))[:reste]:
            base[i] += 1
        self.quotas = np.minimum(base, 10 * self.C)
        self.budget = budget
        self.quotas_actuels = actuel
        self.ouvres_lits = np.ones(self.L, dtype=bool)      # toutes les nuits

    def termes(self, tvo, nvac, salles_jours):
        t = super().termes(tvo, nvac, salles_jours)
        # recalcul du terme lits sur TOUTES les nuits
        TVO = tvo.sum(1)
        cap_eff = np.maximum(0.0, TVO - nvac * self.frag)
        servi = np.minimum(self.demande, cap_eff)
        f = np.divide(servi, TVO, out=np.zeros_like(TVO), where=TVO > 0)
        lits = np.einsum("snj,sj->n", self.K, tvo * f[:, None])
        t["lits"] = float(lits.var()) / 42 ** 2
        t["lits_moyens"] = float(lits.mean())
        return t

    def termes_grille_actuelle(self, annee: int = 2022) -> dict:
        ref = type(self).__new__(type(self))
        ref.__dict__.update(self.__dict__)
        ref.C, ref.L = 4, 28
        ref.demande = np.array([p.minutes_semaine * 4 for p in self.profils])
        ref.K = np.zeros((self.S, 28, 28))
        for s, p in enumerate(self.profils):
            for o, val in p.noyau_lits.items():
                for j in range(28):
                    ref.K[s, (j + o) % 28, j] += val
        ref.ouvres = np.array([j % 7 < 5 for j in range(28)])
        indice = {p.code: s for s, p in enumerate(self.profils)}
        tvo = np.zeros((self.S, 28)); nvac = np.zeros(self.S); ouvertes = set()
        for w in range(1, 5):
            k = w % 4
            for d in range(5):
                for c in creneaux_du(dt.date.fromisocalendar(annee, w, d + 1)):
                    if c.code in indice:
                        tvo[indice[c.code], 7 * k + d] += c.fin - c.debut
                        nvac[indice[c.code]] += 1
                        ouvertes.add((k, d, c.salle))
        return ref.termes(tvo, nvac, len(ouvertes))


def probleme_hybride(don, reg, **kw) -> ProblemeHybride:
    from analyse_couts import Instance as _I  # noqa: F401
    from genetique_grille import profils_chirurgiens
    from modele import Instance
    t0, t1 = don.app
    profils = profils_chirurgiens(don.patients_de(t0, t1), don.med_codes, (t1 - t0 + 1) / 7,
                                  Instance().tis)
    prob = ProblemeHybride(profils, C=reg.cycle_semaines, taux_cible=reg.taux_cible,
                           capacites=(reg.capacite_lits - reg.reserve_lits,
                                      reg.capacite_places_jour - reg.reserve_places), **kw)
    prob.calibrer()
    return prob


# ---------------------------------------------------------------------------
# L'algorithme hybride
# ---------------------------------------------------------------------------

def hybride(prob, w, budget, graine, n_elites=6, part1=0.40, part2=0.35, tabou_enfant=150, **kw) -> Sortie:
    t = time.perf_counter()
    rng = random.Random(graine + 99)
    a = AGGrille(prob, w, ParametresAGGrille(graine=graine))
    cpt = Compteur(prob, w, budget)
    trace = []

    def sous_budget(n):
        return max(50, int(n))

    # Phase 1 : recuits courts, départs aléatoires distincts
    elites = []
    b1 = sous_budget(part1 * budget / n_elites)
    for e in range(n_elites):
        s = recuit_x(prob, w, b1, graine * 100 + e)
        cpt.n += s.evaluations
        elites.append((s.cout, s.X))
        cpt(s.X)
    # Phase 2 : AG stationnaire mémétique (croisement par jours + réparation + tabou court)
    fin2 = cpt.n + part2 * budget
    while cpt.n < fin2:
        elites.sort(key=lambda e: e[0])
        p1, p2 = (min(rng.sample(elites, 2), key=lambda e: e[0])[1] for _ in range(2))
        e1, _ = a.croiser(p1, p2)
        if not a.reparer(e1):
            continue
        s = tabou_x(prob, w, tabou_enfant, rng.randrange(10 ** 6), k=15, depart=e1)
        cpt.n += s.evaluations
        c = cpt(s.X)
        if c < elites[-1][0] - 1e-9 and all(not np.array_equal(s.X, x) for _, x in elites):
            elites[-1] = (c, s.X)
        trace.append((cpt.n, min(e[0] for e in elites)))
    # Phase 3 : tabou d'intensification sur la meilleure
    elites.sort(key=lambda e: e[0])
    reste = budget - cpt.n
    if reste > 50:
        s = tabou_x(prob, w, reste, graine + 5, depart=elites[0][1])
        cpt.n += s.evaluations
        cpt(s.X)
    return Sortie("hybride", graine, cpt.meilleur, cpt.n, time.perf_counter() - t, cpt.X, cpt.trace + trace)
