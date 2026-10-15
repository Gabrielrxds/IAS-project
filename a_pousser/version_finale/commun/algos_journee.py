r"""
algos_journee.py — Ordonnancement d'UNE journée figée : AG, recuit, tabou (et hybride)
sur la MÊME fonction de coût, avec le MÊME budget d'évaluations distinctes.

Fonction de coût : celle de genetique_jour.py (Killian), la plus complète du groupe :
dépassement interdit (scénario P90) >> dépassement, sorties UCA tardives,
surcapacité des places >> robustesse P90 >> pic et écart-type des places,
temps perdu, dispersion du remplissage.  Codage : permutation des patients du
jour + vacation de chaque patient (parmi celles de SON chirurgien ce jour-là).

  AG      AGJournee de Killian (OX, mutations, élitisme, immigration, descente).
  Recuit  une mutation de l'AG tirée, Metropolis, T0 = 0,5 x dégradation
          moyenne, Tmin = 0,01 x (règle de recuit.py, Lucie), refroidissement
          géométrique calé sur le budget.
  Tabou   mouvements explicites (insertion d'un patient à une autre position,
          échange de deux patients, changement de vacation) ; k voisins par
          itération ; un patient déplacé est tabou 7 itérations ; aspiration
          (schéma du TabouLocal de Gabriel).
Tous partent des mêmes individus de départ (les graines heuristiques de l'AG).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from protocole import AG  # noqa: F401
from genetique_jour import AGJournee, Individu, ParametresAG, Poids, REPORT
from modele import deriver_creneaux


# Le niveau global cumule les marges quadratiquement (protocole.CUMUL_MARGES) : une
# journée acceptée peut donc dépasser dans le scénario « tous les actes au P90 ».
# Le dépassement est interdit avec les durées ESTIMÉES ; le scénario P90 reste
# pénalisé dans le coût (robustesse).
MODE_DEPASSEMENT = "median"


class Epuise(Exception):
    pass


class Juge(AGJournee):
    """AGJournee dont l'évaluateur s'arrête au budget (évaluations distinctes)."""

    def __init__(self, *a, budget: int = 10 ** 9, graines: bool = True, **k):
        k.setdefault("depassement", MODE_DEPASSEMENT)
        super().__init__(*a, **k)
        self.graines = graines
        self.budget = budget
        self.meilleur = None
        self.appels = 0

    def population_initiale(self):
        if self.graines:
            return super().population_initiale()
        # sans graines heuristiques : l'ordre d'inscription + du hasard
        pop = [self._ind_depuis_ordre(list(range(self.n)))]
        while len(pop) < self.P.taille_population:
            pop.append(self._aleatoire())
        return pop

    def evaluer(self, ind):
        self.appels += 1
        if self.appels > 20 * self.budget:          # petit espace : tout est déjà vu
            raise Epuise
        if self.nb_evaluations >= self.budget and ind.eval is None:
            files = self.decoder(ind)
            cle = tuple(tuple(files[v]) for v in self.vids)
            if cle not in self._cache:
                raise Epuise
        ev = super().evaluer(ind)
        if self.meilleur is None or ev.cout < self.meilleur.eval.cout - 1e-9:
            self.meilleur = ind.copie()
            self.meilleur.eval = ev
        return ev


@dataclass
class SortieJour:
    algo: str
    cout: float
    cout_depart: float
    evaluations: int
    secondes: float
    sequences: dict
    evaluation: object


def _fin(j: Juge, algo, t0, depart_cout):
    best = j.meilleur
    if best.eval.violation > 0:
        try:
            best = j.reparer(best)
        except Epuise:
            pass
    return SortieJour(algo, best.eval.cout, depart_cout, j.nb_evaluations,
                      time.perf_counter() - t0, j.sequences(best), best.eval)


def _depart(j: Juge):
    if not j.graines:                         # départ = ordre d'inscription
        ind = j._ind_depuis_ordre(list(range(j.n)))
        return ind.copie(), [ind]
    pop = j.population_initiale()
    return min(pop, key=lambda i: i.cout).copie(), pop


def genetique(inst, jour, seq, poids, budget, graine, **kw) -> SortieJour:
    t0 = time.perf_counter()
    P = ParametresAG(graine=graine, nb_generations=10 ** 6, stagnation_max=10 ** 6,
                     taille_population=kw.get("population", 40))
    j = Juge(inst, jour, seq, poids=poids, params=P, budget=budget, graines=kw.get('graines', True))
    j.verifier_faisabilite()
    dep = None
    try:
        dep, _ = _depart(j)
        j.resoudre()
    except Epuise:
        pass
    return _fin(j, "génétique", t0, dep.cout if dep else math.nan)


def recuit(inst, jour, seq, poids, budget, graine, T0_facteur=0.5, Tmin_facteur=0.01,
           depart=None, **kw) -> SortieJour:
    t0 = time.perf_counter()
    j = Juge(inst, jour, seq, poids=poids, params=ParametresAG(graine=graine), budget=budget,
              graines=kw.get('graines', True))
    j.verifier_faisabilite()
    rng = random.Random(graine + 7)
    X, _ = _depart(j) if depart is None else (depart, None)
    c0 = X.cout
    try:
        ds = []
        for _ in range(60):
            Y = X.copie(); j.muter(Y)
            d = j.evaluer(Y).cout - X.cout
            if 0 < d < 1e6:
                ds.append(d)
        dm = sum(ds) / len(ds) if ds else 1.0
        T, Tmin = T0_facteur * dm, Tmin_facteur * dm
        reste = max(1, budget - j.nb_evaluations)
        alpha = (Tmin / T) ** (1 / reste)
        while True:
            Y = X.copie(); j.muter(Y)
            d = j.evaluer(Y).cout - X.cout
            if d <= 0 or (d / T < 700 and rng.random() < math.exp(-d / T)):
                X = Y
            T = max(Tmin, T * alpha)
    except Epuise:
        pass
    return _fin(j, "recuit", t0, c0)


def _voisins_tabou(j: Juge, X: Individu, rng, k):
    """k mouvements explicites, chacun avec le patient qu'il déplace."""
    n = j.n
    reaff = [i for i in range(n) if len([v for v in j.choix[i] if v != REPORT]) > 1]
    for _ in range(k):
        Y = Individu(list(X.ordre), list(X.vac))
        r = rng.random()
        if reaff and r < 0.25:
            i = rng.choice(reaff)
            Y.vac[i] = rng.choice([v for v in j.choix[i] if v not in (REPORT, X.vac[i])])
            yield Y, (i,)
        elif r < 0.65:
            a, b = rng.sample(range(n), 2)
            p = Y.ordre.pop(a); Y.ordre.insert(b, p)
            yield Y, (p,)
        else:
            a, b = rng.sample(range(n), 2)
            Y.ordre[a], Y.ordre[b] = Y.ordre[b], Y.ordre[a]
            yield Y, (Y.ordre[a], Y.ordre[b])


def tabou(inst, jour, seq, poids, budget, graine, k=20, tenure=7, depart=None, **kw) -> SortieJour:
    t0 = time.perf_counter()
    j = Juge(inst, jour, seq, poids=poids, params=ParametresAG(graine=graine), budget=budget,
              graines=kw.get('graines', True))
    j.verifier_faisabilite()
    rng = random.Random(graine + 11)
    X, _ = _depart(j) if depart is None else (depart, None)
    c0 = X.cout
    libere: dict[int, int] = {}
    it = 0
    try:
        if j.n < 2:
            raise Epuise
        while True:
            it += 1
            choix = None
            for Y, bouges in _voisins_tabou(j, X, rng, k):
                c = j.evaluer(Y).cout
                tab = any(libere.get(p, 0) > it for p in bouges)
                if tab and c >= j.meilleur.eval.cout - 1e-9:
                    continue
                if choix is None or c < choix[0]:
                    choix = (c, Y, bouges)
            if choix is None:
                continue
            _, X, bouges = choix
            for p in bouges:
                libere[p] = it + tenure
    except Epuise:
        pass
    return _fin(j, "tabou", t0, c0)


ALGOS = {"génétique": genetique, "recuit": recuit, "tabou": tabou}


def metriques(inst, jour, sequences) -> dict:
    """Mesures communes d'un planning de journée (durées estimées et RÉELLES)."""
    from analyse_couts import metriques_planning
    return metriques_planning(inst, deriver_creneaux(inst, jour, sequences))
