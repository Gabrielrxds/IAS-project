r"""
algos_grille.py — Trois métaheuristiques (et leur hybride) sur LE MÊME problème de
grille, avec LE MÊME budget d'évaluations.

Problème : celui de genetique_grille.py (Killian). Individu X[C, 5, R, 2] :
(semaine du cycle, jour, salle, demi-journée) -> praticien, VIDE ou INTERDIT ;
quotas de demi-journées fixés par la demande ; jamais deux salles à la fois.
Fitness analytique ProblemeGrille.cout (lits, places, remplissage, délai, salles).

Les voisinages sont les MUTATIONS de l'AG (échange de deux cases, alignement
matin/après-midi, échange de deux jours) : elles conservent les quotas et
l'absence de conflit, donc tous les algorithmes explorent le même espace.

  AG        AGGrille de Killian (tournoi, croisement par jours, réparation,
            mutation, élitisme, descente finale), budget converti en générations.
  Recuit    une mutation tirée, Metropolis ; T0 et Tmin calés sur le 1er
            décile des dégradations (règle de recuit_grille.py, Lucie) ;
            refroidissement géométrique réglé pour finir au budget.
  Tabou     à chaque itération, k voisins échantillonnés, le meilleur non tabou
            (attribut : (case, ancien occupant), tenure 15), aspiration ;
            schéma de planning_vacations.py (Gabriel).
  Hybride   voir hybride_grille.py.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from protocole import AG  # noqa: F401
from genetique_grille import AGGrille, ParametresAGGrille, PoidsGrille, ProblemeGrille


class Compteur:
    """Enveloppe de la fitness : compte les évaluations, garde la trace du meilleur."""

    def __init__(self, prob: ProblemeGrille, w: PoidsGrille, budget: int):
        self.prob, self.w, self.budget = prob, w, budget
        self.f = type(prob).cout.__get__(prob)      # la fitness d'origine, jamais l'enveloppe
        self.n = 0
        self.meilleur = math.inf
        self.X = None
        self.trace: list[tuple[int, float]] = []

    def __call__(self, X) -> float:
        self.n += 1
        c = self.f(X, self.w)
        if c < self.meilleur - 1e-12:
            self.meilleur, self.X = c, X.copy()
        if self.n % 250 == 0:
            self.trace.append((self.n, self.meilleur))
        return c

    @property
    def epuise(self) -> bool:
        return self.n >= self.budget


@dataclass
class Sortie:
    algo: str
    graine: int
    cout: float
    evaluations: int
    secondes: float
    X: np.ndarray
    trace: list = field(default_factory=list)


# ---------------------------------------------------------------------------

def ag(prob, w, budget, graine, **kw) -> Sortie:
    """AG de Killian. Un budget de B évaluations = (B - descente) / population générations."""
    t = time.perf_counter()
    pop = kw.get("population", 40)
    desc = kw.get("descente", max(0, budget // 10))
    gens = max(1, (budget - desc - pop) // pop)
    a = AGGrille(prob, w, ParametresAGGrille(taille_population=pop, nb_generations=gens,
                                             iterations_descente=desc, stagnation_max=10 ** 9,
                                             graine=graine))
    cpt = Compteur(prob, w, budget)
    prob.cout = lambda X, W: cpt(X)              # on compte toutes les évaluations
    try:
        r = a.resoudre()
    finally:
        del prob.cout
    return Sortie("génétique", graine, r.cout, cpt.n, time.perf_counter() - t, r.X, cpt.trace)


def _voisin(a: AGGrille, X):
    Y = X.copy()
    a.muter(Y)
    return Y


def recuit(prob, w, budget, graine, T0_facteur=1.0, Tmin_facteur=0.005, depart=None, **kw) -> Sortie:
    t = time.perf_counter()
    a = AGGrille(prob, w, ParametresAGGrille(graine=graine))
    rng = random.Random(graine + 1000)
    cpt = Compteur(prob, w, budget)
    X = depart.copy() if depart is not None else a.aleatoire()
    c = cpt(X)
    # calibrage : 1er décile des dégradations (recuit_grille.py)
    ds = []
    for _ in range(min(300, budget // 20)):
        d = cpt(_voisin(a, X)) - c
        if d > 1e-12:
            ds.append(d)
    ds.sort()
    dm = ds[len(ds) // 10] if ds else 1e-6
    T, Tmin = T0_facteur * dm, Tmin_facteur * dm
    reste = max(1, budget - cpt.n)
    alpha = (Tmin / T) ** (1 / reste)
    while not cpt.epuise:
        Y = _voisin(a, X)
        n = cpt(Y)
        d = n - c
        if d <= 0 or (d / T < 700 and rng.random() < math.exp(-d / T)):
            X, c = Y, n
        T *= alpha
    return Sortie("recuit", graine, cpt.meilleur, cpt.n, time.perf_counter() - t, cpt.X, cpt.trace)


def tabou(prob, w, budget, graine, k=30, tenure=15, depart=None, **kw) -> Sortie:
    t = time.perf_counter()
    a = AGGrille(prob, w, ParametresAGGrille(graine=graine))
    cpt = Compteur(prob, w, budget)
    X = depart.copy() if depart is not None else a.aleatoire()
    c = cpt(X)
    interdit: dict[tuple, int] = {}          # (case, occupant) -> itération de levée
    it = 0
    while not cpt.epuise:
        it += 1
        meilleur = None
        for _ in range(k):
            if cpt.epuise:
                break
            Y = _voisin(a, X)
            diff = [tuple(p) for p in np.argwhere(Y != X)]
            if not diff:
                continue
            n = cpt(Y)
            tabou_ = any(interdit.get((p, int(Y[p])), 0) > it for p in diff)
            if tabou_ and n >= cpt.meilleur - 1e-12:      # aspiration
                continue
            if meilleur is None or n < meilleur[0]:
                meilleur = (n, Y, diff)
        if meilleur is None:
            continue
        n, Y, diff = meilleur
        for p in diff:                        # interdit de remettre l'ancien occupant
            interdit[(p, int(X[p]))] = it + tenure
        X, c = Y, n
    return Sortie("tabou", graine, cpt.meilleur, cpt.n, time.perf_counter() - t, cpt.X, cpt.trace)


ALGOS = {"génétique": ag, "recuit": recuit, "tabou": tabou}
