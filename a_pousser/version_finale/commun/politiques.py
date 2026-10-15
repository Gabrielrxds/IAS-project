r"""
politiques.py — Les règles qui donnent la DATE à la consultation, et le
simulateur en ligne commun.

Paradigme commun (celui du projet) : le patient reçoit sa date À LA
CONSULTATION, elle ne bouge plus. Une journée est FIGÉE 7 jours avant (délai
minimum) : plus personne n'y entre, l'ordonnancement de la journée tourne.

Règles comparées
  PremierCreneau   la secrétaire sans outil : premier créneau réalisable.
  RegleTabou       « ancien régime » (chaîne tabou) : date au plus tôt, la
                   meilleure vacation du jour selon f(x) = w_r Var(τ) + w_l
                   Var(L)/L², journée fermée dès 85 % de remplissage.
  RegleGenetique   chaîne génétique (Killian) : les 12 premières vacations
                   réalisables notées par CoutTotal, les deux meilleures
                   proposées, le patient prend la plus tôt.
La chaîne recuit (Lucie) optimise HORS LIGNE (tous les patients de l'année
connus, ±30 j autour de la date réelle) : elle est évaluée à part
(chaine_recuit.py), comme borne, pas comme règle de consultation.
"""
from __future__ import annotations

import random
from collections import defaultdict

from protocole import AG  # noqa: F401  (chemin vers le modèle du groupe)
from analyse_couts import Config, consulter as consulter_ag
from modele import Solution


class PremierCreneau:
    nom = "premier créneau"

    def preparer(self, sol, t):
        pass

    def consulter(self, sol, pid, t, figes):
        inst = sol.inst
        for vid in inst.vacations_possibles(pid):
            if inst.vacations[vid].jour in figes:
                continue
            sol.affecter(pid, vid)
            if not sol.viole(pid, vid):
                return vid
            sol.affecter(pid, None)
        return None

    def apres(self, sol, vid, t, figes):
        pass


class RegleGenetique(PremierCreneau):
    """`consulter` de analyse_couts.py, tel quel."""
    nom = "règle génétique (CoutTotal, 12 candidates)"

    def __init__(self, ref: dict, config: Config | None = None, nb_candidats: int = 12,
                 fenetre_cout: int = 91):
        self.cout = (config or Config()).cout(ref)
        self.nb, self.fen = nb_candidats, fenetre_cout

    def preparer(self, sol, t):
        sol.definir_fenetre(t, min(sol.inst.nb_jours - 1, t + self.fen))

    def consulter(self, sol, pid, t, figes):
        inst = sol.inst
        cands = []
        for vid in inst.vacations_possibles(pid):
            if inst.vacations[vid].jour in figes:
                continue
            sol.affecter(pid, vid)
            if not sol.viole(pid, vid):
                cands.append((self.cout(sol), inst.vacations[vid].jour, vid))
            sol.affecter(pid, None)
            if len(cands) >= self.nb:
                break
        if not cands:
            return None
        cands.sort()
        vid = min(cands[:2], key=lambda c: c[1])[2]
        sol.affecter(pid, vid)
        return vid


class RegleTabou(PremierCreneau):
    """Ancien régime de la chaîne tabou, transposé dans le modèle commun."""
    nom = "règle tabou (au plus tôt, fermeture à 85 %)"

    def __init__(self, w_r: float = 1.0, w_l: float = 1.0, l_ref: float = 42.0,
                 seuil: float = 0.85, fenetre_cout: int = 91):
        self.w_r, self.w_l, self.L2, self.seuil, self.fen = w_r, w_l, l_ref ** 2, seuil, fenetre_cout

    def preparer(self, sol, t):
        sol.definir_fenetre(t, min(sol.inst.nb_jours - 1, t + self.fen))

    def f(self, sol):
        return self.w_r * sol.variance_remplissage() + self.w_l * sol.variance_lits() / self.L2

    def consulter(self, sol, pid, t, figes):
        inst = sol.inst
        jour, best = None, None
        for vid in inst.vacations_possibles(pid):
            j = inst.vacations[vid].jour
            if j in figes:
                continue
            if jour is not None and j > jour:
                break                       # on ne regarde que le premier jour réalisable
            sol.affecter(pid, vid)
            if not sol.viole(pid, vid):
                c = self.f(sol)
                if best is None or c < best[0]:
                    best, jour = (c, vid), j
            sol.affecter(pid, None)
        if best is None:
            return None
        sol.affecter(pid, best[1])
        return best[1]

    def apres(self, sol, vid, t, figes):
        inst = sol.inst
        j = inst.vacations[vid].jour
        vids = [v for v in inst.vacations_du_jour.get(j, ()) if not inst.vacations[v].urgence]
        tvo = sum(inst.vacations[v].tvo for v in vids)
        if tvo and j > t + 1 and sum(sol.charge(v) for v in vids) / tvo >= self.seuil:
            figes.add(j)


def simuler(inst, regle, graine: int = 0, apres_figer=None, delai_fige: int = 7,
            jours_figes_hook=None) -> Solution:
    """Flux des consultations jour après jour. Après les consultations du jour
    t, la journée t + delai_fige est figée : `apres_figer(sol, j)` y tourne
    (ordonnancement de la journée)."""
    sol = Solution(inst)
    rng = random.Random(graine)
    par_jour = defaultdict(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    figes: set[int] = set()
    deja = set()
    for t in range(inst.nb_jours):
        pids = par_jour.get(t, [])
        if pids:
            rng.shuffle(pids)
            sol.jour_courant = t
            regle.preparer(sol, t)
            for pid in pids:
                vid = regle.consulter(sol, pid, t, figes)
                if vid is None:
                    sol.hors_horizon.add(pid)
                else:
                    regle.apres(sol, vid, t, figes)
        j = t + delai_fige
        if j < inst.nb_jours:
            figes.add(j)
            if apres_figer is not None and j not in deja:
                deja.add(j)
                apres_figer(sol, j)
    sol.jour_courant = None
    sol.definir_fenetre(0, inst.nb_jours - 1)
    return sol
