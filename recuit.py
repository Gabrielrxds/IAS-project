r"""
recuit.py — Recuit simulé, indépendant de la fonction coût.

    RecuitSimule(evaluateur, fonction_cout, ParamsRecuit(...), mobiles, jour_max).lancer()

À chaque itération on tire UN mouvement :
  - déplacement : un patient va dans une autre vacation de SON chirurgien,
    dans sa fenêtre [jour_min, jour_max] ;
  - échange (part_echanges) : deux patients du même chirurgien échangent
    leurs vacations, si chacun reste dans sa fenêtre.
Δ <= 0 : accepté ; Δ > 0 : accepté avec la probabilité exp(-Δ/T).
Tous les `longueur_palier` itérations : T <- alpha . T ; arrêt quand T < Tmin
ou après max_iter itérations. On renvoie la MEILLEURE solution rencontrée.

T0 et Tmin sont des multiples de la dégradation moyenne d'un mouvement,
mesurée au démarrage (T0_facteur = 1 : une dégradation moyenne est acceptée
avec une probabilité e^-1 = 37 % au début). Ainsi les mêmes réglages
conviennent à toutes les fonctions coût, quelle que soit leur échelle.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from modele import PENALITE


@dataclass
class ParamsRecuit:
    T0_facteur: float = 0.5
    alpha: float = 0.995
    longueur_palier: int = 100
    Tmin_facteur: float = 0.01
    part_echanges: float = 0.5
    max_iter: int = 60_000
    graine: int = 0
    journal_periode: int = 500

    def iterations_prevues(self) -> int:
        paliers = math.ceil(math.log(self.Tmin_facteur / self.T0_facteur) / math.log(self.alpha))
        return min(self.max_iter, paliers * self.longueur_palier)


@dataclass
class Resultat:
    cout_initial: float
    cout_final: float
    iterations: int = 0
    temps: float = 0.0
    T0: float = 0.0
    taux_acceptation: float = 0.0
    journal: list = field(default_factory=list)


class RecuitSimule:

    def __init__(self, ev, cout, params: ParamsRecuit, mobiles: list[int], jour_max: dict[int, int]):
        self.ev, self.cout, self.p = ev, cout, params
        self.rng = random.Random(params.graine)
        inst = ev.inst
        self.cands = {}
        for pid in mobiles:
            c = [v for v in inst.vacations_possibles(pid) if inst.vacations[v].jour <= jour_max[pid]]
            if len(c) > 1:
                self.cands[pid] = c
        self.pids = list(self.cands)
        self.ens = {pid: set(c) for pid, c in self.cands.items()}
        self.par_med = {}
        for pid in self.pids:
            self.par_med.setdefault(inst.patients[pid].med_id, []).append(pid)

    def _tirer(self):
        rng, aff = self.rng, self.ev.sol.affectation
        p1 = rng.choice(self.pids)
        if rng.random() < self.p.part_echanges:
            p2 = rng.choice(self.par_med[self.ev.inst.patients[p1].med_id])
            v1, v2 = aff[p1], aff[p2]
            if p1 != p2 and v1 != v2 and v2 in self.ens[p1] and v1 in self.ens[p2]:
                return ("e", p1, p2)
            return None
        v = rng.choice(self.cands[p1])
        return None if v == aff[p1] else ("d", p1, v)

    def _faire(self, mv):
        aff = self.ev.sol.affectation
        if mv[0] == "d":
            ancien = aff[mv[1]]
            self.ev.deplacer(mv[1], mv[2])
            return ("d", mv[1], ancien)
        v1, v2 = aff[mv[1]], aff[mv[2]]
        self.ev.deplacer(mv[1], v2)
        self.ev.deplacer(mv[2], v1)
        return mv

    def _defaire(self, inv):
        if inv[0] == "d":
            self.ev.deplacer(inv[1], inv[2])
        else:
            self._faire(inv)

    def lancer(self) -> Resultat:
        ev, cout, P, rng = self.ev, self.cout, self.p, self.rng
        t0 = time.perf_counter()
        c = cout(ev)
        res = Resultat(c, c)
        if not self.pids:
            return res
        # calibrage : dégradation moyenne d'un mouvement (hors contraintes dures)
        ds = []
        for _ in range(3000):
            mv = self._tirer()
            if mv is None:
                continue
            inv = self._faire(mv)
            d = cout(ev) - c
            self._defaire(inv)
            if 0 < d < PENALITE / 10:
                ds.append(d)
            if len(ds) >= 300:
                break
        dm = sum(ds) / len(ds) if ds else 1e-9
        T = res.T0 = P.T0_facteur * dm
        Tmin = P.Tmin_facteur * dm

        meilleur = c
        best = dict(ev.sol.affectation)
        acc = prop = 0
        it = 0
        while it < P.max_iter and T >= Tmin:
            it += 1
            mv = self._tirer()
            if mv is not None:
                inv = self._faire(mv)
                n = cout(ev)
                d = n - c
                if d <= 0:
                    c = n
                    if c < meilleur - 1e-12 * abs(meilleur):
                        meilleur = c
                        best = dict(ev.sol.affectation)
                else:
                    prop += 1
                    if d / T < 700 and rng.random() < math.exp(-d / T):
                        acc += 1
                        c = n
                    else:
                        self._defaire(inv)
            if it % P.longueur_palier == 0:
                T *= P.alpha
            if it % P.journal_periode == 0:
                res.journal.append({"iteration": it, "temps": time.perf_counter() - t0, "temperature": T,
                                    "cout": c, "meilleur": meilleur, **ev.indicateurs()}
                                   if it % (P.journal_periode * 10) == 0 else
                                   {"iteration": it, "temps": time.perf_counter() - t0, "temperature": T,
                                    "cout": c, "meilleur": meilleur})
        # retour à la meilleure solution
        for pid in self.pids:
            if ev.sol.affectation[pid] != best[pid]:
                ev.deplacer(pid, best[pid])
        res.cout_final = cout(ev)
        res.iterations = it
        res.temps = time.perf_counter() - t0
        res.taux_acceptation = acc / max(1, prop)
        return res
