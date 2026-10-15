r"""
regle_hybride.py — La règle de CONSULTATION de la méthode hybride.

Le patient reçoit sa date à la consultation et elle ne bouge plus (paradigme du
projet). Trois idées sont combinées (cf. les « méthodes proposées pour la suite ») :

1. INSERTION TEMPS RÉEL (règles tabou et génétique) — chaque vacation candidate
   est simulée, son impact sur la courbe des lits est calculé en O(nuits du patient).
   Mesuré : la règle génétique lisse les lits mais repousse les dates (délai médian
   ~ 100 j) car rien ne borne la recherche ; la règle tabou prend le plus tôt et ne
   lisse rien. D'où une FENÊTRE DE TOLÉRANCE : on ne regarde que les vacations entre
   la première date réalisable et Δ jours après (Δ = 7 par défaut).

2. PATIENTS FANTÔMES — la courbe jugée n'est pas celle des seuls patients déjà
   datés : chaque vacation future (non figée, à moins de H jours) contient des
   fantômes, les patients qu'on attend encore pour elle (capacité restante x taux de
   remplissage final attendu de ce chirurgien), avec le profil de séjour de ce
   chirurgien (lits par minute opérée, nuits avant/après), par classe de séjour :
   ambulatoire, court (1-2 nuits), long (3 nuits et plus). Un vrai patient REMPLACE
   des fantômes de sa classe. On lisse donc la courbe ATTENDUE, pas celle du jour :
   sans fantômes, le premier patient d'une semaine vide la trouve toujours idéale.

3. GABARIT OPTIMISÉ LA NUIT — chaque semaine, un recuit (le meilleur algorithme sur
   ce type de voisinage échantillonné, mesuré à l'étage grille) répartit les
   fantômes ENTRE les vacations d'un même chirurgien : il échange des minutes de
   « long » contre des minutes de « court » ou d'« ambulatoire » entre deux de ses
   vacations, pour lisser les lits attendus. Il ne touche JAMAIS un vrai patient.
   Le résultat est un gabarit : où l'on attend les longs séjours. Un patient long
   est alors attiré là où le gabarit a gardé de la place pour lui.

Les deux meilleures dates de la fenêtre sont proposées ; le patient prend la plus
proche (même hypothèse que la règle génétique).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import math
import random
from collections import defaultdict

import numpy as np

from politiques import PremierCreneau

CLASSES = ("ambulatoire", "court", "long")


def classe(p) -> int:
    n = p.nb_nuits
    return 0 if n == 0 else (1 if n <= 2 else 2)


class RegleHybride(PremierCreneau):

    def __init__(self, don, delta: int = 7, horizon: int = 63, fantomes: str = "optimises",
                 choix: str = "plus_proche", remplissage_max: float = 0.95,
                 iterations_nuit: int = 4000, graine: int = 0, quantile_T: float = 0.1):
        assert fantomes in ("aucun", "uniformes", "optimises")
        self.don, self.delta, self.H = don, delta, horizon
        self.mode, self.choix = fantomes, choix
        self.fmax, self.iter_nuit, self.quantile_T = remplissage_max, iterations_nuit, quantile_T
        self.rng = random.Random(graine)
        self.nom = f"hybride (Δ={delta} j, fantômes {fantomes})"
        self._apprendre()
        self.pret = False

    # -- apprentissage (période d'apprentissage seulement) ----------------------

    def _apprendre(self):
        don = self.don
        t0, t1 = don.app
        ps = don.patients_de(t0, t1)
        tis = 15
        num = defaultdict(lambda: defaultdict(float))     # (m, c) -> offset -> nuits
        mins = defaultdict(float)                         # (m, c) -> minutes engagées
        tot_m = defaultdict(float)
        for p in ps:
            c, e = classe(p), p.duree_op + tis
            mins[(p.med_id, c)] += e
            tot_m[p.med_id] += e
            for o in range(-p.nuits_avant, p.nb_nuits - p.nuits_avant):
                num[(p.med_id, c)][o] += 1
        # repli : profil global par classe
        glob_num, glob_min = defaultdict(lambda: defaultdict(float)), defaultdict(float)
        for (m, c), d in num.items():
            for o, v in d.items():
                glob_num[c][o] += v
        for (m, c), v in mins.items():
            glob_min[c] += v
        self.K, self.part = {}, {}
        for m in don.med_codes:
            for c in range(3):
                src, den = (num[(m, c)], mins[(m, c)]) if mins[(m, c)] > 0 else (glob_num[c], glob_min[c])
                offs = sorted(src)
                self.K[(m, c)] = (np.array(offs, dtype=int),
                                  np.array([src[o] / den for o in offs]) if den else np.zeros(len(offs)))
            tot = tot_m[m]
            self.part[m] = (np.array([mins[(m, c)] / tot for c in range(3)]) if tot
                            else np.array([0.5, 0.3, 0.2]))
        self.minutes_semaine = {m: tot_m[m] / ((t1 - t0 + 1) / 7) for m in don.med_codes}

    def _init_instance(self, sol):
        inst = self.inst = sol.inst
        N = inst.nb_jours + 64
        self.E = np.zeros(N)                  # lits attendus = réels + fantômes
        self.r = {}                           # vid -> np.array(3) minutes fantômes par classe
        self.dans = set()                     # vacations qui portent des fantômes
        # taux de remplissage final attendu par chirurgien : demande / offre de la grille
        weeks = inst.nb_jours / 7
        offre = defaultdict(float)
        for v in inst.vacations.values():
            if not v.urgence:
                offre[v.med_id] += inst.capacite_programme(v.id)
        self.f = {m: min(self.fmax, self.minutes_semaine.get(m, 0) / (offre[m] / weeks))
                  if offre[m] else 0.0 for m in inst.medecins}
        self.vac_med = defaultdict(list)
        for vid in sorted(inst.vacations, key=lambda v: inst.vacations[v].jour):
            v = inst.vacations[vid]
            if not v.urgence:
                self.vac_med[v.med_id].append(vid)
        self.pret = True

    # -- outils sur la courbe attendue -------------------------------------------

    def _ajouter_noyau(self, m, c, jour, minutes):
        offs, val = self.K[(m, c)]
        if len(offs) and minutes:
            idx = jour + offs
            ok = (idx >= 0) & (idx < len(self.E))
            np.add.at(self.E, idx[ok], minutes * val[ok])

    def _poser_fantomes(self, vid, r):
        v = self.inst.vacations[vid]
        for c in range(3):
            self._ajouter_noyau(v.med_id, c, v.jour, r[c])

    def _retirer_fantomes(self, vid):
        r = self.r.pop(vid, None)
        if r is not None:
            self._poser_fantomes(vid, -r)
        self.dans.discard(vid)

    def _fenetre(self, t):
        a, b = t + 1, min(len(self.E), t + self.H + 30)
        x = self.E[a:b]
        return a, b, float(x.sum()), float((x * x).sum()), b - a

    # -- cycle quotidien ------------------------------------------------------------

    def preparer(self, sol, t):
        if not self.pret:
            self._init_instance(sol)
        inst = self.inst
        self.t = t
        if self.mode == "aucun":
            return
        # les fantômes des journées figées disparaissent ; ceux de l'horizon apparaissent
        for vid in list(self.dans):
            if inst.vacations[vid].jour <= t + 7:
                self._retirer_fantomes(vid)
        for m, vids in self.vac_med.items():
            for vid in vids:
                j = inst.vacations[vid].jour
                if j <= t + 7:
                    continue
                if j > t + self.H:
                    break
                if vid in self.dans:
                    continue
                reste = max(0.0, self.f.get(m, 0) * inst.capacite_programme(vid) - sol.charge(vid))
                r = reste * self.part[m]
                self.r[vid] = r
                self.dans.add(vid)
                self._poser_fantomes(vid, r)
        if self.mode == "optimises" and t % 7 == 0:
            self._nuit(t)

    def _nuit(self, t):
        """Recuit sur la répartition des fantômes entre vacations d'un même chirurgien."""
        inst, rng = self.inst, self.rng
        par_med = defaultdict(list)
        for vid in self.dans:
            par_med[inst.vacations[vid].med_id].append(vid)
        meds = [m for m, l in par_med.items() if len(l) >= 2]
        if not meds:
            return
        a, b, S1, S2, n = self._fenetre(t)

        def delta_var(changes):
            """changes : liste (m, c, jour, minutes). Renvoie (Δvar, vecteur creux)."""
            d = defaultdict(float)
            for m, c, j, x in changes:
                offs, val = self.K[(m, c)]
                for o, v in zip(offs, val):
                    k = j + o
                    if a <= k < b:
                        d[k] += x * v
            dS1 = sum(d.values())
            dS2 = sum(2 * self.E[k] * x + x * x for k, x in d.items())
            var0 = S2 / n - (S1 / n) ** 2
            var1 = (S2 + dS2) / n - ((S1 + dS1) / n) ** 2
            return var1 - var0, d, dS1, dS2

        # température calée sur les dégradations typiques
        T = None
        ds = []
        for it in range(self.iter_nuit):
            m = rng.choice(meds)
            v, w = rng.sample(par_med[m], 2)
            c1, c2 = rng.sample(range(3), 2)
            x = min(self.r[v][c1], self.r[w][c2], 30.0)
            if x <= 1e-6:
                continue
            jv, jw = inst.vacations[v].jour, inst.vacations[w].jour
            dv, d, dS1, dS2 = delta_var([(m, c1, jv, -x), (m, c2, jv, x), (m, c2, jw, -x), (m, c1, jw, x)])
            if T is None:
                if dv > 0:
                    ds.append(dv)
                if len(ds) < 30:
                    continue
                ds.sort()
                T = ds[int(self.quantile_T * (len(ds) - 1))] or 1e-9
                T_fin = T / 100
                alpha = (T_fin / T) ** (1 / max(1, self.iter_nuit - it))
            if dv <= 0 or rng.random() < math.exp(-dv / T):
                for k, val in d.items():
                    self.E[k] += val
                S1 += dS1; S2 += dS2
                self.r[v][c1] -= x; self.r[v][c2] += x
                self.r[w][c2] -= x; self.r[w][c1] += x
            T *= alpha

    # -- consultation -----------------------------------------------------------------

    def _score(self, sol, pid, vid, fen):
        """Δ variance des lits attendus si pid entre dans vid (et y remplace des fantômes)."""
        a, b, S1, S2, n = fen
        inst, p = self.inst, self.inst.patients[pid]
        v = inst.vacations[vid]
        d = defaultdict(float)
        for k in inst.nuits_occupees(pid, v.jour):
            if a <= k < b:
                d[k] += 1.0
        retrait = None
        if vid in self.dans:
            r = self.r[vid]
            e = p.duree_op + inst.tis
            c = classe(p)
            pris = np.zeros(3)
            pris[c] = min(e, r[c])
            manque = e - pris[c]
            autres = r.copy(); autres[c] = 0
            if manque > 0 and autres.sum() > 0:
                pris += np.minimum(autres, manque * autres / autres.sum())
            retrait = pris
            for cc in range(3):
                if pris[cc]:
                    offs, val = self.K[(v.med_id, cc)]
                    for o, x in zip(offs, val):
                        k = v.jour + o
                        if a <= k < b:
                            d[k] -= pris[cc] * x
        dS1 = sum(d.values())
        dS2 = sum(2 * self.E[k] * x + x * x for k, x in d.items())
        var0 = S2 / n - (S1 / n) ** 2
        var1 = (S2 + dS2) / n - ((S1 + dS1) / n) ** 2
        return var1 - var0, retrait

    def consulter(self, sol, pid, t, figes):
        inst = self.inst
        fen = self._fenetre(t)
        cands, premier = [], None
        for vid in inst.vacations_possibles(pid):
            j = inst.vacations[vid].jour
            if j in figes:
                continue
            if premier is not None and j > premier + self.delta:
                break
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if not ok:
                continue
            if premier is None:
                premier = j
            s, retrait = self._score(sol, pid, vid, fen)
            cands.append((s, j, vid, retrait))
        if not cands:
            return None
        # meilleure vacation par jour, puis les deux meilleurs jours
        par_jour = {}
        for c in cands:
            if c[1] not in par_jour or c[0] < par_jour[c[1]][0]:
                par_jour[c[1]] = c
        deux = sorted(par_jour.values(), key=lambda c: (c[0], c[1]))[:2]
        s, j, vid, retrait = (min(deux, key=lambda c: c[1]) if self.choix == "plus_proche"
                              else deux[0])
        sol.affecter(pid, vid)
        # mise à jour de la courbe attendue : + le patient, - les fantômes remplacés
        for k in inst.nuits_occupees(pid, j):
            if k < len(self.E):
                self.E[k] += 1
        if retrait is not None:
            m = inst.vacations[vid].med_id
            for c in range(3):
                if retrait[c]:
                    self._ajouter_noyau(m, c, j, -retrait[c])
            self.r[vid] = np.maximum(0.0, self.r[vid] - retrait)
        return vid
