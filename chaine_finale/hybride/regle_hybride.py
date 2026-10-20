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
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
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


class RegleHybrideV1(PremierCreneau):

    def __init__(self, don, delta: int = 7, horizon: int = 63, fantomes: str = "optimises",
                 choix: str = "plus_proche", remplissage_max: float = 0.95,
                 iterations_nuit: int = 4000, graine: int = 0, quantile_T: float = 0.1,
                 cible="defaut"):
        """cible = (f_we, f_vac) : les lits sont jugés par leur écart à une cible nuit par nuit
        (calendrier.py) ; "defaut" = calendrier.F_WE, F_VAC ; None = variance pure (f ≡ 1)."""
        assert fantomes in ("aucun", "uniformes", "optimises")
        self.don, self.delta, self.H = don, delta, horizon
        self.mode, self.choix = fantomes, choix
        self.fmax, self.iter_nuit, self.quantile_T = remplissage_max, iterations_nuit, quantile_T
        self.rng = random.Random(graine)
        if cible == "defaut":
            import calendrier
            cible = (calendrier.F_WE, calendrier.F_VAC)
        self.cible = cible
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
        import calendrier
        self.prof = (np.ones(len(self.E)) if self.cible is None
                     else calendrier.profil(inst.jour_zero, len(self.E), *self.cible))
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

    def lits_patient(self, pid, jour, signe=1):
        """Ajoute (signe=+1) ou retire (signe=-1) les nuits d'un patient réel daté au jour donné."""
        for k in self.inst.nuits_occupees(pid, jour):
            if k < len(self.E):
                self.E[k] += signe

    def _retirer_fantomes(self, vid):
        r = self.r.pop(vid, None)
        if r is not None:
            self._poser_fantomes(vid, -r)
        self.dans.discard(vid)

    def _fenetre(self, t):
        a, b = t + 1, min(len(self.E), t + self.H + 30)
        x = self.E[a:b]
        f = self.prof[a:b]
        self._SEf, self._Sff = float(x @ f), float(f @ f)
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
        dS2 = sum(2 * self.E[k] * x + x * x for k, x in d.items())
        # écart à la forme de la cible : (1/n)[Σ E² − (Σ E f)² / Σ f²] ; = variance si f ≡ 1
        dSEf = sum(self.prof[k] * x for k, x in d.items())
        e0 = (S2 - self._SEf ** 2 / self._Sff) / n
        e1 = (S2 + dS2 - (self._SEf + dSEf) ** 2 / self._Sff) / n
        return e1 - e0, retrait

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


# =====================================================================================
# V3 — fantômes « prévisions » : modèle de prévision explicite, mis à jour chaque jour,
# et fantômes d'urgence.
# =====================================================================================

def _minutes(inst, sol, vid) -> float:
    """Minutes réservées dans une vacation (durées estimées + temps entre patients), sans marge."""
    n = sol.nb[vid]
    return float(sol.somme_durees[vid] + inst.tis * n) if n else 0.0


def apprendre_urgences(df, don, annees=(2019, 2021)):
    """Fantômes d'urgence : taux d'arrivée des urgences par jour de semaine et profil de lits
    d'une urgence (lits occupés o jours après l'opération), appris sur l'historique
    (2019 et 2021, 2020 exclue pour les confinements, plus janvier-avril 2022)."""
    import pandas as pd
    from protocole import est_urgent_diag
    d = df[df["annee"].isin(annees) | ((df["annee"] == 2022) & (df["Date Inter"] <= pd.Timestamp(don.date(don.app[1]))))]
    d = d[d["Date Inter"].notna()]
    u = d[est_urgent_diag(d["CIM Diag Pr."].astype("string").fillna("").tolist())]
    jours = pd.to_datetime(d["Date Inter"]).dt.normalize()
    nb_jours = {w: 0 for w in range(7)}
    for x in pd.date_range(jours.min(), jours.max()):
        nb_jours[x.weekday()] += 1
    du = pd.to_datetime(u["Date Inter"]).dt.weekday
    lam = np.array([(du == w).sum() / max(1, nb_jours[w]) for w in range(7)])
    noyau = defaultdict(float)
    for _, r in u.iterrows():
        n, a = int(r["nuits"]), int(min(r["nuits_avant"], r["nuits"]))
        for o in range(-a, n - a):
            noyau[o] += 1
    offs = np.array(sorted(noyau), dtype=int)
    val = np.array([noyau[o] / len(u) for o in offs]) if len(u) else np.zeros(0)
    return dict(lam=lam, offs=offs, val=val, n=len(u))


def apprendre_saison(df, annees=(2019, 2021)):
    """Facteur des vacances scolaires (zone B), ponts et fériés : minutes programmées par jour
    ouvré pendant les vacances / hors vacances, sur l'historique (urgences exclues)."""
    import pandas as pd
    import calendrier
    from protocole import est_urgent_diag
    d = df[df["annee"].isin(annees) & df["Date Inter"].notna()].copy()
    d = d[~est_urgent_diag(d["CIM Diag Pr."].astype("string").fillna("").tolist())]
    d["j"] = pd.to_datetime(d["Date Inter"]).dt.date
    fer, ponts = calendrier.feries_et_ponts(range(min(annees) - 1, max(annees) + 2))
    par_jour = d.groupby("j")["tros"].sum() + 15 * d.groupby("j").size()
    vac, hors = [], []
    for j, x in par_jour.items():
        if j.weekday() >= 5:
            continue
        (vac if calendrier.type_nuit(j, fer, ponts) == "vacances" else hors).append(x)
    return float(np.mean(vac) / np.mean(hors))


class JournalReservations(PremierCreneau):
    """Enveloppe une règle et note chaque réservation (jour t, vacation, minutes)."""

    def __init__(self, regle):
        self.r, self.log = regle, []
        self.nom = getattr(regle, "nom", "")

    def preparer(self, sol, t):
        self.r.preparer(sol, t)

    def consulter(self, sol, pid, t, figes):
        vid = self.r.consulter(sol, pid, t, figes)
        if vid is not None:
            p = sol.inst.patients[pid]
            self.log.append((t, vid, p.duree_op + sol.inst.tis))
        return vid

    def apres(self, sol, vid, t, figes):
        self.r.apres(sol, vid, t, figes)


def apprendre_courbe_reservation(don, lignes, reg=None, lmax: int = 120):
    """Courbe de réservation B(ℓ) : part des minutes finales d'une vacation déjà réservées
    ℓ jours avant elle, et facteur saisonnier κ (remplissage final des vacations de vacances
    scolaires / des autres). Apprises en rejouant la période d'apprentissage avec la règle
    hybride V1 : seules les vacations de janvier-avril et les réservations faites avant le
    30 avril servent."""
    from protocole import instance
    from politiques import simuler
    import calendrier
    inst = instance(don, lignes, reg)
    J = JournalReservations(RegleHybrideV1(don, delta=14, fantomes="uniformes"))
    simuler(inst, J, graine=0)
    a0, a1 = don.app
    fin = defaultdict(float)
    par_v = defaultdict(list)
    for t, vid, x in J.log:
        if t <= a1 and inst.vacations[vid].jour <= a1:
            fin[vid] += x
            par_v[vid].append((inst.vacations[vid].jour - t, x))
    tot = sum(fin.values()) or 1.0
    B = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        B[l] = sum(x for v in par_v for lead, x in par_v[v] if lead >= l) / tot
    # facteur saisonnier : remplissage final / capacité, vacances scolaires contre le reste
    fer, ponts = calendrier.feries_et_ponts(range(don.date(a0).year - 1, don.date(a1).year + 2))
    vac, hors = [], []
    for v, x in inst.vacations.items():
        if x.urgence or not (a0 + 14 <= x.jour <= a1 - 7):
            continue
        r = fin.get(v, 0.0) / max(1, inst.capacite_programme(v))
        (vac if calendrier.type_nuit(don.date(x.jour), fer, ponts) == "vacances" else hors).append(r)
    kappa = float(np.mean(vac) / np.mean(hors)) if vac and hors and np.mean(hors) > 0 else 1.0
    # alpha(ℓ) : part de l'écart au remplissage habituel qui sera encore comblée ℓ jours avant
    R = J.r
    num, den = np.zeros(lmax + 1), np.zeros(lmax + 1)
    for v, x in inst.vacations.items():
        if x.urgence or not (a0 + 14 <= x.jour <= a1 - 7):
            continue
        cap = inst.capacite_programme(v)
        F = R.f.get(x.med_id, 0.0) * (kappa if calendrier.type_nuit(don.date(x.jour), fer, ponts) == "vacances" else 1.0)
        for l in range(8, lmax + 1):
            booked = sum(m for lead, m in par_v.get(v, ()) if lead >= l)
            num[l] += fin.get(v, 0.0) - booked
            den[l] += max(0.0, F * cap - booked)
    alpha = np.where(den > 0, num / np.maximum(den, 1e-9), 1.0)
    alpha[:8] = 0.0
    return dict(B=B, kappa=kappa, alpha=alpha, n_vac=len(fin))


class RegleHybrideV2(RegleHybrideV1):
    """Règle hybride avec fantômes « prévisions » :

    - chaque jour, pour chaque vacation encore ouverte (au-delà de J−7, à moins de H jours), la
      charge encore attendue est PRÉVUE :
          reste_v = min(capacité − réservé,  F_m · κ_v · (1 − B(ℓ_v)) · capacité)
      F_m = remplissage final habituel du chirurgien, κ_v = facteur des vacances scolaires,
      B(ℓ) = part des réservations déjà faites ℓ jours avant la vacation ;
      elle est répartie entre ambulatoire, séjour court et long selon le chirurgien, puis
      convertie en lits par son profil (nuit par nuit) ;
    - les fantômes sont recalculés chaque jour : un fantôme qui n'est pas remplacé par un vrai
      patient diminue à mesure que la date approche et disparaît quand la journée est close ;
      il ne bloque jamais de place ;
    - fantômes d'urgence : lits attendus des urgences futures (taux par jour de semaine × profil
      de lits) ; chaque jour, les urgences réellement arrivées remplacent leur prévision.
    """

    def __init__(self, don, df=None, lignes_calage=None, urgences: bool = True, calage=None,
                 journal: bool = False, modele: str = "ecart", adaptatif: bool = True,
                 saison: bool = True, memoire: int = 56, **kw):
        kw.setdefault("delta", 14)
        kw["fantomes"] = "uniformes"
        super().__init__(don, **kw)
        self.mode = "previsions"
        self.nom = f"hybride V2 (Δ={self.delta} j, fantômes prévisions{', urgences' if urgences else ''})"
        if calage is None:
            calage = apprendre_courbe_reservation(don, lignes_calage)
        self.B, self.kappa = np.asarray(calage["B"]), float(calage["kappa"])
        if df is not None and saison:
            self.kappa = apprendre_saison(df)
        self.adaptatif, self.memoire = adaptatif, memoire
        self.alpha = np.asarray(calage.get("alpha", np.ones(len(self.B))))
        self.modele = modele
        self.urg = apprendre_urgences(df, don) if (urgences and df is not None) else None
        self.journal = journal
        self.snap = {}             # t -> (indices de nuits, courbe attendue) si journal

    def _init_instance(self, sol):
        super()._init_instance(sol)
        import calendrier
        inst = self.inst
        fer, ponts = calendrier.feries_et_ponts(range(inst.jour_zero.year - 1, inst.jour_zero.year + 3))
        self.vac = {v: (calendrier.type_nuit(inst.jour_zero + __import__("datetime").timedelta(x.jour), fer, ponts)
                        == "vacances") for v, x in inst.vacations.items()}
        self.Er = np.zeros(len(self.E))          # lits des patients réellement datés
        self.Eg = np.zeros(len(self.E))          # lits fantômes programmés
        self.Eu = np.zeros(len(self.E))          # lits attendus des urgences
        self._urg_jour = defaultdict(list)
        if self.urg is not None:
            don = self.don
            for p, j in getattr(don, "jour_urgent", {}).items():
                self._urg_jour[j].append(don.urgents[p])
            lam, offs, val = self.urg["lam"], self.urg["offs"], self.urg["val"]
            for d in range(len(self.E)):
                w = (inst.jour_zero.weekday() + d) % 7
                idx = d + offs
                ok = (idx >= 0) & (idx < len(self.E))
                self.Eu[idx[ok]] += lam[w] * val[ok]

    def _maj_remplissage(self, sol, t):
        """F_m = remplissage final moyen des vacations de m closes dans les `memoire` derniers
        jours, ramené hors vacances (÷ κ pour une vacation de vacances), mélangé à l'a priori
        appris (poids de 4 vacations) quand l'historique récent est court."""
        inst = self.inst
        self.F = {}
        for m, vids in self.vac_med.items():
            obs = []
            for vid in vids:
                j = inst.vacations[vid].jour
                if j > t - 7:
                    break
                if j < t - 7 - self.memoire:
                    continue
                cap = inst.capacite_programme(vid)
                if cap:
                    obs.append(_minutes(inst, sol, vid) / cap / (self.kappa if self.vac[vid] else 1.0))
            prior = self.f.get(m, 0.0)
            self.F[m] = min(self.fmax, (sum(obs) + 4 * prior) / (len(obs) + 4))

    def _maj_E(self):
        self.E = self.Er + self.Eg + self.Eu

    def preparer(self, sol, t):
        if not self.pret:
            self._init_instance(sol)
        inst = self.inst
        self.t = t
        # urgences : pour chaque jour écoulé depuis le dernier passage (week-ends compris), la
        # prévision est remplacée par les urgences réellement arrivées
        if self.urg is not None:
            lam, offs, val = self.urg["lam"], self.urg["offs"], self.urg["val"]
            for d in range(getattr(self, "_urg_fait", -1) + 1, t + 1):
                w = (inst.jour_zero.weekday() + d) % 7
                idx = d + offs
                ok = (idx >= 0) & (idx < len(self.E))
                self.Eu[idx[ok]] -= lam[w] * val[ok]
                for p in self._urg_jour.get(d, ()):
                    deb = d - p.nuits_avant
                    for k in range(max(0, deb), min(deb + p.nb_nuits, len(self.Eu))):
                        self.Eu[k] += 1
            self._urg_fait = t
        # remplissage habituel de chaque chirurgien, réestimé sur ses vacations closes récentes
        if self.adaptatif:
            self._maj_remplissage(sol, t)
        # lits des patients réellement datés : ceux de la Solution (déplacements par les aléas compris)
        n = min(len(self.Er), len(sol.lits_jour))
        self.Er[:] = 0
        self.Er[:n] = np.asarray(sol.lits_jour[:n], dtype=float)
        # fantômes programmés recalculés chaque jour
        self.Eg[:] = 0
        self.r, self.dans = {}, set()
        for m, vids in self.vac_med.items():
            F = self.F.get(m, self.f.get(m, 0.0)) if self.adaptatif else self.f.get(m, 0.0)
            for vid in vids:
                j = inst.vacations[vid].jour
                if j <= t + 7:
                    continue
                if j > t + self.H:
                    break
                cap = inst.capacite_programme(vid)
                lead = min(j - t, len(self.B) - 1)
                deja = _minutes(inst, sol, vid)
                Fv = F * (self.kappa if self.vac[vid] else 1.0)
                if self.modele == "ecart":       # part de l'écart au remplissage habituel encore comblée
                    attendu = self.alpha[lead] * max(0.0, Fv * cap - deja)
                else:                            # arrivées attendues d'après la courbe de réservation
                    attendu = Fv * (1.0 - self.B[lead]) * cap
                reste = max(0.0, min(cap - deja, attendu))
                if reste <= 0:
                    continue
                r = reste * self.part[m]
                self.r[vid] = r
                self.dans.add(vid)
                v = inst.vacations[vid]
                for c in range(3):
                    offs, val = self.K[(v.med_id, c)]
                    if len(offs) and r[c]:
                        idx = j + offs
                        ok = (idx >= 0) & (idx < len(self.Eg))
                        np.add.at(self.Eg, idx[ok], r[c] * val[ok])
        self._maj_E()
        if self.journal:
            a, b = t + 1, min(len(self.E), t + 64)
            self.snap[t] = (a, self.E[a:b].copy(), self.Er[a:b].copy())

    def lits_patient(self, pid, jour, signe=1):
        for k in self.inst.nuits_occupees(pid, jour):
            if k < len(self.E):
                self.E[k] += signe
                self.Er[k] += signe

    def _retirer_fantomes(self, vid):
        super()._retirer_fantomes(vid)
        self.Eg = self.E - self.Er - self.Eu

    def consulter(self, sol, pid, t, figes):
        vid = super().consulter(sol, pid, t, figes)
        if vid is not None:
            j = self.inst.vacations[vid].jour
            for k in self.inst.nuits_occupees(pid, j):
                if k < len(self.Er):
                    self.Er[k] += 1
            # super() a mis à jour self.E (patient + retrait de fantômes) ; Eg suit
            self.Eg = self.E - self.Er - self.Eu
        return vid


# -------------------------------------------------------------------------------------
# Fabrique : la règle hybride du rapport V3 (fantômes « prévisions » + urgences) par défaut.
#   fantomes = "previsions" (ou "uniformes", ancien nom) : V2 avec fantômes d'urgence
#              "previsions_sans_urgences"                : V2 sans fantômes d'urgence
#              "v1"                                      : fantômes V1 (remplissage figé)
#              "aucun"                                   : aucun fantôme
# -------------------------------------------------------------------------------------
_CACHE = {}


def _df():
    if "df" not in _CACHE:
        from protocole import charger
        _CACHE["df"] = charger()[0]
    return _CACHE["df"]


def calage_fantomes(don=None):
    """Courbe de réservation, facteur α(ℓ) et κ, appris une fois (grille actuelle, janvier-avril)."""
    if "calage" in _CACHE:
        return _CACHE["calage"]
    import json
    from protocole import RACINE, charger, grille_actuelle_neutre, Reglages
    f = RACINE / "resultats" / "calage_fantomes.json"
    if f.exists():
        c = json.loads(f.read_text())
    else:
        _, d = charger()
        c = apprendre_courbe_reservation(d, grille_actuelle_neutre(d), Reglages())
        c = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in c.items()}
        f.write_text(json.dumps(c))
    _CACHE["calage"] = c
    return c


def RegleHybride(don, delta: int = 14, horizon: int = 63, fantomes: str = "previsions", **kw):
    if fantomes in ("v1", "optimises"):
        return RegleHybrideV1(don, delta=delta, horizon=horizon,
                              fantomes="uniformes" if fantomes == "v1" else "optimises", **kw)
    if fantomes == "aucun":
        return RegleHybrideV1(don, delta=delta, horizon=horizon, fantomes="aucun", **kw)
    urg = fantomes != "previsions_sans_urgences"
    calage = kw.pop("calage", None) or calage_fantomes(don)
    return RegleHybrideV2(don, df=_df(), calage=calage, delta=delta, horizon=horizon,
                          urgences=urg, **kw)


def calage_fantomes_ml():
    """Même calage, appris avec les durées et marges prévues par apprentissage (rapport V5)."""
    import json
    from protocole import RACINE
    return json.loads((RACINE / "resultats" / "calage_fantomes_ml.json").read_text())
