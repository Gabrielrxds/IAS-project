r"""
planning_vacations.py — Construire la grille des vacations À PARTIR DE ZÉRO.

Différence avec grille_tabou.py : là-bas on réattribuait les blocs existants, horaires
compris. Ici on efface la grille. On découpe le bloc en créneaux élémentaires et on
décide, pour chacun, s'il est ouvert et à qui il appartient. Le planning obtenu est un
cycle de 4 semaines répété toute l'année, sans modification.

DÉCOUPAGE
    créneau = (semaine du cycle w ∈ 0..3, jour d ∈ lun..ven, salle r ∈ {2,3,4,5},
               demi-journée : matin 8h–13h (5 h) ou après-midi 13h30–17h30 (4 h))
    soit 160 créneaux par cycle. Ceux qui chevauchent une plage réservée de la grille
    actuelle (urgences, vacataires) sont retirés : ces plages sont conservées telles
    quelles. Budget d'heures : le planning n'a pas le droit d'ouvrir plus d'heures de
    salle que la grille actuelle n'en donne aux praticiens (594 h par cycle). À moyens
    égaux. Chaque praticien garde au moins une demi-journée par cycle.

BESOIN (heures par semaine)
    Pour chaque intervention de l'historique : durée estimée + marge de risque + temps
    inter-salle, comme les simulations les consomment. Somme par praticien et par
    semaine calendaire, puis moyenne, médiane, P80. Le besoin retenu est l'ENVELOPPE
    HAUTE des moyennes (2019, 2020, 2021, dernier trimestre 2021) : l'expérience de
    grille_tabou.py a montré qu'il faut se tromper vers le haut.

OBJECTIF
    F = Σ_m (manque_m / 4 h)² + γ Σ_m (excès_m / 4 h)²       adéquation au besoin
      + w_ℓ · CV²(lits attendus sur les 28 jours du cycle)     lissage des lits
      + w_r · Σ_m irrégularité_m                                créneaux habituels
    irrégularité_m = nombre de (jour, demi-journée) distincts utilisés par m dans le
    cycle, moins le minimum possible : un médecin qui opère « tous les mardis matin »
    vaut mieux qu'un médecin éparpillé sur quatre jours différents.

    Le profil de lits attendu : chaque heure opérée par m le jour c occupe π_m(k) lits
    k jours plus tard, π_m étant mesuré sur l'historique du praticien (sorties réelles).
    Le week-end compte : un praticien à séjours longs placé le vendredi remplit samedi
    et dimanche, placé le lundi il libère ses lits avant le week-end.

MÉTAHEURISTIQUE HYBRIDE
    1. GRASP : construction gloutonne randomisée, plusieurs départs différents.
    2. Tabou à OSCILLATION STRATÉGIQUE : le budget d'heures peut être dépassé d'un
       créneau pendant la recherche, avec une pénalité λ qui s'adapte (elle monte si la
       recherche reste hors budget, descend si elle reste dedans). On traverse ainsi
       des zones interdites pour atteindre des solutions inaccessibles autrement.
    3. PATH RELINKING entre les meilleures solutions : on transforme une solution élite
       en une autre créneau par créneau, et on garde le meilleur point du chemin.
    4. SÉLECTION PAR SIMULATION (simheuristique) : les meilleures solutions élites sont
       départagées par une vraie simulation d'un an, pas par le modèle analytique.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from donnees import ALIAS, GRILLE, RESERVES, MotifVacation

CYCLE = 28
SALLES = (2, 3, 4, 5)
DEMI = {0: (8.0, 13.0), 1: (13.5, 17.5)}          # matin, après-midi
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]
FERME = None


# ---------------------------------------------------------------------------
# 1. Découpage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Creneau:
    id: int
    semaine: int
    jour: int
    salle: int
    demi: int                 # 0 matin, 1 après-midi

    @property
    def debut(self): return DEMI[self.demi][0]

    @property
    def fin(self): return DEMI[self.demi][1]

    @property
    def heures(self): return self.fin - self.debut

    @property
    def jour_cycle(self): return self.semaine * 7 + self.jour

    @property
    def instant(self):
        """Clé de simultanéité : un praticien n'est qu'à un endroit à la fois."""
        return (self.semaine, self.jour, self.demi)

    @property
    def motif(self):
        """Clé de régularité : même jour de la semaine, même demi-journée."""
        return (self.jour, self.demi)


def reserves_actuels(grille=GRILLE) -> list[MotifVacation]:
    return [m for m in grille if ALIAS.get(m.praticien, m.praticien) in RESERVES
            and m.praticien != "LIBRE"]


def decouper(grille=GRILLE) -> list[Creneau]:
    """Tous les créneaux libres de réservation."""
    res = reserves_actuels(grille)
    out = []
    for w in range(4):
        for d in range(5):
            for r in SALLES:
                for h, (a, b) in DEMI.items():
                    pris = any(m.jour == d and m.salle == r and w in m.semaines
                               and m.debut < b and a < m.fin for m in res)
                    if not pris:
                        out.append(Creneau(len(out), w, d, r, h))
    return out


def heures_actuelles(grille=GRILLE) -> dict[str, float]:
    """Heures par cycle que la grille actuelle donne à chaque praticien."""
    h = defaultdict(float)
    for m in grille:
        pr = ALIAS.get(m.praticien, m.praticien)
        if pr not in RESERVES:
            h[pr] += (m.fin - m.debut) * len(m.semaines)
    return dict(h)


# ---------------------------------------------------------------------------
# 2. Besoin hebdomadaire et profil de lits
# ---------------------------------------------------------------------------


@dataclass
class Besoin:
    heures_semaine: dict[str, float]           # besoin retenu (h / semaine)
    stats: dict[str, dict]                     # moyenne, médiane, P80 par période
    profil_lits: dict[str, np.ndarray]         # π_m(k), lits par heure opérée
    source: str = ""
    scenarios: dict[str, np.ndarray] | None = None   # h / semaine, un tirage par scénario
    volatilite: dict | None = None


def series_hebdo(df, estimateur, praticiens, debut, fin, tis=15):
    """Heures nécessaires par praticien et par semaine calendaire (lundi), sur [debut, fin[.
    Heures = durée estimée + marge de risque + TIS, comme dans les simulations."""
    import pandas as pd
    from estimation import estimer
    d0, d1 = pd.Timestamp(debut), pd.Timestamp(fin)
    sel = df[(df["date_inter"] >= d0) & (df["date_inter"] < d1)]
    n_sem = max(1, (d1 - d0).days // 7)
    S = {m: np.zeros(n_sem) for m in praticiens}
    lits = {m: np.zeros(CYCLE) for m in praticiens}
    heures = defaultdict(float)
    cache = {}
    for l in sel.itertuples(index=False):
        m = l.praticien
        if m not in S:
            continue
        cle = (m, l.acte)
        if cle not in cache:
            e = estimer(estimateur, m, l.acte, facteur_mutualisation=0.45)
            cache[cle] = (max(5, e.duree) + e.marge + tis) / 60
        h = cache[cle]
        s = (l.date_inter - d0).days // 7
        if 0 <= s < n_sem:
            S[m][s] += h
        heures[m] += h
        for k in range(min(int(l.nuits), CYCLE)):
            lits[m][k] += 1
    return S, lits, heures


def estimer_besoin(df, estimateur, praticiens, oracle=False, scenarios=0,
                   graine=0) -> Besoin:
    """Besoin hebdomadaire par praticien.

    Par défaut : enveloppe haute des moyennes 2019, 2020, 2021 et T4 2021 (aucune donnée
    de 2022 n'est lue). `oracle=True` : moyenne réelle de 2022, pour mesurer le plafond.
    Le profil de lits π_m est mesuré sur 2019–2021 dans les deux cas."""
    periodes = {"2019": ("2019-01-07", "2020-01-06"), "2020": ("2020-01-06", "2021-01-04"),
                "2021": ("2021-01-04", "2022-01-03"), "T4 2021": ("2021-10-04", "2022-01-03"),
                "2022": ("2022-01-03", "2023-01-02")}
    stats = defaultdict(dict)
    lits_tot = {m: np.zeros(CYCLE) for m in praticiens}
    h_tot = defaultdict(float)
    moy = {}
    for nom, (a, b) in periodes.items():
        S, lits, heures = series_hebdo(df, estimateur, praticiens, a, b)
        moy[nom] = {m: float(S[m].mean()) for m in praticiens}
        for m in praticiens:
            x = S[m]
            stats[m][nom] = dict(moyenne=round(float(x.mean()), 2),
                                 mediane=round(float(np.median(x)), 2),
                                 p80=round(float(np.percentile(x, 80)), 2),
                                 semaines_actives=int((x > 0).sum()), semaines=len(x))
            if nom in ("2019", "2020", "2021"):
                lits_tot[m] += lits[m]
                h_tot[m] += heures[m]
    if oracle:
        besoin = moy["2022"]; src = "moyenne réelle 2022 (oracle)"
    else:
        besoin = {m: max(moy[p][m] for p in ("2019", "2020", "2021", "T4 2021"))
                  for m in praticiens}
        src = "enveloppe haute 2019, 2020, 2021, T4 2021"
    pi = {m: lits_tot[m] / h_tot[m] if h_tot[m] else np.zeros(CYCLE) for m in praticiens}
    if not scenarios or oracle:
        return Besoin(besoin, dict(stats), pi, src)

    # --- OPTIMISATION PAR SCÉNARIOS -------------------------------------------------
    # Base : moyenne de 2021 et du dernier trimestre 2021 (le plus récent).
    # Variation d'une année sur l'autre : log(besoin suivant / besoin précédent), mesurée
    # sur les paires 2019→2020, 2020→2021, 2021→T4 2021 de tous les praticiens.
    # Plus un praticien opère peu, plus il varie en proportion (effet de petit nombre,
    # arrivées, départs) : on ajuste  Var(log r) ≈ a + b / volume  par moindres carrés.
    paires = []
    for m in praticiens:
        for u, v in (("2019", "2020"), ("2020", "2021"), ("2021", "T4 2021")):
            x, y = moy[u][m], moy[v][m]
            if x >= 0.3:
                paires.append((x, math.log(max(y, 0.05) / x)))
    X = np.array([[1.0, 1.0 / x] for x, _ in paires]); lr = np.array([l for _, l in paires])
    mu = float(np.median(lr))
    coef, *_ = np.linalg.lstsq(X, (lr - mu) ** 2, rcond=None)
    a, b = max(0.0, float(coef[0])), max(0.0, float(coef[1]))
    alea = np.random.default_rng(graine)
    sc, sig_m = {}, {}
    for m in praticiens:
        base = 0.5 * (moy["2021"][m] + moy["T4 2021"][m])
        # garde-fous : volume plancher de 1 h/semaine pour la volatilité, σ ≤ 1, et aucun
        # scénario au-delà de 4 fois la base (sinon un tirage extrême sur un petit
        # praticien écrase tout le reste, le manque étant pénalisé au carré)
        sig_m[m] = min(1.0, math.sqrt(a + b / max(base, 1.0)))
        sc[m] = (np.zeros(scenarios) if base <= 0
                 else np.minimum(4 * base,
                                 base * np.exp(mu + sig_m[m] * alea.standard_normal(scenarios))))
    besoin = {m: float(sc[m].mean()) for m in praticiens}
    return Besoin(besoin, dict(stats), pi,
                  f"{scenarios} scénarios, base 2021 et T4 2021, volatilité selon le volume",
                  scenarios=sc, volatilite=dict(a=a, b=b, mu=mu, n_paires=len(paires), sigma=sig_m))


# ---------------------------------------------------------------------------
# 3. Évaluation incrémentale
# ---------------------------------------------------------------------------


@dataclass
class Parametres:
    rho: float = 0.85
    rho_min: float = 0.50
    gamma: float = 0.1
    w_lits: float = 20.0
    w_reg: float = 0.3
    max_demi_semaine: int = 8          # au plus 8 demi-journées par semaine et par praticien
    budget_heures: float = 594.0       # heures de salle par cycle, comme la grille actuelle
    # recherche
    departs: int = 6
    iterations: int = 250
    duree_tabou: int = 15
    echantillon_reloc: int = 1200
    echantillon_echange: int = 800
    elites: int = 6
    graine: int = 0


class Modele:
    """État d'un planning et coût, mis à jour par différence."""

    def __init__(self, creneaux: list[Creneau], besoin: Besoin, P: Parametres):
        self.C = creneaux
        self.P = P
        self.prat = sorted(besoin.heures_semaine)
        self.D = {m: 4 * besoin.heures_semaine[m] for m in self.prat}     # par cycle
        # scénarios de besoin (par cycle) : l'adéquation devient une MOYENNE sur scénarios
        self.Dsc = ({m: 4 * np.asarray(besoin.scenarios[m], dtype=float) for m in self.prat}
                    if besoin.scenarios else None)
        self.decale = {m: np.stack([np.roll(besoin.profil_lits[m], c) for c in range(CYCLE)])
                       for m in self.prat}

    # -- coûts élémentaires --------------------------------------------------

    def terme_adequation(self, m, T):
        P = self.P
        if self.Dsc is not None:
            d = self.Dsc[m]
            ma = np.maximum(0.0, d - P.rho * T)
            ex = np.maximum(0.0, P.rho_min * T - d)
            return float(np.mean((ma / 4) ** 2 + P.gamma * (ex / 4) ** 2))
        d = self.D[m]
        ma = max(0.0, d - P.rho * T)
        ex = max(0.0, P.rho_min * T - d)
        return (ma / 4) ** 2 + P.gamma * (ex / 4) ** 2

    @staticmethod
    def irregularite(motifs: Counter, n: int) -> float:
        if n == 0:
            return 0.0
        return len(motifs) - math.ceil(n / 4)

    def f(self, m, T):
        return min(1.0, self.D[m] / T) if T > 0 else 0.0

    # -- état complet ---------------------------------------------------------

    def etat(self, x):
        """x[id créneau] = praticien ou None. Retourne un état complet pour la recherche."""
        T = defaultdict(float); V = {m: np.zeros(CYCLE) for m in self.prat}
        motifs = {m: Counter() for m in self.prat}; n = Counter()
        occ = {m: Counter() for m in self.prat}; parsem = {m: Counter() for m in self.prat}
        for c in self.C:
            m = x[c.id]
            if m is None:
                continue
            T[m] += c.heures; V[m] += c.heures * self.decale[m][c.jour_cycle]
            motifs[m][c.motif] += 1; n[m] += 1; occ[m][c.instant] += 1; parsem[m][c.semaine] += 1
        A = {m: self.terme_adequation(m, T[m]) for m in self.prat}
        R = {m: self.irregularite(motifs[m], n[m]) for m in self.prat}
        B = sum(self.f(m, T[m]) * V[m] for m in self.prat)
        return dict(x=list(x), T=T, V=V, motifs=motifs, n=n, occ=occ, parsem=parsem,
                    A=A, R=R, B=B, heures=sum(T.values()))

    def cout(self, e, lam=0.0):
        """F, plus la pénalité d'oscillation λ · dépassement du budget."""
        P = self.P
        moy = e["B"].mean()
        cv2 = float(e["B"].var() / moy ** 2) if moy > 0 else 0.0
        F = sum(e["A"].values()) + P.w_lits * cv2 + P.w_reg * sum(e["R"].values())
        return F + lam * max(0.0, e["heures"] - P.budget_heures)

    def detail(self, e):
        moy = e["B"].mean()
        return dict(adequation=sum(e["A"].values()),
                    cv2_lits=float(e["B"].var() / moy ** 2) if moy > 0 else 0.0,
                    irregularite=sum(e["R"].values()), heures=e["heures"],
                    lits_attendus=e["B"].tolist(),
                    praticiens={m: dict(besoin=self.D[m], offre=e["T"][m],
                                        occupation=self.D[m] / e["T"][m] if e["T"][m] else None,
                                        demi_journees=e["n"][m], motifs=len(e["motifs"][m]))
                                for m in self.prat})

    # -- mouvement : liste de (créneau, ancien, nouveau) -----------------------

    def admissible(self, e, chg):
        """Contraintes dures après application de chg (sans l'appliquer)."""
        P = self.P
        ajout = defaultdict(list); retrait = defaultdict(list)
        for cid, a, b in chg:
            c = self.C[cid]
            if a is not None: retrait[a].append(c)
            if b is not None: ajout[b].append(c)
        for m, cs in ajout.items():
            libere = {c.instant for c in retrait.get(m, [])}
            vus = set()
            for c in cs:
                if (e["occ"][m][c.instant] > 0 and c.instant not in libere) or c.instant in vus:
                    return False                                 # deux salles à la fois
                vus.add(c.instant)
            sem = Counter(e["parsem"][m])
            for c in cs: sem[c.semaine] += 1
            for c in retrait.get(m, []): sem[c.semaine] -= 1
            if max(sem.values()) > P.max_demi_semaine:
                return False
        for m, cs in retrait.items():                        # chacun garde au moins 1 créneau
            if e["n"][m] - len(cs) + len(ajout.get(m, [])) < 1:
                return False
        return True

    def evaluer(self, e, chg, lam):
        """Coût après chg, sans modifier e. Ne recalcule que les praticiens touchés."""
        P = self.P
        tou = {m for _, a, b in chg for m in (a, b) if m is not None}
        T = {m: e["T"][m] for m in tou}; V = {m: e["V"][m].copy() for m in tou}
        mot = {m: Counter(e["motifs"][m]) for m in tou}; n = {m: e["n"][m] for m in tou}
        h = e["heures"]
        for cid, a, b in chg:
            c = self.C[cid]
            if a is not None:
                T[a] -= c.heures; V[a] -= c.heures * self.decale[a][c.jour_cycle]
                mot[a][c.motif] -= 1; n[a] -= 1; h -= c.heures
                if mot[a][c.motif] == 0: del mot[a][c.motif]
            if b is not None:
                T[b] += c.heures; V[b] += c.heures * self.decale[b][c.jour_cycle]
                mot[b][c.motif] += 1; n[b] += 1; h += c.heures
        A = sum(e["A"].values()) - sum(e["A"][m] for m in tou) + sum(self.terme_adequation(m, T[m]) for m in tou)
        R = sum(e["R"].values()) - sum(e["R"][m] for m in tou) + sum(self.irregularite(mot[m], n[m]) for m in tou)
        B = e["B"] - sum(self.f(m, e["T"][m]) * e["V"][m] for m in tou) + sum(self.f(m, T[m]) * V[m] for m in tou)
        moy = B.mean(); cv2 = float(B.var() / moy ** 2) if moy > 0 else 0.0
        F = A + P.w_lits * cv2 + P.w_reg * R
        return F + lam * max(0.0, h - P.budget_heures), h

    def appliquer(self, e, chg):
        for cid, a, b in chg:
            c = self.C[cid]
            if a is not None:
                e["B"] -= self.f(a, e["T"][a]) * e["V"][a]
                e["T"][a] -= c.heures; e["V"][a] -= c.heures * self.decale[a][c.jour_cycle]
                e["motifs"][a][c.motif] -= 1
                if e["motifs"][a][c.motif] == 0: del e["motifs"][a][c.motif]
                e["n"][a] -= 1; e["occ"][a][c.instant] -= 1; e["parsem"][a][c.semaine] -= 1
                e["heures"] -= c.heures
                e["B"] += self.f(a, e["T"][a]) * e["V"][a]
                e["A"][a] = self.terme_adequation(a, e["T"][a])
                e["R"][a] = self.irregularite(e["motifs"][a], e["n"][a])
            if b is not None:
                e["B"] -= self.f(b, e["T"][b]) * e["V"][b]
                e["T"][b] += c.heures; e["V"][b] += c.heures * self.decale[b][c.jour_cycle]
                e["motifs"][b][c.motif] += 1; e["n"][b] += 1
                e["occ"][b][c.instant] += 1; e["parsem"][b][c.semaine] += 1
                e["heures"] += c.heures
                e["B"] += self.f(b, e["T"][b]) * e["V"][b]
                e["A"][b] = self.terme_adequation(b, e["T"][b])
                e["R"][b] = self.irregularite(e["motifs"][b], e["n"][b])
            e["x"][cid] = b


# ---------------------------------------------------------------------------
# 4. La métaheuristique hybride
# ---------------------------------------------------------------------------


@dataclass
class Journal:
    departs: list[dict] = field(default_factory=list)
    relinking: list[dict] = field(default_factory=list)
    elites: list[float] = field(default_factory=list)
    lambda_trace: list[float] = field(default_factory=list)


class Planificateur:

    def __init__(self, creneaux, besoin, P: Parametres | None = None):
        self.P = P or Parametres()
        self.M = Modele(creneaux, besoin, self.P)
        self.C = creneaux
        self.alea = random.Random(self.P.graine)
        self.journal = Journal()

    # -- 1. GRASP -------------------------------------------------------------

    def grasp(self, alpha=3):
        """Construction gloutonne randomisée. À chaque pas : un praticien parmi les
        `alpha` plus en manque, puis un créneau parmi les `alpha` qui augmentent le moins
        le coût. Le hasard contrôlé donne des départs différents et bons."""
        M, P = self.M, self.P
        e = M.etat([None] * len(self.C))
        # amorçage : chaque praticien reçoit d'abord un créneau (contrainte « au moins un »)
        for m in sorted(M.prat, key=lambda m: -M.D[m]):
            cands = [(M.evaluer(e, [(c.id, None, m)], 0.0)[0], c.id) for c in self.C
                     if e["x"][c.id] is None and M.admissible(e, [(c.id, None, m)])]
            cands.sort()
            M.appliquer(e, [(self.alea.choice(cands[:alpha])[1], None, m)])
        bloques = set()                       # praticiens qui ne peuvent plus rien recevoir
        while True:
            manques = sorted(((M.D[m] - P.rho * e["T"][m], m) for m in M.prat
                              if m not in bloques), reverse=True)
            manques = [(v, m) for v, m in manques if v > 0]
            if not manques:
                break
            m = self.alea.choice(manques[:alpha])[1]
            cands = []
            for c in self.C:
                if e["x"][c.id] is not None or e["heures"] + c.heures > P.budget_heures + 1e-9:
                    continue
                chg = [(c.id, None, m)]
                if not M.admissible(e, chg):
                    continue
                cands.append((M.evaluer(e, chg, 0.0)[0], c.id))
            if not cands:
                bloques.add(m)
                continue
            cands.sort()
            _, cid = self.alea.choice(cands[:alpha])
            M.appliquer(e, [(cid, None, m)])
        return e

    # -- 2. Tabou à oscillation stratégique ---------------------------------------

    def voisinage(self, e):
        M, P, C = self.M, self.P, self.C
        prat = M.prat
        ouverts = [c.id for c in C if e["x"][c.id] is not None]
        fermes = [c.id for c in C if e["x"][c.id] is None]
        # transferts et ouvertures : complet
        for c in C:
            a = e["x"][c.id]
            for b in prat:
                if b != a:
                    yield ("T", [(c.id, a, b)])
            if a is not None:
                yield ("F", [(c.id, a, None)])                 # fermer
        # relocalisations : un praticien quitte un créneau pour un créneau fermé
        for _ in range(P.echantillon_reloc):
            if not ouverts or not fermes:
                break
            s, t = self.alea.choice(ouverts), self.alea.choice(fermes)
            a = e["x"][s]
            yield ("R", [(s, a, None), (t, None, a)])
        # échanges entre deux praticiens
        for _ in range(P.echantillon_echange):
            if len(ouverts) < 2:
                break
            s, t = self.alea.sample(ouverts, 2)
            a, b = e["x"][s], e["x"][t]
            if a != b:
                yield ("E", [(s, a, b), (t, b, a)])

    def tabou(self, e, iterations=None):
        """Tabou avec oscillation stratégique sur le budget d'heures.
        Le meilleur n'est enregistré que parmi les solutions DANS le budget."""
        M, P = self.M, self.P
        iterations = iterations or P.iterations
        lam, hist_faisable = 2.0, []
        tabou: dict[tuple, int] = {}
        best, best_x = (M.cout(e) if e["heures"] <= P.budget_heures + 1e-9 else math.inf), list(e["x"])
        for it in range(1, iterations + 1):
            choisi = None
            for typ, chg in self.voisinage(e):
                if not M.admissible(e, chg):
                    continue
                F, h = M.evaluer(e, chg, lam)
                if h > P.budget_heures + 5.0 + 1e-9:
                    continue                                      # oscillation bornée à ~1 créneau
                interdit = any(tabou.get((cid, b), 0) > it for cid, _, b in chg if b is not None)
                faisable = h <= P.budget_heures + 1e-9
                Fvrai = F if faisable else math.inf
                if interdit and not Fvrai < best - 1e-12:         # aspiration
                    continue
                if choisi is None or F < choisi[0]:
                    choisi = (F, chg, h)
            if choisi is None:
                break
            F, chg, h = choisi
            M.appliquer(e, chg)
            for cid, a, _ in chg:
                if a is not None:
                    tabou[(cid, a)] = it + P.duree_tabou      # ne pas rendre ce créneau à a
            faisable = e["heures"] <= P.budget_heures + 1e-9
            if faisable:
                Fv = M.cout(e)
                if Fv < best - 1e-12:
                    best, best_x = Fv, list(e["x"])
            # oscillation : λ s'adapte à la proportion de temps passé hors budget
            hist_faisable.append(faisable)
            if len(hist_faisable) >= 10:
                recent = hist_faisable[-10:]
                if all(recent):
                    lam = max(0.2, lam / 1.5)
                elif not any(recent):
                    lam = min(200.0, lam * 1.5)
            self.journal.lambda_trace.append(lam)
        return best, best_x

    # -- 3. Path relinking ------------------------------------------------------

    def relinking(self, xa, xb):
        """De xa vers xb, un créneau à la fois : à chaque pas, la différence dont
        l'application coûte le moins. On garde le meilleur point FAISABLE du chemin."""
        M, P = self.M, self.P
        e = M.etat(xa)
        diff = [c.id for c in self.C if xa[c.id] != xb[c.id]]
        best, best_x = M.cout(e), list(e["x"])
        while diff:
            cands = []
            for cid in diff:
                chg = [(cid, e["x"][cid], xb[cid])]
                if not M.admissible(e, chg):
                    continue
                F, h = M.evaluer(e, chg, 0.0)
                if h > P.budget_heures + 5.0:
                    continue
                cands.append((F, cid, h))
            if not cands:
                break
            F, cid, h = min(cands)
            M.appliquer(e, [(cid, e["x"][cid], xb[cid])])
            diff.remove(cid)
            if e["heures"] <= P.budget_heures + 1e-9 and M.cout(e) < best - 1e-12:
                best, best_x = M.cout(e), list(e["x"])
        return best, best_x

    # -- orchestration -------------------------------------------------------

    def resoudre(self, verbeux=False):
        M, P = self.M, self.P
        elites = []                                   # (coût, x)
        for k in range(P.departs):
            e = self.grasp()
            F0 = M.cout(e)
            F, x = self.tabou(e)
            self.journal.departs.append(dict(depart=k, grasp=F0, tabou=F))
            elites.append((F, x))
            if verbeux:
                print(f"    départ {k}: GRASP {F0:8.3f} -> tabou {F:8.3f}", flush=True)
        elites.sort(key=lambda t: t[0])
        # path relinking entre les élites deux à deux, puis court tabou sur le résultat
        pool = elites[:P.elites]
        nouveaux = []
        for i in range(len(pool)):
            for j in range(len(pool)):
                if i == j:
                    continue
                F, x = self.relinking(pool[i][1], pool[j][1])
                e = M.etat(x)
                F2, x2 = self.tabou(e, iterations=40)
                nouveaux.append((F2, x2))
                self.journal.relinking.append(dict(de=i, vers=j, chemin=F, apres_tabou=F2))
        tous = sorted(elites + nouveaux, key=lambda t: t[0])
        # élites distinctes
        vus, uniques = set(), []
        for F, x in tous:
            # deux plannings qui ne diffèrent que par l'échange de salles à la même heure
            # sont identiques pour les patients et les lits : on ne les compte qu'une fois
            cle = tuple(sorted((x[c.id], c.semaine, c.jour, c.demi)
                               for c in self.C if x[c.id] is not None))
            if cle not in vus:
                vus.add(cle); uniques.append((F, x))
        self.journal.elites = [F for F, _ in uniques[:P.elites]]
        if verbeux:
            print(f"    relinking : meilleur {uniques[0][0]:.3f}", flush=True)
        return uniques[:P.elites]


# ---------------------------------------------------------------------------
# 5. Retour au format grille, pour les simulations
# ---------------------------------------------------------------------------


def planning_vers_grille(creneaux, x, grille_origine=GRILLE) -> list[MotifVacation]:
    """Créneaux attribués + plages réservées d'origine, au format de donnees.py."""
    out = list(reserves_actuels(grille_origine))
    for c in creneaux:
        m = x[c.id]
        if m is not None:
            out.append(MotifVacation(c.jour, c.salle, m, c.debut, c.fin, (c.semaine,)))
    return out
