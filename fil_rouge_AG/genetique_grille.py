r"""
genetique_grille.py — Algorithme génétique qui construit une NOUVELLE GRILLE
DE VACATIONS (niveau tactique) à partir d'une période d'apprentissage.

La grille est CYCLIQUE : elle se répète à l'identique toutes les C semaines.
Elle dit quel chirurgien a quelle salle, quel jour, le matin ou l'après-midi.
Le moteur d'insertion (analyse_couts.py) et l'AG journalier tournent ensuite
DANS cette grille, sans rien y changer.

CE QUI EST FIXÉ (jamais optimisé)
---------------------------------
Le VOLUME de chaque chirurgien. On mesure sur la période d'apprentissage le
temps ENGAGÉ par semaine (durée estimée + marge P90 + TIS, comme
`Solution.charge`), et on lui donne

    quota(s) = max(1, arrondi(minutes_hebdo(s) · C / (taux_cible · 270)))

demi-journées par cycle (270 min = demi-journée moyenne). On compte le temps
engagé et non le TROS réel parce qu'aucun dépassement n'est accepté : la
vacation doit contenir les marges P90.

CE QUI EST OPTIMISÉ
-------------------
La position des demi-journées : jour, salle, matin / après-midi. On peut
ouvrir ou fermer des salles (une salle sans demi-journée est fermée).
Les créneaux URGENCES de la grille actuelle sont conservés à l'identique
(cases interdites au programmé).

CODAGE
------
Un individu est un tableau d'entiers X[C, 5, R, 2] :
    (semaine du cycle, jour lun..ven, salle, demi-journée 0 = matin / 1 = après-midi)
    valeur = indice du chirurgien, VIDE (-1) ou INTERDIT (-2).
Même chirurgien matin + après-midi dans la même salle = une vacation de
JOURNÉE 8h00-17h30 (pause comprise, comme la grille actuelle).
Contraintes dures, vraies PAR CONSTRUCTION (les opérateurs les conservent) :
    - chaque chirurgien a exactement son quota de demi-journées ;
    - un chirurgien n'est jamais dans deux salles à la même demi-journée.

FITNESS (analytique, régime permanent — pas de simulation, donc rapide)
------------------------------------------------------------------------
Pour chaque chirurgien : sa demande par cycle est répartie sur ses vacations
au prorata de leur TVO. On en déduit, jour par jour du cycle :
    places  admissions ambulatoires attendues ;
    lits    lits occupés attendus = convolution CIRCULAIRE du temps opéré
            par le profil de séjour du chirurgien (nuits avant et après) ;
    remplissage  temps perdu / TVO, avec une perte de fin de vacation égale
            à la moitié de la durée moyenne d'un de ses actes (une JOURNÉE
            perd donc moins qu'un matin + un après-midi séparés) ;
    délai   attente moyenne jusqu'à sa prochaine vacation (Σ écart² / 2L),
            + 182 j pour la part de la demande qui ne tient pas dans le TVO ;
    salles  nombre de salles-jours ouvertes par cycle.
Les mêmes termes que `CoutTotal` (lits, places, remplissage, délai),
normalisés par leur valeur sur la GRILLE ACTUELLE (version 28), plus un coût
par salle-jour ouverte :

    coût = w_lits·L/L0 + w_places·P/P0 + w_rempl·R/R0 + w_délai·D/D0
         + w_salles·S/S0 + PENALITE_CAPACITE · (lits et places attendus au-delà
                                                 des capacités hors réserves)

OPÉRATEURS
----------
    Sélection    tournoi ; élitisme.
    Croisement   par blocs JOUR : chaque (semaine, jour) de l'enfant vient en
                 entier de l'un des deux parents, puis réparation des quotas.
    Mutations    échange de deux cases ; « aligner » (regrouper le matin et
                 l'après-midi d'un chirurgien dans la même salle) ; échange de
                 deux jours entiers de même profil de cases interdites.
    Descente     recherche locale sur le meilleur (AG mémétique).
"""

from __future__ import annotations

import datetime as dt
import random
import time
from dataclasses import dataclass, field

import numpy as np

from grille import creneaux_du
from modele import Vacation

VIDE, INTERDIT = -1, -2
MATIN = (8 * 60, 13 * 60)                 # 300 min
APREM = (13 * 60 + 30, 17 * 60 + 30)      # 240 min
JOURNEE = (8 * 60, 17 * 60 + 30)          # 570 min, pause comprise
DEMI = (MATIN, APREM)
DEMI_MOYENNE = 270
PENALITE_CAPACITE = 100.0
NOMS_JOURS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi")


# ---------------------------------------------------------------------------
# 0. Calendrier
# ---------------------------------------------------------------------------

def paques(annee: int) -> dt.date:
    """Dimanche de Pâques (algorithme de Meeus / Jones / Butcher)."""
    a, b, c = annee % 19, annee // 100, annee % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mois = (h + l - 7 * m + 114) // 31
    jour = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(annee, mois, jour)


def jours_feries(annees) -> set[dt.date]:
    """Jours fériés français (métropole) : pas de vacation ces jours-là."""
    res = set()
    for a in annees:
        p = paques(a)
        res |= {dt.date(a, 1, 1), p + dt.timedelta(days=1), dt.date(a, 5, 1),
                dt.date(a, 5, 8), p + dt.timedelta(days=39), p + dt.timedelta(days=50),
                dt.date(a, 7, 14), dt.date(a, 8, 15), dt.date(a, 11, 1),
                dt.date(a, 11, 11), dt.date(a, 12, 25)}
    return res


def semaine_du_cycle(d: dt.date, C: int) -> int:
    """Semaine du cycle d'une date : numéro de semaine ISO modulo C (même
    convention de parité que grille.py)."""
    return d.isocalendar()[1] % C


# ---------------------------------------------------------------------------
# 1. Demande de chaque chirurgien (apprise sur la période d'apprentissage)
# ---------------------------------------------------------------------------

@dataclass
class ProfilChirurgien:
    code: str
    med_id: int
    nb_patients: int
    minutes_semaine: float           # temps engagé par semaine (durée + marge + TIS)
    frag: float                      # perte de fin de vacation (min)
    ambu_par_minute: float           # admissions ambulatoires par minute engagée
    noyau_lits: dict[int, float]     # décalage / jour opératoire -> lits par minute engagée


def profils_chirurgiens(patients, med_codes: dict[int, str], nb_semaines: float,
                        tis: int) -> list[ProfilChirurgien]:
    """`patients` : Patients de la période d'apprentissage. Un chirurgien sans
    patient sur la période garde un profil minimal (quota de 1)."""
    par_med: dict[int, list] = {m: [] for m in med_codes}
    for p in patients:
        par_med.setdefault(p.med_id, []).append(p)
    profils = []
    for m, code in sorted(med_codes.items()):
        ps = par_med.get(m, [])
        engage = [p.duree_op + p.marge_perso + tis for p in ps]
        tot = float(sum(engage))
        noyau: dict[int, float] = {}
        for p in ps:
            for o in range(-p.nuits_avant, p.nb_nuits - p.nuits_avant):
                noyau[o] = noyau.get(o, 0.0) + 1.0
        profils.append(ProfilChirurgien(
            code=code, med_id=m, nb_patients=len(ps),
            minutes_semaine=tot / nb_semaines if nb_semaines else 0.0,
            frag=(tot / len(ps) / 2) if ps else 45.0,
            ambu_par_minute=(sum(p.ambulatoire for p in ps) / tot) if tot else 0.0,
            noyau_lits={o: n / tot for o, n in noyau.items()} if tot else {}))
    return profils


# ---------------------------------------------------------------------------
# 2. Le problème et son évaluateur
# ---------------------------------------------------------------------------

@dataclass
class PoidsGrille:
    """Mêmes coefficients que `CoutTotal`, plus le coût des salles ouvertes."""
    lits: float = 1.0
    places: float = 1.0
    remplissage: float = 1.0
    delai: float = 1.0
    salles: float = 1.0


def _cases_interdites(C: int, salles: list[int], annee: int = 2022) -> np.ndarray:
    """Cases occupées par un créneau URGENCES (ou « urg ») de la grille
    actuelle, pour au moins une semaine de l'année ayant cette position dans
    le cycle. Elles restent réservées aux urgences."""
    R = len(salles)
    interdit = np.zeros((C, 5, R, 2), dtype=bool)
    idx = {s: i for i, s in enumerate(salles)}
    for w in range(1, 53):
        k = w % C
        for d in range(5):
            date = dt.date.fromisocalendar(annee, w, d + 1)
            for c in creneaux_du(date, types=("urgence",)):
                if c.salle not in idx:
                    continue
                for h, (a, b) in enumerate(DEMI):
                    if c.debut < b and a < c.fin:
                        interdit[k, d, idx[c.salle], h] = True
    return interdit


class ProblemeGrille:
    """Données fixes du problème de grille + évaluation analytique.

    `profils`   : ProfilChirurgien (un par chirurgien du périmètre) ;
    `C`         : longueur du cycle en semaines ;
    `salles`    : salles utilisables (une salle sans vacation est fermée) ;
    `capacites` : (lits, places/jour) utilisables par le PROGRAMMÉ (hors réserves).
    """

    def __init__(self, profils: list[ProfilChirurgien], C: int = 4,
                 salles=(1, 2, 3, 4, 5), taux_cible: float = 0.9,
                 capacites: tuple[float, float] = (40, 16), annee_urgences: int = 2022):
        self.profils = profils
        self.C, self.salles = C, list(salles)
        self.R, self.S, self.L = len(self.salles), len(profils), 7 * C
        self.taux_cible = taux_cible
        self.cap_lits, self.cap_places = capacites
        self.interdit = _cases_interdites(C, self.salles, annee_urgences)
        self.alertes: list[str] = []

        # --- quotas -------------------------------------------------------
        libres = int((~self.interdit).sum())
        q = [max(1, round(p.minutes_semaine * C / (taux_cible * DEMI_MOYENNE)))
             for p in profils]
        q = [min(x, 10 * C) for x in q]           # jamais deux salles à la fois
        if sum(q) > libres:
            self.alertes.append(
                f"demande = {sum(q)} demi-journées par cycle pour {libres} cases "
                f"libres : quotas réduits proportionnellement (les patients en "
                f"trop attendront, cf. terme délai)")
            facteur = libres / sum(q)
            q = [max(1, int(x * facteur)) for x in q]
            while sum(q) > libres:
                q[q.index(max(q))] -= 1
        self.quotas = np.array(q, dtype=int)

        # --- tableaux pour l'évaluation -----------------------------------
        self.demande = np.array([p.minutes_semaine * C for p in profils])   # min / cycle
        self.frag = np.array([p.frag for p in profils])
        self.ambu = np.array([p.ambu_par_minute for p in profils])
        L = self.L
        # matrice circulante des lits : lits[n] = Σ_s Σ_j K[s, n, j] · opéré[s, j]
        self.K = np.zeros((self.S, L, L))
        for s, p in enumerate(profils):
            for o, val in p.noyau_lits.items():
                for j in range(L):
                    self.K[s, (j + o) % L, j] += val
        self.ouvres = np.array([j % 7 < 5 for j in range(L)])
        jc = np.arange(C)[:, None] * 7 + np.arange(5)[None, :]
        self._jc = np.broadcast_to(jc[:, :, None], (C, 5, self.R))
        self.ref: dict[str, float] | None = None

    # -- évaluation ---------------------------------------------------------

    def _tvo(self, X: np.ndarray):
        """X -> (TVO par chirurgien et jour du cycle, nb de vacations par
        chirurgien, salles-jours ouvertes)."""
        M, A = X[..., 0], X[..., 1]
        tvo = np.zeros((self.S, self.L))
        nvac = np.zeros(self.S)
        jour = M >= 0
        same = jour & (M == A)
        mo = jour & ~same
        ao = (A >= 0) & ~same
        for masque, val, Z in ((same, JOURNEE[1] - JOURNEE[0], M),
                               (mo, MATIN[1] - MATIN[0], M),
                               (ao, APREM[1] - APREM[0], A)):
            if masque.any():
                np.add.at(tvo, (Z[masque], self._jc[masque]), val)
                np.add.at(nvac, Z[masque], 1)
        salles_jours = int(((M >= 0) | (A >= 0)).sum())
        return tvo, nvac, salles_jours

    def termes(self, tvo: np.ndarray, nvac: np.ndarray, salles_jours: float) -> dict:
        """Termes BRUTS de la fitness (cf. docstring du module)."""
        TVO = tvo.sum(1)
        cap_eff = np.maximum(0.0, TVO - nvac * self.frag)
        servi = np.minimum(self.demande, cap_eff)
        f = np.divide(servi, TVO, out=np.zeros_like(TVO), where=TVO > 0)
        opere = tvo * f[:, None]                            # minutes opérées / jour
        places = (self.ambu[:, None] * opere).sum(0)
        lits = np.einsum("snj,sj->n", self.K, opere)
        po, lo = places[self.ouvres], lits[self.ouvres]

        # délai : attente jusqu'à la prochaine vacation + part non servie
        attente = np.zeros(self.S)
        for s in range(self.S):
            jours = np.flatnonzero(tvo[s] > 0)
            if len(jours) == 0:
                attente[s] = self.L
                continue
            ecarts = np.diff(np.append(jours, jours[0] + self.L))
            attente[s] = (ecarts ** 2).sum() / (2 * self.L)
        non_servi = np.divide(self.demande - servi, self.demande,
                              out=np.zeros_like(self.demande), where=self.demande > 0)
        delai_s = attente + 182 * non_servi
        poids = self.demande if self.demande.sum() > 0 else np.ones(self.S)
        return {
            "lits": float(lo.var()) / 42 ** 2,
            "places": float(po.var()) / 18 ** 2,
            "remplissage": float((TVO - servi).sum() / TVO.sum()) if TVO.sum() else 0.0,
            "delai": float((poids * delai_s).sum() / poids.sum()) / 182,
            "salles": salles_jours * 7 / self.L,             # par semaine
            "exces_capacite": float(np.maximum(0, lits - self.cap_lits).sum()
                                    + np.maximum(0, po - self.cap_places).sum()),
            "lits_moyens": float(lo.mean()), "places_moyennes": float(po.mean()),
        }

    def termes_individu(self, X: np.ndarray) -> dict:
        return self.termes(*self._tvo(X))

    def cout(self, X: np.ndarray, w: PoidsGrille) -> float:
        t = self.termes_individu(X)
        r = self.ref or {}
        return (w.lits * t["lits"] / r.get("lits", 1.0)
                + w.places * t["places"] / r.get("places", 1.0)
                + w.remplissage * t["remplissage"] / r.get("remplissage", 1.0)
                + w.delai * t["delai"] / r.get("delai", 1.0)
                + w.salles * t["salles"] / r.get("salles", 1.0)
                + PENALITE_CAPACITE * t["exces_capacite"])

    # -- référence : la grille actuelle (version 28) -----------------------

    def termes_grille_actuelle(self, annee: int = 2022) -> dict:
        """Termes de la grille actuelle (grille.py, cycle de 4 semaines),
        calculés avec le MÊME évaluateur, pour normaliser la fitness."""
        ref = ProblemeGrille.__new__(ProblemeGrille)
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
        tvo = np.zeros((self.S, 28))
        nvac = np.zeros(self.S)
        ouvertes = set()
        for w in range(1, 5):                     # une semaine par position du cycle
            k = w % 4
            for d in range(5):
                for c in creneaux_du(dt.date.fromisocalendar(annee, w, d + 1)):
                    if c.code in indice:
                        tvo[indice[c.code], 7 * k + d] += c.fin - c.debut
                        nvac[indice[c.code]] += 1
                        ouvertes.add((k, d, c.salle))
        return ref.termes(tvo, nvac, len(ouvertes))

    def calibrer(self) -> dict:
        t = self.termes_grille_actuelle()
        self.ref = {k: (t[k] if t[k] > 1e-12 else 1.0)
                    for k in ("lits", "places", "remplissage", "delai", "salles")}
        return self.ref

    # -- décodage ------------------------------------------------------------

    def derouler(self, X: np.ndarray, jour_zero: dt.date, nb_jours: int,
                 feries=frozenset(), premier_id: int = 0) -> list[Vacation]:
        """Grille -> vacations datées sur [jour_zero, jour_zero + nb_jours)."""
        vacs, vid = [], premier_id
        for j in range(nb_jours):
            d = jour_zero + dt.timedelta(days=j)
            if d.weekday() > 4 or d in feries:
                continue
            k = semaine_du_cycle(d, self.C)
            for r, salle in enumerate(self.salles):
                m, a = int(X[k, d.weekday(), r, 0]), int(X[k, d.weekday(), r, 1])
                if m >= 0 and m == a:
                    morceaux = [(m, JOURNEE)]
                else:
                    morceaux = [(x, cr) for x, cr in ((m, MATIN), (a, APREM)) if x >= 0]
                for s, (debut, fin) in morceaux:
                    p = self.profils[s]
                    vacs.append(Vacation(vid, bloc_id=salle, med_id=p.med_id, jour=j,
                                         debut=debut, fin=fin,
                                         etiquette=f"{p.code} S{salle} {d:%d/%m} (nouvelle grille)"))
                    vid += 1
        return vacs

    def en_tableau(self, X: np.ndarray):
        """Grille lisible : une ligne par (semaine du cycle, jour), une colonne
        par salle ; « JT » = journée, « JT / CL » = matin / après-midi."""
        import pandas as pd
        lignes = []
        for k in range(self.C):
            for d in range(5):
                ligne = {"semaine du cycle": k + 1, "jour": NOMS_JOURS[d]}
                for r, salle in enumerate(self.salles):
                    m, a = int(X[k, d, r, 0]), int(X[k, d, r, 1])
                    nom = lambda x: ("URG" if x == INTERDIT else "—" if x == VIDE
                                     else self.profils[x].code)
                    ligne[f"salle {salle}"] = (nom(m) + " (journée)" if m >= 0 and m == a
                                               else f"{nom(m)} / {nom(a)}")
                lignes.append(ligne)
        return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 3. L'algorithme génétique
# ---------------------------------------------------------------------------

@dataclass
class ParametresAGGrille:
    taille_population: int = 40
    nb_generations: int = 300
    p_croisement: float = 0.9
    p_mutation: float = 0.7
    taille_tournoi: int = 3
    elitisme: int = 2
    stagnation_max: int = 80
    iterations_descente: int = 1500
    p_journee: float = 0.6          # à l'initialisation : part des quotas posés en journées
    graine: int | None = 0


@dataclass
class ResultatGrille:
    X: np.ndarray
    cout: float
    termes: dict
    termes_normalises: dict
    generations: int
    duree_s: float
    convergence: list = field(default_factory=list)     # (génération, meilleur, moyen)


class AGGrille:
    def __init__(self, prob: ProblemeGrille, poids: PoidsGrille | None = None,
                 params: ParametresAGGrille | None = None):
        if prob.ref is None:
            prob.calibrer()
        self.prob = prob
        self.W = poids or PoidsGrille()
        self.P = params or ParametresAGGrille()
        self.rng = random.Random(self.P.graine)
        self._libres = [tuple(c) for c in np.argwhere(~prob.interdit)]

    # -- outils ----------------------------------------------------------------

    def _present(self, X, s, k, d, h) -> bool:
        return bool((X[k, d, :, h] == s).any())

    def _vide(self) -> np.ndarray:
        X = np.full((self.prob.C, 5, self.prob.R, 2), VIDE, dtype=np.int16)
        X[self.prob.interdit] = INTERDIT
        return X

    def _poser(self, X, s: int, nb: int) -> bool:
        """Ajoute nb demi-journées au chirurgien s dans des cases vides, en
        privilégiant les journées complètes. False si impossible."""
        rng, cases = self.rng, list(self._libres)
        rng.shuffle(cases)
        while nb > 0:
            pose = False
            if nb >= 2 and rng.random() < self.P.p_journee:
                for (k, d, r, h) in cases:
                    if (h == 0 and X[k, d, r, 0] == VIDE and X[k, d, r, 1] == VIDE
                            and not self._present(X, s, k, d, 0)
                            and not self._present(X, s, k, d, 1)):
                        X[k, d, r, 0] = X[k, d, r, 1] = s
                        nb -= 2
                        pose = True
                        break
            if not pose:
                # compléter une journée si l'autre moitié est déjà à lui
                for (k, d, r, h) in cases:
                    if (X[k, d, r, h] == VIDE and X[k, d, r, 1 - h] == s
                            and not self._present(X, s, k, d, h)):
                        X[k, d, r, h] = s
                        nb -= 1
                        pose = True
                        break
            if not pose:
                for (k, d, r, h) in cases:
                    if X[k, d, r, h] == VIDE and not self._present(X, s, k, d, h):
                        X[k, d, r, h] = s
                        nb -= 1
                        pose = True
                        break
            if not pose:
                return False
        return True

    def aleatoire(self) -> np.ndarray:
        q = self.prob.quotas
        for _ in range(50):
            X = self._vide()
            ordre = sorted(range(self.prob.S), key=lambda s: (-q[s], self.rng.random()))
            if all(self._poser(X, s, int(q[s])) for s in ordre):
                return X
        raise RuntimeError("impossible de construire une grille respectant les quotas "
                           "(trop de demi-journées pour les salles disponibles)")

    def reparer(self, X) -> bool:
        """Remet chaque chirurgien exactement à son quota."""
        q = self.prob.quotas
        n = np.bincount(X[X >= 0].ravel(), minlength=self.prob.S)
        for s in np.flatnonzero(n > q):
            cases = [tuple(c) for c in np.argwhere(X == s)]
            self.rng.shuffle(cases)
            # retirer d'abord les demi-journées isolées (pas en journée)
            cases.sort(key=lambda c: X[c[0], c[1], c[2], 1 - c[3]] == s)
            for c in cases[: n[s] - q[s]]:
                X[c] = VIDE
        for s in np.flatnonzero(n < q):
            if not self._poser(X, int(s), int(q[s] - n[s])):
                return False
        return True

    # -- opérateurs ------------------------------------------------------------

    def croiser(self, A, B):
        e1, e2 = A.copy(), B.copy()
        for k in range(self.prob.C):
            for d in range(5):
                if self.rng.random() < 0.5:
                    e1[k, d], e2[k, d] = B[k, d], A[k, d]
        return e1, e2

    def _echange_valide(self, X, a, b) -> bool:
        x, y = X[a], X[b]
        if x == INTERDIT or y == INTERDIT or x == y:
            return False
        (k1, d1, _, h1), (k2, d2, _, h2) = a, b
        if (k1, d1, h1) == (k2, d2, h2):
            return True
        if x >= 0 and self._present(X, x, k2, d2, h2):
            return False
        if y >= 0 and self._present(X, y, k1, d1, h1):
            return False
        return True

    def _echanger(self, X, a, b) -> bool:
        if not self._echange_valide(X, a, b):
            return False
        X[a], X[b] = X[b], X[a]
        return True

    def muter(self, X) -> None:
        rng, libres = self.rng, self._libres
        op = rng.random()
        if op < 0.5:                                   # échange de deux cases
            for _ in range(20):
                if self._echanger(X, rng.choice(libres), rng.choice(libres)):
                    return
        elif op < 0.85:                                # aligner matin / après-midi
            occ = [tuple(c) for c in np.argwhere(X >= 0)]
            for _ in range(20):
                k, d, r, h = rng.choice(occ)
                s = X[k, d, r, h]
                if X[k, d, r, 1 - h] == s:
                    continue
                autre = (k, d, r, 1 - h)
                # s déjà présent ce jour-là à l'autre moitié, dans une autre salle ?
                meme_jour = [(k, d, r2, 1 - h) for r2 in range(self.prob.R)
                             if X[k, d, r2, 1 - h] == s]
                cible = meme_jour[0] if meme_jour else rng.choice(
                    [tuple(c) for c in np.argwhere(X == s)])
                if self._echanger(X, autre, cible):
                    return
        else:                                          # échange de deux jours
            C = self.prob.C
            for _ in range(20):
                k1, d1, k2, d2 = rng.randrange(C), rng.randrange(5), rng.randrange(C), rng.randrange(5)
                if (k1, d1) != (k2, d2) and (self.prob.interdit[k1, d1] == self.prob.interdit[k2, d2]).all():
                    X[k1, d1], X[k2, d2] = X[k2, d2].copy(), X[k1, d1].copy()
                    return

    def descente(self, X, cout: float, iterations: int):
        """Recherche locale : on garde toute mutation qui améliore."""
        for _ in range(iterations):
            Y = X.copy()
            self.muter(Y)
            c = self.prob.cout(Y, self.W)
            if c < cout - 1e-12:
                X, cout = Y, c
        return X, cout

    def _tournoi(self, pop, couts):
        ids = self.rng.sample(range(len(pop)), min(self.P.taille_tournoi, len(pop)))
        return pop[min(ids, key=lambda i: couts[i])]

    # -- boucle principale --------------------------------------------------------

    def resoudre(self, verbeux: bool = False) -> ResultatGrille:
        P, prob, t0 = self.P, self.prob, time.perf_counter()
        pop = [self.aleatoire() for _ in range(P.taille_population)]
        couts = [prob.cout(X, self.W) for X in pop]
        i0 = int(np.argmin(couts))
        meilleur, c_meilleur = pop[i0].copy(), couts[i0]
        convergence, stagnation, gen = [], 0, 0
        for gen in range(1, P.nb_generations + 1):
            ordre = np.argsort(couts)
            nouvelle = [pop[i].copy() for i in ordre[:P.elitisme]]
            while len(nouvelle) < P.taille_population:
                a, b = self._tournoi(pop, couts), self._tournoi(pop, couts)
                if self.rng.random() < P.p_croisement:
                    e1, e2 = self.croiser(a, b)
                    if not self.reparer(e1):
                        e1 = a.copy()
                    if not self.reparer(e2):
                        e2 = b.copy()
                else:
                    e1, e2 = a.copy(), b.copy()
                for e in (e1, e2):
                    if self.rng.random() < P.p_mutation:
                        self.muter(e)
                    nouvelle.append(e)
            pop = nouvelle[:P.taille_population]
            couts = [prob.cout(X, self.W) for X in pop]
            i = int(np.argmin(couts))
            if couts[i] < c_meilleur - 1e-12:
                meilleur, c_meilleur, stagnation = pop[i].copy(), couts[i], 0
            else:
                stagnation += 1
            convergence.append((gen, c_meilleur, float(np.mean(couts))))
            if verbeux and gen % 50 == 0:
                print(f"  génération {gen} : meilleur coût {c_meilleur:.4f}")
            if stagnation >= P.stagnation_max:
                break
        meilleur, c_meilleur = self.descente(meilleur, c_meilleur, P.iterations_descente)
        t = prob.termes_individu(meilleur)
        n = {k: t[k] / prob.ref[k] for k in prob.ref}
        return ResultatGrille(meilleur, c_meilleur, t, n, gen,
                              time.perf_counter() - t0, convergence)
