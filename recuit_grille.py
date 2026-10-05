r"""
recuit_grille.py — Recuit simulé qui propose une NOUVELLE GRILLE DE VACATIONS.

DÉCISION : qui occupe chaque créneau de la grille (salles et horaires gardés).
MOUVEMENTS : échanger les occupants de deux créneaux, ou donner un créneau à
un autre chirurgien.

ÉVALUATION RAPIDE D'UNE GRILLE (« modèle attendu »)
---------------------------------------------------
À partir de la base (2019-2022), pour chaque chirurgien m :
  - sa demande : patients opérés par semaine en 2022, répartis entre ses
    créneaux au prorata de leur durée ;
  - la probabilité qu'un de ses patients occupe encore un lit s nuits après
    l'opération (courbe de survie des durées de séjour) ;
  - sa part d'ambulatoire.
(sa demande hebdomadaire est fixe : plus de créneaux = vacations moins pleines).
Une grille donne alors, sur le cycle de 4 semaines (28 nuits), le nombre
ATTENDU de lits occupés chaque nuit et d'ambulatoires chaque jour. Évaluer
une grille coûte quelques microsecondes : le recuit peut en tester des
centaines de milliers. La grille retenue est ensuite VALIDÉE en rejouant
toute l'année 2022 avec les vrais patients (etude_grille.py).

FONCTION COÛT (normalisée : la grille actuelle vaut 1 sur les termes lits et ambu)
  w_lits   x écart des lits attendus à la cible (moins de lits les nuits de
             week-end : facteur_weekend)
  w_ambu   x variance des ambulatoires attendus par jour ouvré
  w_temps  x (écart de temps de travail au-delà de la tolérance)² par chirurgien
  w_capacite x (heures manquantes / heures nécessaires à son activité)² : on
             ne retire pas de temps à un chirurgien qui en manque déjà
  w_changements x part des créneaux modifiés (une grille trop bouleversée
             ne sera pas acceptée par les chirurgiens)
  + 1e6    x conflits (un chirurgien à deux endroits en même temps)
  + 1e6    x chirurgiens de la grille actuelle qui n'auraient plus aucun
             créneau (tous doivent continuer à travailler à la clinique)
Les vacances scolaires ne peuvent pas être gérées par une grille
hebdomadaire identique toute l'année : elles le sont au niveau des patients
(cible plus basse en vacances, voir etude_grille.py).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

from grille_vacations import FIXES, SEMAINES_ACTIVES, Grille

PENALITE = 1e6
NUITS = 28
NUITS_WEEKEND = (4, 5, 6)   # nuits du vendredi, samedi et dimanche : personnel de week-end


class ModeleAttendu:
    """Demande, survie des séjours et part d'ambulatoire de chaque chirurgien.

    La DEMANDE d'un chirurgien (patients par semaine, activité réelle de
    `annee`) ne dépend pas de la grille : si on lui donne un créneau de plus,
    il n'a pas plus de patients, ses vacations sont juste moins pleines. Ses
    patients sont répartis entre ses créneaux au prorata de leur durée."""

    def __init__(self, base, grille_ref: Grille, annee: int = 2022, horizon_nuits: int = 30):
        act = base[base.inter.dt.year == annee]
        semaines = act.inter.dt.isocalendar().week.nunique()
        self.demande, self.survie, self.p_ambu, self.besoin_h = {}, {}, {}, {}
        med = base.tros.median()
        for m in grille_ref.heures_par_semaine():
            self.demande[m] = (act.chir == m).sum() / max(1, semaines)
            # heures de bloc nécessaires par semaine pour son activité réelle,
            # avec des vacations remplies à 85 %
            self.besoin_h[m] = float((act[act.chir == m].tros.fillna(med) + 15).sum()) / 60 / max(1, semaines) / 0.85
            g = base[base.chir == m]
            n = g.nuits.values if len(g) else []
            self.survie[m] = [float((n > s).mean()) if len(n) else 0.0 for s in range(horizon_nuits)]
            self.p_ambu[m] = float((n == 0).mean()) if len(n) else 1.0
        self._cache: dict = {}

    def unitaire(self, ligne, m):
        """Lits et ambulatoires d'UN patient par heure de créneau, sur le cycle."""
        cle = (ligne, m)
        if cle not in self._cache:
            lits, amb = [0.0] * NUITS, [0.0] * NUITS
            for w in SEMAINES_ACTIVES[ligne.regle]:
                t = 7 * w + ligne.jour
                for s, p in enumerate(self.survie.get(m, ())):
                    lits[(t + s) % NUITS] += ligne.heures * p
                amb[t] += ligne.heures * self.p_ambu.get(m, 0.0)
            self._cache[cle] = (lits, amb)
        return self._cache[cle]


@dataclass
class PoidsGrille:
    lits: float = 0.7
    ambu: float = 0.3
    temps: float = 50.0
    tolerance_temps: float = 0.10       # ±10 % de temps de travail sans pénalité
    facteur_weekend: float = 0.85       # cible des nuits de week-end (vendredi, samedi, dimanche)
    capacite: float = 20.0              # pénalité si un chirurgien a moins d'heures que son activité n'en demande
    changements: float = 1.0            # coût d'avoir modifié TOUS les créneaux (stabilité)


class EvaluationGrille:
    """Évaluation incrémentale d'une grille : pour chaque chirurgien on garde la
    somme de ses vecteurs unitaires U_m et ses heures H_m ; ses lits attendus
    valent demande_m / H_m x U_m."""

    def __init__(self, grille: Grille, modele: ModeleAttendu, poids: PoidsGrille, reference: Grille,
                 heures_cible: dict | None = None):
        self.g, self.mod, self.p = grille, modele, poids
        # temps de travail visé par chirurgien : celui de la grille actuelle,
        # ou une répartition imposée (scénario « temps rééquilibré »)
        self.h_ref = heures_cible or reference.heures_par_semaine()
        # chirurgiens de la grille actuelle : ils doivent TOUS garder au moins un créneau
        self.presents = [m for m, h in reference.heures_par_semaine().items() if h > 0]
        self.chir_ref = list(reference.chirurgiens)
        self.n_modifiables = max(1, len(reference.modifiables()))
        self.f = [poids.facteur_weekend if n % 7 in NUITS_WEEKEND else 1.0 for n in range(NUITS)]
        self.jours_ouvres = [n for n in range(NUITS) if n % 7 < 5]
        self._recalculer()
        self.ref = self.termes_bruts()   # grille de départ = référence de normalisation

    def _part(self, m):
        H = self.H.get(m, 0.0)
        k = self.mod.demande.get(m, 0.0) / H if H > 0 else 0.0
        return [k * x for x in self.U[m]], [k * x for x in self.A[m]]

    def _recalculer(self):
        self.U, self.A, self.H = {}, {}, {}
        for l, c in zip(self.g.lignes, self.g.chirurgiens):
            if c in FIXES:
                continue
            u, a = self.mod.unitaire(l, c)
            U = self.U.setdefault(c, [0.0] * NUITS)
            A = self.A.setdefault(c, [0.0] * NUITS)
            for n in range(NUITS):
                U[n] += u[n]
                A[n] += a[n]
            self.H[c] = self.H.get(c, 0.0) + l.heures_par_semaine
        self.lits, self.amb = [0.0] * NUITS, [0.0] * NUITS
        for m in self.U:
            a, b = self._part(m)
            for n in range(NUITS):
                self.lits[n] += a[n]
                self.amb[n] += b[n]
        self.heures = self.H
        self.nb_conflits = len(self.g.conflits())
        self.nb_changes = sum(a != b for a, b in zip(self.g.chirurgiens, self.chir_ref))

    def termes_bruts(self) -> dict:
        moy = sum(self.lits) / NUITS
        mf = sum(self.f) / NUITS
        cible = [moy * f / mf for f in self.f]
        L = sum((x - c) ** 2 for x, c in zip(self.lits, cible)) / NUITS
        A = [self.amb[n] for n in self.jours_ouvres]
        mA = sum(A) / len(A)
        V = sum((a - mA) ** 2 for a in A) / len(A)
        T = K = 0.0
        for m, h0 in self.h_ref.items():
            d = abs(self.H.get(m, 0.0) - h0) / max(h0, 2.0)
            T += max(0.0, d - self.p.tolerance_temps) ** 2
            b = self.mod.besoin_h.get(m, 0.0)
            if b > 0:
                K += (max(0.0, b - self.H.get(m, 0.0)) / max(b, 2.0)) ** 2
        return {"lits": L, "ambu": V, "temps": T, "capacite": K, "changes": self.nb_changes / self.n_modifiables}

    def absents(self) -> int:
        """Chirurgiens de la grille actuelle qui n'auraient plus aucun créneau."""
        return sum(1 for m in self.presents if self.H.get(m, 0.0) <= 1e-9)

    def cout(self) -> float:
        t = self.termes_bruts()
        return (self.p.lits * t["lits"] / (self.ref["lits"] or 1) + self.p.ambu * t["ambu"] / (self.ref["ambu"] or 1)
                + self.p.temps * t["temps"] + self.p.capacite * (t["capacite"] - self.ref["capacite"])
                + self.p.changements * t["changes"] + PENALITE * (self.nb_conflits + self.absents()))

    def changer(self, i: int, nouveau: str) -> None:
        """Le créneau i passe au chirurgien `nouveau`."""
        l, ancien = self.g.lignes[i], self.g.chirurgiens[i]
        if ancien == nouveau:
            return
        for m in (ancien, nouveau):           # on retire la part des deux chirurgiens
            if m in self.U:
                a, b = self._part(m)
                for n in range(NUITS):
                    self.lits[n] -= a[n]
                    self.amb[n] -= b[n]
        u, a = self.mod.unitaire(l, ancien)
        for n in range(NUITS):
            self.U[ancien][n] -= u[n]
            self.A[ancien][n] -= a[n]
        self.H[ancien] -= l.heures_par_semaine
        u, a = self.mod.unitaire(l, nouveau)
        U = self.U.setdefault(nouveau, [0.0] * NUITS)
        A = self.A.setdefault(nouveau, [0.0] * NUITS)
        for n in range(NUITS):
            U[n] += u[n]
            A[n] += a[n]
        self.H[nouveau] = self.H.get(nouveau, 0.0) + l.heures_par_semaine
        for m in (ancien, nouveau):           # on remet leur nouvelle part
            a, b = self._part(m)
            for n in range(NUITS):
                self.lits[n] += a[n]
                self.amb[n] += b[n]
        self.nb_conflits -= self._conflits_de(i, ancien)
        self.nb_changes += (nouveau != self.chir_ref[i]) - (ancien != self.chir_ref[i])
        self.g.chirurgiens[i] = nouveau
        self.nb_conflits += self._conflits_de(i, nouveau)

    def _conflits_de(self, i, m):
        l = self.g.lignes[i]
        return sum(1 for k, (lk, ck) in enumerate(zip(self.g.lignes, self.g.chirurgiens))
                   if k != i and ck == m and lk.chevauche(l))


@dataclass
class ParamsRecuitGrille:
    T0_facteur: float = 1.0
    alpha: float = 0.995
    longueur_palier: int = 100
    Tmin_facteur: float = 0.005
    part_echanges: float = 0.8
    max_iter: int = 100_000
    graine: int = 0
    journal_periode: int = 250


@dataclass
class ResultatGrille:
    grille: Grille
    cout_initial: float
    cout_final: float
    iterations: int = 0
    temps: float = 0.0
    journal: list = field(default_factory=list)


def recuit_grille(depart: Grille, modele: ModeleAttendu, pr: ParamsRecuitGrille,
                  poids: PoidsGrille | None = None, reference: Grille | None = None,
                  heures_cible: dict | None = None) -> ResultatGrille:
    poids = poids or PoidsGrille()
    reference = reference or depart
    g = depart.copie("grille proposée")
    ev = EvaluationGrille(g, modele, poids, reference, heures_cible)
    rng = random.Random(pr.graine)
    mods = g.modifiables()
    chirs = sorted(reference.heures_par_semaine())
    t0 = time.perf_counter()
    c = ev.cout()
    res = ResultatGrille(g, c, c)

    def tirer():
        if rng.random() < pr.part_echanges:
            i, k = rng.sample(mods, 2)
            if g.chirurgiens[i] == g.chirurgiens[k]:
                return None
            return [(i, g.chirurgiens[k]), (k, g.chirurgiens[i])]
        i = rng.choice(mods)
        m = rng.choice(chirs)
        return None if m == g.chirurgiens[i] else [(i, m)]

    def appliquer(mv):
        inv = [(i, g.chirurgiens[i]) for i, _ in mv]
        for i, m in mv:
            ev.changer(i, m)
        return inv[::-1]

    # calibrage de la température sur la dégradation typique d'un mouvement
    ds = []
    for _ in range(2000):
        mv = tirer()
        if mv:
            inv = appliquer(mv)
            d = ev.cout() - c
            appliquer(inv)
            if 0 < d < PENALITE / 10:
                ds.append(d)
    # 1er DÉCILE et non moyenne : la plupart des mouvements cassent le temps de
    # travail d'un chirurgien (dégradations de 10 à 1 000) ; les mouvements
    # intéressants dégradent de ~0,1. Calibrer sur la moyenne donnerait une
    # température beaucoup trop haute : le recuit errerait sans jamais converger.
    ds.sort()
    dm = ds[len(ds) // 10] if ds else 1e-6
    T, Tmin = pr.T0_facteur * dm, pr.Tmin_facteur * dm
    meilleur, best = c, list(g.chirurgiens)
    it = 0
    while it < pr.max_iter and T >= Tmin:
        it += 1
        mv = tirer()
        if mv:
            inv = appliquer(mv)
            n = ev.cout()
            d = n - c
            if d <= 0 or (d / T < 700 and rng.random() < math.exp(-d / T)):
                c = n
                if c < meilleur - 1e-12:
                    meilleur, best = c, list(g.chirurgiens)
            else:
                appliquer(inv)
        if it % pr.longueur_palier == 0:
            T *= pr.alpha
        if it % pr.journal_periode == 0:
            res.journal.append({"iteration": it, "temperature": T, "cout": c, "meilleur": meilleur})
    for i, m in enumerate(best):
        ev.changer(i, m)
    res.cout_final = ev.cout()
    res.iterations = it
    res.temps = time.perf_counter() - t0
    return res


def lits_attendus(grille: Grille, modele: ModeleAttendu) -> tuple[list[float], list[float]]:
    ev = EvaluationGrille(grille.copie(), modele, PoidsGrille(), grille)
    return ev.lits, ev.amb


def heures_selon_activite(base, grille_ref: Grille, annee: int = 2022, part_min: float = 0.5) -> dict[str, float]:
    """Même temps de bloc TOTAL que la grille actuelle, réparti entre les
    chirurgiens au prorata du temps opératoire réel de `annee`
    (durées réelles + 15 min d'inter-salle par patient).
    Chaque chirurgien de la grille actuelle garde AU MOINS `part_min` de son
    temps actuel : tous continuent de travailler à la clinique (y compris
    ceux qui n'ont pas d'activité dans la base, arrivés depuis)."""
    act = base[base.inter.dt.year == annee]
    med = base.tros.median()
    besoin = {m: float((act[act.chir == m].tros.fillna(med) + 15).sum()) for m in grille_ref.heures_par_semaine()}
    total_h = sum(grille_ref.heures_par_semaine().values())
    total_b = sum(besoin.values())
    h0 = grille_ref.heures_par_semaine()
    return {m: max(part_min * h0[m], total_h * b / total_b) for m, b in besoin.items()}
