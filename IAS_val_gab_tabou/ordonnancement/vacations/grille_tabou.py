r"""
grille_tabou.py — Construire la grille des vacations (le « master surgical schedule ») par tabou.

LE PROBLÈME
-----------
Jusqu'ici la grille des vacations était une DONNÉE : qui opère quel jour, dans quelle
salle, était fixé, et on cherchait seulement à y placer les patients. Or le diagnostic
praticien par praticien l'a montré : la grille ne correspond pas à la demande. Certains
praticiens manquent de temps de salle (leurs patients attendent ou restent sans date),
d'autres en ont trop (des vacations à moitié vides). Aucun ordonnancement des patients
ne corrige ça. Il faut changer la grille elle-même.

LA MODÉLISATION
---------------
La grille est un cycle de 4 semaines. On la découpe en BLOCS : une salle, un jour, une
plage horaire, une semaine du cycle. Chaque bloc a un propriétaire : un praticien, ou
personne (bloc libre). Les blocs réservés (urgences, vacataires hors file élective) ne
bougent pas. La structure horaire des blocs n'est pas modifiée : ouvrir une salle ou
changer ses horaires engage des équipes d'anesthésie et de soins, c'est une autre
décision. On ne change que le PROPRIÉTAIRE.

Variables : proprio(b) ∈ praticiens ∪ {libre}, pour chaque bloc b non réservé.

Contraintes dures
  (G1) un praticien n'est jamais dans deux salles en même temps ;
  (G2) un praticien n'opère que les jours de la semaine où il opère déjà
       (ses autres jours sont pris : consultations, autre établissement) ;
  (G3) chaque praticien garde au moins un bloc ;
  (G4) au plus K blocs changent de propriétaire (budget de changement).

Fonction objectif
  Soit D_m la demande du praticien m (heures par cycle, marges et TIS compris) et T_m
  les heures de blocs qu'il possède.

      manque_m = max(0, D_m − ρ* · T_m)      heures qui manquent pour tenir ρ*
      excès_m  = max(0, ρ_min · T_m − D_m)    heures offertes au-delà du besoin

      F = Σ_m (manque_m / h₀)²  +  γ · Σ_m (excès_m / h₀)²  +  w_ℓ · CV²(lits)

  Le carré répartit le manque : il vaut mieux deux praticiens à 2 h de manque qu'un à
  4 h. h₀ = 4 h (une demi-journée) rend le terme lisible. Le terme des lits prend le
  profil d'occupation attendu sur les 28 jours du cycle, week-ends compris : un
  praticien dont les patients restent 4 nuits, placé le vendredi, remplit les lits du
  week-end. CV² = variance / moyenne².

Mouvements
  TRANSFERT  b : m → m'   (m' peut être « libre »)
  ÉCHANGE    b₁ ↔ b₂       deux praticiens permutent deux blocs
  Attribut tabou : (b, ancien propriétaire) — on ne rend pas b à m tout de suite.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from donnees import ALIAS, GRILLE, RESERVES, MotifVacation

LIBRE = None
CYCLE = 28


# ---------------------------------------------------------------------------
# Blocs
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Bloc:
    id: int
    semaine: int          # 0..3 dans le cycle
    jour: int             # 0 = lundi .. 4 = vendredi
    salle: int
    debut: float          # heures décimales
    fin: float
    proprio: str | None   # code praticien, ou None si libre
    reserve: bool         # urgences / vacataires : ne bouge jamais
    etiquette: str = ""

    @property
    def heures(self) -> float:
        return self.fin - self.debut

    @property
    def jour_cycle(self) -> int:
        return self.semaine * 7 + self.jour

    def chevauche(self, autre: "Bloc") -> bool:
        return (self.semaine == autre.semaine and self.jour == autre.jour
                and self.debut < autre.fin and autre.debut < self.fin)


def blocs_depuis_grille(grille=GRILLE) -> list[Bloc]:
    """Un bloc par (motif, semaine du cycle)."""
    blocs = []
    for m in grille:
        pr = ALIAS.get(m.praticien, m.praticien)
        libre = pr == "LIBRE"
        reserve = pr in RESERVES and not libre
        for s in m.semaines:
            blocs.append(Bloc(len(blocs), s, m.jour, m.salle, m.debut, m.fin,
                              None if libre else pr, reserve, m.etiquette))
    return blocs


def grille_depuis_blocs(blocs: list[Bloc], proprio: list[str | None] | None = None
                        ) -> list[MotifVacation]:
    """Retour au format de `donnees.py`, pour réinjecter la grille dans les simulations.
    Un bloc libre devient un motif « LIBRE » (compté dans les réservés, donc fermé à la
    file élective, comme dans la grille d'origine)."""
    out = []
    for b in blocs:
        pr = b.proprio if proprio is None else proprio[b.id]
        out.append(MotifVacation(b.jour, b.salle, pr if pr is not None else "LIBRE",
                                 b.debut, b.fin, (b.semaine,), b.etiquette))
    return out


# ---------------------------------------------------------------------------
# Demande
# ---------------------------------------------------------------------------


@dataclass
class Demande:
    """Ce que chaque praticien consomme, calibré sur une ou plusieurs années."""
    heures_cycle: dict[str, float]            # D_m : heures de bloc par cycle de 4 semaines
    profil_lits: dict[str, np.ndarray]        # π_m(k) : lits occupés k jours après, par heure opérée
    source: str = ""


def calibrer_demande(df, estimateur, annees: list[int], praticiens: set[str]) -> Demande:
    """D_m et π_m à partir des patients réellement opérés pendant `annees`.

    On passe par `construire_instance` pour que la demande soit mesurée EXACTEMENT comme
    les simulations la consomment : durée estimée + marge de risque + TIS."""
    from donnees import construire_instance, id_vers_code
    import pandas as pd
    heures = defaultdict(float)
    lits = defaultdict(lambda: np.zeros(CYCLE))
    for an in annees:
        # premier lundi de l'année
        d0 = pd.Timestamp(f"{an}-01-01")
        d0 = d0 + pd.Timedelta(days=(7 - d0.weekday()) % 7)
        inst, _ = construire_instance(df, estimateur, debut_periode=str(d0.date()),
                                      nb_semaines=52, graine=0)
        for p in inst.patients.values():
            m = id_vers_code(p.med_id)
            if m not in praticiens:
                continue
            h = (p.duree_op + p.marge_perso + inst.tis) / 60
            heures[m] += h
            for k in range(min(p.nb_nuits, CYCLE)):
                lits[m][k] += 1
    n_cycles = 13 * len(annees)
    D = {m: heures.get(m, 0.0) / n_cycles for m in praticiens}
    # π_m(k) : lits par heure de demande — multiplié par les heures RÉELLEMENT opérées
    pi = {m: (lits[m] / heures[m]) if heures.get(m) else np.zeros(CYCLE) for m in praticiens}
    return Demande(D, pi, source=f"patients opérés en {', '.join(map(str, annees))}")


# ---------------------------------------------------------------------------
# Le tabou
# ---------------------------------------------------------------------------


@dataclass
class ParametresGrille:
    rho_cible: float = 0.85
    """Taux d'occupation visé. Au-delà, le praticien manque de temps de salle : ses
    marges de risque ne tiennent plus et ses patients attendent."""
    rho_min: float = 0.50
    """En dessous, le praticien a trop de temps : des vacations à moitié vides."""
    gamma: float = 0.1
    """Poids de l'excès, faible : il sert surtout à choisir À QUI reprendre un bloc."""
    w_lits: float = 10.0
    budget: int | None = None
    """K, nombre maximal de blocs qui changent de propriétaire. None = pas de limite."""
    jours_imposes: bool = True
    liberer: bool = False
    """Autoriser un bloc à devenir libre (fermé à la file élective). Désactivé par défaut :
    la validation sur 2022 a montré que libérer du temps de salle sur la foi de la
    demande passée le fait manquer l'année suivante aux praticiens dont l'activité monte."""
    iterations: int = 400
    duree_tabou: int = 12
    echanges_max: int = 1500
    graine: int = 0


@dataclass
class ResultatGrille:
    proprio: list[str | None]
    cout: float
    detail: dict
    historique: list[float] = field(default_factory=list)
    changements: int = 0


class TabouGrille:

    def __init__(self, blocs: list[Bloc], demande: Demande,
                 params: ParametresGrille | None = None,
                 scenarios: list[Demande] | None = None):
        """`scenarios` : si fourni, le coût est la MOYENNE des coûts sur ces demandes
        (une par année d'historique). C'est la version ROBUSTE : la grille n'est pas
        taillée pour une seule prévision, qui peut se tromper d'une année sur l'autre.
        Le profil de lits reste celui de `demande` (toutes années confondues)."""
        self.blocs = blocs
        self.D = demande.heures_cycle
        self.pi = demande.profil_lits
        self.scenarios = [sc.heures_cycle for sc in scenarios] if scenarios else [self.D]
        self.p = params or ParametresGrille()
        self.alea = random.Random(self.p.graine)
        self.praticiens = sorted(self.D)
        self.origine = [b.proprio for b in blocs]
        self.mobiles = [b.id for b in blocs if not b.reserve]
        # jours de la semaine où chaque praticien opère déjà (G2)
        self.jours = defaultdict(set)
        for b in blocs:
            if b.proprio is not None and not b.reserve:
                self.jours[b.proprio].add(b.jour)
        # profil de lits décalé, précalculé : decale[m][c] = π_m roulé de c jours
        self.decale = {m: np.stack([np.roll(self.pi[m], c) for c in range(CYCLE)])
                       for m in self.praticiens}

    # -- évaluation ---------------------------------------------------------

    def _agregats(self, proprio):
        T = defaultdict(float)
        V = {m: np.zeros(CYCLE) for m in self.praticiens}
        for b in self.blocs:
            m = proprio[b.id]
            if m is None or b.reserve or m not in V:
                continue
            T[m] += b.heures
            V[m] += b.heures * self.decale[m][b.jour_cycle]
        return T, V

    def _cout(self, T, V, details=False):
        if not details:
            return sum(self._cout_un(T, V, D) for D in self.scenarios) / len(self.scenarios)
        F = sum(self._cout_un(T, V, D) for D in self.scenarios) / len(self.scenarios)
        _, det = self._cout_un(T, V, self.D, details=True)
        return F, det

    def _cout_un(self, T, V, Dm, details=False):
        P = self.p
        manque = excès = 0.0
        L = np.zeros(CYCLE)
        par_m = {}
        for m in self.praticiens:
            d, t = Dm.get(m, 0.0), T.get(m, 0.0)
            ma = max(0.0, d - P.rho_cible * t)
            ex = max(0.0, P.rho_min * t - d)
            manque += (ma / 4) ** 2
            excès += (ex / 4) ** 2
            f = min(1.0, d / t) if t > 0 else 0.0
            L += f * V[m]
            if details:
                par_m[m] = dict(demande=d, offre=t, occupation=d / t if t else float("inf"),
                                manque=ma, exces=ex)
        moy = L.mean()
        cv2 = float(L.var() / moy ** 2) if moy > 0 else 0.0
        F = manque + P.gamma * excès + P.w_lits * cv2
        if not details:
            return F
        return F, dict(manque=manque, exces=excès, cv2_lits=cv2, lits=L.tolist(),
                       praticiens=par_m)

    def evaluer(self, proprio):
        T, V = self._agregats(proprio)
        return self._cout(T, V, details=True)

    # -- admissibilité ------------------------------------------------------

    def _libre_pour(self, m, b, proprio, ignore=()):
        """(G1) et (G2) pour donner le bloc b au praticien m."""
        if m is None:
            return True
        if self.p.jours_imposes and b.jour not in self.jours[m]:
            return False
        for c in self._par_proprio[m]:
            if c != b.id and c not in ignore and self.blocs[c].chevauche(b):
                return False
        return True

    # -- recherche ----------------------------------------------------------

    def resoudre(self, verbeux=False) -> ResultatGrille:
        P, B = self.p, self.blocs
        proprio = list(self.origine)
        self._par_proprio = defaultdict(set)
        for b in B:
            if proprio[b.id] is not None:
                self._par_proprio[proprio[b.id]].add(b.id)
        T, V = self._agregats(proprio)
        cout = self._cout(T, V)
        meilleur, meilleur_proprio = cout, list(proprio)
        changes = 0
        tabou: dict[tuple, int] = {}
        hist = [cout]
        cibles = self.praticiens + ([None] if P.liberer else [])

        def delta_cout(mouvs):
            """mouvs : liste de (bloc, ancien, nouveau). Coût après, sans rien modifier."""
            T2, V2 = dict(T), dict(V)
            for bid, a, n in mouvs:
                b = B[bid]
                if a is not None:
                    T2[a] = T2.get(a, 0) - b.heures
                    V2[a] = V2[a] - b.heures * self.decale[a][b.jour_cycle]
                if n is not None:
                    T2[n] = T2.get(n, 0) + b.heures
                    V2[n] = V2[n] + b.heures * self.decale[n][b.jour_cycle]
            return self._cout(T2, V2), T2, V2

        for it in range(1, P.iterations + 1):
            candidats = []
            # 1. transferts : voisinage complet
            for bid in self.mobiles:
                a = proprio[bid]
                if a is not None and len(self._par_proprio[a]) <= 1:
                    continue                                    # (G3)
                for n in cibles:
                    if n == a or not self._libre_pour(n, B[bid], proprio):
                        continue
                    ch = changes + (n != self.origine[bid]) - (a != self.origine[bid])
                    if P.budget is not None and ch > P.budget:
                        continue                                # (G4)
                    candidats.append(([(bid, a, n)], ch, [(bid, a)]))
            # 2. échanges : échantillonnés
            occupes = [bid for bid in self.mobiles if proprio[bid] is not None]
            for _ in range(P.echanges_max):
                b1, b2 = self.alea.sample(occupes, 2)
                a1, a2 = proprio[b1], proprio[b2]
                if a1 == a2:
                    continue
                if not (self._libre_pour(a2, B[b1], proprio, ignore={b2})
                        and self._libre_pour(a1, B[b2], proprio, ignore={b1})):
                    continue
                ch = (changes + (a2 != self.origine[b1]) - (a1 != self.origine[b1])
                      + (a1 != self.origine[b2]) - (a2 != self.origine[b2]))
                if P.budget is not None and ch > P.budget:
                    continue
                candidats.append(([(b1, a1, a2), (b2, a2, a1)], ch, [(b1, a1), (b2, a2)]))

            choisi = None
            for mouvs, ch, attrs in candidats:
                c, T2, V2 = delta_cout(mouvs)
                interdit = any(tabou.get((bid, n), 0) > it for bid, _, n in mouvs)
                if interdit and not c < meilleur - 1e-12:            # aspiration
                    continue
                if choisi is None or c < choisi[0]:
                    choisi = (c, mouvs, ch, attrs, T2, V2)
            if choisi is None:
                break
            cout, mouvs, changes, attrs, T, V = choisi
            for bid, a, n in mouvs:
                proprio[bid] = n
                if a is not None:
                    self._par_proprio[a].discard(bid)
                if n is not None:
                    self._par_proprio[n].add(bid)
            for bid, a in attrs:                       # ne pas rendre b à a tout de suite
                tabou[(bid, a)] = it + P.duree_tabou
            hist.append(cout)
            if cout < meilleur - 1e-12:
                meilleur, meilleur_proprio = cout, list(proprio)
            if verbeux and it % 50 == 0:
                print(f"    it {it:4d}  coût {cout:9.3f}  meilleur {meilleur:9.3f}  "
                      f"changements {changes}", flush=True)

        F, detail = self.evaluer(meilleur_proprio)
        n_ch = sum(1 for b in B if meilleur_proprio[b.id] != self.origine[b.id])
        return ResultatGrille(meilleur_proprio, F, detail, hist, n_ch)
