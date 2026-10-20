r"""
estimation.py — Estimation de la durée opératoire, validée par le chirurgien.

C'est la PREMIÈRE étape de votre chaîne, et de loin la plus rentable : un
tabou parfait sur des durées fausses produit un planning faux. Sur votre
historique, le même acte varie du simple au double selon l'opérateur ; une
durée « moyenne établissement » est donc structurellement inutilisable.

PRINCIPE
--------
On ne remplace pas le chirurgien, on l'OUTILLE. Le système :
  1. calcule les statistiques du couple (praticien × type d'acte) ;
  2. propose une durée — la MÉDIANE, pas la moyenne ;
  3. affiche la dispersion et laisse le praticien trancher ;
  4. retient la valeur validée, et mémorise la MARGE DE RISQUE de l'acte.

Pourquoi la médiane et pas la moyenne : la distribution des durées
opératoires est asymétrique à droite (quelques interventions qui dérapent,
jamais d'intervention deux fois plus rapide que prévu). La moyenne est tirée
vers le haut par la queue ; la médiane décrit le cas courant. C'est un
résultat classique de la littérature sur le *case duration prediction*
(la loi log-normale est le modèle de référence).

LA MARGE DE RISQUE : LE VRAI APPORT
-----------------------------------
Votre version réservait 30 minutes par vacation, forfaitairement. Le problème :
30 minutes ne veulent pas dire la même chose selon ce qu'on opère. Un canal
carpien chez un praticien entraîné, c'est 32 min médian avec 11 min d'écart-
type — il ne dérapera pas. Une ostéosynthèse de membre supérieur, c'est 76 min
médian avec 34 min d'écart-type — elle dérapera.

On remplace donc la marge fixe par `marge_perso = P90 - médiane`, portée par
CHAQUE patient et sommée dans la charge de la vacation. Conséquence :
  - une vacation d'actes prévisibles se remplit davantage (moins de creux) ;
  - une vacation d'actes variables se protège toute seule.
C'est une somme de marges indépendantes, donc pessimiste (on suppose que tout
dérape en même temps) : `facteur_mutualisation` permet de la réduire, en
s'appuyant sur le fait que les aléas se compensent partiellement — la variance
d'une somme de n variables indépendantes croît en n, son écart-type en √n.

CHAÎNE DE REPLI
---------------
  (praticien × acte)  si n >= n_min          le plus précis
  (acte, tous praticiens)  si n >= n_min     on perd l'effet opérateur
  (spécialité du praticien)                  ordre de grandeur
  défaut paramétrable                        acte totalement nouveau
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol


# ---------------------------------------------------------------------------
# 1. Le résumé statistique montré au praticien
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatsActe:
    """Distribution observée d'un acte. Toutes les durées en minutes."""

    n: int
    mediane: float
    moyenne: float
    ecart_type: float
    p25: float
    p75: float
    p90: float
    mini: float
    maxi: float
    source: str          # sur quoi on s'est rabattu
    libelle: str = ""

    @property
    def marge_risque(self) -> int:
        """P90 - médiane : ce qu'il faut réserver pour ne pas déborder 9 fois
        sur 10. Plancher à 5 minutes, sinon un acte ultra-régulier n'aurait
        aucune réserve et le moindre aléa ferait déborder la vacation."""
        return max(5, int(round(self.p90 - self.mediane)))

    @property
    def fiable(self) -> bool:
        return self.n >= 10 and self.source.startswith("praticien")

    def tableau(self) -> str:
        """Ce qu'on met sous les yeux du chirurgien."""
        return (
            f"  {self.libelle}\n"
            f"  source : {self.source}   (n = {self.n} interventions)\n"
            f"  ┌──────────┬──────────┬──────────┬──────────┬──────────┐\n"
            f"  │   P25    │ MÉDIANE  │   P75    │   P90    │   max    │\n"
            f"  ├──────────┼──────────┼──────────┼──────────┼──────────┤\n"
            f"  │ {self.p25:6.0f}   │ {self.mediane:6.0f}   │ {self.p75:6.0f}   │"
            f" {self.p90:6.0f}   │ {self.maxi:6.0f}   │\n"
            f"  └──────────┴──────────┴──────────┴──────────┴──────────┘\n"
            f"  moyenne {self.moyenne:.0f} min, écart-type {self.ecart_type:.0f} min\n"
            f"  marge de risque retenue (P90 - médiane) : {self.marge_risque} min"
        )


def _quantile(valeurs: list[float], q: float) -> float:
    """Quantile par interpolation linéaire. On n'importe pas numpy pour ça :
    le module doit tourner sur n'importe quelle machine de l'hôpital."""
    if not valeurs:
        return 0.0
    v = sorted(valeurs)
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    bas = math.floor(pos)
    haut = min(bas + 1, len(v) - 1)
    return v[bas] + (v[haut] - v[bas]) * (pos - bas)


def resumer(durees: Iterable[float], source: str, libelle: str = "") -> StatsActe:
    d = [float(x) for x in durees]
    if not d:
        return StatsActe(0, 0, 0, 0, 0, 0, 0, 0, 0, source, libelle)
    return StatsActe(
        n=len(d),
        mediane=statistics.median(d),
        moyenne=statistics.fmean(d),
        ecart_type=statistics.pstdev(d) if len(d) > 1 else 0.0,
        p25=_quantile(d, 0.25),
        p75=_quantile(d, 0.75),
        p90=_quantile(d, 0.90),
        mini=min(d),
        maxi=max(d),
        source=source,
        libelle=libelle,
    )


# ---------------------------------------------------------------------------
# 2. L'estimateur
# ---------------------------------------------------------------------------


class EstimateurDuree:
    """Construit les statistiques et propose une durée.

    `historique` : itérable de tuples (praticien, type_acte, duree_minutes).
    `specialites` : praticien -> spécialité, pour le dernier repli.
    """

    def __init__(self, historique: Iterable[tuple[str, str, float]],
                 specialites: dict[str, str] | None = None,
                 n_min: int = 5, defaut: float = 60.0):
        self.n_min = n_min
        self.defaut = defaut
        self.specialites = specialites or {}

        par_couple: dict[tuple[str, str], list[float]] = {}
        par_acte: dict[str, list[float]] = {}
        par_spe: dict[str, list[float]] = {}
        toutes: list[float] = []

        for prat, acte, duree in historique:
            if duree is None or duree <= 0:
                continue          # les 172 lignes à TROS <= 0 du fichier source
            par_couple.setdefault((prat, acte), []).append(duree)
            par_acte.setdefault(acte, []).append(duree)
            par_spe.setdefault(self.specialites.get(prat, "?"), []).append(duree)
            toutes.append(duree)

        self._couple = {k: resumer(v, f"praticien {k[0]} × acte", f"{k[1]} — {k[0]}")
                        for k, v in par_couple.items()}
        self._acte = {k: resumer(v, "acte, tous praticiens", k) for k, v in par_acte.items()}
        self._spe = {k: resumer(v, f"spécialité {k}", k) for k, v in par_spe.items()}
        self._global = resumer(toutes, "tout l'historique", "tous actes")

    # -- chaîne de repli ---------------------------------------------------

    def stats(self, praticien: str, acte: str) -> StatsActe:
        s = self._couple.get((praticien, acte))
        if s is not None and s.n >= self.n_min:
            return s
        s = self._acte.get(acte)
        if s is not None and s.n >= self.n_min:
            return s
        s = self._spe.get(self.specialites.get(praticien, "?"))
        if s is not None and s.n >= self.n_min:
            return s
        if self._global.n:
            return self._global
        return StatsActe(0, self.defaut, self.defaut, 0, self.defaut, self.defaut,
                         self.defaut * 1.3, self.defaut, self.defaut,
                         "valeur par défaut", acte)

    def proposition(self, praticien: str, acte: str) -> tuple[int, StatsActe]:
        """Durée proposée (médiane arrondie à 5 min) + les stats qui l'étayent.

        L'arrondi à 5 min n'est pas cosmétique : proposer « 47 minutes » donne
        une fausse impression de précision et invite le praticien à discuter le
        chiffre plutôt que l'ordre de grandeur.
        """
        s = self.stats(praticien, acte)
        return int(round(s.mediane / 5) * 5), s

    def couverture(self) -> dict:
        """Diagnostic : sur quelle proportion des cas a-t-on une vraie stat ?"""
        fiables = sum(1 for s in self._couple.values() if s.n >= self.n_min)
        return {
            "couples_praticien_acte": len(self._couple),
            "couples_exploitables": fiables,
            "actes_distincts": len(self._acte),
            "actes_exploitables": sum(1 for s in self._acte.values() if s.n >= self.n_min),
            "interventions": self._global.n,
        }


# ---------------------------------------------------------------------------
# 3. La validation par le chirurgien
# ---------------------------------------------------------------------------


class Valideur(Protocol):
    """Contrat : on montre les stats, le praticien rend une durée.

    Le fait que ce soit un protocole (et pas du code en dur) est ce qui permet
    d'avoir le même pipeline en production (console / formulaire web) et en
    simulation (acceptation automatique), sans jamais dupliquer la logique
    d'estimation.
    """

    def __call__(self, praticien: str, acte: str,
                 proposition: int, stats: StatsActe) -> int: ...


def valideur_auto(praticien: str, acte: str, proposition: int, stats: StatsActe) -> int:
    """Simulation : on accepte la proposition. Sert à rejouer l'historique."""
    return proposition


def valideur_quantile(q: float = 0.75) -> Valideur:
    """Politique « prudente » : on programme sur le P75 au lieu de la médiane.

    À comparer avec la médiane dans vos expériences : le P75 réduit les
    dépassements de vacation mais crée mécaniquement du creux. C'est
    exactement l'arbitrage que la fonction objectif est censée arbitrer, il est
    donc instructif de le déplacer volontairement pour voir ce que ça coûte.
    """
    def _v(praticien: str, acte: str, proposition: int, stats: StatsActe) -> int:
        return int(round(_quantile_depuis(stats, q) / 5) * 5)
    return _v


def _quantile_depuis(s: StatsActe, q: float) -> float:
    """Interpolation grossière entre les quantiles mémorisés."""
    points = [(0.25, s.p25), (0.5, s.mediane), (0.75, s.p75), (0.90, s.p90)]
    if q <= 0.25:
        return s.p25
    if q >= 0.90:
        return s.p90
    for (q1, v1), (q2, v2) in zip(points, points[1:]):
        if q1 <= q <= q2:
            return v1 + (v2 - v1) * (q - q1) / (q2 - q1)
    return s.mediane


def valideur_console(praticien: str, acte: str, proposition: int, stats: StatsActe) -> int:
    """Vrai usage : on affiche les statistiques et on demande au praticien.

    La séquence d'affichage est délibérée — les stats d'ABORD, la proposition
    ENSUITE. L'inverse produit un effet d'ancrage : le praticien valide le
    chiffre sans regarder la dispersion.
    """
    print()
    print("─" * 70)
    print(f"Patient à programmer — {acte}  /  praticien {praticien}")
    print(stats.tableau())
    if not stats.fiable:
        print("  ⚠ peu de données pour ce praticien sur cet acte : "
              "la proposition est indicative.")
    print(f"\n  Durée proposée (entrée salle → sortie salle) : {proposition} min")
    reponse = input("  Validez-vous ? [Entrée = oui, ou saisissez une durée en min] ").strip()
    if not reponse:
        return proposition
    try:
        valeur = int(float(reponse.replace(",", ".")))
    except ValueError:
        print("  Saisie non comprise, on garde la proposition.")
        return proposition
    if valeur <= 0:
        return proposition
    if stats.n and (valeur < 0.4 * stats.mediane or valeur > 2.5 * stats.mediane):
        conf = input(f"  {valeur} min s'écarte beaucoup de l'historique "
                     f"({stats.mediane:.0f} min). Confirmer ? [o/N] ").strip().lower()
        if conf != "o":
            return proposition
    return valeur


# ---------------------------------------------------------------------------
# 4. Application à un patient
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Estimation:
    duree: int
    marge: int
    stats: StatsActe
    valide_par_praticien: bool


def estimer(estimateur: EstimateurDuree, praticien: str, acte: str,
            valideur: Valideur = valideur_auto,
            facteur_mutualisation: float = 1.0) -> Estimation:
    """Chaîne complète pour UN patient : stats -> proposition -> validation.

    `facteur_mutualisation` < 1 réduit la marge individuelle. Justification :
    les dépassements sont approximativement indépendants, donc l'écart-type de
    la somme de n actes croît en √n et non en n. Réserver la somme des marges
    individuelles est correct mais très pessimiste. Une valeur autour de
    1/√(nb moyen de patients par vacation) ≈ 0.4 est un bon point de départ,
    à calibrer sur vos propres dépassements observés.
    """
    proposition, stats = estimateur.proposition(praticien, acte)
    duree = valideur(praticien, acte, proposition, stats)
    marge = max(5, int(round(stats.marge_risque * facteur_mutualisation)))
    return Estimation(duree=duree, marge=marge, stats=stats,
                      valide_par_praticien=(duree != proposition))
