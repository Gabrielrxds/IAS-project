r"""
calendrier.py — Week-ends, vacances scolaires, ponts et fériés, et la CIBLE de lits nuit par nuit.

Objectif du service : des lits lissés EN SEMAINE, et moins de lits occupés les nuits du week-end
et pendant les vacances scolaires (moins de personnel). On l'exprime par un profil de cible f :

    f(nuit) = 1        nuit de semaine (dimanche soir exclu : la nuit j est celle qui suit le jour j)
            = F_WE     nuits du samedi et du dimanche
            = F_VAC    nuits de vacances scolaires (zone B), de pont ou de jour férié
            (le plus petit des deux si les deux s'appliquent)

et par l'écart des lits L à la forme de cette cible, au facteur d'échelle près (le niveau moyen
reste celui de la demande) :

    écart(L) = min_κ  (1/n) Σ (L_j − κ f_j)²  =  (1/n) [ Σ L² − (Σ L f)² / Σ f² ]

Avec f ≡ 1, c'est exactement la variance : le critère d'avant est un cas particulier.
Valeurs par défaut : celles retenues dans le rapport (etude_cibles.py) : 0,70 et 0,65.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

F_WE = 0.70
F_VAC = 0.65

# Capacité POST-OPÉRATOIRE en vacances scolaires (moins d'infirmiers, de kinés...) :
# contrainte DURE sur les LITS, en plus de la cible. Le service est armé, en période normale,
# pour LITS_ARMES lits ; en vacances, ponts et fériés, pour CAPA_VAC de ce niveau (réserve
# d'urgence comprise). Hypothèses à fixer avec l'établissement (écran « calendrier »).
# Pas de plafond sur l'ambulatoire : aucune consigne, et le planning lissé n'en a pas besoin
# (5,7 admissions par jour ouvré de vacances contre 8,0 hors vacances, max. 12).
CAPA_VAC = 0.70
LITS_ARMES = 17          # pic des nuits de semaine du planning lissé

# Vacances scolaires zone B (premier et dernier jour sans classe).
VACANCES_ZONE_B = [
    ("2018-10-20", "2018-11-04"), ("2018-12-22", "2019-01-06"), ("2019-02-09", "2019-02-24"),
    ("2019-04-06", "2019-04-22"), ("2019-07-06", "2019-09-01"), ("2019-10-19", "2019-11-03"),
    ("2019-12-21", "2020-01-05"), ("2020-02-15", "2020-03-01"), ("2020-04-11", "2020-04-26"),
    ("2020-07-04", "2020-08-31"), ("2020-10-17", "2020-11-01"), ("2020-12-19", "2021-01-03"),
    ("2021-02-20", "2021-03-07"), ("2021-04-10", "2021-04-25"), ("2021-07-06", "2021-09-01"),
    ("2021-10-23", "2021-11-07"), ("2021-12-18", "2022-01-02"), ("2022-02-12", "2022-02-27"),
    ("2022-04-16", "2022-05-01"), ("2022-07-07", "2022-08-31"), ("2022-10-22", "2022-11-06"),
    ("2022-12-17", "2023-01-02"), ("2023-02-11", "2023-02-26"), ("2023-04-15", "2023-05-01"),
    ("2023-07-08", "2023-09-03"),
]
_VAC = [(dt.date.fromisoformat(a), dt.date.fromisoformat(b)) for a, b in VACANCES_ZONE_B]


def _paques(a: int) -> dt.date:
    b, c = a // 100, a % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * (a % 19) + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = ((a % 19) + 11 * h + 22 * l) // 451
    return dt.date(a, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def feries_et_ponts(annees):
    fer = set()
    for a in annees:
        p = _paques(a)
        fer |= {dt.date(a, 1, 1), p + dt.timedelta(1), dt.date(a, 5, 1), dt.date(a, 5, 8),
                p + dt.timedelta(39), p + dt.timedelta(50), dt.date(a, 7, 14), dt.date(a, 8, 15),
                dt.date(a, 11, 1), dt.date(a, 11, 11), dt.date(a, 12, 25)}
    ponts = ({f - dt.timedelta(1) for f in fer if f.weekday() == 1}
             | {f + dt.timedelta(1) for f in fer if f.weekday() == 3})
    return fer, ponts


def type_nuit(d: dt.date, fer=None, ponts=None) -> str:
    """'vacances', 'week-end' ou 'semaine' pour la nuit qui suit le jour d."""
    if fer is None:
        fer, ponts = feries_et_ponts(range(d.year - 1, d.year + 2))
    if d in fer or d in ponts or any(a <= d <= b for a, b in _VAC):
        return "vacances"
    return "week-end" if d.weekday() >= 5 else "semaine"


def profil(jour_zero: dt.date, nb_jours: int, f_we: float = None, f_vac: float = None) -> np.ndarray:
    """Cible relative de chaque nuit de l'horizon (indice j = nuit qui suit le jour j)."""
    f_we = F_WE if f_we is None else f_we
    f_vac = F_VAC if f_vac is None else f_vac
    fin = jour_zero + dt.timedelta(nb_jours)
    fer, ponts = feries_et_ponts(range(jour_zero.year - 1, fin.year + 2))
    out = np.ones(nb_jours)
    for j in range(nb_jours):
        d = jour_zero + dt.timedelta(j)
        if d.weekday() >= 5:
            out[j] = f_we
        if d in fer or d in ponts or any(a <= d <= b for a, b in _VAC):
            out[j] = min(out[j], f_vac)
    return out


def profil_cycle(L: int = 28, f_we: float = None) -> np.ndarray:
    """Cible relative des nuits d'un cycle de grille (jour 0 = lundi). Les vacances ne sont pas
    représentables dans une grille cyclique : seule la cible du week-end s'y applique."""
    f_we = F_WE if f_we is None else f_we
    return np.array([f_we if j % 7 >= 5 else 1.0 for j in range(L)])


def ecart_cible(L, f) -> float:
    """(1/n) min_κ Σ (L − κ f)² ; égal à la variance si f ≡ 1."""
    L = np.asarray(L, float)
    f = np.asarray(f, float)
    n = len(L)
    return float((L @ L - (L @ f) ** 2 / (f @ f)) / n)


def capacites_calendrier(jour_zero: dt.date, nb_jours: int, capacite_lits: int,
                         capa_vac: float = None, lits_armes: int = None):
    """Lits possibles chaque nuit : capacite_lits hors vacances, capa_vac × lits_armes pendant
    les vacances, ponts et fériés."""
    capa_vac = CAPA_VAC if capa_vac is None else capa_vac
    lits_armes = LITS_ARMES if lits_armes is None else lits_armes
    fin = jour_zero + dt.timedelta(nb_jours)
    fer, ponts = feries_et_ponts(range(jour_zero.year - 1, fin.year + 2))
    lits = []
    for j in range(nb_jours):
        vac = type_nuit(jour_zero + dt.timedelta(j), fer, ponts) == "vacances"
        lits.append(min(capacite_lits, round(capa_vac * lits_armes)) if vac else capacite_lits)
    return lits
