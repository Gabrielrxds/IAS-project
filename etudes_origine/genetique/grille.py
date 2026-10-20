r"""
grille.py — Grille des vacations (« Propo CDB à venir », version 28) transcrite
en table, et déroulée en objets `Vacation` sur un horizon.

La feuille Excel est une grille en TEXTE LIBRE (cellules fusionnées, mentions
« semaine paire », « 1 sem. impaire sur 2 »...). Elle ne se lit pas de façon
fiable automatiquement : elle est donc transcrite à la main ci-dessous, une
ligne par créneau, avec la cellule source en commentaire. À re-vérifier à
chaque nouvelle version de la grille.

CONVENTIONS D'INTERPRÉTATION (à valider avec le bloc)
------------------------------------------------------
  - Parité = parité du numéro de semaine ISO.
  - « 1 sem. impaire sur 2 » : les semaines impaires alternent entre deux
    plans, notés IMPAIRE_A (semaine ISO ≡ 1 mod 4) et IMPAIRE_B (≡ 3 mod 4).
    Le lundi, MT apparaît en salle 3 (J8) ET en salle 4 (O8) l'après-midi :
    il ne peut pas être dans les deux, donc le plan J (GA puis MT, salle 3)
    va avec le plan N (TDO/BS puis DS, salle 4), et le plan K (GHREA puis
    DEVOS, salle 3) avec le plan O (RL puis MT, salle 4).
  - DEVOS (K6) = praticien « DE » de l'historique.
  - URGENCES, créneaux « (urg) », GHREA et LIBRE ne sont pas des vacations
    programmées : ils sont gardés dans la table (type ≠ "programme") mais
    `derouler_grille` ne crée pas de vacation pour eux par défaut.
  - Salle 1 : aucune vacation dans cette version de la grille.
  - Les vacations de journée incluent la pause repas (comme les TVO de la
    feuille : 8h-17h30 = 9,5 h).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from modele import Vacation

TOUTES, PAIRE, IMPAIRE, IMPAIRE_A, IMPAIRE_B = "toutes", "paire", "impaire", "impaireA", "impaireB"
LUN, MAR, MER, JEU, VEN = range(5)


def h(texte: str) -> int:
    """'7h45' -> 465 minutes."""
    hh, _, mm = texte.partition("h")
    return int(hh) * 60 + (int(mm) if mm else 0)


@dataclass(frozen=True)
class Creneau:
    jour_semaine: int
    salle: int
    code: str          # code praticien (celui de la colonne « Praticien »)
    debut: int         # minutes
    fin: int
    parite: str = TOUTES
    type: str = "programme"   # "programme" | "urgence" | "libre" | "autre"
    source: str = ""          # cellule(s) de la feuille


GRILLE: list[Creneau] = [
    # ------------------------------------------------------------- LUNDI
    Creneau(LUN, 2, "JT", h("7h45"), h("17h30"), source="F5:G9"),
    Creneau(LUN, 3, "SR", h("8h"), h("17h30"), PAIRE, source="I5:I9"),
    Creneau(LUN, 3, "GA", h("8h"), h("13h"), IMPAIRE_A, source="J5:J6"),
    Creneau(LUN, 3, "MT", h("13h30"), h("17h30"), IMPAIRE_A, source="J8:J9"),
    Creneau(LUN, 3, "GHREA", h("8h"), h("10h"), IMPAIRE_B, "autre", source="K5"),
    Creneau(LUN, 3, "DE", h("10h"), h("15h30"), IMPAIRE_B, source="K6:K8 (DEVOS)"),
    Creneau(LUN, 4, "GA", h("8h"), h("13h"), PAIRE, source="M5:M6 (2 prothèses)"),
    Creneau(LUN, 4, "URGENCES", h("13h30"), h("15h30"), PAIRE, "urgence", source="M8"),
    Creneau(LUN, 4, "TDO/BS", h("8h"), h("10h"), IMPAIRE_A, "urgence", source="N5"),
    Creneau(LUN, 4, "DS", h("10h"), h("15h30"), IMPAIRE_A, source="N6:N8"),
    Creneau(LUN, 4, "RL", h("8h"), h("13h"), IMPAIRE_B, source="O5:O6"),
    Creneau(LUN, 4, "MT", h("13h30"), h("17h30"), IMPAIRE_B, source="O8:O9"),
    Creneau(LUN, 5, "FN", h("8h"), h("15h30"), source="Q6:R8 (TVO S6 = 7,5)"),
    # ------------------------------------------------------------- MARDI
    Creneau(MAR, 2, "JT", h("7h45"), h("17h30"), IMPAIRE, source="F12:G15"),
    Creneau(MAR, 2, "JT", h("7h45"), h("15h30"), PAIRE, source="F12:G15"),
    Creneau(MAR, 3, "TR", h("8h"), h("13h"), source="I12:K12"),
    Creneau(MAR, 3, "MT", h("13h30"), h("17h30"), PAIRE, source="I14:I15"),
    Creneau(MAR, 3, "URGENCES", h("13h30"), h("15h30"), IMPAIRE, "urgence", source="J14:K14"),
    Creneau(MAR, 4, "CT", h("8h"), h("15h30"), source="M12:O14"),
    Creneau(MAR, 5, "MO", h("8h"), h("17h30"), source="Q12:R15"),
    # ---------------------------------------------------------- MERCREDI
    Creneau(MER, 2, "CL", h("7h45"), h("17h30"), source="F18:G21"),
    Creneau(MER, 3, "DE", h("8h"), h("17h30"), source="I18:K21"),
    Creneau(MER, 4, "DN", h("8h"), h("13h"), source="M18:O18"),
    Creneau(MER, 4, "URGENCES", h("13h30"), h("15h30"), type="urgence", source="M20"),
    Creneau(MER, 5, "MT", h("8h"), h("13h"), PAIRE, source="Q18"),
    Creneau(MER, 5, "URGENCES", h("13h30"), h("15h30"), PAIRE, "urgence", source="Q20"),
    Creneau(MER, 5, "JE", h("8h"), h("15h30"), IMPAIRE, source="R18:R20"),
    # ------------------------------------------------------------- JEUDI
    Creneau(JEU, 2, "DE", h("7h45"), h("17h30"), source="F24:G27"),
    # SR : bloc fusionné I24:K26 + « Stop 15h30 » en I27, mais TVO L24 = 9,5 h
    # (= 8h-17h30). Incohérence de la feuille : on retient 15h30.
    Creneau(JEU, 3, "SR", h("8h"), h("15h30"), source="I24:K26 (L24 dit 9,5 h)"),
    Creneau(JEU, 4, "CT", h("8h"), h("15h30"), source="M24:O26"),
    Creneau(JEU, 5, "LZ", h("8h"), h("17h30"), source="Q24:R27"),
    # ---------------------------------------------------------- VENDREDI
    Creneau(VEN, 2, "CL", h("7h45"), h("13h"), source="F30:G30"),
    Creneau(VEN, 2, "URGENCES", h("13h30"), h("15h30"), type="urgence", source="F32:G32"),
    Creneau(VEN, 3, "CU", h("8h"), h("13h"), source="I30:K30"),
    Creneau(VEN, 3, "MT", h("13h30"), h("15h30"), source="I32:K32"),
    Creneau(VEN, 4, "HA", h("8h"), h("13h"), PAIRE, source="M30"),
    Creneau(VEN, 4, "FP", h("13h30"), h("17h30"), PAIRE, source="M32:M33"),
    Creneau(VEN, 4, "SM", h("8h"), h("17h30"), IMPAIRE, source="N30:O33"),
    Creneau(VEN, 5, "LR", h("8h"), h("13h"), source="Q30:R30"),
    Creneau(VEN, 5, "LIBRE", h("13h30"), h("17h30"), type="libre", source="Q32:R33"),
]


def parite_semaine(d: dt.date) -> str:
    s = d.isocalendar()[1]
    if s % 2 == 0:
        return PAIRE
    return IMPAIRE_A if s % 4 == 1 else IMPAIRE_B


def s_applique(c: Creneau, d: dt.date) -> bool:
    p = parite_semaine(d)
    return (c.parite == TOUTES or c.parite == p
            or (c.parite == IMPAIRE and p in (IMPAIRE_A, IMPAIRE_B)))


def creneaux_du(d: dt.date, types=("programme",)) -> list[Creneau]:
    if d.weekday() > 4:
        return []
    return [c for c in GRILLE
            if c.jour_semaine == d.weekday() and c.type in types and s_applique(c, d)]


def derouler_grille(jour_zero: dt.date, nb_jours: int, med_ids: dict[str, int],
                    premier_id: int = 0, feries: set[dt.date] = frozenset(),
                    types=("programme",)) -> list[Vacation]:
    """Déroule la grille sur [jour_zero, jour_zero + nb_jours).

    `med_ids` : code praticien -> med_id de l'instance (les codes absents de
    `med_ids` sont ignorés, ce qui permet de ne garder que les praticiens connus).
    """
    vacs, vid = [], premier_id
    for j in range(nb_jours):
        d = jour_zero + dt.timedelta(days=j)
        if d in feries:
            continue
        for c in creneaux_du(d, types):
            if c.code not in med_ids:
                continue
            vacs.append(Vacation(vid, bloc_id=c.salle, med_id=med_ids[c.code], jour=j,
                                 debut=c.debut, fin=c.fin,
                                 etiquette=f"{c.code} S{c.salle} {d:%d/%m}"))
            vid += 1
    return vacs


def tvo_hebdo_moyen(par_salle: bool = True, types=("programme", "urgence", "libre", "autre")):
    """TVO moyen par semaine (sur le cycle de 4 semaines), en heures —
    pour contrôler la transcription contre les colonnes TVO de la feuille."""
    lundis = [dt.date(2023, 1, 2) + dt.timedelta(weeks=k) for k in range(4)]
    tot: dict[int, float] = {}
    for l in lundis:
        for k in range(5):
            for c in creneaux_du(l + dt.timedelta(days=k), types):
                tot[c.salle] = tot.get(c.salle, 0) + (c.fin - c.debut) / 60 / 4
    return tot if par_salle else sum(tot.values())
