r"""
grille_vacations.py — La grille de vacations (planning type des salles), telle
qu'elle est dans « Vacations_Opératoires_anonymiséees.xlsx » (Version 28).

Une LIGNE = un créneau fixe de la grille : (jour de semaine, salle, début, fin,
règle de semaine) + le chirurgien qui l'occupe.
  règle : T toutes les semaines, P semaines paires, I semaines impaires,
          IA / IB une semaine impaire sur deux (alternance A / B).
Le cycle complet dure donc 4 semaines : paire, impaire A, paire, impaire B.
C'est ce cycle qui s'applique toute l'année.

Le recuit (recuit_grille.py) garde les créneaux (salles, horaires, pauses,
fins à 15h30 liées au personnel) et change QUI les occupe. Les créneaux
URGENCES et LIBRE restent fixes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, timedelta

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi"]
SEMAINES_ACTIVES = {"T": (0, 1, 2, 3), "P": (0, 2), "I": (1, 3), "IA": (1,), "IB": (3,)}
NOMS_SEMAINES = ["paire", "impaire A", "paire", "impaire B"]
FIXES = {"URG", "LIBRE"}

# (jour 0=lundi, salle, chirurgien, début, fin, règle) — grille Version 28
V28 = [
    (0, 2, "JT", "7:45", "17:30", "T"), (0, 3, "SR", "8:00", "17:30", "P"),
    (0, 3, "GA", "8:00", "13:00", "IA"), (0, 3, "MT", "13:30", "17:30", "IA"),
    (0, 3, "GA", "8:00", "10:00", "IB"), (0, 3, "DEVOS", "10:00", "15:30", "IB"),
    (0, 4, "GA", "8:00", "13:00", "P"), (0, 4, "URG", "13:30", "15:30", "P"),
    (0, 4, "URG", "8:00", "10:00", "IA"), (0, 4, "DS", "10:00", "15:30", "IA"),
    (0, 4, "RL", "8:00", "13:00", "IB"), (0, 4, "MT", "13:30", "17:30", "IB"),
    (0, 5, "FN", "8:00", "15:30", "T"),
    (1, 2, "JT", "7:45", "17:30", "I"), (1, 2, "JT", "7:45", "15:30", "P"),
    (1, 3, "TR", "8:00", "13:00", "T"), (1, 3, "MT", "13:30", "17:30", "P"),
    (1, 3, "URG", "13:30", "15:30", "I"), (1, 4, "CT", "8:00", "15:30", "T"),
    (1, 5, "MO", "8:00", "17:30", "T"),
    (2, 2, "CL", "7:45", "17:30", "T"), (2, 3, "DE", "8:00", "17:30", "T"),
    (2, 4, "DN", "8:00", "13:00", "T"), (2, 4, "URG", "13:30", "15:30", "T"),
    (2, 5, "MT", "8:00", "13:00", "P"), (2, 5, "URG", "13:30", "15:30", "P"),
    (2, 5, "JE", "8:00", "15:30", "I"),
    (3, 2, "DE", "7:45", "17:30", "T"), (3, 3, "SR", "8:00", "17:30", "T"),
    (3, 4, "CT", "8:00", "15:30", "T"), (3, 5, "LZ", "8:00", "17:30", "T"),
    (4, 2, "CL", "7:45", "13:00", "T"), (4, 2, "URG", "13:30", "15:30", "T"),
    (4, 3, "CU", "8:00", "13:00", "T"), (4, 3, "MT", "13:30", "15:30", "T"),
    (4, 4, "HA", "8:00", "13:00", "P"), (4, 4, "FP", "13:30", "17:30", "P"),
    (4, 4, "SM", "8:00", "17:30", "I"),
    (4, 5, "LR", "8:00", "13:00", "T"), (4, 5, "LIBRE", "13:30", "17:30", "T"),
]


def hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def fmt(m: int) -> str:
    return f"{m // 60}h{m % 60:02d}"


@dataclass(frozen=True)
class Ligne:
    jour: int          # 0 = lundi
    salle: int
    debut: int         # minutes
    fin: int
    regle: str

    @property
    def heures(self) -> float:
        return (self.fin - self.debut) / 60

    @property
    def heures_par_semaine(self) -> float:
        return self.heures * len(SEMAINES_ACTIVES[self.regle]) / 4

    def chevauche(self, autre: "Ligne") -> bool:
        """Même jour, horaires qui se recouvrent, et au moins une semaine du
        cycle où les deux sont actives."""
        return (self.jour == autre.jour and self.debut < autre.fin and autre.debut < self.fin
                and bool(set(SEMAINES_ACTIVES[self.regle]) & set(SEMAINES_ACTIVES[autre.regle])))


@dataclass
class Grille:
    lignes: list[Ligne]
    chirurgiens: list[str]                 # occupant de chaque ligne
    nom: str = ""

    @classmethod
    def actuelle(cls) -> "Grille":
        return cls([Ligne(j, s, hm(d), hm(f), r) for j, s, c, d, f, r in V28], [c for _, _, c, _, _, _ in V28],
                   "grille actuelle (V28)")

    def copie(self, nom: str = "") -> "Grille":
        return Grille(list(self.lignes), list(self.chirurgiens), nom or self.nom)

    def modifiables(self) -> list[int]:
        return [i for i, c in enumerate(self.chirurgiens) if c not in FIXES]

    def heures_par_semaine(self) -> dict[str, float]:
        h: dict[str, float] = {}
        for l, c in zip(self.lignes, self.chirurgiens):
            if c not in FIXES:
                h[c] = h.get(c, 0.0) + l.heures_par_semaine
        return h

    def conflits(self) -> list[tuple[int, int]]:
        """Paires de lignes où un même chirurgien serait à deux endroits à la fois."""
        res = []
        for i in range(len(self.lignes)):
            for k in range(i + 1, len(self.lignes)):
                if (self.chirurgiens[i] == self.chirurgiens[k] and self.chirurgiens[i] not in FIXES
                        and self.lignes[i].chevauche(self.lignes[k])):
                    res.append((i, k))
        return res

    # -- déroulement sur un calendrier ------------------------------------------

    @staticmethod
    def semaine_du_cycle(d: date) -> int:
        """0 paire, 1 impaire A, 2 paire, 3 impaire B (numéro de semaine ISO)."""
        iso = d.isocalendar()[1]
        if iso % 2 == 0:
            return 0 if (iso // 2) % 2 == 0 else 2
        return 1 if (iso // 2) % 2 == 0 else 3

    def creneaux_du(self, d: date):
        """(chirurgien, salle, début, fin) actifs le jour d (hors URGENCES, LIBRE)."""
        if d.weekday() > 4:
            return []
        w = self.semaine_du_cycle(d)
        return [(c, l.salle, l.debut, l.fin) for l, c in zip(self.lignes, self.chirurgiens)
                if l.jour == d.weekday() and w in SEMAINES_ACTIVES[l.regle] and c not in FIXES]

    # -- export -------------------------------------------------------------------

    def tableau(self, semaine: int) -> dict[tuple[int, int], list[str]]:
        """(jour, salle) -> textes des créneaux actifs la semaine `semaine` du cycle."""
        t: dict[tuple[int, int], list[str]] = {}
        for l, c in sorted(zip(self.lignes, self.chirurgiens), key=lambda x: x[0].debut):
            if semaine in SEMAINES_ACTIVES[l.regle]:
                t.setdefault((l.jour, l.salle), []).append(f"{c} {fmt(l.debut)}-{fmt(l.fin)}")
        return t

    def exporter_excel(self, chemin, reference: "Grille | None" = None) -> None:
        """Une feuille par type de semaine, même présentation que la grille de
        l'hôpital (jours en lignes, salles en colonnes). Les créneaux modifiés
        par rapport à `reference` sont marqués d'une étoile."""
        import pandas as pd
        change = set()
        if reference is not None:
            change = {i for i, (a, b) in enumerate(zip(self.chirurgiens, reference.chirurgiens)) if a != b}
        salles = sorted({l.salle for l in self.lignes})
        with pd.ExcelWriter(chemin) as w:
            for s, nom in ((0, "semaine paire"), (1, "semaine impaire A"), (3, "semaine impaire B")):
                lignes = []
                for j in range(5):
                    row = {"jour": JOURS[j]}
                    for sa in salles:
                        txt = [f"{'* ' if i in change else ''}{c} {fmt(l.debut)}-{fmt(l.fin)}"
                               for i, (l, c) in enumerate(zip(self.lignes, self.chirurgiens))
                               if l.jour == j and l.salle == sa and s in SEMAINES_ACTIVES[l.regle]]
                        row[f"salle {sa}"] = " / ".join(txt)
                    lignes.append(row)
                pd.DataFrame(lignes).to_excel(w, sheet_name=nom, index=False)
            h0 = reference.heures_par_semaine() if reference else {}
            h1 = self.heures_par_semaine()
            pd.DataFrame([{"chirurgien": c, "heures/semaine actuelles": round(h0.get(c, 0), 2),
                           "heures/semaine proposées": round(h1.get(c, 0), 2),
                           "écart (%)": round(100 * (h1.get(c, 0) - h0.get(c, 0)) / h0[c], 1) if h0.get(c) else None}
                          for c in sorted(set(h0) | set(h1))]).to_excel(w, sheet_name="temps par chirurgien",
                                                                         index=False)
