"""Trois nuits exemplaires pour expliquer les fantômes (rapport V5, figure 8) :
  1 un vrai patient prend la place du fantôme ; 2 personne ne vient, le fantôme s'efface ;
  3 la demande dépasse la prévision. Chaîne hybride, graine 0, nuits de semaine hors vacances
  de septembre à novembre 2022. Sortie : resultats/fantomes_cas.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import datetime as dt
import json

import numpy as np

from protocole import charger, instance, lire_grille, Reglages, RACINE, fond_urgences
from politiques import simuler
from regle_hybride import RegleHybride
from qualite_fantomes import Espion
import calendrier

AVANCES = (49, 35, 21, 14, 7)

_, don = charger()
fl, _ = fond_urgences(don)
S = Espion(RegleHybride(don, delta=14, fantomes="previsions"))
sol = simuler(instance(don, lire_grille("hybride")["vacations"], Reglages()), S, graine=0)
final = np.asarray(sol.lits_jour, float)[:len(fl)] + fl[:len(sol.lits_jour)]
fer, ponts = calendrier.feries_et_ponts(range(2021, 2024))
snaps = sorted(S.snap)


def vue(k, avance):
    """Dernière vue prise au plus tard `avance` jours avant la nuit k."""
    ts = [t for t in snaps if k - t >= avance and 0 <= k - S.snap[t][0] < len(S.snap[t][1]["E"])]
    if not ts:
        return None
    a, d = S.snap[ts[-1]]
    i = k - a
    return dict(avance=k - ts[-1], Er=float(d["Er"][i]), Eg=float(d["Eg"][i]), Eu=float(d["Eu"][i]))


nuits = []
for k in range(len(final)):
    j = don.date(k)
    if not (dt.date(2022, 9, 5) <= j <= dt.date(2022, 11, 30)) or j.weekday() >= 4:
        continue
    if calendrier.type_nuit(j, fer, ponts) != "semaine":
        continue
    v = [vue(k, a) for a in AVANCES]
    if None in v:
        continue
    nuits.append(dict(nuit=k, date=str(j), final=float(final[k]), vues=v))

def tot(x):
    return x["Er"] + x["Eg"] + x["Eu"]

cas = {}
# 1 remplacé : beaucoup de fantômes au début, les patients arrivent et prennent leur place, prévision juste
c1 = [n for n in nuits if n["vues"][0]["Eg"] >= 4 and n["vues"][-1]["Er"] >= n["vues"][0]["Er"] + 3
      and all(abs(tot(x) - n["final"]) <= 1.2 for x in n["vues"])]
# 2 non remplacé : des fantômes, peu d'arrivées, les fantômes s'effacent jusqu'aux lits finaux
c2 = [n for n in nuits if n["vues"][1]["Eg"] >= 2 and n["vues"][-1]["Er"] - n["vues"][1]["Er"] <= 1
      and tot(n["vues"][1]) - n["final"] >= 1.5 and abs(tot(n["vues"][-1]) - n["final"]) <= 0.8]
# 3 dépassé : les patients datés dépassent la prévision du début
c3 = [n for n in nuits if n["vues"][-1]["Er"] - tot(n["vues"][0]) >= 2 and abs(tot(n["vues"][-1]) - n["final"]) <= 1]
for nom, L, cle in (("remplace", c1, lambda n: sum(abs(tot(x) - n["final"]) for x in n["vues"])),
                    ("efface", c2, lambda n: -n["vues"][1]["Eg"]),
                    ("depasse", c3, lambda n: -(n["vues"][-1]["Er"] - tot(n["vues"][0])))):
    print(nom, len(L), [(n["date"], n["final"], [round(tot(x), 1) for x in n["vues"]]) for n in sorted(L, key=cle)[:4]])
    if L:
        cas[nom] = sorted(L, key=cle)[0]
(RACINE / "resultats" / "fantomes_cas.json").write_text(json.dumps(dict(avances=AVANCES, cas=cas, nuits=nuits)))
