"""Grille de la chaîne RECUIT (recuit_grille.py de Lucie), grille « A » (temps de chaque
chirurgien conservé à ±10 %), meilleure de 3 exécutions comme dans etude_grille.py.
Différence avec l'étude d'origine : le modèle attendu n'apprend que sur la base
tronquée au 30 avril 2022 (l'original utilisait l'activité de toute l'année 2022,
c'est-à-dire la période de test).
Processus séparé (les modules du recuit portent les mêmes noms que ceux du groupe).
    python3 grille_recuit.py <jour_zero ISO> <nb_jours> [restarts]
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys, json, time, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemins import RACINE, RECUIT, DONNEES
sys.path.insert(0, str(RECUIT))
import pandas as pd
from donnees_reelles import charger_base
from grille_vacations import Grille
from recuit_grille import ModeleAttendu, ParamsRecuitGrille, recuit_grille

j0, n = dt.date.fromisoformat(sys.argv[1]), int(sys.argv[2])
restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 3
base = charger_base(DONNEES)
base = base[base.inter <= pd.Timestamp("2022-04-30")]
G = Grille.actuelle()
mod = ModeleAttendu(base, G, annee=2022)
t = time.time()
best = None
for g in range(restarts):
    r = recuit_grille(G, mod, ParamsRecuitGrille(graine=g))
    print(f"  exécution {g}: {r.cout_initial:.3f} -> {r.cout_final:.3f} ({r.iterations} it, {r.temps:.0f} s)")
    if best is None or r.cout_final < best.cout_final:
        best = r
g = best.grille
lignes = []
for k in range(n):
    d = j0 + dt.timedelta(days=k)
    for c, salle, deb, fin in g.creneaux_du(d):
        lignes.append([d.isoformat(), salle, deb, fin, c])
modifs = sum(a != b for a, b in zip(g.chirurgiens, G.chirurgiens))
out = {"nom": "recuit", "meta": {"methode": "recuit simulé (Lucie), grille A", "secondes": time.time() - t,
       "cout": best.cout_final, "cout_initial": best.cout_initial, "creneaux_modifies": modifs,
       "conflits": len(g.conflits())},
       "vacations": lignes}
from protocole import sauver_grille  # noqa: E402  (codes anonymisés C1..C18 sur disque)
sauver_grille(out["nom"], out["vacations"], json.loads(json.dumps(out["meta"], default=str)))
print(f"recuit: coût {best.cout_final:.3f}, {modifs} créneaux modifiés, {len(lignes)} vacations, {time.time()-t:.0f} s")
