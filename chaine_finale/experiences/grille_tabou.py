"""Grille de la chaîne TABOU (planning_vacations.py de Gabriel) : reconstruction de zéro
par demi-journées, volume horaire actuel de chaque praticien conservé (variante « volume »),
GRASP -> tabou à oscillation stratégique -> path relinking -> sélection par simulation.
Différence avec l'étude d'origine : la sélection simheuristique des élites rejoue 2021
(et non 2022, qui est la période de test du protocole commun).
Processus séparé.   python3 grille_tabou.py <jour_zero ISO> <nb_jours>
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys, json, time, datetime as dt
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chemins import RACINE, TABOU, DONNEES
sys.path[:0] = [str(TABOU / "coeur"), str(TABOU / "vacations")]
import numpy as np
from donnees import charger_historique, construire_estimateur, construire_instance
from planning_vacations import (decouper, heures_actuelles, estimer_besoin, Parametres,
                                Planificateur, planning_vers_grille, DEMI)
from propositions import choix_plus_proche
from simulation import simuler_annee

j0, n = dt.date.fromisoformat(sys.argv[1]), int(sys.argv[2])
df = charger_historique(str(DONNEES))
df_app = df[df["date_inter"] <= "2022-04-30"] if "date_inter" in df.columns else df
est = construire_estimateur(df_app)
CR = decouper(); HA = heures_actuelles(); PRAT = sorted(HA)
B = estimer_besoin(df_app, est, PRAT)
rho = Parametres().rho
B.heures_semaine = {m: rho * HA[m] / 4 for m in PRAT}      # variante « volume »
B.scenarios = None
P = Parametres(w_lits=20.0, budget_heures=sum(HA.values()), departs=6, iterations=200, elites=4)
pl = Planificateur(CR, B, P)
t = time.time()
elites = pl.resoudre(verbeux=False)
t_meta = time.time() - t

def variance_2021(x):
    inst, _ = construire_instance(df, est, debut_periode="2021-01-04", nb_semaines=65,
                                  inscriptions_sur=364, graine=0, grille=planning_vers_grille(CR, x))
    inst.capacite_places_jour = 18
    sol, _, j = simuler_annee(inst, choix=choix_plus_proche, seuil=0.85, seuil_densite=0.50,
                              iterations_locales=10)
    L = np.array(sol.lits_jour[35:357], dtype=float)
    return float(L.var()), len(inst.patients) - j.resume()["dates_donnees"]

cands = []
for rang, (F, x) in enumerate(elites[:3]):
    v, sans = variance_2021(x)
    cands.append((rang, F, x, v, sans))
    print(f"  élite {rang}: coût {F:.3f}, 2021 : variance lits {v:.2f}, {sans} sans date", flush=True)
mini = min(c[4] for c in cands)
rang, F, x, v, sans = min((c for c in cands if c[4] <= mini + 5), key=lambda c: c[3])

# déroulage daté : semaine du cycle = (semaines écoulées depuis le 3 janvier 2022) mod 4,
# la convention de planning_vacations / donnees.py ; matin + après-midi du même praticien
# dans la même salle = une vacation de journée (même règle que la chaîne génétique)
par = defaultdict(dict)
for c in CR:
    if x[c.id] is not None:
        par[(c.semaine, c.jour, c.salle)][c.demi] = x[c.id]
ref = dt.date(2022, 1, 3)
lignes = []
for k in range(n):
    d = j0 + dt.timedelta(days=k)
    if d.weekday() > 4:
        continue
    w = ((d - ref).days // 7) % 4
    for salle in (2, 3, 4, 5):
        h = par.get((w, d.weekday(), salle), {})
        if 0 in h and 1 in h and h[0] == h[1]:
            lignes.append([d.isoformat(), salle, 8 * 60, 17 * 60 + 30, h[0]])
        else:
            for demi, m in h.items():
                a, b = DEMI[demi]
                lignes.append([d.isoformat(), salle, int(a * 60), int(b * 60), m])
out = {"nom": "tabou", "meta": {"methode": "GRASP + tabou à oscillation + path relinking (Gabriel)",
       "secondes": time.time() - t, "secondes_meta": t_meta, "cout": F, "elite": rang,
       "variance_2021": v, "x": x, "creneaux": [[c.id, c.semaine, c.jour, c.salle, c.demi] for c in CR],
       "journal": {"departs": pl.journal.departs, "relinking": pl.journal.relinking}},
       "vacations": lignes}
from protocole import sauver_grille  # noqa: E402  (codes anonymisés C1..C18 sur disque)
sauver_grille(out["nom"], out["vacations"], json.loads(json.dumps(out["meta"], default=str)))
print(f"tabou: élite {rang}, coût {F:.3f}, {len(lignes)} vacations, méta {t_meta:.0f} s, total {time.time()-t:.0f} s")
