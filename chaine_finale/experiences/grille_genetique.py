"""Grille de la chaîne GÉNÉTIQUE (genetique_grille.py de Killian), paramètres par défaut,
apprise sur janvier-avril 2022. Sortie : grilles/genetique.json (vacations datées)."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys, time
from protocole import charger, sauver_grille, vacations_vers_neutre, Reglages
from analyse_couts import probleme_grille, Config
from genetique_grille import AGGrille, ParametresAGGrille

graine = int(sys.argv[1]) if len(sys.argv) > 1 else 0
df, don = charger()
reg = Reglages()
prob = probleme_grille(don, reg)
t = time.time()
rg = AGGrille(prob, Config().poids_grille(reg.poids_salles), ParametresAGGrille(graine=graine)).resoudre()
vacs = prob.derouler(rg.X, don.jour_zero, don.nb_jours, don.feries)
nom = "genetique" if graine == 0 else f"genetique_g{graine}"
sauver_grille(nom, vacations_vers_neutre(don, vacs),
              {"methode": "AG (Killian)", "secondes": time.time() - t, "cout": rg.cout,
               "generations": rg.generations, "termes": rg.termes_normalises,
               "X": rg.X.tolist(), "quotas": prob.quotas.tolist(),
               "codes": [p.code for p in prob.profils], "salles": prob.salles})
print(f"{nom}: coût {rg.cout:.4f}, {rg.generations} générations, {time.time()-t:.0f} s, {len(vacs)} vacations")
