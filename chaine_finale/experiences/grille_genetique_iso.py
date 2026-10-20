"""Grille GÉNÉTIQUE à ressources égales : même algorithme et même fitness que Killian,
mais salles 2 à 5 seulement (la grille d'origine ouvrait la salle 1, +33 % d'heures)."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from protocole import charger, sauver_grille, vacations_vers_neutre, Reglages
from analyse_couts import Config
from genetique_grille import AGGrille, ParametresAGGrille, ProblemeGrille, profils_chirurgiens
from modele import Instance
df, don = charger(); reg = Reglages()
t0, t1 = don.app
prof = profils_chirurgiens(don.patients_de(t0, t1), don.med_codes, (t1 - t0 + 1) / 7, Instance().tis)
prob = ProblemeGrille(prof, C=4, salles=(2, 3, 4, 5), taux_cible=reg.taux_cible,
                      capacites=(reg.capacite_lits - reg.reserve_lits, reg.capacite_places_jour - reg.reserve_places))
prob.calibrer()
print("alertes:", prob.alertes, "quotas", int(prob.quotas.sum()))
t = time.time()
rg = AGGrille(prob, Config().poids_grille(reg.poids_salles), ParametresAGGrille(graine=0)).resoudre()
vacs = prob.derouler(rg.X, don.jour_zero, don.nb_jours, don.feries)
sauver_grille("genetique_iso", vacations_vers_neutre(don, vacs),
              {"methode": "AG (Killian), salles 2-5", "secondes": time.time() - t, "cout": rg.cout,
               "X": rg.X.tolist(), "quotas": prob.quotas.tolist(), "codes": [p.code for p in prob.profils]})
print("ok", round(time.time() - t), "s")
