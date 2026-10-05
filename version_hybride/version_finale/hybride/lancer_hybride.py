r"""
lancer_hybride.py — La chaîne HYBRIDE de bout en bout, sans aléa, jugée par le protocole commun.

    python3 hybride/lancer_hybride.py                      # grille enregistrée, 1 graine, journées ordonnées
    python3 hybride/lancer_hybride.py --graines 0 1 2      # plusieurs ordres de consultation
    python3 hybride/lancer_hybride.py --sans-journee       # plus rapide : dates seulement
    python3 hybride/lancer_hybride.py --recalculer-grille  # refait la grille hybride (~15-30 min)
    python3 hybride/lancer_hybride.py --delta 7 --fantomes aucun   # variantes de la règle

Étages :
  1. GRILLE      grilles/hybride.json (chaine_hybride.py la reconstruit) ;
  2. CONSULTATION RegleHybride : fenêtre de tolérance Δ + patients fantômes ;
  3. JOURNÉE      graines heuristiques de l'AG + tabou, 7 jours avant, chaque jour ouvré de test.
Comparée à la référence : grille actuelle + premier créneau libre.
Sortie : resultats/hybride.json (toutes les mesures) et un tableau à l'écran.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import statistics as st
import time

from protocole import (charger, instance, grille_actuelle_neutre, lire_grille, mesurer,
                       mesurer_reel, Reglages, RACINE)
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
import algos_journee as AJ

AFFICHE = [("lits_variance", "variance des lits (nuit)", 1), ("lits_p95", "lits pour 95 % des nuits", 1),
           ("lits_max", "pic de lits", 1), ("places_ecart_type", "écart-type des admissions / jour", 2),
           ("places_max", "pic d'admissions / jour", 1), ("sans_date", "patients sans date", 1),
           ("delai_median", "délai médian (j)", 1), ("delai_p90", "délai pour 90 % (j)", 1),
           ("heures_offertes_semaine", "heures de bloc offertes / semaine", 1),
           ("occupation_reelle_pct", "occupation réelle des vacations (%)", 1),
           ("patients_par_semaine", "patients opérés / semaine", 1), ("conflits", "conflits de règles", 0),
           ("jour_pic_places", "pic de places simultanées (moy./jour)", 2),
           ("jour_creux_min", "temps creux (min / jour)", 0),
           ("jour_depassement_reel_min", "dépassement réel (min / jour)", 1)]


def ordonner_journees(don, budget=1000):
    """Crochet appelé quand une journée est figée (7 jours avant) : AG (graines) + tabou."""
    JT = set(don.jours_ouvres(*don.test))
    lignes = []

    def apres(sol, j):
        if j not in JT:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        try:
            s = AJ.tabou(sol.inst, j, seq, Poids(), budget, 0, graines=True)
        except JourneeInfaisable:
            return
        lignes.append(dict(jour=j, cout=s.cout, **AJ.metriques(sol.inst, j, s.sequences)))
    return apres, lignes


def lancer(graines=(0,), journee=True, delta=14, fantomes="uniformes", choix="plus_proche", comparer=True):
    df, don = charger()
    reg = Reglages()
    grille = lire_grille("hybride")["vacations"]
    out = {"parametres": dict(graines=list(graines), journee=journee, delta=delta, fantomes=fantomes,
                              choix=choix), "reel": mesurer_reel(don), "hybride": [], "reference": []}
    for g in graines:
        t = time.time()
        apres, lignes = ordonner_journees(don) if journee else (None, None)
        regle = RegleHybride(don, delta=delta, fantomes=fantomes, choix=choix)
        sol = simuler(instance(don, grille, reg), regle, graine=g, apres_figer=apres)
        m = mesurer(don, sol, "hybride", journees=lignes)
        m["secondes"] = time.time() - t
        out["hybride"].append(m)
        print(f"hybride   graine {g} : {m['secondes']:.0f} s", flush=True)
        if comparer:
            t = time.time()
            apres_r, lignes_r = ordonner_journees(don) if journee else (None, None)
            sol_r = simuler(instance(don, grille_actuelle_neutre(don), reg), PremierCreneau(), graine=g,
                            apres_figer=apres_r)
            m = mesurer(don, sol_r, "référence", journees=lignes_r)
            m["secondes"] = time.time() - t
            out["reference"].append(m)
            print(f"référence graine {g} : {m['secondes']:.0f} s", flush=True)
    return out


def tableau(out):
    def moy(liste, k):
        v = [m[k] for m in liste if k in m]
        return st.mean(v) if v else None
    print(f"\n{'indicateur (période de test mai-déc. 2022)':42} {'réel':>8} {'actuelle':>9} {'hybride':>9}")
    for k, lab, nd in AFFICHE:
        h, r = moy(out["hybride"], k), moy(out["reference"], k)
        if h is None:
            continue
        reel = out["reel"].get(k)
        f = lambda x: "" if x is None else f"{x:.{nd}f}"
        print(f"{lab:42} {f(reel):>8} {f(r):>9} {f(h):>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chaîne hybride, protocole commun")
    ap.add_argument("--graines", type=int, nargs="+", default=[0])
    ap.add_argument("--sans-journee", action="store_true")
    ap.add_argument("--recalculer-grille", action="store_true")
    ap.add_argument("--delta", type=int, default=14)
    ap.add_argument("--fantomes", default="uniformes", choices=("aucun", "uniformes", "optimises"))
    ap.add_argument("--choix", default="plus_proche", choices=("plus_proche", "meilleur"))
    ap.add_argument("--sans-reference", action="store_true")
    a = ap.parse_args()
    if a.recalculer_grille or not (RACINE / "grilles" / "hybride.json").exists():
        import chaine_hybride
        chaine_hybride.main()
    out = lancer(a.graines, not a.sans_journee, a.delta, a.fantomes, a.choix, not a.sans_reference)
    tableau(out)
    (RACINE / "resultats").mkdir(exist_ok=True)
    for cle in ("hybride", "reference", "reel"):
        for m in (out[cle] if isinstance(out[cle], list) else [out[cle]]):
            m.pop("lits_serie", None)
    (RACINE / "resultats" / "hybride.json").write_text(json.dumps(out, default=float, indent=1))
    print("\nrésultats : resultats/hybride.json")
