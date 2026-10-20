r"""
evaluation_finale.py — Les mesures de la partie « Expérimentations et résultats » (V3).

Urgences séparées (diagnostic FSSA) : elles ne passent pas par la consultation, sont opérées à
leur date réelle et s'ajoutent aux lits et aux admissions mesurés (protocole.mesurer_profils).
Critère n°1 : variance des lits des nuits de semaine (lundi-vendredi soir), vacances comprises.

  A. Matrice grille x règle de date (premier créneau, règle hybride), 3 graines.
  B. Les chaînes mono-méthode : chaîne X = grille X (étude statique, plus faible coût)
     + dates par lots du jour avec X (étude dynamique) + ordre de la journée avec X ;
     la référence (grille actuelle + premier créneau) et la chaîne hybride.
     Trois graines pour les dates, ordre des journées un jour ouvré sur deux (graine 0).
  C. Ablations de la règle hybride : fenêtre Δ, patients fantômes, choix du patient.
Sortie : resultats/final.json
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import statistics as st
import sys
import time
from multiprocessing import Pool

from protocole import (charger, instance, grille_actuelle_neutre, lire_grille, mesurer,
                       mesurer_reel, Reglages, RACINE)
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
import algos_journee as AJ
from etude_dynamique import dater_par_lots

GRAINES = (0, 1, 2)
BUDGET_LOT, BUDGET_JOUR = 400, 1000
CLES = ["lits_crit1", "lits_crit1_max", "lits_semaine_variance", "lits_moyenne", "lits_variance",
        "lits_ecart_type", "lits_max", "lits_p95", "lits_variation_nuit", "lits_semaine_moy",
        "lits_weekend_moy", "places_ecart_type", "places_max", "sans_date", "delai_median",
        "delai_moyen", "delai_p90", "heures_offertes_semaine", "occupation_reelle_pct",
        "remplissage_estime_pct", "patients_par_semaine", "vacations_inutilisees_pct", "conflits",
        "depassements_h"]
GRILLES = ("actuelle", "statique_tabou", "statique_recuit", "statique_genetique", "hybride")
CHAINES = [  # nom, grille, dates, journée, graines heuristiques de l'AG pour la journée
    ("Référence", "actuelle", "premier", None, False),
    ("Chaîne tabou", "statique_tabou", "tabou", "tabou", False),
    ("Chaîne recuit", "statique_recuit", "recuit", "recuit", False),
    ("Chaîne génétique", "statique_genetique", "génétique", "génétique", False),
    ("Chaîne hybride", "hybride", "hybride", "tabou", True),
]
ABLATIONS = [(0, "previsions", "plus_proche"), (7, "previsions", "plus_proche"),
             (14, "aucun", "plus_proche"), (14, "v1", "plus_proche"),
             (14, "previsions_sans_urgences", "plus_proche"), (14, "previsions", "plus_proche"),
             (21, "previsions", "plus_proche"), (28, "previsions", "plus_proche"),
             (14, "previsions", "meilleur")]


def _donnees():
    df, don = charger()
    return df, don, Reglages()


def _lignes(don, g):
    return grille_actuelle_neutre(don) if g == "actuelle" else lire_grille(g)["vacations"]


def _agreger(ms, **meta):
    agg = dict(meta)
    for k in CLES:
        vals = [m[k] for m in ms]
        agg[k] = st.mean(vals)
        agg[k + "_sd"] = st.pstdev(vals)
    agg["lits_serie"] = ms[0]["lits_serie"]
    return agg


def matrice(args):
    g, r = args
    _, don, reg = _donnees()
    ms = []
    for s in GRAINES:
        R = PremierCreneau() if r == "premier" else RegleHybride(don, delta=14)
        ms.append(mesurer(don, simuler(instance(don, _lignes(don, g), reg), R, graine=s)))
    return _agreger(ms, grille=g, regle=r, nom=f"grille {g} + {r}")


def chaine(args):
    nom, g, dates, algo, graines_ag, graine = args
    _, don, reg = _donnees()
    JT = set(don.jours_ouvres(*don.test))
    lignes_j = []

    def apres(sol, j):
        if algo is None or graine != 0 or j not in JT or j % 2:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        try:
            s = AJ.ALGOS[algo](sol.inst, j, seq, Poids(), BUDGET_JOUR, 0, graines=graines_ag)
        except JourneeInfaisable:
            return
        lignes_j.append(dict(jour=j, cout=s.cout, cout_depart=s.cout_depart, secondes=s.secondes,
                             **AJ.metriques(sol.inst, j, s.sequences)))

    t = time.time()
    traces = [] if graine == 0 and dates not in ("premier", "hybride") else None
    if dates == "premier":
        sol = simuler(instance(don, _lignes(don, g), reg), PremierCreneau(), graine=graine, apres_figer=apres)
    elif dates == "hybride":
        sol = simuler(instance(don, _lignes(don, g), reg), RegleHybride(don, delta=14), graine=graine,
                      apres_figer=apres)
    else:
        sol, _ = dater_par_lots(don, _lignes(don, g), dates, BUDGET_LOT, graine, reg=reg,
                                apres_figer=apres, traces=traces)
    m = mesurer(don, sol, nom, journees=lignes_j or None)
    m.update(grille=g, dates=dates, journee=algo, graine=graine, secondes=time.time() - t)
    if traces is not None:
        m["traces_lots"] = traces
    return m


def ablation(args):
    g, d, fa, ch = args
    _, don, reg = _donnees()
    ms = [mesurer(don, simuler(instance(don, _lignes(don, g), reg),
                               RegleHybride(don, delta=d, fantomes=fa, choix=ch), graine=s))
          for s in GRAINES]
    return _agreger(ms, grille=g, delta=d, fantomes=fa, choix=ch)


def main(parties="ABC"):
    t_tot = time.time()
    df, don, _ = _donnees()
    f = RACINE / "resultats" / "final.json"
    out = json.loads(f.read_text()) if f.exists() and parties != "ABC" else {}
    out["reel"] = mesurer_reel(don)
    with Pool(2) as P:
        if "A" in parties:
            out["matrice"] = P.map(matrice, [(g, r) for g in GRILLES for r in ("premier", "hybride")],
                                   chunksize=1)
            for a in out["matrice"]:
                print(f"{a['grille']:20} {a['regle']:8} crit1 {a['lits_crit1']:6.2f}±{a['lits_crit1_sd']:.2f} "
                      f"pic {a['lits_max']:4.1f} sans {a['sans_date']:5.1f} délai {a['delai_median']:5.1f}",
                      flush=True)
        if "B" in parties:
            taches = [(n, g, d, a, ga, s) for (n, g, d, a, ga) in CHAINES for s in GRAINES]
            res = P.map(chaine, taches, chunksize=1)
            out["chaines"] = []
            for (n, g, d, a, ga) in CHAINES:
                ms = [r for r in res if r["nom"] == n]
                ms.sort(key=lambda r: r["graine"])
                agg = _agreger(ms, nom=n, grille=g, dates=d, journee=a)
                m0 = ms[0]
                for k, v in m0.items():
                    if k.startswith("jour_") or k in ("traces_lots",):
                        agg[k] = v
                agg["secondes"] = st.mean(r["secondes"] for r in ms)
                out["chaines"].append(agg)
                print(f"{n:18} crit1 {agg['lits_crit1']:6.2f}±{agg['lits_crit1_sd']:.2f} pic {agg['lits_max']:4.1f} "
                      f"sans {agg['sans_date']:5.1f} délai {agg['delai_median']:5.1f} "
                      f"creux {agg.get('jour_creux_min', 0):5.0f}", flush=True)
        if "C" in parties:
            taches = [("hybride", d, fa, ch) for d, fa, ch in ABLATIONS]
            taches += [("actuelle", 14, fa, "plus_proche") for fa in ("aucun", "previsions")]
            out["ablations"] = P.map(ablation, taches, chunksize=1)
            for a in out["ablations"]:
                print(f"ablation {a['grille']:9} Δ={a['delta']:2d} {a['fantomes']:25} {a['choix']:11} "
                      f"crit1 {a['lits_crit1']:6.2f} sans {a['sans_date']:5.1f} délai {a['delai_median']:5.1f}",
                      flush=True)
    out["secondes"] = time.time() - t_tot
    f.write_text(json.dumps(out, default=float))
    print(f"terminé en {out['secondes']:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABC")
