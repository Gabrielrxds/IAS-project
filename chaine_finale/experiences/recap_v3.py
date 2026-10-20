r"""
recap_v3.py — Ce qu'apporte chaque brique, pas à pas, et toutes les combinaisons (rapport V3).

ESCALIER (sans aléa du jour, lits mesurés avec les séjours réels, urgences réelles ajoutées) :
  1 référence          grille actuelle, première date libre, séjour supposé connu
  2 + grille hybride   première date libre
  3 + règle de date    fenêtre de 14 jours, sans fantômes
  4 + fantômes         = chaîne hybride, séjour supposé connu
  5 séjour prévu       on planifie avec la médiane de l'acte (durée, nuits) : situation réaliste
  6 + apprentissage    on planifie avec les prévisions du modèle d'apprentissage
  7 jour J sans agents mêmes aléas que 8, plan suivi tel quel, urgences par règle de capacité
  8 jour J avec agents système multi-agents
COMBINAISONS : fantômes (sans / avec) x prévision (médiane / apprentissage)
               x jour J (sans aléa / aléas sans agents / aléas avec agents), chaîne hybride.
Sortie : resultats/recap_v3.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

from protocole import charger, instance, lire_grille, grille_actuelle_neutre, mesurer, mesurer_profils, \
    Reglages, RACINE
from politiques import simuler, PremierCreneau
from regle_hybride import RegleHybride, calage_fantomes_ml
from ml_prevision import don_prevu, lits_reels

CALAGE_ML = False
GRAINES_PLAN = (0, 1, 2)
GRAINES_J = tuple(range(1, 11))
K = ["lits_crit1", "lits_crit1_max", "lits_max", "places_ecart_type", "lits_ratio_weekend"]


def _regle_h(dp, fant, prev):
    """Règle hybride ; avec les durées prévues par apprentissage, fantômes recalés sur ces durées."""
    # CALAGE_ML : fantômes recalés sur les durées prévues ; testé (recap_ml_calage.py), sans gain net,
    # donc non retenu : un seul calage pour toutes les configurations.
    kw = dict(calage=calage_fantomes_ml()) if (CALAGE_ML and fant == "previsions" and prev == "ml") else {}
    return RegleHybride(dp, delta=14, fantomes=fant, **kw)


def _plan(args):
    """Planification seule : mesures avec les séjours réels."""
    nom, grille, regle, fant, prev, g = args
    _, don = charger()
    dp = don_prevu(don, prev)
    lignes = grille_actuelle_neutre(dp) if grille == "actuelle" else lire_grille("hybride")["vacations"]
    R = PremierCreneau() if regle == "premier" else _regle_h(dp, fant, prev)
    sol = simuler(instance(dp, lignes, Reglages()), R, graine=g)
    m = mesurer(dp, sol)
    lits, places = lits_reels(don, sol)
    mr = mesurer_profils(don, lits, places)
    return dict(nom=nom, grille=grille, regle=regle, fantomes=fant, prevision=prev, graine=g,
                sans_date=m["sans_date"], delai_median=m["delai_median"], delai_p90=m["delai_p90"],
                depassements_h=m["depassements_h"], **{k: mr[k] for k in K})


def _matin(args):
    """Plan du matin (chaîne hybride complète, journée ordonnée la veille par tabou)."""
    fant, prev = args
    import algos_journee as AJ
    from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
    from sma_aleas import optimiseur_glouton
    _, don = charger()
    dp = don_prevu(don, prev)
    t0, t1 = dp.test
    JT = set(dp.jours_ouvres(t0, t1))
    plans = {}

    def ordre(sol, j):
        if j not in JT:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        try:
            plans[j] = AJ.ALGOS["tabou"](sol.inst, j, seq, Poids(), 1000, 0, graines=True).sequences
        except JourneeInfaisable:
            plans[j] = optimiseur_glouton(sol.inst, j, seq)

    inst = instance(dp, lire_grille("hybride")["vacations"], Reglages())
    sol = simuler(inst, _regle_h(dp, fant, prev), apres_figer=ordre)
    f = RACINE / "cache" / f"recap_matin_{fant}_{prev}.pkl"
    f.write_bytes(pickle.dumps({"sol": sol, "plans": plans, "plans_b": {}, "jours": sorted(JT)}))
    return str(f)


def _jour(args):
    fant, prev, pol, g = args
    import sma_v3 as V
    df, don = charger()
    m = pickle.loads((RACINE / "cache" / f"recap_matin_{fant}_{prev}.pkl").read_bytes())
    inst = m["sol"].inst
    jours = [j for j in inst.jours_ouvres if min(m["jours"]) <= j <= max(m["jours"])]
    ctx = V.contexte(df, don, inst)
    gen = V.GenerateurV3(V.reservoir_urgences(don, *don.test), ctx.classes_reelles, graine=g)
    out = V.simuler(m, gen, pol, jours, ctx, don_reel=don)
    mr = mesurer_profils(don, np.array(out["lits"]), np.array(out["places"]), avec_urgences=False)
    R = out["resume"]
    n = max(1, R["urgences_arrivees"])
    sal = [x for x in out["salles"] if not x["urgence"]]
    return dict(fantomes=fant, prevision=prev, politique=pol, graine=g, **{k: mr[k] for k in K},
                urg_delai=100 * R["urgences_dans_delai"] / n, urg_jour=100 * R["urgences_operees_jour_meme"] / n,
                urg_echec=100 * R["urgences_echec"] / n, attente=R["attente_urgence_moy"],
                u1=R["urgences_u1"], u1_jour=R["urgences_u1_jour_meme"],
                depassement_jour=R["depassement_min"] / R["jours"],
                salles_30=100 * np.mean([x["depassement"] > 30 for x in sal]),
                reportes=R["reportes"], reportes_jours=R["reportes_jours"] / max(1, R["reportes"]),
                a_arbitrer=R["a_arbitrer"], absences=R["absences"], relogements=R["relogements"],
                creneau_libre=R["urgences_par_creneau_libre"], conversions=R["conversions_nuit"],
                annulations=R["annulations"], secondes=out["secondes"])


ESCALIER = [("1 référence", "actuelle", "premier", "aucun", "connu"),
            ("2 + grille hybride", "hybride", "premier", "aucun", "connu"),
            ("3 + règle de date", "hybride", "hybride", "aucun", "connu"),
            ("4 + fantômes", "hybride", "hybride", "previsions", "connu"),
            ("5 séjour prévu (médiane)", "hybride", "hybride", "previsions", "mediane"),
            ("6 + nuits par apprentissage", "hybride", "hybride", "previsions", "ml_nuits"),
            ("7 + durées par apprentissage", "hybride", "hybride", "previsions", "ml"),
            ("c aucun mediane", "hybride", "hybride", "aucun", "mediane"),
            ("c aucun ml_nuits", "hybride", "hybride", "aucun", "ml_nuits"),
            ("c aucun ml", "hybride", "hybride", "aucun", "ml")]
COMBOS = [(f, p) for f in ("aucun", "previsions") for p in ("mediane", "ml_nuits", "ml")]


def main(parties="PMJ"):
    t = time.time()
    f = RACINE / "resultats" / "recap_v3.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    with Pool(2) as P:
        if "P" in parties:
            out["plans"] = P.map(_plan, [(*e, g) for e in ESCALIER for g in GRAINES_PLAN], chunksize=1)
            for e in ESCALIER:
                rs = [r for r in out["plans"] if r["nom"] == e[0]]
                print(e[0], {k: round(float(np.mean([r[k] for r in rs])), 2) for k in
                             ("lits_crit1", "lits_crit1_max", "places_ecart_type", "delai_p90", "sans_date")}, flush=True)
        if "M" in parties:
            print(P.map(_matin, COMBOS, chunksize=1), flush=True)
        if "J" in parties:
            out["jour"] = P.map(_jour, [(fa, pr, pol, g) for fa, pr in COMBOS for pol in ("statique", "sma")
                                        for g in GRAINES_J], chunksize=2)
            for fa, pr in COMBOS:
                for pol in ("statique", "sma"):
                    rs = [r for r in out["jour"] if r["fantomes"] == fa and r["prevision"] == pr and r["politique"] == pol]
                    print(fa, pr, pol, {k: round(float(np.mean([r[k] for r in rs])), 2) for k in
                                        ("lits_crit1", "lits_crit1_max", "urg_delai", "urg_jour", "urg_echec",
                                         "depassement_jour", "reportes", "reportes_jours")}, flush=True)
    out["secondes"] = time.time() - t
    f.write_text(json.dumps(out, default=float))
    print(f"fin ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "PMJ")
