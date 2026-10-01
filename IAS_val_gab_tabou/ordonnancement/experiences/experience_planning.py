"""Construire le planning de vacations de zéro, puis le faire tourner un an (ancien régime).
    python3 experience_planning.py [--donnees X] <variante> [<variante> ...]
Variantes : actuelle, zero, oracle, scenarios, volume, volume_sans_lits
"""
import json, statistics as st, sys, time
from collections import defaultdict, Counter
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import chemins  # rend coeur/ et vacations/ importables
import numpy as np
from donnees import (charger_historique, construire_estimateur, construire_instance,
                     id_vers_code, planning_reel, GRILLE)
from planning_vacations import (decouper, heures_actuelles, estimer_besoin, Parametres,
                                Planificateur, planning_vers_grille)
from propositions import choix_plus_proche
from simulation import simuler_annee

C = chemins.fichier_donnees()
df = charger_historique(C); est = construire_estimateur(df)
CR = decouper(); HA = heures_actuelles(); PRAT = sorted(HA)
BUDGET = sum(HA.values())


def stats_lits(serie):
    x = np.array(serie, dtype=float)
    return dict(moyenne=float(x.mean()), mediane=float(np.median(x)), variance=float(x.var()),
                ecart_type=float(x.std()), cv=float(x.std() / x.mean()) if x.mean() else 0.0,
                p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)),
                min=float(x.min()), max=float(x.max()))


def simuler(grille):
    """2022, ancien régime, consultations uniformes : le protocole de tout le projet."""
    inst, _ = construire_instance(df, est, debut_periode="2022-01-03", nb_semaines=65,
                                  inscriptions_sur=364, graine=0, grille=grille)
    inst.capacite_places_jour = 18
    t = time.perf_counter()
    sol, pl, j = simuler_annee(inst, choix=choix_plus_proche, seuil=0.85,
                               seuil_densite=0.50, iterations_locales=10)
    R = j.resume(); ind = sol.indicateurs()
    d = sorted(inst.vacations[v].jour - inst.patients[p].jour_demande
               for p, v in sol.affectation.items() if v is not None)
    sans = Counter(id_vers_code(inst.patients[p].med_id)
                   for p, v in sol.affectation.items() if v is None)
    for p in sol.hors_horizon:
        pass
    L = list(sol.lits_jour)
    # lits par jour de la semaine (régime établi)
    jds = defaultdict(list)
    for jj in range(35, 357):
        jds[jj % 7].append(L[jj])
    return dict(patients=len(inst.patients), dates=R["dates_donnees"],
                sans_date=len(inst.patients) - R["dates_donnees"],
                sans_date_par_praticien=dict(sans),
                delai_median=d[len(d) // 2], delai_p90=d[int(.9 * len(d))],
                part_creux=ind["part_creux"],
                lits_etabli=stats_lits(L[35:357]),       # semaines 5 à 50
                lits_annee=stats_lits(L[0:364]),         # année 2022 complète
                lits_jour=L, lits_sem=[round(float(np.mean(L[i*7:(i+1)*7])), 2) for i in range(65)],
                lits_par_jour_semaine={k: round(float(np.mean(v)), 2) for k, v in sorted(jds.items())},
                places_pic=ind["places_pic"], conflits=len(sol.verifier()),
                secondes=round(time.perf_counter() - t, 1))


def construire(oracle=False, w_lits=20.0, scenarios=0, volume_actuel=False):
    B = estimer_besoin(df, est, PRAT, oracle=oracle, scenarios=scenarios)
    if volume_actuel:
        # chaque praticien garde EXACTEMENT son volume d'heures actuel : le besoin est fixé
        # à ρ* × heures actuelles, si bien que l'offre visée égale l'offre d'aujourd'hui.
        # Seule la RÉPARTITION dans la semaine est reconstruite de zéro.
        rho = Parametres().rho
        B.heures_semaine = {m: rho * HA[m] / 4 for m in PRAT}
        B.scenarios = None
        B.source = "volume actuel de chaque praticien, répartition reconstruite"
    P = Parametres(w_lits=w_lits, budget_heures=BUDGET, departs=6, iterations=200, elites=4)
    pl = Planificateur(CR, B, P)
    t = time.perf_counter()
    elites = pl.resoudre(verbeux=True)
    t_meta = time.perf_counter() - t
    return B, pl, elites, t_meta


for nom in sys.argv[1:]:
    if nom == "actuelle":
        res = dict(nom=nom, simulation=simuler(GRILLE))
    else:
        B, pl, elites, t_meta = construire(
            oracle=(nom == "oracle"),
            w_lits=0.0 if nom.endswith("sans_lits") else 20.0,
            scenarios=40 if nom.startswith("scenarios") else 0,
            volume_actuel=nom.startswith("volume"))
        # sélection par simulation (simheuristique) : les 3 meilleures élites
        cands = []
        for rang, (F, x) in enumerate(elites[:3]):
            sim = simuler(planning_vers_grille(CR, x))
            cands.append(dict(rang=rang, cout=F, x=x, simulation=sim,
                              detail=pl.M.detail(pl.M.etat(x))))
            print(f"   élite {rang}: coût {F:.3f} | {sim['sans_date']} sans date, "
                  f"lits var {sim['lits_etabli']['variance']:.2f} méd {sim['lits_etabli']['mediane']:.1f}", flush=True)
        # règle : la plus faible variance des lits parmi celles qui ne laissent pas plus de
        # patients sans date que la meilleure élite + 5 ; sinon le moins de sans date
        mini = min(c["simulation"]["sans_date"] for c in cands)
        ok = [c for c in cands if c["simulation"]["sans_date"] <= mini + 5]
        choisie = min(ok, key=lambda c: c["simulation"]["lits_etabli"]["variance"])
        res = dict(nom=nom, besoin=B.heures_semaine, besoin_stats=B.stats, source=B.source,
                   volatilite=B.volatilite,
                   scenarios={m: v.tolist() for m, v in B.scenarios.items()} if B.scenarios else None,
                   journal=dict(departs=pl.journal.departs, relinking=pl.journal.relinking,
                                elites=pl.journal.elites, lambda_trace=pl.journal.lambda_trace[::5]),
                   secondes_meta=round(t_meta, 1), elites=[{k: v for k, v in c.items() if k != "x"}
                                                         | {"x": c["x"]} for c in cands],
                   choisie=choisie["rang"], x=choisie["x"], detail=choisie["detail"],
                   simulation=choisie["simulation"])
    json.dump(res, open(chemins.RESULTATS / f"planning_{nom}.json", "w"), default=str)
    S = res["simulation"]; L = S["lits_etabli"]
    print(f"{nom:15} {S['dates']} datés, {S['sans_date']} sans date, délai méd {S['delai_median']} j p90 {S['delai_p90']} j | "
          f"lits moy {L['moyenne']:.2f} méd {L['mediane']:.1f} var {L['variance']:.2f} max {L['max']:.0f}", flush=True)
