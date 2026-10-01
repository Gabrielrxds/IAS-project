"""Construire plusieurs grilles par tabou, puis les tester sur 2022.
    python3 experience_grille.py [--donnees X] <nom> [<nom> ...]
"""
import json, statistics as st, sys, time
from collections import defaultdict, Counter
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import chemins  # rend coeur/ et vacations/ importables
from donnees import (charger_historique, construire_estimateur, construire_instance,
                     id_vers_code)
from grille_tabou import (blocs_depuis_grille, grille_depuis_blocs, calibrer_demande,
                          TabouGrille, ParametresGrille)
from propositions import choix_plus_proche
from simulation import simuler_annee

C = chemins.fichier_donnees()
df = charger_historique(C); est = construire_estimateur(df)
blocs = blocs_depuis_grille()
prat = {b.proprio for b in blocs if b.proprio and not b.reserve}
DEM = {an: calibrer_demande(df, est, [an], prat) for an in (2019, 2020, 2021, 2022)}
POOL = calibrer_demande(df, est, [2019, 2020, 2021], prat)

def demande_periode(d0, sem):
    """Demande sur une période quelconque, ramenée au cycle de 4 semaines."""
    from grille_tabou import Demande
    inst, _ = construire_instance(df, est, debut_periode=d0, nb_semaines=sem, graine=0)
    h = defaultdict(float)
    for p in inst.patients.values():
        m = id_vers_code(p.med_id)
        if m in prat:
            h[m] += (p.duree_op + p.marge_perso + inst.tis) / 60
    return Demande({m: h.get(m, 0.0) / (sem / 4) for m in prat}, POOL.profil_lits,
                   source=f"{d0}, {sem} semaines")

from grille_tabou import Demande
T4 = demande_periode("2021-10-04", 13)
ENV = Demande({m: max(DEM[2019].heures_cycle[m], DEM[2020].heures_cycle[m],
                      DEM[2021].heures_cycle[m], T4.heures_cycle[m]) for m in prat},
              POOL.profil_lits, source="enveloppe haute")
TOUT = [DEM[2019], DEM[2020], DEM[2021]]

VARIANTES = {
    "actuelle":  None,
    "prevision": dict(demande=POOL),
    "derniere":  dict(demande=DEM[2021]),
    "robuste":   dict(demande=POOL, scenarios=[DEM[2019], DEM[2020], DEM[2021]]),
    "oracle":    dict(demande=DEM[2022]),
    **{f"robuste_K{k}": dict(demande=POOL, scenarios=[DEM[2019], DEM[2020], DEM[2021]], budget=k)
       for k in (4, 8, 16, 32)},
    # --- sans libération de blocs (défaut du code désormais) ---
    "robuste_nl":   dict(demande=POOL, scenarios=TOUT, liberer=False),
    "recente_nl":   dict(demande=T4, liberer=False),
    "enveloppe_nl": dict(demande=ENV, liberer=False),
    "enveloppe_K8":  dict(demande=ENV, liberer=False, budget=8),
    **{f"enveloppe_K{k}": dict(demande=ENV, liberer=False, budget=k) for k in (2, 4, 6, 12, 32)},
    "enveloppe_K16": dict(demande=ENV, liberer=False, budget=16),
    "enveloppe_K24": dict(demande=ENV, liberer=False, budget=24),
    "oracle_nl":    dict(demande=DEM[2022], liberer=False),
}


def valider(proprio):
    """Année 2022, ancien régime, consultations uniformes : même protocole que partout."""
    grille = grille_depuis_blocs(blocs, proprio)
    inst, _ = construire_instance(df, est, debut_periode="2022-01-03", nb_semaines=65,
                                  inscriptions_sur=364, graine=0, grille=grille)
    inst.capacite_places_jour = 18
    t = time.perf_counter()
    sol, pl, j = simuler_annee(inst, choix=choix_plus_proche, seuil=0.85,
                               seuil_densite=0.50, iterations_locales=10)
    ind = sol.indicateurs(); R = j.resume()
    L = sol.lits_jour[35:357]
    d_par = defaultdict(list); sans = Counter(); tot = Counter()
    for p in inst.patients.values():
        m = id_vers_code(p.med_id); tot[m] += 1
        v = sol.affectation.get(p.id)
        if v is None:
            sans[m] += 1
        else:
            d_par[m].append(inst.vacations[v].jour - p.jour_demande)
    d = sorted(x for l in d_par.values() for x in l)
    q = lambda s, pp: sorted(s)[int(pp * len(s))] if s else None
    return dict(
        patients=len(inst.patients), dates=R["dates_donnees"],
        sans_date=len(inst.patients) - R["dates_donnees"],
        delai_median=q(d, .5), delai_p90=q(d, .9), delai_moyen=round(st.fmean(d), 1),
        part_creux=ind["part_creux"], remplissage=ind["taux_remplissage_moyen"],
        lits_moy=st.fmean(L), lits_cv=st.pstdev(L) / st.fmean(L), lits_pic=max(L),
        lits_sem=[round(st.fmean(sol.lits_jour[i * 7:(i + 1) * 7]), 2) for i in range(65)],
        par_praticien={m: dict(patients=tot[m], sans_date=sans[m],
                               delai_median=q(d_par[m], .5), delai_p90=q(d_par[m], .9))
                       for m in tot},
        conflits=len(sol.verifier()), secondes=round(time.perf_counter() - t, 1))


for nom in sys.argv[1:]:
    v = VARIANTES[nom]
    t = time.perf_counter()
    if v is None:
        proprio = [b.proprio for b in blocs]; hist = []; tg = TabouGrille(blocs, POOL)
    else:
        P = ParametresGrille(iterations=600, budget=v.get("budget"), liberer=v.get("liberer", True))
        tg = TabouGrille(blocs, v["demande"], P, scenarios=v.get("scenarios"))
        r = tg.resoudre(); proprio = r.proprio; hist = r.historique
    t_tabou = time.perf_counter() - t
    # adéquation mesurée sur chaque année, avec la même fonction de coût
    adeq = {}
    for an, dm in DEM.items():
        ev = TabouGrille(blocs, dm)
        F, det = ev.evaluer(proprio)
        adeq[str(an)] = dict(F=F, manque=det["manque"], exces=det["exces"], cv2=det["cv2_lits"],
                             occupation={m: round(x["occupation"], 3) for m, x in det["praticiens"].items()})
    heures = defaultdict(float)
    for b in blocs:
        if not b.reserve and proprio[b.id] is not None:
            heures[proprio[b.id]] += b.heures
    libres = sum(b.heures for b in blocs if not b.reserve and proprio[b.id] is None)
    res = dict(nom=nom, secondes_tabou=round(t_tabou, 1),
               changements=sum(1 for b in blocs if proprio[b.id] != b.proprio),
               heures_libres=libres, heures=dict(heures), proprio=proprio,
               historique=hist[::5], adequation=adeq,
               demande={an: dm.heures_cycle for an, dm in DEM.items()},
               demande_prevision=POOL.heures_cycle,
               validation=valider(proprio))
    json.dump(res, open(chemins.RESULTATS / f"grille_{nom}.json", "w"), default=str)
    V = res["validation"]
    print(f"{nom:12} {res['changements']:3d} changements, {libres:4.1f} h libres | 2022 : F={adeq['2022']['F']:6.1f} | "
          f"{V['dates']} datés, {V['sans_date']} sans date, délai méd {V['delai_median']} j p90 {V['delai_p90']} j, "
          f"lits cv {V['lits_cv']:.3f} pic {V['lits_pic']}, creux {V['part_creux']:.1%} | "
          f"tabou {t_tabou:.0f}s sim {V['secondes']:.0f}s", flush=True)
