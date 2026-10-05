r"""
sma_experiences.py — Le SMA de Kyllian branché sur la chaîne hybride, période de test (mai-déc. 2022).

  A. Comparaison principale : 2 plans du matin (grille actuelle + premier créneau + ordre
     glouton ; chaîne hybride complète) x 3 politiques (statique, réoptimisation, SMA),
     MÊMES aléas pour les 3 politiques d'un même plan, 20 graines.
     + chaque plan sans aléa (durées estimées puis durées réelles).
  B. Intensité des urgences : 0,5 -> 3 urgences par jour, statique vs SMA, 10 graines.
  C. Taille des réserves (tampon de TVO, lits, admissions) : le plan du matin est refait
     pour chaque réglage, statique vs SMA, 10 graines.
  D. Une journée racontée : la plus riche en événements, hybride, statique et SMA.
Sortie : resultats/sma.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import pickle
import sys
import time
from multiprocessing import Pool

import sma_pont as SP
import sma_aleas as S
from protocole import RACINE
from sma_matin import charger_ou_construire

GRAINES_A, GRAINES_BC = tuple(range(1, 21)), tuple(range(1, 11))
URG = 1.5
_CACHE = {}


RESERVES = {"": None, "_r0_0_0": (0.0, 0, 0), "_r10_3_3": (0.10, 3, 3), "_r15_4_4": (0.15, 4, 4)}


def matin(nom):
    """'hybride', 'reference', ou avec un suffixe de réserves ('hybride_r10_3_3'...)."""
    if nom not in _CACHE:
        base, _, suf = nom.partition("_")
        _CACHE[nom] = charger_ou_construire(base, RESERVES["_" + suf if suf else ""])
    return _CACHE[nom]


def jours_de(m):
    inst = m["sol"].inst
    a, b = min(m["jours"]), max(m["jours"])
    return [j for j in inst.jours_ouvres if a <= j <= b]


def optimiseur(chaine):
    return SP.optimiseur_tabou(300) if chaine.startswith("hybride") else S.optimiseur_glouton


def tache(args):
    exp, chaine, pol, graine, urg, durees = args
    m = matin(chaine)
    gen = None if graine is None else S.GenerateurAleas(urgences_par_jour=urg, p_annulation=0.04,
                                                         p_lits=0.15, graine=graine)
    mes, _ = SP.simuler(m, gen, pol, optimiseur(chaine), chaine.startswith("hybride"),
                        jours_de(m), durees=durees)
    sol = m["sol"]
    B = mes["bilans"]
    sal = [x for x in mes["salles"] if not x.get("urgence")]
    mes.update(pic_places_moy=sum(b["pic_places"] for b in B) / len(B),
               jours_10_places=sum(b["pic_places"] >= 10 for b in B),
               jours_surcap_places=sum(b["pic_places"] > 12 for b in B),
               opere_min=sum(x["opere"] for x in sal), tvo_utilise_min=sum(x["tvo"] for x in sal),
               opere_urg_min=sum(x["opere"] for x in mes["salles"] if x.get("urgence")),
               salles_depassement_30=sum(x["depassement"] > 30 for x in sal),
               dep_par_jour=[sum(x["depassement"] for x in mes["salles"] if x["jour"] == b["jour"]) for b in B])
    mes.update(exp=exp, chaine=chaine, politique=pol, graine=graine, urgences_par_jour=urg,
               durees=durees, sans_date=len(sol.hors_horizon))
    if exp != "A" or graine not in (None, 1):
        mes.pop("salles"); mes.pop("bilans")
    return mes


def plan(nom):
    """Indicateurs du plan du matin (cohorte de test) : dates, délais, volume horaire."""
    import numpy as np
    from protocole import charger
    _, don = charger()
    m = matin(nom); sol, inst = m["sol"], m["sol"].inst
    t0, t1 = don.test
    coh = [p for p, j in don.jour_reel.items() if t0 <= j <= t1]
    dl = [inst.vacations[sol.affectation[p]].jour - inst.patients[p].jour_demande
          for p in coh if sol.affectation.get(p) is not None]
    vacs = [v for v, x in inst.vacations.items() if not x.urgence and t0 <= x.jour <= t1]
    urg = [v for v, x in inst.vacations.items() if x.urgence and t0 <= x.jour <= t1]
    sem = (t1 - t0 + 1) / 7
    return dict(sans_date=len(coh) - len(dl), delai_median=float(np.median(dl)), delai_moyen=float(np.mean(dl)),
                delai_p90=float(np.percentile(dl, 90)), patients_sem=len(dl) / sem,
                heures_offertes_sem=sum(inst.vacations[v].tvo for v in vacs) / 60 / sem,
                heures_urgences_sem=sum(inst.vacations[v].tvo for v in urg) / 60 / sem,
                remplissage_pct=100 * float(np.mean([sol.charge(v) / inst.vacations[v].tvo for v in vacs if sol.nb[v]])),
                tampon=inst.tampon_urgence, reserve_lits=inst.reserve_lits, reserve_places=inst.reserve_places,
                delais=dl)


def journee_racontee():
    """Hybride, graine 1 : la journée qui mêle le plus d'événements, en statique et en SMA."""
    m = matin("hybride")
    jours = jours_de(m)
    gen = S.GenerateurAleas(urgences_par_jour=URG, p_annulation=0.04, p_lits=0.15, graine=1)
    _, J = SP.simuler(m, gen, "sma", optimiseur("hybride"), True, jours, garder=set(jours))

    def interet(j):
        b = j.bilan
        perf = {x.performatif for x in j.pf.journal}
        return (2 * ("ACCEPT_PROPOSAL" in perf) + b.annulations + b.urgences_operees_jour_meme
                + (b.urgences_reprises > 0) + min(b.depassement_min, 60) / 30 - len(j.pf.journal) / 200
                + 3 * (b.urgences_operees_jour_meme >= 2))
    jour = max(J.values(), key=interet).jour
    out = {"jour": jour}
    for pol in ("statique", "sma"):
        gen = S.GenerateurAleas(urgences_par_jour=URG, p_annulation=0.04, p_lits=0.15, graine=1)
        _, G = SP.simuler(m, gen, pol, optimiseur("hybride"), True, jours, garder={jour})
        j = G[jour]
        inst = j.inst
        salles = []
        for s in j.salles.values():
            if not s.realises and s.v.urgence:
                continue
            salles.append(dict(nom=s.nom, urgence=s.v.urgence, salle=s.v.bloc_id, debut=s.v.debut, fin=s.v.fin,
                               chir=j.inst.medecins[s.v.med_id].nom if s.v.med_id in j.inst.medecins else "URG",
                               actes=[dict(pid=p, d=d, f=f, urg=p in j.urgents, ambu=j.patient(p).ambulatoire,
                                           prevu=j.prevu.get(p), estime=j.patient(p).duree_op,
                                           vac0=j.vac_initiale.get(p))
                                      for p, d, f in s.realises]))
        out[pol] = dict(date=j.bilan.date, bilan=j.bilan.en_dict(), salles=salles,
                        aleas=[dict(t=a.t, type=a.type, pid=(a.patient.id if a.patient else a.pid),
                                    nb=a.nb_lits) for a in j.aleas],
                        arrivee={str(k): v for k, v in j.arrivee.items()},
                        journal=[str(x) for x in j.pf.journal], gantt=j.gantt())
    return out


def main():
    t0 = time.time()
    for nom in ("reference", "hybride") + tuple(f"{c}{r}" for c in ("reference", "hybride")
                                                  for r in ("_r0_0_0", "_r10_3_3", "_r15_4_4")):
        matin(nom)                     # plans du matin construits (ou lus) avant le parallélisme
    taches = []
    for ch in ("reference", "hybride"):
        taches += [("A", ch, "sma", None, 0.0, "estimees"), ("A", ch, "sma", None, 0.0, "reelles")]
        taches += [("A", ch, p, g, URG, "reelles") for p in S.POLITIQUES for g in GRAINES_A]
        taches += [("B", ch, p, g, u, "reelles") for u in (0.5, 1.0, 2.0, 3.0)
                   for p in ("statique", "sma") for g in GRAINES_BC]
        taches += [("C", f"{ch}{r}", p, g, URG, "reelles") for r in ("_r0_0_0", "_r10_3_3", "_r15_4_4")
                   for p in ("statique", "sma") for g in GRAINES_BC]
    with Pool(2) as pool:
        res = []
        for k, r in enumerate(pool.imap_unordered(tache, taches, chunksize=2), 1):
            res.append(r)
            if k % 10 == 0:
                print(f"{k}/{len(taches)} {time.time() - t0:.0f}s", flush=True)
    plans = {n: plan(n) for n in ("reference", "hybride") + tuple(f"{c}{r}" for c in ("reference", "hybride")
                                                                    for r in ("_r0_0_0", "_r10_3_3", "_r15_4_4"))}
    out = {"runs": res, "plans": plans, "journee": journee_racontee(), "secondes": time.time() - t0}
    (RACINE / "resultats" / "sma.json").write_text(json.dumps(out, default=float))
    print(f"terminé en {out['secondes']:.0f} s")


if __name__ == "__main__":
    main()
