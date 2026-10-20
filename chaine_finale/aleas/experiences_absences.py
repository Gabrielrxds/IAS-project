"""Absences des chirurgiens et reprogrammation — protocole commun, grille hybride.
  A. sans absence (référence)
  B. congés « réels » (semaines sans activité en 2022), déclarés 42 / 14 / 3 jours avant,
     reprogrammation hybride vs premier créneau
  C. arrêts maladie (1 % / 2 % des journées, déclarés 0-2 j avant), 5 graines, hybride vs premier
  D. congés (42 j) + maladie 1 %
  E. grille actuelle + premier créneau, avec / sans congés (réalisme : volume d'août)
Sortie : resultats/absences.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, statistics as st, datetime as dt
import numpy as np
from protocole import charger, instance, lire_grille, grille_actuelle_neutre, mesurer, Reglages, RACINE
from politiques import PremierCreneau
from regle_hybride import RegleHybride
from absences import absences_reelles, absences_maladie, simuler_absences

df, don = charger(); reg = Reglages(); t0, t1 = don.test
GR = {"hybride": lire_grille("hybride")["vacations"], "actuelle": grille_actuelle_neutre(don)}
K = ["lits_crit1", "lits_variance", "lits_max", "lits_p95", "places_ecart_type", "sans_date", "delai_median", "delai_moyen", "delai_p90",
     "patients_par_semaine", "occupation_reelle_pct", "conflits"]
reel_aout = int(((df["Date Inter"].dt.year == 2022) & (df["Date Inter"].dt.month == 8)).sum())


def un(grille, regle_nom, absences, politique, graine=0):
    ratt = politique.endswith("+rattrapage"); politique = politique.replace("+rattrapage", "")
    inst = instance(don, GR[grille], reg)
    R = RegleHybride(don, delta=14, fantomes="uniformes") if regle_nom == "hybride" else PremierCreneau()
    sol, RP = simuler_absences(inst, R, absences, politique, graine, rattrapage=ratt, feries=don.feries)
    m = {k: mesurer(don, sol)[k] for k in K}
    J = [r for r in RP.journal if t0 <= r.jour_initial <= t1]
    d = [r.jour_nouveau - r.jour_initial for r in J if r.jour_nouveau is not None]
    pre = [r.jour_initial - r.declaration for r in J]
    m.update(reprogrammes=len(J), a_arbitrer=sum(r.jour_nouveau is None for r in J),
             decalage_moyen=float(np.mean(d)) if d else 0.0, decalage_median=float(np.median(d)) if d else 0.0,
             avances=sum(x < 0 for x in d), sous_7j=sum(0 <= x <= 7 for x in d), plus_30j=sum(x > 30 for x in d),
             rattrapages=len(RP.rattrapages), heures_rattrapage=sum(inst.vacations[v].tvo for v in RP.rattrapages) / 60,
             vacations_fermees=sum(1 for v in RP.fermees if t0 <= inst.vacations[v].jour <= t1),
             aout=sum(1 for p, v in sol.affectation.items() if v is not None and don.date(inst.vacations[v].jour).year == 2022
                      and don.date(inst.vacations[v].jour).month == 8),
             decalages=d)
    return m


def moyenne(ms):
    out = {}
    for k in ms[0]:
        if k == "decalages":
            out[k] = sum((m[k] for m in ms), []); continue
        out[k] = st.mean(m[k] for m in ms); out[k + "_sd"] = st.pstdev(m[k] for m in ms)
    return out


out = {"reel_aout": reel_aout, "lignes": []}
def ajoute(nom, ms, **meta):
    r = moyenne(ms); r.update(nom=nom, n=len(ms), **meta); out["lignes"].append(r)
    print(f"{nom:55} var {r['lits_variance']:5.2f} pic {r['lits_max']:4.1f} sans {r['sans_date']:5.1f} "
          f"dméd {r['delai_median']:4.1f} ratt {r['rattrapages']:4.1f} reprog {r['reprogrammes']:6.1f} arb {r['a_arbitrer']:4.1f} "
          f"décal {r['decalage_moyen']:5.1f} août {r['aout']:5.0f} confl {r['conflits']}", flush=True)

ajoute("A. sans absence", [un("hybride", "hybride", [], "hybride")], groupe="A")
POL = ("premier", "hybride", "hybride+rattrapage")
for pre in (42, 14, 3):
    A = absences_reelles(don, pre)
    for pol in POL:
        ajoute(f"B. congés déclarés {pre} j avant · reprog. {pol}", [un("hybride", "hybride", A, pol)], groupe="B", preavis=pre, politique=pol)
for taux in (0.01, 0.02):
    for pol in POL:
        ajoute(f"C. maladie {int(taux*100)} % · reprog. {pol}",
               [un("hybride", "hybride", absences_maladie(instance(don, GR["hybride"], reg), taux, g), pol, g) for g in range(5)],
               groupe="C", taux=taux, politique=pol)
A = absences_reelles(don, 42)
for pol in ("hybride", "hybride+rattrapage"):
    ajoute(f"D. congés (42 j) + maladie 1 % · reprog. {pol}",
           [un("hybride", "hybride", A + absences_maladie(instance(don, GR["hybride"], reg), 0.01, g), pol, g) for g in range(5)],
           groupe="D", politique=pol)
ajoute("E. grille actuelle + 1er créneau · sans absence", [un("actuelle", "premier", [], "premier")], groupe="E")
ajoute("E. grille actuelle + 1er créneau · congés 42 j", [un("actuelle", "premier", A, "premier")], groupe="E")
(RACINE / "resultats" / "absences.json").write_text(json.dumps(out, default=float))
print("août réel :", reel_aout)
