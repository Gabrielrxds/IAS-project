r"""
experiences_aleas.py — Ce que coûte chaque aléa, et ce que la gestion récupère.
Grille hybride + règle hybride, patients de 2022, mesures sur mai-décembre 2022, 5 graines.

  0. référence sans aléa
  1. médecin : congés (déclarés 42 j avant) + maladie 1 % — sans / avec rattrapage
  2. médecin : indisponibilités de quelques heures (raison + durée estimée), 2 % des journées
  3. salles : pannes 0,5 % + 2 maintenances/an — relogement (salles 2-5) / sans relogement
  4. matériel : 1 manque par mois
  5. lits : 0,5 fermeture par mois (2-6 lits) ; stress : 32 lits pendant 3 semaines, 2 fois
  6. annulations 3 % — sans / avec comblement
  7. non aptes 1 % — sans / avec comblement
  8. non venus 2 %
  9. tout ensemble — gestion minimale / gestion complète (relogement, décalage, raccourcissement, comblement ; pas de rattrapage)
Sortie : resultats/aleas.json, et un tableau en Markdown à l'écran.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import statistics as st
from collections import Counter
from multiprocessing import Pool

import numpy as np

from protocole import charger, instance, lire_grille, mesurer, Reglages, RACINE
from regle_hybride import RegleHybride
import gestion_aleas as GA
import scenarios_aleas as SC
from gestion_aleas import JOUR, FermetureLits

GRAINES = range(5)
TOUT = dict(annulation=0.03, non_apte=0.01, non_venue=0.02)


def generer(nom, inst, don, g):
    A = []
    if nom in ("medecin_longues", "tout"):
        A += SC.medecin_absences(don, inst, 42, graine=g)
    if nom in ("medecin_temporaires", "tout"):
        A += SC.medecin_temporaires(inst, graine=g)
    if nom in ("salles", "tout"):
        A += SC.salles(inst, graine=g)
    if nom in ("materiel", "tout"):
        A += SC.materiel(inst, graine=g)
    if nom in ("lits", "tout"):
        A += SC.lits(inst, graine=g)
    if nom == "lits_stress":
        t0, t1 = don.test
        for j in (t0 + 40, t0 + 150):
            A.append(FermetureLits(j - 5, "epidemie", duree_estimee="21 j", debut=j * JOUR, fin=(j + 21) * JOUR, nb=32))
    return A


SCENARIOS = [
    # nom affiché, générateur, taux patients, options du gestionnaire
    ("0. Référence sans aléa", "aucun", {}, {}),
    ("1. Congés + maladie 1 % · sans rattrapage", "medecin_longues", {}, dict(rattrapage=False)),
    ("1. Congés + maladie 1 % · avec rattrapage", "medecin_longues", {}, dict(rattrapage=True)),
    ("2. Médecin indisponible quelques heures (2 %)", "medecin_temporaires", {}, {}),
    ("3. Salles · relogement", "salles", {}, {}),
    ("3. Salles · sans relogement", "salles", {}, dict(salles=())),
    ("4. Matériel manquant (1/mois)", "materiel", {}, {}),
    ("5. Lits fermés (2-6 lits, 0,5/mois)", "lits", {}, {}),
    ("5. Lits fermés · stress 32 lits × 3 sem.", "lits_stress", {}, {}),
    ("6. Annulations 3 % · sans comblement", "aucun", dict(annulation=0.03), dict(comblement=False)),
    ("6. Annulations 3 % · avec comblement", "aucun", dict(annulation=0.03), dict(comblement=True)),
    ("7. Non aptes 1 % · sans comblement", "aucun", dict(non_apte=0.01), dict(comblement=False)),
    ("7. Non aptes 1 % · avec comblement", "aucun", dict(non_apte=0.01), dict(comblement=True)),
    ("8. Non venus 2 %", "aucun", dict(non_venue=0.02), {}),
    ("9. Tout · gestion minimale", "tout", TOUT, dict(comblement=False, rattrapage=False, salles=())),
    ("9. Tout · gestion complète", "tout", TOUT, dict(comblement=True, rattrapage=False)),   # V5 : pas de rattrapage, les patients sont reprogrammés dans la grille
]


def un(args):
    k, g = args
    nom, gen, taux, opts = SCENARIOS[k]
    df, don = charger()
    inst = instance(don, lire_grille("hybride")["vacations"], Reglages())
    A = generer(gen, inst, don, g)
    sol, G = GA.simuler_aleas(inst, RegleHybride(don, delta=14, fantomes="uniformes"), A, graine=g,
                              feries=don.feries, taux=GA.TauxPatients(**taux), **opts)
    t0, t1 = don.test
    m = mesurer(don, sol)
    cohorte = [p for p, j in don.jour_reel.items() if t0 <= j <= t1]
    dans = [a for a in G.actions if t0 <= a.declaration <= t1]
    n = Counter((a.famille, a.action) for a in dans)
    act = Counter(a.action for a in dans)
    rep = [a.jour_nouveau - a.jour_initial for a in dans if a.action == "reprogramme"]
    av = [a.jour_initial - a.jour_nouveau for a in dans if a.action == "avance"]
    touches = {a.pid for a in dans if a.pid is not None and a.action in
               ("reprogramme", "a_arbitrer", "annule", "nouvelle_demande")}
    return dict(
        k=k, graine=g,
        sans_date=sum(sol.affectation[p] is None and p not in G.annules for p in cohorte),
        annules=sum(p in G.annules for p in cohorte),
        delai_median=m["delai_median"], delai_p90=m["delai_p90"],
        lits_variance=m["lits_variance"], lits_crit1=m["lits_crit1"], lits_max=m["lits_max"], conflits=m["conflits"],
        patients_semaine=m["patients_par_semaine"],
        touches=len(touches), reprogrammes=act["reprogramme"], a_arbitrer=act["a_arbitrer"],
        decalage_moyen=float(np.mean(rep)) if rep else 0.0,
        decalage_median=float(np.median(rep)) if rep else 0.0,
        plus_30j=sum(x > 30 for x in rep),
        avances=act["avance"], jours_gagnes=float(np.mean(av)) if av else 0.0,
        relogees=act["relogee"], fermees=act["fermee"], raccourcies=act["raccourcie"], decalees=act["decalee"],
        rattrapages=act["rattrapage"], deficit_lits=act["deficit_lits"], rouvertes=act["rouverte"],
        nouvelles_demandes=act["nouvelle_demande"],
        detail={f"{a}/{b}": c for (a, b), c in n.items()},
    )


def main():
    taches = [(k, g) for k in range(len(SCENARIOS)) for g in GRAINES]
    with Pool() as P:
        R = P.map(un, taches, chunksize=1)
    lignes = []
    for k, (nom, *_rest) in enumerate(SCENARIOS):
        rs = [r for r in R if r["k"] == k]
        moy = {c: st.mean(r[c] for r in rs) for c in rs[0] if c not in ("k", "graine", "detail")}
        moy.update(nom=nom, n=len(rs), runs=rs)
        lignes.append(moy)
    (RACINE / "resultats").mkdir(exist_ok=True)
    (RACINE / "resultats" / "aleas.json").write_text(json.dumps(lignes, ensure_ascii=False, indent=1, default=float))
    cols = [("touches", "Patients touchés", 0), ("reprogrammes", "Reprogrammés", 0), ("a_arbitrer", "À arbitrer", 1),
            ("decalage_moyen", "Décalage moyen (j)", 0), ("avances", "Avancés", 0), ("relogees", "Vacations relogées", 0),
            ("fermees", "Fermées", 0), ("raccourcies", "Raccourcies", 0), ("decalees", "Décalées", 0), ("deficit_lits", "Nuits en déficit", 1),
            ("sans_date", "Sans date", 1), ("annules", "Annulés", 0), ("delai_median", "Délai médian (j)", 0),
            ("lits_variance", "Variance lits", 2), ("conflits", "Conflits", 0)]
    print("| Scénario | " + " | ".join(c[1] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for l in lignes:
        print(f"| {l['nom']} | " + " | ".join(f"{l[c]:.{d}f}".replace(".", ",") for c, _, d in cols) + " |")


if __name__ == "__main__":
    main()
