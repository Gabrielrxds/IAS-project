r"""
demo_sma.py — Démonstration du SMA pour la partie 5 de la soutenance.

    python demo_sma.py donees_bloc_anonyme_pour_centrale_2026.xlsx          # glouton, rapide
    python demo_sma.py donees_bloc_anonyme_pour_centrale_2026.xlsx --ag     # avec l'AG journalier

Base : modele.py, charger_historique.py (TROS calimed, nuits par les dates),
genetique_jour.py, grille.py et scenario.py du groupe. Modèle V1 sans patients
mobiles, réserves et créneaux URGENCES actifs, SM exclu du niveau global.

La Solution du matin est construite ici par la règle du premier créneau
réalisable (`Solution.viole`, réserves respectées). Remplacez
`construire_solution` par le moteur du groupe (propositions.py, AG global...) :
le SMA prend n'importe quelle Solution.

Trois sorties :
  A. le JOURNAL DES MESSAGES d'une journée réelle de 2022 et ce qui s'est
     réellement passé (Gantt) ;
  B. la COMPARAISON des trois politiques, mêmes aléas, même plan du matin ;
  C. l'effet de la TAILLE DES RÉSERVES (tampon de TVO, lits, places), qui
     règle le compromis urgences / programmé puisqu'aucun dépassement n'est
     accepté.
Les tableaux B et C sont aussi écrits en CSV.
"""

import dataclasses
import sys
import time

import pandas as pd

from modele import Solution, ajouter_vacations_urgence
from scenario import construire_scenario
from sma_aleas import (POLITIQUES, GenerateurAleas, ParametresSMA,
                       construire_plans_b, optimiseur_ag_journalier,
                       optimiseur_glouton, preparer_plans, resumer,
                       sequences_du_jour, simuler_periode)

ANNEE = 2022
PRATICIENS_EXCLUS = ("SM",)        # exclus du niveau global

COLONNES = {
    "urgences_arrivees": "urgences (nouvelles)",
    "urgences_operees_jour_meme": "urgences opérées le jour même",
    "urgences_operees_lendemain": "urgences opérées le jour ouvré suivant",
    "urgences_echec": "urgences sans place (échec)",
    "attente_urgence_moy": "attente urgence, jour même (min)",
    "depassement_min": "dépassement salles (min)",
    "salles_en_depassement": "salles en dépassement",
    "conversions_nuit": "ambu convertis en nuit",
    "hebergements": "hébergements hors service",
    "transferts": "transferts de salle",
    "reordonnancements": "réordonnancements",
    "patients_decales": "programmés décalés >30 min",
    "messages": "messages",
    "temps_optimiseur_s": "temps optimiseur (s)",
}


def construire_instance(chemin: str, tampon=0.05, reserve_lits=2, reserve_places=2,
                        exclus=PRATICIENS_EXCLUS):
    inst, rapport = construire_scenario(chemin, ANNEE, (2019, 2020, 2021))
    ids = {m for m, x in inst.medecins.items() if x.nom in exclus}
    for pid in [p for p, x in inst.patients.items() if x.med_id in ids]:
        del inst.patients[pid]
    inst.indexer()
    n_urg = ajouter_vacations_urgence(inst)          # AVANT la Solution
    inst.tampon_urgence, inst.reserve_lits, inst.reserve_places = tampon, reserve_lits, reserve_places
    return inst, n_urg


def construire_solution(inst) -> Solution:
    """Premier créneau réalisable, dans l'ordre des consultations. Une date
    donnée ne change plus (V1)."""
    sol = Solution(inst)
    for pid in sorted(inst.patients, key=lambda p: (inst.patients[p].jour_demande, p)):
        if not any(sol._essayer(pid, vid) for vid in inst.vacations_possibles(pid)):
            sol.sans_date.discard(pid)
            sol.hors_horizon.add(pid)
    return sol


def tableau(resultats: dict) -> pd.DataFrame:
    return pd.DataFrame({nom: {COLONNES[k]: r.get(k, 0) for k in COLONNES}
                         for nom, r in resultats.items()}).T.round(1)


def main(chemin: str, avec_ag: bool = False):
    t0 = time.perf_counter()
    inst, n_urg = construire_instance(chemin)
    sol = construire_solution(inst)
    assert not sol.verifier(), sol.verifier()[:3]
    ind = sol.indicateurs()
    print(f"{ANNEE} : {len(inst.patients)} patients (sans {', '.join(PRATICIENS_EXCLUS)}), "
          f"{n_urg} créneaux URGENCES, tampon {inst.tampon_urgence:.0%}, réserves "
          f"{inst.reserve_lits} lits / {inst.reserve_places} places ; "
          f"{ind['patients_programmes']} programmés, 0 conflit ({time.perf_counter() - t0:.0f} s)")

    jours = [j for j in inst.jours_ouvres
             if inst.date_du_jour(j).year == ANNEE and sequences_du_jour(sol, j)]
    gen = GenerateurAleas(urgences_par_jour=1.5, p_annulation=0.04, p_lits=0.15, graine=1)
    if avec_ag:
        optimiseur = optimiseur_ag_journalier(temps_max=0.3)
        print("Plans du matin par l'AG journalier...")
        plans = preparer_plans(sol, jours, optimiseur_ag_journalier(temps_max=1.0))
        plans_b = {j: construire_plans_b(inst, j, plans[j], {"glouton": optimiseur_glouton})
                   for j in jours}
    else:
        optimiseur, plans, plans_b = optimiseur_glouton, None, None
    P = ParametresSMA()

    # ---- A. une journée racontée par ses messages ---------------------------
    _, J, _ = simuler_periode(sol, jours, gen, dataclasses.replace(P, politique="sma"),
                              optimiseur, plans, plans_b, garder_journees=True)

    def interet(j):
        perf = {m.performatif for m in j.pf.journal}
        actions = {m.contenu.get("action") for m in j.pf.journal}
        return (2 * ("ACCEPT_PROPOSAL" in perf) + j.bilan.annulations + ("avancer" in actions)
                + (j.bilan.conversions_nuit > 0) + (j.bilan.urgences_operees_jour_meme >= 2)
                + (j.bilan.urgences_reprises > 0) - len(j.pf.journal) / 150)
    jour = max(J, key=interet)
    print("\n" + "=" * 100)
    print(f"A. JOURNAL DES MESSAGES — J{jour.jour} ({jour.bilan.date}), politique SMA")
    print("=" * 100)
    print(jour.pf.texte(max_lignes=70))
    print()
    print(jour.gantt())

    # ---- B. comparaison des politiques ---------------------------------------
    print("\n" + "=" * 100)
    print(f"B. COMPARAISON SUR {len(jours)} JOURS (mêmes aléas, même plan du matin)")
    print("=" * 100)
    res = {}
    for pol in POLITIQUES:
        t1 = time.perf_counter()
        b, _, work = simuler_periode(sol, jours, gen, dataclasses.replace(P, politique=pol),
                                     optimiseur, plans, plans_b)
        work.controle_coherence()
        res[pol] = resumer(b)
        print(f"   {pol:<15} simulé en {time.perf_counter() - t1:.1f} s")
    df = tableau(res)
    print(df.T.to_string())
    df.to_csv("comparaison_politiques.csv", encoding="utf-8-sig")

    # ---- C. taille des réserves ----------------------------------------------
    print("\n" + "=" * 100)
    print("C. TAILLE DES RÉSERVES (le programmé est replanifié pour chaque réglage)")
    print("=" * 100)
    lignes = {}
    for tampon, lits, places in ((0.0, 0, 0), (0.05, 2, 2), (0.10, 3, 3), (0.15, 4, 4)):
        i2, _ = construire_instance(chemin, tampon, lits, places)
        s2 = construire_solution(i2)
        j2 = [j for j in i2.jours_ouvres
              if i2.date_du_jour(j).year == ANNEE and sequences_du_jour(s2, j)]
        for pol in ("statique", "sma"):
            b, _, _ = simuler_periode(s2, j2, gen, dataclasses.replace(P, politique=pol),
                                      optimiseur_glouton)
            r = resumer(b)
            r["programmés sans date"] = len(s2.hors_horizon)
            lignes[f"tampon {tampon:.0%}, {lits} lits, {places} places — {pol}"] = r
    df = tableau(lignes)
    df.insert(0, "programmés sans date", [lignes[k]["programmés sans date"] for k in df.index])
    print(df[["programmés sans date", "urgences opérées le jour même",
              "urgences opérées le jour ouvré suivant", "urgences sans place (échec)",
              "attente urgence, jour même (min)", "dépassement salles (min)",
              "ambu convertis en nuit"]].to_string())
    df.to_csv("effet_reserves.csv", encoding="utf-8-sig")
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s. CSV : comparaison_politiques.csv, "
          f"effet_reserves.csv")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "donees_bloc_anonyme_pour_centrale_2026.xlsx",
         avec_ag="--ag" in sys.argv)
