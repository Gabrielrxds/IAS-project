r"""
etude_couts.py — Version script du notebook « Fil Rouge — analyse des
coefficients de la fonction coût (méthode : algorithmes génétiques) ».

Chaîne testée pour chaque jeu de coefficients de `CoutTotal` :
  1. un AG construit une nouvelle grille de vacations sur janvier-avril 2022 ;
  2. les patients de 2022 sont insérés un par un (modèle V1, sans patients
     mobiles), notés par `CoutTotal` ;
  3. l'AG journalier ordonne chaque journée dès qu'elle est figée ;
  4. mesures sur mai-décembre 2022 (hors apprentissage).

Usage :
    python etude_couts.py                       # analyse complète (~10 min)
    python etude_couts.py --rapide              # vérification rapide (~1 min)
    python etude_couts.py --donnees chemin.xlsx --sortie resultats

Produit dans le dossier de sortie : les tableaux (CSV), les figures (PNG) et
le rapport HTML (rapport_html.py).
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # pas d'affichage : les figures sont enregistrées
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyse_couts
from analyse_couts import (COEFS, Config, Reglages, audit_praticiens, audit_sejour,
                           calibrer_cout, configs_un_facteur, etude_globale,
                           etude_journaliere, executer_global, gains_vs_reel,
                           indicateurs_reel, preparer_donnees, probleme_grille,
                           profils_reels, resume_journalier, tableau_global,
                           tracer_convergence_grilles, tracer_famille, tracer_journalier,
                           tracer_pic_places_jours, tracer_sensibilite)
from charger_historique import lire_historique
from genetique_grille import ParametresAGGrille
from genetique_jour import ParametresAG

warnings.filterwarnings("ignore")         # avertissements d'openpyxl


# ---------------------------------------------------------------------------
# Enregistrement des figures : les fonctions de tracé d'analyse_couts.py
# appellent plt.show() ; on le remplace par un enregistrement en PNG.
# ---------------------------------------------------------------------------

class Figures:
    def __init__(self, dossier: Path):
        self.dossier = dossier
        self.dossier.mkdir(parents=True, exist_ok=True)
        self.nom = "figure"

        def show(*_, **__):               # une fonction, pas une méthode : pyplot
            self._enregistrer()           # lui attribue une signature
        analyse_couts.plt.show = show

    def __call__(self, nom: str):
        self.nom = nom
        return self

    def _enregistrer(self, *_, **__):
        chemin = self.dossier / f"{self.nom}.png"
        plt.gcf().savefig(chemin, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"    figure -> {chemin}")


def titre(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def afficher(df: pd.DataFrame) -> None:
    with pd.option_context("display.float_format", lambda x: f"{x:,.2f}",
                           "display.max_columns", 40, "display.width", 200):
        print(df)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--donnees", default="donees_bloc_anonyme_pour_centrale_2026.xlsx")
    ap.add_argument("--sortie", default="resultats")
    ap.add_argument("--rapide", action="store_true", help="paramètres réduits (vérification)")
    ap.add_argument("--annee", type=int, default=2022)
    ap.add_argument("--fin-apprentissage", default="2022-04-30")
    ap.add_argument("--exclus", nargs="*", default=["SM"], help="praticiens exclus")
    ap.add_argument("--sans-rapport", action="store_true", help="ne pas générer le rapport HTML")
    args = ap.parse_args()

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    fig = Figures(sortie / "figures")
    REG = Reglages()
    ANNEES_APP = (2019, 2020, 2021)
    t_debut = time.perf_counter()

    # ================================================================ Partie 2
    titre("2.1 Paramètres")
    print(REG)
    print(f"mode {'RAPIDE' if args.rapide else 'COMPLET'} ; données : {args.donnees}")
    if not os.path.exists(args.donnees):
        raise SystemExit(f"fichier introuvable : {args.donnees}")

    titre("2.2 Lecture et contrôle de la durée de séjour")
    df_brut = pd.read_excel(args.donnees)
    col_annee = next(c for c in df_brut.columns if "Ann" in c)
    df = lire_historique(args.donnees)
    n_brut, n_ok = int((df_brut[col_annee] == args.annee).sum()), int((df["annee"] == args.annee).sum())
    print(f"{args.annee} : {n_brut} lignes, {n_ok} avec des heures de salle valides "
          f"({n_brut - n_ok} écartées)")
    afficher(audit_sejour(df, args.annee))

    titre("2.3 Praticiens absents de la grille (arrêt si c'est le cas)")
    audit_praticiens(df, args.annee)

    titre("2.4 Patients de l'étude")
    don = preparer_donnees(df, args.annee, ANNEES_APP, args.fin_apprentissage, tuple(args.exclus))
    print(f"{len(don.patients)} patients, {len(don.med_codes)} chirurgiens "
          f"(exclus : {', '.join(don.exclus) or 'aucun'})")
    print(f"apprentissage {don.date(don.app[0])} -> {don.date(don.app[1])} : "
          f"{len(don.patients_de(*don.app))} patients")
    print(f"test          {don.date(don.test[0])} -> {don.date(don.test[1])} : "
          f"{len(don.patients_de(*don.test))} patients")

    titre("2.5 Hôpital réel : lits et places")
    LITS_REEL, PLACES_REEL = profils_reels(don)
    reel = pd.DataFrame({"apprentissage": indicateurs_reel(don, LITS_REEL, PLACES_REEL, REG, "app"),
                         "test": indicateurs_reel(don, LITS_REEL, PLACES_REEL, REG, "test")})
    afficher(reel)

    # ================================================================ Partie 3
    titre("3.0 Préparation commune")
    t = time.perf_counter()
    _, sol_actuelle, REF = calibrer_cout(don, REG)
    print(f"grille actuelle simulée en {time.perf_counter() - t:.0f} s ; référence de CoutTotal : {REF}")
    PROB = probleme_grille(don, REG)
    for a in PROB.alertes:
        print("ATTENTION :", a)
    afficher(pd.DataFrame({
        "chirurgien": [p.code for p in PROB.profils],
        "patients (apprentissage)": [p.nb_patients for p in PROB.profils],
        "heures engagées / semaine": [p.minutes_semaine / 60 for p in PROB.profils],
        "demi-journées / cycle": PROB.quotas}).set_index("chirurgien"))

    if args.rapide:
        MULTS = (0, 4)
        P_GRILLE = ParametresAGGrille(taille_population=20, nb_generations=60, iterations_descente=300)
        P_JOUR = P_JOUR_GLOBAL = ParametresAG(taille_population=30, nb_generations=100, stagnation_max=30)
        PAS_31, PAS_32 = 10, 20
    else:
        MULTS = (0, 0.25, 4, 16)
        P_GRILLE = ParametresAGGrille()
        P_JOUR = ParametresAG()
        P_JOUR_GLOBAL = ParametresAG(taille_population=40, nb_generations=200, stagnation_max=50)
        PAS_31, PAS_32 = 1, 2
    JOURS_TEST = don.jours_ouvres(*don.test)

    # ---------------------------------------------------------------- 3.1
    titre("3.1 Optimisation journalière")
    RES_REF = executer_global(don, Config(), PROB, REF, REG, P_GRILLE)
    print("grille de référence :")
    afficher(PROB.en_tableau(RES_REF.grille.X))
    CONFIGS_JOUR = configs_un_facteur(("places", "remplissage"), MULTS)
    DF_JOUR = etude_journaliere(RES_REF.sol, JOURS_TEST[::PAS_31], CONFIGS_JOUR, P_JOUR)
    RESUME_JOUR = resume_journalier(DF_JOUR)
    afficher(RESUME_JOUR)
    fig("31_journalier_barres")
    tracer_journalier(RESUME_JOUR)
    fig("31_pic_places_par_jour")
    tracer_pic_places_jours(DF_JOUR, [c.nom for c in CONFIGS_JOUR if c.coef_varie in (None, "places")])

    # ---------------------------------------------------------------- 3.2
    titre("3.2 Optimisation globale")
    CONFIGS_GLOBAL = configs_un_facteur(COEFS, MULTS)
    RESULTATS = etude_globale(don, CONFIGS_GLOBAL, PROB, REF, REG, P_GRILLE,
                              P_JOUR_GLOBAL, JOURS_TEST[::PAS_32])
    TAB = tableau_global(don, RESULTATS, LITS_REEL, PLACES_REEL, REG, base=sol_actuelle, periode="test")
    TAB_APP = tableau_global(don, RESULTATS, LITS_REEL, PLACES_REEL, REG, base=sol_actuelle, periode="app")
    print("\nPériode de test (mai-déc.) :")
    afficher(TAB.drop(columns=["coef", "mult"]).astype(float))
    print("\nGains par rapport à l'hôpital réel (%, négatif = mieux) :")
    afficher(gains_vs_reel(TAB))

    for coef in COEFS:
        fig(f"32_lits_{coef}")
        tracer_famille(don, RESULTATS, LITS_REEL, coef, "lits", base=sol_actuelle)
        fig(f"32_places_{coef}")
        tracer_famille(don, RESULTATS, PLACES_REEL, coef, "places", base=sol_actuelle)
    INDICATEURS = ["Lits occupés : écart-type", "Lits occupés : pic",
                   "Places ambulatoires / jour : écart-type", "Places ambulatoires / jour : pic",
                   "Remplissage des vacations utilisées (%)", "Temps perdu dans les vacations utilisées (%)",
                   "Délai consultation → opération (j)", "Patients sans date", "Salles-jours utilisées"]
    if "Pic de places simultanées (moyenne par jour)" in TAB.columns:
        INDICATEURS.append("Pic de places simultanées (moyenne par jour)")
    fig("32_sensibilite")
    tracer_sensibilite(TAB, INDICATEURS)
    fig("32_convergence_grilles")
    tracer_convergence_grilles(RESULTATS)

    # ---------------------------------------------------------------- export
    titre("Export")
    TAB.to_csv(sortie / "resultats_global_test.csv", encoding="utf-8-sig")
    TAB_APP.to_csv(sortie / "resultats_global_apprentissage.csv", encoding="utf-8-sig")
    RESUME_JOUR.to_csv(sortie / "resultats_journalier.csv", encoding="utf-8-sig")
    DF_JOUR.to_csv(sortie / "resultats_journalier_detail.csv", encoding="utf-8-sig", index=False)
    PROB.en_tableau(RESULTATS[0].grille.X).to_csv(sortie / "grille_reference.csv",
                                                  encoding="utf-8-sig", index=False)
    print(f"CSV écrits dans {sortie}/")

    if not args.sans_rapport:
        from rapport_html import generer_rapport
        etat = {"don": don, "REG": REG, "REF": REF, "PROB": PROB, "LITS_REEL": LITS_REEL,
                "PLACES_REEL": PLACES_REEL, "sol_actuelle": sol_actuelle, "RESULTATS": RESULTATS,
                "RESUME_JOUR": RESUME_JOUR, "DF_JOUR": DF_JOUR, "MULTS": MULTS}
        chemin = generer_rapport(etat, str(sortie / "Rapport_grille_AG.html"))
        print(f"rapport -> {chemin}")
        if args.rapide:
            print("NB : le texte du rapport est écrit pour l'exécution complète ; en mode rapide, "
                  "seuls les graphiques et tableaux correspondent aux données.")

    print(f"\nterminé en {time.perf_counter() - t_debut:.0f} s")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
