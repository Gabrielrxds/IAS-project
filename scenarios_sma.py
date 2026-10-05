r"""
scenarios_sma.py — Simulations & scénarios réalistes (charge, aléas, échelle).

    python scenarios_sma.py donees_bloc_anonyme_pour_centrale_2026.xlsx
    python scenarios_sma.py donees_bloc_anonyme_pour_centrale_2026.xlsx --rapide

À placer dans le dossier du projet (à côté de modele.py, scenario.py, grille.py,
demo_sma.py, sma_aleas.py). Il réutilise le moteur du groupe tel quel : aucune
formule inventée, tous les chiffres sortent de la simulation.

Les 8 scénarios = charge × perturbations × échelle
--------------------------------------------------
  charge     normale   : les patients de 2022 (données réelles)
             surcharge : +35 % de patients programmés (clones de patients réels,
                         même consultation, même chirurgien) -> `--surcharge`
  aléas      stable    : aucune perturbation, les actes durent la durée estimée
             aléas     : SMA de sma_aleas.py : urgences du jour (Poisson),
                         annulations, lits fermés, et RETARDS réels (durées
                         observées dans l'historique, inconnues des agents)
  échelle    grande    : tous les praticiens, capacités réelles
             petite    : les 2 praticiens les plus actifs, lits et places
                         réduits au prorata de leur part de patients

Pour chaque scénario avec aléas, les trois politiques sont comparées avec les
MÊMES aléas et le MÊME plan du matin : statique / réoptimisation / SMA.
Plusieurs graines sont tirées et moyennées (`--graines`).

Sorties
-------
  tableau console, resultats_scenarios_brut.csv (une ligne par scénario ×
  politique × graine), synthese_scenarios.csv (moyennes) et quatre figures :

  courbes_lits.png                 occupation des lits au fil de l'année (moyenne
                                   glissante 7 j) : plan du matin vs après aléas,
                                   statique vs SMA, avec la capacité, pour
                                   charge normale/surcharge × petite/grande échelle
  courbes_sensibilite_aleas.png    urgences par jour (0 -> 4) en abscisse : échecs,
                                   dépassements et attente des urgences, par politique
  courbes_sensibilite_charge.png   surcharge (0 -> +60 %) en abscisse : patients sans
                                   date, σ des lits, pic de lits, échecs d'urgences
  synthese_scenarios.png           barres : trois politiques sur les 4 scénarios à aléas

  `--sans-balayage` ne produit que les deux premières (plus rapide).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import random
import sys
import time
from collections import Counter

import pandas as pd

from demo_sma import ANNEE, PRATICIENS_EXCLUS, construire_solution
from modele import ajouter_vacations_urgence
from scenario import construire_scenario
from sma_aleas import (POLITIQUES, GenerateurAleas, ParametresSMA,
                       optimiseur_glouton, resumer, sequences_du_jour,
                       simuler_periode)

SANS_PERTURBATION = "aucune perturbation"

# Colonnes affichées dans le tableau de synthèse (clé -> libellé)
COLONNES = {
    "programmes_sans_date": "sans date",
    "taux_remplissage": "remplissage",
    "lits_ecart_type": "σ lits",
    "lits_pic": "pic lits",
    "urgences_arrivees": "urgences",
    "urgences_operees_jour_meme": "urg. J",
    "urgences_operees_lendemain": "urg. J+1",
    "urgences_echec": "urg. échec",
    "attente_urgence_moy": "attente urg. (min)",
    "depassement_min": "dépass. (min)",
    "conversions_nuit": "ambu→nuit",
    "patients_decales": "décalés >30 min",
    "temps_s": "temps (s)",
}


# ---------------------------------------------------------------------------
# 1. Construction d'une instance de scénario
# ---------------------------------------------------------------------------

def charger_base(chemin: str):
    """Instance 2022 du groupe (durées estimées sur 2019-2021). Construite
    UNE fois, puis copiée pour chaque scénario."""
    inst, _ = construire_scenario(chemin, ANNEE, (2019, 2020, 2021))
    return inst


def preparer_instance(base, charge: str, echelle: str, *, surcharge: float,
                      n_chirurgiens_petite: int, graine: int,
                      tampon: float = 0.05, reserve_lits: int = 2,
                      reserve_places: int = 2):
    """Copie de `base` mise à l'échelle et à la charge du scénario."""
    inst = copy.deepcopy(base)

    exclus = {m for m, x in inst.medecins.items() if x.nom in PRATICIENS_EXCLUS}
    compte = Counter(p.med_id for p in inst.patients.values()
                     if p.med_id not in exclus and not p.est_urgent)
    if echelle == "petite":
        garde = {m for m, _ in compte.most_common(n_chirurgiens_petite)}
    else:
        garde = set(compte)
    for pid in [pid for pid, p in inst.patients.items() if p.med_id not in garde]:
        del inst.patients[pid]

    part = sum(compte[m] for m in garde) / max(1, sum(compte.values()))

    if charge == "surcharge":
        rng = random.Random(graine)
        programmes = sorted(pid for pid, p in inst.patients.items() if not p.est_urgent)
        suivant = max(inst.patients, default=0) + 1
        for pid in rng.choices(programmes, k=round(len(programmes) * surcharge)):
            inst.patients[suivant] = dataclasses.replace(inst.patients[pid], id=suivant)
            suivant += 1

    if echelle == "petite":
        # petite structure : capacités au prorata des patients conservés
        inst.capacite_lits = max(6, round(inst.capacite_lits * part))
        inst.capacite_places = max(3, round(inst.capacite_places * part))
        inst.capacite_places_jour = max(4, round(inst.capacite_places_jour * part))
        reserve_lits = min(reserve_lits, 1)
        reserve_places = min(reserve_places, 1)

    inst.indexer()
    n_urg = ajouter_vacations_urgence(inst)          # AVANT la Solution
    inst.tampon_urgence = tampon
    inst.reserve_lits = reserve_lits
    inst.reserve_places = reserve_places
    info = {"patients": len(inst.patients), "chirurgiens": len(garde),
            "creneaux_urgence": n_urg, "capacite_lits": inst.capacite_lits,
            "capacite_places": inst.capacite_places}
    return inst, info


# ---------------------------------------------------------------------------
# 2. Un scénario
# ---------------------------------------------------------------------------

def jours_a_simuler(sol, jours_max: int | None) -> list[int]:
    inst = sol.inst
    jours = [j for j in inst.jours_ouvres
             if inst.date_du_jour(j).year == ANNEE and sequences_du_jour(sol, j)]
    return jours[:jours_max] if jours_max else jours


def lancer_scenario(base, charge: str, aleas: str, echelle: str, args,
                    courbes: dict | None = None) -> list[dict]:
    """Renvoie une ligne de résultats par (politique, graine). Si `courbes` est
    fourni, y range les profils de lits du scénario (pour les courbes)."""
    t0 = time.perf_counter()
    inst, info = preparer_instance(
        base, charge, echelle, surcharge=args.surcharge,
        n_chirurgiens_petite=args.petite_chirurgiens, graine=args.graine_base)
    sol = construire_solution(inst)
    assert not sol.verifier(), sol.verifier()[:3]
    ind = sol.indicateurs()
    t_plan = time.perf_counter() - t0
    jours = jours_a_simuler(sol, args.jours)
    cc = None
    if courbes is not None:
        cc = courbes.setdefault((charge, echelle), {})
        cc["dates"] = [inst.date_du_jour(j) for j in range(inst.nb_jours)]
        cc["cap"] = [inst.cap_lits(j) for j in range(inst.nb_jours)]
        if aleas == "stable":
            cc["plan"] = list(sol.lits_jour)

    plan = {
        "programmes_sans_date": len(sol.hors_horizon),
        "patients_programmes": ind["patients_programmes"],
        "taux_remplissage": ind["taux_remplissage_moyen"],
        "lits_ecart_type": ind["lits_ecart_type"],
        "lits_pic": ind["lits_pic"],
        "creux_total_h": ind["creux_total_h"],
    }
    base_row = {"charge": charge, "aleas": aleas, "echelle": echelle, **info, **plan,
                "jours_simules": len(jours)}

    P0 = ParametresSMA()
    lignes = []
    if aleas == "stable":
        # aucune perturbation : durées estimées, ni urgences ni annulations
        t1 = time.perf_counter()
        b, _, work = simuler_periode(sol, jours, None, dataclasses.replace(P0, politique="sma"),
                                     optimiseur_glouton, durees="estimees")
        work.controle_coherence()
        lignes.append({**base_row, "politique": SANS_PERTURBATION, "graine": 0,
                       **resumer(b), "temps_s": time.perf_counter() - t1})
    else:
        for g in range(1, args.graines + 1):
            gen = GenerateurAleas(urgences_par_jour=args.urgences,
                                  p_annulation=args.annulation,
                                  p_lits=args.lits, graine=g)
            for pol in POLITIQUES:
                t1 = time.perf_counter()
                b, _, work = simuler_periode(sol, jours, gen,
                                             dataclasses.replace(P0, politique=pol),
                                             optimiseur_glouton, durees="reelles")
                work.controle_coherence()
                if cc is not None and g == 1 and pol in ("statique", "sma"):
                    cc[pol] = list(work.lits_jour)
                lignes.append({**base_row, "politique": pol, "graine": g,
                               **resumer(b), "temps_s": time.perf_counter() - t1})
    print(f"   {info['patients']} patients, {info['chirurgiens']} chirurgiens, "
          f"{plan['programmes_sans_date']} sans date, plan en {t_plan:.0f} s, "
          f"{len(jours)} jours simulés, total {time.perf_counter() - t0:.0f} s")
    return lignes


# ---------------------------------------------------------------------------
# 3. Synthèse et figure
# ---------------------------------------------------------------------------

def synthese(brut: pd.DataFrame) -> pd.DataFrame:
    cles = ["charge", "aleas", "echelle", "politique"]
    moy = brut.groupby(cles, sort=False).mean(numeric_only=True)
    ordre = [c for c in COLONNES if c in moy.columns]
    return moy[ordre].rename(columns=COLONNES).round(2)


def figure(brut: pd.DataFrame, fichier: str) -> None:
    plt = _plt()
    if plt is None:
        return
    al = brut[brut["aleas"] == "aleas"].copy()
    if al.empty:
        return
    al["scenario"] = al["charge"] + "\n" + al["echelle"]
    mesures = [("urgences_echec", "Urgences sans place (échecs)"),
               ("depassement_min", "Dépassement des salles (min)"),
               ("attente_urgence_moy", "Attente d'une urgence (min)")]
    scenarios = list(dict.fromkeys(al["scenario"]))
    couleurs = {"statique": "#9aa5b1", "reoptimisation": "#e8a33d", "sma": "#2c7a7b"}
    fig, axes = plt.subplots(1, len(mesures), figsize=(15, 4.2))
    largeur = 0.26
    for ax, (col, titre) in zip(axes, mesures):
        for i, pol in enumerate(POLITIQUES):
            vals = [al[(al["scenario"] == s) & (al["politique"] == pol)][col].mean()
                    for s in scenarios]
            ax.bar([k + (i - 1) * largeur for k in range(len(scenarios))], vals,
                   largeur, label=pol, color=couleurs[pol])
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, fontsize=9)
        ax.set_title(titre, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("Trois politiques face aux mêmes aléas (moyenne sur les graines)", y=1.02)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.close(fig)


STYLE = {"statique": ("#e8a33d", "statique"), "reoptimisation": ("#9aa5b1", "réoptimisation"),
         "sma": ("#2c7a7b", "SMA")}


def _axe(ax, titre=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    if titre:
        ax.set_title(titre, fontsize=11)


def courbes_lits(courbes: dict, fichier: str) -> None:
    """Occupation des lits au fil de l'année, un graphique par (charge, échelle)."""
    plt = _plt()
    if plt is None or not courbes:
        return
    cles = [(c, e) for c in ("normale", "surcharge") for e in ("petite", "grande")
            if (c, e) in courbes]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, squeeze=False)
    for ax, cle in zip(axes.flat, cles):
        cc = courbes[cle]
        dates = pd.to_datetime(cc["dates"])

        def lisse(v):
            return pd.Series(list(v), index=dates).rolling(7, center=True, min_periods=1).mean()

        if "plan" in cc:
            ax.plot(dates, lisse(cc["plan"]), color="#4a5568", lw=1.6, label="plan du matin")
        for pol in ("statique", "sma"):
            if pol in cc:
                col, nom = STYLE[pol]
                ax.plot(dates, lisse(cc[pol]), color=col, lw=1.4, label=f"après aléas — {nom}")
        ax.plot(dates, cc["cap"], color="#c53030", ls="--", lw=1, label="capacité")
        _axe(ax, f"charge {cle[0]} · échelle {cle[1]}")
        ax.set_ylabel("lits occupés (moy. 7 j)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Occupation des lits au fil de l'année", y=0.995)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib absent : pas de figure (pip install matplotlib)")
        return None


def tracer_balayage(df: pd.DataFrame, x: str, xlabel: str, mesures: list, fichier: str,
                    titre: str, par_politique: bool) -> None:
    """Courbes mesure = f(x), moyenne sur les graines et min-max en bande."""
    plt = _plt()
    if plt is None or df.empty:
        return
    fig, axes = plt.subplots(1, len(mesures), figsize=(4.6 * len(mesures), 4.2), squeeze=False)
    for ax, (col, lib) in zip(axes[0], mesures):
        groupes = df.groupby("politique") if par_politique else [(None, df)]
        for pol, d in groupes:
            g = d.groupby(x)[col].agg(["mean", "min", "max"])
            col_, nom = STYLE.get(pol, ("#2c7a7b", None))
            ax.plot(g.index, g["mean"], marker="o", color=col_, label=nom, lw=1.8)
            if (g["max"] > g["min"]).any():
                ax.fill_between(g.index, g["min"], g["max"], color=col_, alpha=0.15)
        _axe(ax, lib)
        ax.set_xlabel(xlabel)
    if par_politique:
        axes[0][0].legend(frameon=False)
    fig.suptitle(titre, y=1.02)
    fig.tight_layout()
    fig.savefig(fichier, dpi=150, bbox_inches="tight")
    plt.close(fig)


def balayage_aleas(base, args, valeurs) -> pd.DataFrame:
    """Charge normale, grande échelle : on fait varier l'intensité des urgences."""
    inst, _ = preparer_instance(base, "normale", "grande", surcharge=0,
                                n_chirurgiens_petite=args.petite_chirurgiens,
                                graine=args.graine_base)
    sol = construire_solution(inst)
    jours = jours_a_simuler(sol, args.jours)
    P0, lignes = ParametresSMA(), []
    for u in valeurs:
        print(f"   urgences/jour = {u}")
        for g in range(1, args.graines + 1):
            gen = GenerateurAleas(urgences_par_jour=u, p_annulation=args.annulation,
                                  p_lits=args.lits, graine=g)
            for pol in POLITIQUES:
                b, _, _ = simuler_periode(sol, jours, gen, dataclasses.replace(P0, politique=pol),
                                          optimiseur_glouton, durees="reelles")
                lignes.append({"urgences_par_jour": u, "politique": pol, "graine": g,
                               **resumer(b)})
    return pd.DataFrame(lignes)


def balayage_charge(base, args, taux) -> pd.DataFrame:
    """Grande échelle : on fait varier la surcharge (+0 % … +60 % de patients)."""
    P0, lignes = ParametresSMA(), []
    for t in taux:
        print(f"   surcharge = +{t:.0%}")
        inst, _ = preparer_instance(base, "surcharge" if t > 0 else "normale", "grande",
                                    surcharge=t, n_chirurgiens_petite=args.petite_chirurgiens,
                                    graine=args.graine_base)
        sol = construire_solution(inst)
        ind = sol.indicateurs()
        jours = jours_a_simuler(sol, args.jours)
        plan = {"surcharge_pct": round(100 * t), "patients_sans_date": len(sol.hors_horizon),
                "lits_ecart_type": ind["lits_ecart_type"], "lits_pic": ind["lits_pic"],
                "taux_remplissage": ind["taux_remplissage_moyen"]}
        for g in range(1, args.graines + 1):
            gen = GenerateurAleas(urgences_par_jour=args.urgences, p_annulation=args.annulation,
                                  p_lits=args.lits, graine=g)
            for pol in ("statique", "sma"):
                b, _, _ = simuler_periode(sol, jours, gen, dataclasses.replace(P0, politique=pol),
                                          optimiseur_glouton, durees="reelles")
                lignes.append({**plan, "politique": pol, "graine": g, **resumer(b)})
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# 4. Programme principal
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("chemin", nargs="?", default="donees_bloc_anonyme_pour_centrale_2026.xlsx")
    ap.add_argument("--graines", type=int, default=3, help="tirages d'aléas par scénario")
    ap.add_argument("--jours", type=int, default=None, help="limiter le nombre de jours simulés")
    ap.add_argument("--surcharge", type=float, default=0.35, help="part de patients ajoutés")
    ap.add_argument("--petite-chirurgiens", type=int, default=2)
    ap.add_argument("--urgences", type=float, default=1.5, help="urgences du jour / jour (Poisson)")
    ap.add_argument("--annulation", type=float, default=0.04, help="proba d'annulation d'un programmé")
    ap.add_argument("--lits", type=float, default=0.15, help="proba d'un événement « lits fermés »")
    ap.add_argument("--graine-base", type=int, default=42)
    ap.add_argument("--rapide", action="store_true", help="40 jours, 1 graine (test)")
    ap.add_argument("--sans-figure", action="store_true")
    ap.add_argument("--sans-balayage", action="store_true",
                    help="ne pas faire les courbes de sensibilité (plus rapide)")
    args = ap.parse_args(argv)
    if args.rapide:
        args.jours, args.graines = 40, 1

    t0 = time.perf_counter()
    print(f"Chargement de {args.chemin} ...")
    base = charger_base(args.chemin)

    grille = [(c, a, e) for c in ("normale", "surcharge")
              for a in ("stable", "aleas") for e in ("petite", "grande")]
    lignes: list[dict] = []
    courbes: dict = {}
    for k, (charge, aleas, echelle) in enumerate(grille, 1):
        print(f"\n[{k}/{len(grille)}] charge={charge} | {aleas} | échelle={echelle}")
        lignes += lancer_scenario(base, charge, aleas, echelle, args, courbes)

    brut = pd.DataFrame(lignes)
    brut.to_csv("resultats_scenarios_brut.csv", index=False, encoding="utf-8-sig")
    syn = synthese(brut)
    syn.to_csv("synthese_scenarios.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n" + "=" * 100 + "\nSYNTHÈSE (moyenne sur les graines)\n" + "=" * 100)
    print(syn.to_string())
    fichiers = ["resultats_scenarios_brut.csv", "synthese_scenarios.csv"]
    if not args.sans_figure:
        courbes_lits(courbes, "courbes_lits.png")
        figure(brut, "synthese_scenarios.png")
        fichiers += ["courbes_lits.png", "synthese_scenarios.png"]
        if not args.sans_balayage:
            vu = (0, 1.5, 3) if args.rapide else (0, 1, 1.5, 2, 3, 4)
            vt = (0, 0.35, 0.6) if args.rapide else (0, 0.15, 0.35, 0.5, 0.6)
            print("\nBalayage de l'intensité des urgences ...")
            ba = balayage_aleas(base, args, vu)
            ba.to_csv("balayage_aleas.csv", index=False, encoding="utf-8-sig")
            tracer_balayage(ba, "urgences_par_jour", "urgences par jour (Poisson)",
                            [("urgences_echec", "urgences sans place"),
                             ("depassement_min", "dépassement des salles (min)"),
                             ("attente_urgence_moy", "attente d'une urgence (min)")],
                            "courbes_sensibilite_aleas.png",
                            "Effet de l'intensité des aléas (charge normale, grande échelle)", True)
            print("\nBalayage de la surcharge ...")
            bc = balayage_charge(base, args, vt)
            bc.to_csv("balayage_charge.csv", index=False, encoding="utf-8-sig")
            tracer_balayage(bc, "surcharge_pct", "patients ajoutés (%)",
                            [("patients_sans_date", "programmés sans date"),
                             ("lits_ecart_type", "σ de l'occupation des lits"),
                             ("lits_pic", "pic de lits"),
                             ("urgences_echec", "urgences sans place")],
                            "courbes_sensibilite_charge.png",
                            "Effet de la surcharge (grande échelle, aléas actifs)", True)
            fichiers += ["courbes_sensibilite_aleas.png", "courbes_sensibilite_charge.png",
                         "balayage_aleas.csv", "balayage_charge.csv"]
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s. Fichiers : " + ", ".join(fichiers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
