r"""
etude_recuit.py — Étude du recuit simulé sur la VRAIE base de l'hôpital.

Question : quelle FONCTION COÛT permet au recuit de lisser l'occupation des
lits, et avec quels paramètres ?

On rejoue le 1er semestre 2022 (1 954 patients réels, vacations réellement
utilisées) : le planning RÉEL de l'hôpital sert de référence, le recuit part
de ce planning et peut déplacer chaque patient de ±30 jours dans une vacation
de son chirurgien.

UTILISATION (dans le dossier qui contient modele.py et la base Excel)

  python etude_recuit.py tout --rapide    # tout, version courte (~5 min)
  python etude_recuit.py tout             # tout (~15 min)

ou étape par étape : reel, fonctions, poids, parametres, convergence, taille.
Options : --debut 2022-01-03 --semaines 25 --fenetre 30 --graines 3

SORTIES : resultats_recuit/figures/*.png (numérotées dans l'ordre d'une
présentation), resultats_recuit/*.csv, resultats_recuit/synthese.txt
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from donnees_reelles import charger_base, construire_rejeu
from fonctions_cout import (CoutCibleLits, CoutGroupe, CoutGroupeNormalise, Evaluateur, Poids,
                            candidates, verifier_coherence)
from modele import Solution
from recuit import ParamsRecuit, RecuitSimule

DOSSIER = Path(__file__).with_name("resultats_recuit")
FIG = DOSSIER / "figures"
_BASE = None
_REJEUX: dict = {}

INDIC = {  # indicateurs physiques montrés (plus bas = mieux, sauf mention)
    "lits_ecart_type": "Écart-type des lits (lits)",
    "lits_pic": "Pic de lits",
    "variation_nuit_a_nuit": "Variation d'une nuit à l'autre (lits)",
    "nuits_hors_2lits_pct": "Nuits à plus de ±2 lits de la cible (%)",
    "ratio_weekend_semaine": "Lits week-end / semaine",
    "ambu_ecart_type": "Écart-type ambulatoire / jour",
    "bloc_remplissage_pct": "Remplissage du bloc (%) — plus haut = mieux",
    "decalage_moyen_j": "Décalage moyen vs date réelle (j)",
}


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------

def rejeu(args, semaines=None):
    global _BASE
    if _BASE is None:
        print("Lecture de la base...")
        _BASE = charger_base()
    cle = (args.debut, semaines or args.semaines, args.fenetre)
    if cle not in _REJEUX:
        _REJEUX[cle] = construire_rejeu(args.debut, semaines or args.semaines, args.fenetre, base=_BASE)
    return _REJEUX[cle]


def evaluateur(rj):
    s = Solution(rj.inst)
    for pid in rj.inst.patients:
        s.affecter(pid, rj.reel.affectation[pid])
    ev = Evaluateur(rj, s)
    ev.figer_reference()
    return ev


def lancer(rj, cout, pr: ParamsRecuit):
    ev = evaluateur(rj)
    r = RecuitSimule(ev, cout, pr, rj.mobiles, rj.jour_max).lancer()
    return ev, r


def ecrire(nom, lignes):
    DOSSIER.mkdir(exist_ok=True)
    cles = list(dict.fromkeys(k for l in lignes for k in l))
    with open(DOSSIER / nom, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cles, delimiter=";")
        w.writeheader()
        w.writerows(lignes)
    print(f"   -> resultats_recuit/{nom}")


def lire(nom):
    import pandas as pd
    f = DOSSIER / nom
    return pd.read_csv(f, sep=";") if f.exists() else None


def plt_():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
    FIG.mkdir(parents=True, exist_ok=True)
    return plt


def sauver(fig, nom):
    fig.savefig(FIG / nom, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"   figure : {nom}")


def tracer_profil(ax, rj, ev_ref, ev=None, titre="", couleur="C1"):
    import datetime as dt
    inst = rj.inst
    js = list(range(rj.eval_debut, rj.eval_fin))
    x = [inst.date_du_jour(j) for j in js]
    for j, d in zip(js, x):
        if j in rj.periode_reduite:
            ax.axvspan(d, d + dt.timedelta(days=1), color="orange", alpha=0.13, lw=0)
        elif d.weekday() in (4, 5, 6):       # nuits du vendredi, samedi, dimanche
            ax.axvspan(d, d + dt.timedelta(days=1), color="grey", alpha=0.12, lw=0)
    ax.step(x, [ev_ref.sol.lits_jour[j] for j in js], where="post", color="0.55", lw=1, label="planning réel")
    ax.step(x, [ev_ref.cible[j] for j in js], where="post", color="k", ls=":", lw=1, label="cible")
    if ev is not None:
        ax.step(x, [ev.sol.lits_jour[j] for j in js], where="post", color=couleur, lw=1.4, label="recuit")
    ax.set_title(titre, fontsize=10)
    ax.set_ylabel("lits occupés")
    ax.tick_params(axis="x", labelsize=8)


# ---------------------------------------------------------------------------
# 1. Le planning réel
# ---------------------------------------------------------------------------

def etape_reel(args):
    rj = rejeu(args)
    ev = evaluateur(rj)
    print(f"   {len(rj.mobiles)} patients réels, {len(rj.inst.vacations)} vacations réelles, "
          f"du {rj.inst.date_du_jour(rj.eval_debut)} au {rj.inst.date_du_jour(rj.eval_fin - 1)}")
    print(f"   cohérence avec CoutTotal du modele.py : écart {verifier_coherence(ev):.1e}")
    i = ev.indicateurs()
    for k, lib in INDIC.items():
        print(f"   {lib:48s}{i[k]:8.2f}")
    ecrire("reel.csv", [i])
    plt = plt_()
    fig, ax = plt.subplots(figsize=(12, 3.8))
    tracer_profil(ax, rj, ev, titre="Planning réel du 1er semestre 2022 — gris : nuits de week-end (ven., sam., dim.), orange : vacances / fériés")
    ax.legend(fontsize=8, loc="upper right")
    sauver(fig, "1_planning_reel.png")


# ---------------------------------------------------------------------------
# 2. Comparaison des fonctions coût
# ---------------------------------------------------------------------------

def etape_fonctions(args):
    rj = rejeu(args)
    lignes = []
    meilleurs = {}
    for c in candidates():
        for g in range(args.graines):
            ev, r = lancer(rj, c, ParamsRecuit(graine=g, max_iter=args.iterations))
            i = ev.indicateurs()
            lignes.append({"fonction": c.nom, "graine": g, "temps": r.temps, "iterations": r.iterations, **i})
            if g == 0:
                meilleurs[c.nom] = ev
        print(f"   {c.nom:24s} écart-type lits {sum(l['lits_ecart_type'] for l in lignes[-args.graines:]) / args.graines:.2f}")
    ecrire("fonctions.csv", lignes)

    plt = plt_()
    ev_ref = evaluateur(rj)
    fig, axs = plt.subplots(len(meilleurs), 1, figsize=(12, 2.6 * len(meilleurs)), sharex=True)
    for ax, (nom, ev), coul in zip(axs, meilleurs.items(), ["C3", "C0", "C2", "C4"]):
        tracer_profil(ax, rj, ev_ref, ev, titre=nom, couleur=coul)
    axs[0].legend(fontsize=8, loc="upper right", ncol=3)
    fig.suptitle("Lits occupés : planning réel (gris) et après recuit, selon la fonction coût", y=1.0)
    fig.tight_layout()
    sauver(fig, "2_profils_par_fonction_cout.png")
    graph_fonctions()


def graph_fonctions():
    import numpy as np
    df, reel = lire("fonctions.csv"), lire("reel.csv")
    if df is None:
        return
    plt = plt_()
    noms = list(dict.fromkeys(df.fonction))
    fig, axs = plt.subplots(2, 4, figsize=(17, 7))
    for ax, (k, lib) in zip(axs.ravel(), INDIC.items()):
        m = [df[df.fonction == n][k].mean() for n in noms]
        e = [1.96 * df[df.fonction == n][k].std(ddof=1) / np.sqrt(max(1, (df.fonction == n).sum())) for n in noms]
        vals = ([reel[k].iloc[0]] if reel is not None else []) + m
        err = ([0] if reel is not None else []) + [0 if x != x else x for x in e]
        lab = (["réel"] if reel is not None else []) + [n.split(" ")[0] for n in noms]
        coul = (["0.6"] if reel is not None else []) + ["C3", "C0", "C2", "C4"][:len(noms)]
        ax.bar(range(len(vals)), vals, yerr=err, color=coul, capsize=3)
        ax.set_xticks(range(len(vals)), lab)
        ax.set_title(lib, fontsize=9)
        for i, (v, er) in enumerate(zip(vals, err)):
            y = v + er if v >= 0 else v - er
            ax.text(i, y, f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    fig.suptitle("Ce que voit l'hôpital, selon la fonction coût (C0 groupe, C1 groupe normalisée, "
                 "C2 cible lits, C3 cible lits seule)")
    fig.tight_layout()
    sauver(fig, "3_comparaison_fonctions_cout.png")


# ---------------------------------------------------------------------------
# 3. Poids : compromis lissage des lits / reste
# ---------------------------------------------------------------------------

def etape_poids(args):
    rj = rejeu(args)
    lignes = []
    for w in args.poids_lits:
        reste = (1 - w) / 3
        c = CoutCibleLits(Poids(w, reste, reste, reste))
        for g in range(max(1, args.graines - 1)):
            ev, r = lancer(rj, c, ParamsRecuit(graine=g, max_iter=args.iterations))
            lignes.append({"poids_lits": w, "graine": g, **ev.indicateurs()})
        print(f"   poids lits {w:.2f}")
    ecrire("poids.csv", lignes)
    graph_poids()


def graph_poids():
    df, reel = lire("poids.csv"), lire("reel.csv")
    if df is None:
        return
    plt = plt_()
    agg = df.groupby("poids_lits").mean(numeric_only=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (k, lib) in zip(axs, [("bloc_remplissage_pct", "Remplissage du bloc (%)"),
                                  ("ambu_ecart_type", "Écart-type ambulatoire / jour"),
                                  ("decalage_moyen_j", "Décalage moyen vs date réelle (j)")]):
        ax.plot(agg.lits_ecart_type, agg[k], marker="o")
        for w, row in agg.iterrows():
            ax.annotate(f"{w:.0%}", (row.lits_ecart_type, row[k]), fontsize=8, xytext=(4, 4),
                        textcoords="offset points")
        if reel is not None:
            ax.scatter([reel.lits_ecart_type.iloc[0]], [reel[k].iloc[0]], marker="s", c="0.4", s=60, label="réel")
            ax.legend()
        ax.set_xlabel("Écart-type des lits (plus à gauche = plus lisse)")
        ax.set_ylabel(lib)
    fig.suptitle("Compromis selon le poids du terme lits dans C2 (étiquettes : poids des lits)")
    fig.tight_layout()
    sauver(fig, "4_compromis_poids.png")


# ---------------------------------------------------------------------------
# 4. Paramètres du recuit : T0 x alpha, avec la fonction C2
# ---------------------------------------------------------------------------

def etape_parametres(args):
    rj = rejeu(args)
    lignes = []
    for t0 in args.t0:
        for a in args.alphas:
            for g in range(max(1, args.graines - 1)):
                pr = ParamsRecuit(T0_facteur=t0, alpha=a, graine=g, max_iter=400_000)
                ev, r = lancer(rj, CoutCibleLits(), pr)
                lignes.append({"T0_facteur": t0, "alpha": a, "graine": g, "temps": r.temps,
                               "iterations": r.iterations, "cout_final": r.cout_final,
                               "taux_acceptation": r.taux_acceptation, **ev.indicateurs()})
            print(f"   T0 = {t0}, alpha = {a} ({r.iterations} itérations, {r.temps:.1f} s)")
    ecrire("parametres.csv", lignes)
    graph_parametres()


def graph_parametres():
    import numpy as np
    df = lire("parametres.csv")
    if df is None:
        return
    plt = plt_()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3))
    for ax, (col, titre, cmap) in zip(axs, [("lits_ecart_type", "Écart-type des lits (plus bas = mieux)", "RdYlGn_r"),
                                            ("nuits_hors_2lits_pct", "Nuits hors ±2 lits (%)", "RdYlGn_r"),
                                            ("temps", "Temps de calcul (s)", "Blues")]):
        piv = df.groupby(["alpha", "T0_facteur"])[col].mean().unstack()
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xlabel("T0 (x dégradation moyenne)")
        ax.set_ylabel("alpha")
        ax.set_title(titre)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Paramètres du recuit (fonction C2) : température initiale T0 et refroidissement alpha")
    fig.tight_layout()
    sauver(fig, "5_parametres_T0_alpha.png")

    fig, ax = plt.subplots(figsize=(7, 4.3))
    for a, g in df.groupby("alpha"):
        m = g.groupby("T0_facteur").agg(t=("temps", "mean"), e=("lits_ecart_type", "mean"))
        ax.plot(m.t, m.e, marker="o", label=f"alpha = {a}")
    ax.set_xscale("log")
    ax.set_xlabel("Temps de calcul (s, échelle log)")
    ax.set_ylabel("Écart-type des lits")
    ax.set_title("Qualité du lissage en fonction du temps de calcul")
    ax.legend()
    sauver(fig, "6_qualite_temps.png")


# ---------------------------------------------------------------------------
# 5. Convergence
# ---------------------------------------------------------------------------

def etape_convergence(args):
    rj = rejeu(args)
    lignes = []
    for nom, pr in [("T0 = 0,1", ParamsRecuit(T0_facteur=0.1, max_iter=400_000)),
                    ("T0 = 0,5 (référence)", ParamsRecuit(max_iter=400_000)),
                    ("T0 = 5", ParamsRecuit(T0_facteur=5, max_iter=400_000)),
                    ("alpha = 0,998", ParamsRecuit(alpha=0.998, max_iter=400_000))]:
        ev, r = lancer(rj, CoutCibleLits(), replace(pr, journal_periode=500))
        for j in r.journal:
            lignes.append({"config": nom, **j})
        print(f"   convergence {nom}")
    ecrire("convergence.csv", lignes)
    graph_convergence()


def graph_convergence():
    df = lire("convergence.csv")
    if df is None:
        return
    plt = plt_()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    for nom, g in df.groupby("config", sort=False):
        axs[0].plot(g.iteration, g.meilleur, label=nom)
        axs[1].plot(g.iteration, g.temperature, label=nom)
        gi = g.dropna(subset=["lits_ecart_type"])
        axs[2].plot(gi.iteration, gi.lits_ecart_type, marker=".", label=nom)
    axs[0].set_ylabel("Meilleur coût (C2)")
    axs[1].set_ylabel("Température")
    axs[1].set_yscale("log")
    axs[2].set_ylabel("Écart-type des lits (solution courante)")
    for a in axs:
        a.set_xlabel("Itération")
    axs[0].legend(fontsize=8)
    fig.suptitle("Dynamique du recuit simulé")
    fig.tight_layout()
    sauver(fig, "7_convergence.png")


# ---------------------------------------------------------------------------
# 6. Taille de la population (durée de la période rejouée)
# ---------------------------------------------------------------------------

def etape_taille(args):
    lignes = []
    for s in args.tailles_semaines:
        rj = rejeu(args, semaines=s)
        ev0 = evaluateur(rj)
        i0 = ev0.indicateurs()
        for g in range(max(1, args.graines - 1)):
          for budget, it in (("fixe", args.iterations), ("proportionnel", 100 * len(rj.mobiles))):
            # alpha calé pour que la température atteigne Tmin exactement à la fin du budget
            alpha = (0.01 / 0.5) ** (100 / it)
            pr = ParamsRecuit(graine=g, max_iter=it, alpha=alpha)
            ev, r = lancer(rj, CoutCibleLits(), pr)
            i = ev.indicateurs()
            lignes.append({"semaines": s, "patients": len(rj.mobiles), "graine": g, "budget": budget,
                           "iterations": r.iterations, "temps": r.temps,
                           "reel_lits_ecart_type": i0["lits_ecart_type"], "reel_nuits_hors_2lits_pct":
                           i0["nuits_hors_2lits_pct"], **i})
        print(f"   {s} semaines ({len(rj.mobiles)} patients)")
    ecrire("taille.csv", lignes)
    graph_taille()


def graph_taille():
    df = lire("taille.csv")
    if df is None:
        return
    plt = plt_()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    ref = df.groupby("patients").mean(numeric_only=True)
    axs[0].plot(ref.index, ref.reel_lits_ecart_type, marker="s", c="0.5", label="réel")
    axs[1].plot(ref.index, ref.reel_nuits_hors_2lits_pct, marker="s", c="0.5", label="réel")
    libs = {"fixe": "recuit, budget fixe", "proportionnel": "recuit, 100 itérations / patient"}
    for b, g in df.groupby("budget"):
        agg = g.groupby("patients").mean(numeric_only=True)
        axs[0].plot(agg.index, agg.lits_ecart_type, marker="o", label=libs[b])
        axs[1].plot(agg.index, agg.nuits_hors_2lits_pct, marker="o", label=libs[b])
        axs[2].plot(agg.index, agg.temps, marker="o", label=libs[b])
    axs[0].set_ylabel("Écart-type des lits")
    axs[1].set_ylabel("Nuits hors ±2 lits (%)")
    axs[2].set_ylabel("Temps de calcul (s)")
    for a in axs:
        a.set_xlabel("Nombre de patients à planifier")
        a.legend(fontsize=8)
    fig.suptitle("Effet de la taille de la population (périodes de 8 à 25 semaines)")
    fig.tight_layout()
    sauver(fig, "8_taille_population.png")


# ---------------------------------------------------------------------------
# Synthèse
# ---------------------------------------------------------------------------

def synthese(args):
    reel, fo = lire("reel.csv"), lire("fonctions.csv")
    if reel is None or fo is None:
        return
    L = ["SYNTHÈSE — recuit simulé sur le 1er semestre 2022 (données réelles)", ""]
    L.append(f"{'indicateur':48s}{'réel':>9s}" + "".join(f"{n.split(' ')[0]:>9s}" for n in dict.fromkeys(fo.fonction)))
    for k, lib in INDIC.items():
        L.append(f"{lib:48s}{reel[k].iloc[0]:9.2f}" +
                 "".join(f"{fo[fo.fonction == n][k].mean():9.2f}" for n in dict.fromkeys(fo.fonction)))
    pa = lire("parametres.csv")
    if pa is not None:
        b = pa.groupby(["T0_facteur", "alpha"]).lits_ecart_type.mean().idxmin()
        L += ["", f"Meilleurs paramètres (C2) : T0 = {b[0]}, alpha = {b[1]}"]
    DOSSIER.mkdir(exist_ok=True)
    (DOSSIER / "synthese.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etape", choices=["reel", "fonctions", "poids", "parametres", "convergence", "taille",
                                      "graphiques", "tout"])
    ap.add_argument("--debut", default="2022-01-03")
    ap.add_argument("--semaines", type=int, default=25)
    ap.add_argument("--fenetre", type=int, default=30, help="déplacement autorisé autour de la date réelle (j)")
    ap.add_argument("--graines", type=int, default=3)
    ap.add_argument("--iterations", type=int, default=80_000)
    ap.add_argument("--rapide", action="store_true")
    a = ap.parse_args()
    a.poids_lits = [0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
    a.t0 = [0.1, 0.5, 2, 5]
    a.alphas = [0.99, 0.995, 0.998]
    a.tailles_semaines = [8, 13, 25]
    if a.rapide:
        a.graines, a.iterations = 2, 50_000
        a.poids_lits = [0.1, 0.4, 0.8, 1.0]
        a.t0, a.alphas = [0.1, 0.5, 5], [0.99, 0.995]
        a.tailles_semaines = [8, 25]
    t = time.perf_counter()
    etapes = {"reel": etape_reel, "fonctions": etape_fonctions, "poids": etape_poids,
              "parametres": etape_parametres, "convergence": etape_convergence, "taille": etape_taille}
    if a.etape == "graphiques":
        graph_fonctions(); graph_poids(); graph_parametres(); graph_convergence(); graph_taille()
    else:
        for e in (list(etapes) if a.etape == "tout" else [a.etape]):
            print(f"\n=== {e} ===")
            etapes[e](a)
    synthese(a)
    print(f"\nTerminé en {(time.perf_counter() - t) / 60:.1f} min. Figures : {FIG}")
