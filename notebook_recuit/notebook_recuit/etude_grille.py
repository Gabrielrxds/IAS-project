r"""
etude_grille.py — Proposer une NOUVELLE GRILLE DE VACATIONS par recuit simulé
et la valider sur l'activité réelle de 2022.

  python etude_grille.py tout --rapide    (~8 min)
  python etude_grille.py tout             (~20 min)

Étapes (utilisables séparément) :
  grille      recuit sur la grille -> grille A (temps de chaque chirurgien
              conservé à ±10 %) et grille B (même temps total, réparti selon
              l'activité réelle 2022) ; exports Excel au format de l'hôpital
  stabilite   compromis : nombre de créneaux modifiés / gain sur les lits
  parametres  réglage du recuit : T0 x alpha, convergence
  validation  on rejoue TOUTE l'année 2022 (vrais patients) dans la grille
              actuelle et dans les grilles proposées, avec le recuit patients
              (fonction coût C2 « cible lits ») : ce que verrait l'hôpital

Hypothèses (réglables) : chaque patient peut être opéré à ±30 jours de sa
vraie date ; cible de lits à 85 % les nuits de week-end et à 70 % pendant
les vacances scolaires, ponts et fériés (moins de lits en vacances qu'en
week-end) ; créneaux URGENCES et LIBRE inchangés.

SORTIES : resultats_grille/figures/*.png (numérotées pour la présentation),
resultats_grille/grille_A.xlsx, grille_B.xlsx, *.csv, synthese.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

from donnees_reelles import charger_base, construire_rejeu, construire_rejeu_grille
from fonctions_cout import CoutCibleLits, Evaluateur
from grille_vacations import FIXES, JOURS, SEMAINES_ACTIVES, Grille, fmt
from recuit import ParamsRecuit, RecuitSimule
from recuit_grille import (NUITS, EvaluationGrille, ModeleAttendu, ParamsRecuitGrille, PoidsGrille,
                           heures_selon_activite, recuit_grille)

DOSSIER = Path(__file__).with_name("resultats_grille")
FIG = DOSSIER / "figures"
_C: dict = {}


def base():
    if "base" not in _C:
        print("Lecture de la base...")
        _C["base"] = charger_base()
        _C["actuelle"] = Grille.actuelle()
        _C["modele"] = ModeleAttendu(_C["base"], _C["actuelle"])
        _C["cible_B"] = heures_selon_activite(_C["base"], _C["actuelle"])
    return _C["base"], _C["actuelle"], _C["modele"]


def ecrire(nom, lignes):
    DOSSIER.mkdir(exist_ok=True)
    cles = list(dict.fromkeys(k for l in lignes for k in l))
    with open(DOSSIER / nom, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cles, delimiter=";")
        w.writeheader()
        w.writerows(lignes)


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
    import matplotlib.pyplot as plt
    fig.savefig(FIG / nom, bbox_inches="tight")
    plt.close(fig)
    print(f"   figure : {nom}")


def charger_grille(nom) -> Grille | None:
    f = DOSSIER / f"{nom}.json"
    if not f.exists():
        return None
    g = Grille.actuelle().copie(nom)
    g.chirurgiens = json.loads(f.read_text(encoding="utf-8"))
    return g


# ---------------------------------------------------------------------------
# 1. Recuit sur la grille
# ---------------------------------------------------------------------------

def meilleure_de(n, poids=None, heures_cible=None):
    """Le recuit est aléatoire : on le lance n fois et on garde la meilleure grille."""
    _, G, mod = base()
    best = None
    for g in range(n):
        r = recuit_grille(G, mod, ParamsRecuitGrille(graine=g), poids=poids, heures_cible=heures_cible)
        print(f"      exécution {g + 1}/{n} : coût {r.cout_initial:.3f} -> {r.cout_final:.3f} "
              f"({r.iterations} itérations, {r.temps:.0f} s)")
        if best is None or r.cout_final < best.cout_final:
            best = r
    return best


def etape_grille(args):
    b, G, mod = base()
    DOSSIER.mkdir(exist_ok=True)
    for nom, cible in (("grille_A", None), ("grille_B", _C["cible_B"])):
        print(f"   {nom} ({'temps conservé ±10 %' if cible is None else 'temps rééquilibré selon l activité'})")
        r = meilleure_de(args.restarts, heures_cible=cible)
        r.grille.nom = nom
        (DOSSIER / f"{nom}.json").write_text(json.dumps(r.grille.chirurgiens), encoding="utf-8")
        r.grille.exporter_excel(DOSSIER / f"{nom}.xlsx", reference=G)
        print(f"   -> resultats_grille/{nom}.xlsx ({sum(a != c for a, c in zip(r.grille.chirurgiens, G.chirurgiens))}"
              f" créneaux modifiés, {len(r.grille.conflits())} conflit)")
    graph_grille()


def graph_grille():
    b, G, mod = base()
    grilles = {"actuelle": G}
    for n in ("grille_A", "grille_B"):
        g = charger_grille(n)
        if g is not None:
            grilles[n.replace("grille_", "proposée ")] = g
    plt = plt_()
    # -- lits et ambulatoires attendus sur le cycle de 4 semaines
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for n in range(NUITS):
        if n % 7 in (4, 5, 6):
            axs[0].axvspan(n - 0.5, n + 0.5, color="grey", alpha=0.12, lw=0)
    for (nom, g), c in zip(grilles.items(), ["0.4", "C2", "C0"]):
        ev = EvaluationGrille(g.copie(), mod, PoidsGrille(), G)
        axs[0].step(range(NUITS), ev.lits, where="mid", color=c, lw=1.6 if nom != "actuelle" else 1.2, label=nom)
        jo = [n for n in range(NUITS) if n % 7 < 5]
        axs[1].plot(jo, [ev.amb[n] for n in jo], marker="o", color=c, label=nom)
    ev0 = EvaluationGrille(G.copie(), mod, PoidsGrille(), G)
    moy = sum(ev0.lits) / NUITS
    mf = sum(ev0.f) / NUITS
    axs[0].step(range(NUITS), [moy * f / mf for f in ev0.f], where="mid", color="k", ls=":", label="cible")
    axs[0].set_ylabel("lits attendus")
    axs[0].set_title("Lits attendus chaque nuit du cycle de 4 semaines (gris : nuits de vendredi, samedi, dimanche)")
    axs[1].set_ylabel("ambulatoires attendus")
    axs[1].set_title("Ambulatoires attendus par jour opératoire")
    etiq = [f"{['L', 'M', 'M', 'J', 'V', 'S', 'D'][n % 7]}" for n in range(NUITS)]
    axs[1].set_xticks(range(NUITS), etiq)
    axs[1].set_xlabel("semaines : paire | impaire A | paire | impaire B")
    axs[0].legend(fontsize=8, ncol=4)
    fig.tight_layout()
    sauver(fig, "1_lits_attendus_par_grille.png")

    # -- temps par chirurgien
    import numpy as np
    h0 = G.heures_par_semaine()
    chirs = sorted(h0, key=lambda m: -h0[m])
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(chirs))
    series = [("actuelle", h0, "0.6")] + [(n, g.heures_par_semaine(), c) for (n, g), c in
                                           zip(list(grilles.items())[1:], ["C2", "C0"])]
    series.append(("besoin selon l'activité 2022", _C["cible_B"], "C3"))
    w = 0.8 / len(series)
    for k, (n, h, c) in enumerate(series):
        ax.bar(x + k * w - 0.4 + w / 2, [h.get(m, 0) for m in chirs], w, label=n, color=c)
    ax.set_xticks(x, chirs)
    ax.set_ylabel("heures de bloc par semaine")
    ax.set_title("Temps de bloc par chirurgien")
    ax.legend(fontsize=8)
    sauver(fig, "2_temps_par_chirurgien.png")

    # -- les grilles elles-mêmes
    for nom, g in list(grilles.items())[1:]:
        fig, axs = plt.subplots(2, 2, figsize=(17, 9))
        for col, (gg, titre) in enumerate(((G, "actuelle"), (g, nom))):
            for row, (sem, lib) in enumerate(((0, "semaine paire"), (1, "semaine impaire A"))):
                ax = axs[row, col]
                ax.axis("off")
                salles = sorted({l.salle for l in gg.lignes})
                cells, couleurs = [], []
                for j in range(5):
                    ligne, coul = [], []
                    for s in salles:
                        idx = [i for i, l in enumerate(gg.lignes)
                               if l.jour == j and l.salle == s and sem in SEMAINES_ACTIVES[l.regle]]
                        idx.sort(key=lambda i: gg.lignes[i].debut)
                        ligne.append("\n".join(f"{gg.chirurgiens[i]} {fmt(gg.lignes[i].debut)}-{fmt(gg.lignes[i].fin)}"
                                               for i in idx))
                        change = gg is not G and any(gg.chirurgiens[i] != G.chirurgiens[i] for i in idx)
                        coul.append("#ffd6a5" if change else "white")
                    cells.append(ligne)
                    couleurs.append(coul)
                t = ax.table(cellText=cells, rowLabels=JOURS, colLabels=[f"salle {s}" for s in salles],
                             cellColours=couleurs, loc="center", cellLoc="center")
                t.auto_set_font_size(False)
                t.set_fontsize(8)
                t.scale(1, 3.2)
                ax.set_title(f"Grille {titre} — {lib}", fontsize=11, pad=22)
        fig.suptitle("En orange : créneaux dont l'occupant change", y=0.02)
        sauver(fig, f"3_{nom.replace(' ', '_')}_vs_actuelle.png")


# ---------------------------------------------------------------------------
# 2. Stabilité : combien de créneaux faut-il changer ?
# ---------------------------------------------------------------------------

def etape_stabilite(args):
    _, G, mod = base()
    lignes = []
    t0 = EvaluationGrille(G.copie(), mod, PoidsGrille(), G).termes_bruts()     # grille actuelle
    for w in args.poids_changements:
        for g in range(args.restarts):
            r = recuit_grille(G, mod, ParamsRecuitGrille(graine=g), poids=PoidsGrille(changements=w))
            t = EvaluationGrille(r.grille.copie(), mod, PoidsGrille(), G).termes_bruts()
            lignes.append({"poids_changements": w, "graine": g, "cout": r.cout_final,
                           "lits_rel": t["lits"] / t0["lits"], "ambu_rel": t["ambu"] / t0["ambu"],
                           "creneaux_modifies": sum(a != c for a, c in zip(r.grille.chirurgiens, G.chirurgiens))})
        print(f"   poids stabilité {w}")
    ecrire("stabilite.csv", lignes)
    graph_stabilite()


def graph_stabilite():
    df = lire("stabilite.csv")
    if df is None:
        return
    b, G, mod = base()
    plt = plt_()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sc = ax.scatter(df.creneaux_modifies, 100 * (1 - df.lits_rel), c=df.poids_changements, cmap="viridis",
                    s=45, label="irrégularité des lits")
    ax.scatter(df.creneaux_modifies, 100 * (1 - df.ambu_rel), c=df.poids_changements, cmap="viridis",
               marker="^", s=45, label="irrégularité de l'ambulatoire")
    ax.set_xlabel(f"Créneaux modifiés (sur {len(G.modifiables())})")
    ax.set_ylabel("Réduction par rapport à la grille actuelle (%)")
    ax.set_title("Combien de créneaux faut-il changer ?")
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="poids de la stabilité")
    sauver(fig, "4_stabilite_creneaux_modifies.png")


# ---------------------------------------------------------------------------
# 3. Réglage du recuit sur la grille
# ---------------------------------------------------------------------------

def etape_parametres(args):
    _, G, mod = base()
    lignes, conv = [], []
    for t0 in args.t0:
        for a in args.alphas:
            for g in range(2):
                r = recuit_grille(G, mod, ParamsRecuitGrille(T0_facteur=t0, alpha=a, graine=g, max_iter=250_000))
                lignes.append({"T0_facteur": t0, "alpha": a, "graine": g, "cout": r.cout_final,
                               "iterations": r.iterations, "temps": r.temps})
                if g == 0 and a == 0.995:
                    conv += [{"config": f"T0 = {t0}", **j} for j in r.journal]
            print(f"   T0 = {t0}, alpha = {a}")
    ecrire("parametres.csv", lignes)
    ecrire("convergence.csv", conv)
    graph_parametres()


def graph_parametres():
    df, cv = lire("parametres.csv"), lire("convergence.csv")
    if df is None:
        return
    plt = plt_()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3))
    for ax, (col, titre, cmap) in zip(axs[:2], (("cout", "Coût final (plus bas = mieux)", "RdYlGn_r"),
                                                ("temps", "Temps de calcul (s)", "Blues"))):
        piv = df.groupby(["alpha", "T0_facteur"])[col].mean().unstack()
        im = ax.imshow(piv.values, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [str(i) for i in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}" if col == "cout" else f"{piv.values[i, j]:.0f}",
                        ha="center", va="center", fontsize=9)
        ax.set_xlabel("T0 (x dégradation typique)")
        ax.set_ylabel("alpha")
        ax.set_title(titre)
        fig.colorbar(im, ax=ax)
    if cv is not None:
        for nom, g in cv.groupby("config", sort=False):
            axs[2].plot(g.iteration, g.meilleur, label=nom)
        axs[2].set_xlabel("Itération")
        axs[2].set_ylabel("Meilleur coût")
        axs[2].set_title("Convergence (alpha = 0,995)")
        axs[2].legend(fontsize=8)
    fig.suptitle("Réglage du recuit simulé sur la grille de vacations")
    fig.tight_layout()
    sauver(fig, "5_parametres_recuit_grille.png")


# ---------------------------------------------------------------------------
# 4. Validation sur l'année 2022
# ---------------------------------------------------------------------------

INDIC = {
    "lits_ecart_type": "Écart-type des lits",
    "lits_pic": "Pic de lits",
    "nuits_hors_2lits_pct": "Nuits hors ±2 lits de la cible (%)",
    "ambu_ecart_type": "Écart-type ambulatoire / jour",
    "bloc_remplissage_pct": "Remplissage du bloc (%)",
    "patients_sans_creneau": "Patients sans créneau",
}


def etape_validation(args):
    b, G, _ = base()
    lignes, profils = [], {}
    rr = construire_rejeu("2022-01-03", 52, args.fenetre, base=b)
    ev = Evaluateur(rr, rr.reel)
    lignes.append({"scenario": "réel 2022", **ev.indicateurs()})
    profils["réel 2022"] = (rr, list(ev.sol.lits_jour))
    scen = [("actuelle", G)] + [(f"proposée {n[-1]}", charger_grille(n)) for n in ("grille_A", "grille_B")]
    for nom, g in scen:
        if g is None:
            continue
        rj = construire_rejeu_grille(g, b, fenetre=args.fenetre)
        ev = Evaluateur(rj, rj.reel)
        ev.figer_reference()
        r = RecuitSimule(ev, CoutCibleLits(), ParamsRecuit(alpha=0.998, max_iter=args.iter_patients),
                         rj.mobiles, rj.jour_max).lancer()
        lignes.append({"scenario": f"grille {nom} + recuit", "temps": r.temps, **ev.indicateurs()})
        profils[f"grille {nom}"] = (rj, list(ev.sol.lits_jour))
        print(f"   grille {nom} : écart-type des lits {lignes[-1]['lits_ecart_type']:.2f}, "
              f"{lignes[-1]['patients_sans_creneau']} patients sans créneau")
    ecrire("validation.csv", lignes)
    _C["profils"] = profils
    graph_validation()


def graph_validation():
    import datetime as dt
    df = lire("validation.csv")
    if df is None:
        return
    plt = plt_()
    coul = ["0.6", "C7", "C2", "C0"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
    for ax, (k, lib) in zip(axs.ravel(), INDIC.items()):
        vals = df[k].tolist()
        ax.bar(range(len(vals)), vals, color=coul[:len(vals)])
        ax.set_xticks(range(len(vals)), [s.replace(" + recuit", "").replace("grille ", "") for s in df.scenario],
                      fontsize=8)
        ax.set_title(lib, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}" if v < 100 else f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Année 2022 rejouée avec les vrais patients (grilles + recuit patients)")
    fig.tight_layout()
    sauver(fig, "6_validation_2022.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    import numpy as np
    x = np.arange(len(df))
    for k, (col, lib) in enumerate((("lits_semaine", "nuits de semaine"), ("lits_weekend", "nuits de week-end"),
                                     ("lits_vacances", "vacances, ponts, fériés"))):
        ax.bar(x + (k - 1) * 0.27, df[col], 0.27, label=lib)
    ax.set_xticks(x, [s.replace(" + recuit", "") for s in df.scenario], fontsize=9)
    ax.set_ylabel("lits occupés en moyenne")
    ax.set_title("Lits occupés en moyenne : semaine, week-end, vacances (cible : week-end 85 %, vacances 70 %)")
    ax.legend(loc="lower right", framealpha=0.95)
    sauver(fig, "7_semaine_weekend_vacances.png")

    prof = _C.get("profils")
    if prof:
        fig, axs = plt.subplots(len(prof), 1, figsize=(14, 2.3 * len(prof)), sharex=True, sharey=True)
        for ax, (nom, (rj, L)) in zip(axs, prof.items()):
            js = list(range(rj.eval_debut, rj.eval_fin))
            d = [rj.inst.date_du_jour(j) for j in js]
            for j, dd in zip(js, d):
                if j in rj.periode_reduite:
                    ax.axvspan(dd, dd + dt.timedelta(days=1), color="orange", alpha=0.12, lw=0)
            ax.step(d, [L[j] for j in js], where="post", lw=1)
            ax.set_title(nom, fontsize=10)
            ax.set_ylabel("lits")
        fig.suptitle("Lits occupés chaque nuit en 2022 (orange : vacances, ponts, fériés)")
        fig.tight_layout()
        sauver(fig, "8_profils_annee_2022.png")


# ---------------------------------------------------------------------------

def synthese():
    L = ["SYNTHÈSE — nouvelle grille de vacations par recuit simulé", ""]
    b, G, mod = base()
    for n in ("grille_A", "grille_B"):
        g = charger_grille(n)
        if g is None:
            continue
        ev = EvaluationGrille(g.copie(), mod, PoidsGrille(), G)
        ev0 = EvaluationGrille(G.copie(), mod, PoidsGrille(), G)
        t, t0 = ev.termes_bruts(), ev0.termes_bruts()
        h0, h1 = G.heures_par_semaine(), g.heures_par_semaine()
        dev = max(abs(h1.get(m, 0) - h0[m]) / h0[m] for m in h0)
        L.append(f"{n} : {sum(a != c for a, c in zip(g.chirurgiens, G.chirurgiens))} créneaux modifiés, "
                 f"irrégularité des lits attendus {100 * (1 - t['lits'] / t0['lits']):.0f} % plus faible, "
                 f"ambulatoire {100 * (1 - t['ambu'] / t0['ambu']):.0f} % plus régulier, "
                 f"écart max de temps de travail par rapport à aujourd'hui {100 * dev:.0f} %")
    df = lire("validation.csv")
    if df is not None:
        L += ["", "Validation sur 2022 (vrais patients) :"]
        courts = [s.replace(" + recuit", "").replace("grille ", "").replace("proposée ", "grille ") for s in df.scenario]
        L.append(f"{'':38s}" + "".join(f"{s:>18s}" for s in courts))
        for k, lib in list(INDIC.items()) + [("lits_weekend", "Lits moyens week-end"),
                                             ("lits_vacances", "Lits moyens vacances")]:
            L.append(f"{lib:38s}" + "".join(f"{v:18.2f}" for v in df[k]))
    (DOSSIER / "synthese.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("etape", choices=["grille", "stabilite", "parametres", "validation", "graphiques", "tout"])
    ap.add_argument("--fenetre", type=int, default=30)
    ap.add_argument("--restarts", type=int, default=3, help="exécutions du recuit, on garde la meilleure")
    ap.add_argument("--rapide", action="store_true")
    a = ap.parse_args()
    a.poids_changements = [0, 0.3, 0.6, 1, 1.5, 2, 3]
    a.t0, a.alphas = [0.1, 1, 5], [0.99, 0.995, 0.998]
    a.iter_patients = 150_000
    if a.rapide:
        a.restarts, a.poids_changements, a.t0, a.alphas, a.iter_patients = 2, [0, 0.6, 1.5, 3], [0.1, 1, 5], [0.99, 0.995], 80_000
    t = time.perf_counter()
    etapes = {"grille": etape_grille, "stabilite": etape_stabilite, "parametres": etape_parametres,
              "validation": etape_validation}
    if a.etape == "graphiques":
        graph_grille(); graph_stabilite(); graph_parametres(); graph_validation()
    else:
        for e in (list(etapes) if a.etape == "tout" else [a.etape]):
            print(f"\n=== {e} ===")
            etapes[e](a)
    synthese()
    print(f"\nTerminé en {(time.perf_counter() - t) / 60:.1f} min. Figures : {FIG}")
