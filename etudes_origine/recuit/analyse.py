r"""
analyse.py — Fonctions utilisées par le notebook `rapport_recuit_grille.ipynb`
(simulations, statistiques, figures). Aucun calcul n'est lancé à l'import.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from donnees_reelles import construire_rejeu, construire_rejeu_grille
from fonctions_cout import CoutCibleLits, Evaluateur
from grille_vacations import FIXES, JOURS, SEMAINES_ACTIVES, Grille, fmt
from recuit import ParamsRecuit, RecuitSimule
from recuit_grille import NUITS, NUITS_WEEKEND, EvaluationGrille, ParamsRecuitGrille, PoidsGrille, recuit_grille

SORTIE = Path(__file__).with_name("resultats_notebook")
FIG = SORTIE / "figures"


def dossiers():
    FIG.mkdir(parents=True, exist_ok=True)


def sauver_fig(fig, nom):
    dossiers()
    fig.savefig(FIG / nom, bbox_inches="tight", dpi=140)


# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------

def ic95(x) -> tuple[float, float]:
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))


def tableau_ic(df, par, colonnes, libelles=None) -> pd.DataFrame:
    """Moyenne ± IC95 de chaque colonne, par groupe."""
    lignes = []
    for cle, g in df.groupby(par, sort=False):
        row = dict(zip(par if isinstance(par, list) else [par], cle if isinstance(cle, tuple) else [cle]))
        for c in colonnes:
            m, e = ic95(g[c])
            row[(libelles or {}).get(c, c)] = f"{m:.2f} ± {e:.2f}" if e == e else "—"
        row["répétitions"] = len(g)
        lignes.append(row)
    return pd.DataFrame(lignes)


def wilcoxon_apparie(df, col, a, b, par="graine"):
    """Test de Wilcoxon apparié (mêmes graines) entre les scénarios a et b."""
    from scipy.stats import wilcoxon
    x = df[df.scenario == a].set_index(par)[col]
    y = df[df.scenario == b].set_index(par)[col]
    commun = x.index.intersection(y.index)
    if len(commun) < 5:
        return float("nan"), len(commun)
    try:
        return float(wilcoxon(x[commun], y[commun]).pvalue), len(commun)
    except ValueError:
        return 1.0, len(commun)


# ---------------------------------------------------------------------------
# Recuit sur la grille : plusieurs exécutions, décomposition du coût
# ---------------------------------------------------------------------------

def recuits_grille(G, mod, n, poids=None, heures_cible=None, params=None, verbose=True):
    """n exécutions (graines 0..n-1) ; renvoie (meilleure, liste des résultats)."""
    res = []
    for g in range(n):
        pr = replace(params or ParamsRecuitGrille(), graine=g)
        r = recuit_grille(G, mod, pr, poids=poids, heures_cible=heures_cible)
        res.append(r)
        if verbose:
            print(f"   exécution {g + 1}/{n} : coût {r.cout_initial:.3f} -> {r.cout_final:.3f} "
                  f"({r.iterations} itérations, {r.temps:.0f} s)")
    return min(res, key=lambda r: r.cout_final), res


def decomposition(grilles: dict, mod, G, poids=None, heures_cible=None) -> pd.DataFrame:
    """Valeur de chaque terme de la fonction coût de la grille, pour chaque grille."""
    p = poids or PoidsGrille()
    ev0 = EvaluationGrille(G.copie(), mod, p, G, heures_cible)
    lignes = []
    for nom, g in grilles.items():
        ev = EvaluationGrille(g.copie(), mod, p, G, heures_cible)
        ev.ref = ev0.ref
        t = ev.termes_bruts()
        lignes.append({
            "grille": nom,
            "lits (écart à la cible, rel.)": t["lits"] / ev0.ref["lits"],
            "ambulatoire (variance, rel.)": t["ambu"] / ev0.ref["ambu"],
            "temps de travail (pénalité)": t["temps"],
            "capacité manquante": t["capacite"],
            "part de créneaux modifiés": t["changes"],
            "conflits": ev.nb_conflits,
            "chirurgiens sans créneau": ev.absents(),
            "coût total": ev.cout(),
        })
    return pd.DataFrame(lignes).set_index("grille")


def heures_table(grilles: dict, mod) -> pd.DataFrame:
    G = grilles[list(grilles)[0]]
    h0 = G.heures_par_semaine()
    df = pd.DataFrame({"besoin 2022 (h/sem)": {m: mod.besoin_h.get(m, 0) for m in h0}})
    for nom, g in grilles.items():
        df[f"{nom} (h/sem)"] = pd.Series(g.heures_par_semaine())
    for nom, g in list(grilles.items())[1:]:
        h = g.heures_par_semaine()
        df[f"{nom} écart (%)"] = pd.Series({m: 100 * (h.get(m, 0) - h0[m]) / h0[m] for m in h0})
    return df.fillna(0).round(1).sort_values(list(df.columns)[1], ascending=False)


# ---------------------------------------------------------------------------
# Simulations sur une année réelle
# ---------------------------------------------------------------------------

def debut_annee(annee: int) -> str:
    from datetime import date
    d = date(annee, 1, 1)
    d += timedelta(days=(7 - d.weekday()) % 7)      # premier lundi
    return d.isoformat()


def simuler_grille(grille, base, annee, graine, fenetre=30, iter_patients=150_000, facteur_weekend=0.85,
                   facteur_reduit=0.70, facteur_marge=0.5):
    """Vrais patients de `annee` placés dans `grille`, puis recuit patients (C2)."""
    t = time.perf_counter()
    rj = construire_rejeu_grille(grille, base, debut=debut_annee(annee), semaines=52, fenetre=fenetre,
                                 facteur_marge=facteur_marge)
    ev = Evaluateur(rj, rj.reel, facteur_weekend=facteur_weekend, facteur_reduit=facteur_reduit)
    ev.figer_reference()
    alpha = (0.01 / 0.5) ** (100 / iter_patients)
    r = RecuitSimule(ev, CoutCibleLits(), ParamsRecuit(alpha=alpha, max_iter=iter_patients, graine=graine),
                     rj.mobiles, rj.jour_max).lancer()
    return rj, ev, time.perf_counter() - t


def simuler_reel(base, annee, fenetre=30, facteur_weekend=0.85, facteur_reduit=0.70):
    rr = construire_rejeu(debut_annee(annee), 52, fenetre, base=base)
    ev = Evaluateur(rr, rr.reel, facteur_weekend=facteur_weekend, facteur_reduit=facteur_reduit)
    return rr, ev


def validation(grilles: dict, base, annees, graines, iter_patients=150_000, fenetre=30, profils=None, **kw):
    """Pour chaque année et chaque graine : toutes les grilles sont simulées.
    Indicateurs bruts + indicateurs « à patients identiques » (on retire les
    patients restés sans créneau dans au moins une grille)."""
    lignes = []
    for annee in annees:
        _, evr = simuler_reel(base, annee, fenetre, kw.get("facteur_weekend", 0.85), kw.get("facteur_reduit", 0.70))
        lignes.append({"annee": annee, "scenario": "réel", "graine": 0, **evr.indicateurs()})
        if profils is not None:
            profils[(annee, "réel")] = evr
        for g in graines:
            evs = {}
            for nom, gr in grilles.items():
                rj, ev, dt = simuler_grille(gr, base, annee, g, fenetre, iter_patients, **kw)
                evs[nom] = (rj, ev, dt)
            union = set()
            for rj, ev, _ in evs.values():
                union |= {p for p in rj.mobiles if ev.sol.affectation[p] is None}
            for nom, (rj, ev, dt) in evs.items():
                brut = ev.indicateurs()
                if profils is not None and g == graines[0]:
                    profils[(annee, nom)] = ev
                for p in union:
                    if ev.sol.affectation[p] is not None:
                        ev.deplacer(p, None)
                ev.recalibrer_cible()
                cmp = {f"cmp_{k}": v for k, v in ev.indicateurs().items()}
                lignes.append({"annee": annee, "scenario": nom, "graine": g, "temps_s": dt,
                               "patients_exclus_cmp": len(union), **brut, **cmp})
            print(f"   {annee}, graine {g} : " + ", ".join(
                f"{n} {l['nuits_hors_2lits_pct']:.1f} %" for n, l in
                zip(grilles, lignes[-len(grilles):])))
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_lits_attendus(grilles: dict, mod, G):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    for n in range(NUITS):
        if n % 7 in NUITS_WEEKEND:
            axs[0].axvspan(n - 0.5, n + 0.5, color="grey", alpha=0.12, lw=0)
    couleurs = ["0.4", "C2", "C0", "C3", "C4"]
    for (nom, g), c in zip(grilles.items(), couleurs):
        ev = EvaluationGrille(g.copie(), mod, PoidsGrille(), G)
        axs[0].step(range(NUITS), ev.lits, where="mid", color=c, lw=1.5, label=nom)
        jo = [n for n in range(NUITS) if n % 7 < 5]
        axs[1].plot(jo, [ev.amb[n] for n in jo], marker="o", color=c, label=nom)
    ev0 = EvaluationGrille(G.copie(), mod, PoidsGrille(), G)
    moy, mf = sum(ev0.lits) / NUITS, sum(ev0.f) / NUITS
    axs[0].step(range(NUITS), [moy * f / mf for f in ev0.f], where="mid", color="k", ls=":", label="cible")
    axs[0].set_ylabel("lits attendus")
    axs[0].set_title("Lits attendus chaque nuit du cycle de 4 semaines (gris : nuits de week-end)")
    axs[1].set_ylabel("ambulatoires attendus")
    axs[1].set_title("Ambulatoires attendus par jour opératoire")
    axs[1].set_xticks(range(NUITS), [["L", "M", "M", "J", "V", "S", "D"][n % 7] for n in range(NUITS)])
    axs[1].set_xlabel("semaines du cycle : paire | impaire A | paire | impaire B")
    axs[0].legend(fontsize=8, ncol=5)
    fig.tight_layout()
    return fig


def fig_grille(G, g, titre):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(17, 13))
    for col, (gg, t) in enumerate(((G, "actuelle"), (g, titre))):
        for row, (sem, lib) in enumerate(((0, "semaine paire"), (1, "semaine impaire A"), (3, "semaine impaire B"))):
            ax = axs[row, col]
            ax.axis("off")
            salles = sorted({l.salle for l in gg.lignes})
            cells, couleurs = [], []
            for j in range(5):
                ligne, coul = [], []
                for s in salles:
                    idx = sorted((i for i, l in enumerate(gg.lignes)
                                  if l.jour == j and l.salle == s and sem in SEMAINES_ACTIVES[l.regle]),
                                 key=lambda i: gg.lignes[i].debut)
                    ligne.append("\n".join(f"{gg.chirurgiens[i]} {fmt(gg.lignes[i].debut)}-{fmt(gg.lignes[i].fin)}"
                                           for i in idx))
                    change = gg is not G and any(gg.chirurgiens[i] != G.chirurgiens[i] for i in idx)
                    coul.append("#ffd6a5" if change else "white")
                cells.append(ligne)
                couleurs.append(coul)
            tb = ax.table(cellText=cells, rowLabels=JOURS, colLabels=[f"salle {s}" for s in salles],
                          cellColours=couleurs, loc="center", cellLoc="center")
            tb.auto_set_font_size(False)
            tb.set_fontsize(8)
            tb.scale(1, 3.0)
            ax.set_title(f"Grille {t} — {lib}", fontsize=11, pad=20)
    fig.suptitle("En orange : créneaux dont l'occupant change", y=0.01)
    return fig


def fig_heures(table: pd.DataFrame):
    import matplotlib.pyplot as plt
    cols = [c for c in table.columns if c.endswith("(h/sem)")]
    fig, ax = plt.subplots(figsize=(14, 4.3))
    x = np.arange(len(table))
    w = 0.8 / len(cols)
    coul = {"besoin 2022 (h/sem)": "C3"}
    for k, c in enumerate(cols):
        ax.bar(x + k * w - 0.4 + w / 2, table[c], w, label=c.replace(" (h/sem)", ""),
               color=coul.get(c, ["0.6", "C2", "C0", "C4"][min(k - 1, 3)] if k else "C3"))
    ax.set_xticks(x, table.index)
    ax.set_ylabel("heures de bloc par semaine")
    ax.set_title("Temps de bloc par chirurgien (tous gardent des créneaux)")
    ax.legend(fontsize=8)
    return fig


def fig_profils(profils: dict, annee: int):
    import datetime as dt
    import matplotlib.pyplot as plt
    cles = [k for k in profils if k[0] == annee]
    fig, axs = plt.subplots(len(cles), 1, figsize=(14, 2.3 * len(cles)), sharex=True, sharey=True)
    for ax, k in zip(np.atleast_1d(axs), cles):
        ev = profils[k]
        rj = ev.rj
        js = list(range(rj.eval_debut, rj.eval_fin))
        d = [rj.inst.date_du_jour(j) for j in js]
        for j, dd in zip(js, d):
            if j in rj.periode_reduite:
                ax.axvspan(dd, dd + dt.timedelta(days=1), color="orange", alpha=0.12, lw=0)
        ax.step(d, [ev.sol.lits_jour[j] for j in js], where="post", lw=1)
        ax.step(d, [ev.cible[j] for j in js], where="post", color="k", ls=":", lw=0.8)
        ax.set_title(f"{annee} — {k[1]}", fontsize=10)
        ax.set_ylabel("lits")
    fig.suptitle(f"Lits occupés chaque nuit en {annee} (pointillés : cible ; orange : vacances, ponts, fériés)")
    fig.tight_layout()
    return fig


def fig_barres_ic(df, indicateurs: dict, titre, colonne_scenario="scenario"):
    import matplotlib.pyplot as plt
    scen = list(dict.fromkeys(df[colonne_scenario]))
    n = len(indicateurs)
    fig, axs = plt.subplots(math.ceil(n / 3), 3, figsize=(16, 3.8 * math.ceil(n / 3)))
    coul = ["0.6", "0.35", "C2", "C0", "C4"]
    for ax, (k, lib) in zip(np.ravel(axs), indicateurs.items()):
        ms, es = zip(*[ic95(df[df[colonne_scenario] == s][k]) for s in scen])
        ax.bar(range(len(scen)), ms, yerr=[0 if e != e else e for e in es], capsize=4, color=coul[:len(scen)])
        ax.set_xticks(range(len(scen)), scen, fontsize=8)
        ax.set_title(lib, fontsize=10)
        for i, (m, e) in enumerate(zip(ms, es)):
            ax.text(i, m + (e if e == e else 0), f"{m:.1f}" if abs(m) < 100 else f"{m:.0f}", ha="center",
                    va="bottom", fontsize=8)
    for ax in np.ravel(axs)[n:]:
        ax.axis("off")
    fig.suptitle(titre)
    fig.tight_layout()
    return fig
