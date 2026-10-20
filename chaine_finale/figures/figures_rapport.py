r"""
figures_rapport.py — Figures du rapport, toutes lues dans resultats/*.json (aucune valeur recopiée).
    python3 chaine_finale/figures/figures_rapport.py <dossier de sortie> [figure ...]
Critère n°1 : variance des lits des nuits de semaine (lundi-vendredi soir), vacances comprises,
urgences réelles comprises.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import datetime as dt
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protocole import RACINE

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "resultats" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
QUOI = set(sys.argv[2:])
RES = RACINE / "resultats"
C = {"réel": "#3d4148", "référence": "#9aa0a8", "tabou": "#E8663D", "génétique": "#1BAF7A",
     "recuit": "#E5A000", "hybride": "#5A3FC0", "glouton": "#9aa0a8"}
INK, MUTED, GRID = "#1f2328", "#5b616b", "#e6e8eb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "font.family": "DejaVu Sans"})
ALGOS = ("tabou", "recuit", "génétique")


def charger_json(nom):
    return json.loads((RES / f"{nom}.json").read_text())


def sauver(fig, nom):
    fig.savefig(OUT / f"{nom}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nom}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figure", nom)


def fr(x, f="{:.1f}"):
    if f == "{:.0f}":
        return str(int(np.floor(x + 0.5)))
    return f.format(x).replace(".", ",")


def barres(ax, noms, vals, f="{:.1f}", couleurs=None, titre="", err=None):
    x = np.arange(len(noms))
    vals = [np.nan if v is None else v for v in vals]
    v0 = [0 if np.isnan(v) else v for v in vals]
    ax.bar(x, v0, width=0.64, color=couleurs, edgecolor="white", linewidth=2, zorder=3,
           yerr=err, error_kw=dict(ecolor=MUTED, lw=0.8, capsize=2))
    top = max(max(v0), 1e-9)
    for i, (xi, v) in enumerate(zip(x, vals)):
        if np.isnan(v):
            ax.text(xi, 0.02 * top, "n.d.", ha="center", va="bottom", fontsize=7, color=MUTED)
            continue
        h = v + (err[i] if err is not None else 0)
        ax.text(xi, h + 0.02 * top, fr(v, f), ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x, noms, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, top * 1.22)
    ax.set_title(titre, fontsize=9, loc="left", color=INK)


def veut(nom):
    return not QUOI or nom in QUOI


def courbe_meilleur(traces, grille):
    """traces : liste de [(évaluations, meilleur coût)] ; renvoie le meilleur coût sur la grille."""
    out = []
    for tr in traces:
        e = np.array([x[0] for x in tr], float)
        c = np.array([x[1] for x in tr], float)
        idx = np.searchsorted(e, grille, side="right") - 1
        v = np.where(idx >= 0, c[np.clip(idx, 0, None)], np.nan)
        out.append(v)
    return np.array(out)


# ------------------------------------------------------------------------- patients fantômes
if veut("fantomes") and (RES / "fantomes_v3.json").exists():
    F = charger_json("fantomes_v3")
    V = {v["cle"]: v for v in F["variantes"]}
    v = V["previsions"]
    # 1. la courbe vue un jour de consultation
    vue = v["vue"]
    n = len(vue["Er"])
    x = np.arange(n)
    Er, Eg, Eu = (np.array(vue[k]) for k in ("Er", "Eg", "Eu"))
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    ax.bar(x, Er, width=1, color="#c9c3ea", label="patients déjà datés", zorder=2)
    ax.bar(x, Eg, bottom=Er, width=1, color="#5A3FC0", alpha=.55, label="fantômes programmés", zorder=2)
    ax.bar(x, Eu, bottom=Er + Eg, width=1, color="#E8663D", alpha=.6, label="fantômes d'urgence", zorder=2)
    ax.plot(x, vue["final"], color=INK, lw=1.4, label="lits réellement occupés ensuite", zorder=3)
    ax.axvline(6.5, color=MUTED, lw=.8, ls="--")
    ax.text(6.7, ax.get_ylim()[1] * .95, "journées closes\n(J−7)", fontsize=7, color=MUTED, va="top")
    d0 = dt.date.fromisoformat(vue["dates"][0])
    ticks = [i for i in range(n) if (d0 + dt.timedelta(i)).weekday() == 0]
    ax.set_xticks(ticks, [(d0 + dt.timedelta(i)).strftime("%d/%m") for i in ticks], fontsize=7.5)
    ax.set_xlim(-.6, n - .4)
    ax.set_ylabel("lits")
    ax.set_title(f"Courbe des lits vue par la règle de date le {dt.date.fromisoformat(vue['date']).strftime('%d/%m/%Y')}",
                 fontsize=9, loc="left")
    ax.legend(fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12))
    sauver(fig, "fantomes_vue")
    # 2. le suivi d'une nuit
    fig, axs = plt.subplots(1, len(v["suivi"]), figsize=(7.6, 2.5), sharey=True)
    for ax, s in zip(axs, v["suivi"]):
        L = s["lignes"]
        a = np.array([l["avance"] for l in L])
        er, eg, eu = (np.array([l[k] for l in L]) for k in ("Er", "Eg", "Eu"))
        # la dernière vue est la veille (ou le dernier jour de consultation) : on la prolonge jusqu'à la nuit
        a, er, eg, eu = np.append(a, 0), np.append(er, er[-1]), np.append(eg, eg[-1]), np.append(eu, eu[-1])
        ax.stackplot(-a, er, eg, eu, colors=["#c9c3ea", "#8f7fd6", "#f0a487"], lw=0)
        ax.axhline(s["final"], color=INK, lw=1.2)
        ax.axvline(-7, color=MUTED, lw=.8, ls="--")
        d = dt.date.fromisoformat(s["date"])
        ax.set_title(f"nuit du {d.strftime('%a %d/%m').replace('Mon','lun.').replace('Tue','mar.').replace('Wed','mer.').replace('Thu','jeu.').replace('Fri','ven.').replace('Sat','sam.').replace('Sun','dim.')}",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("jours avant la nuit")
        ax.set_xticks([-63, -42, -21, -7, 0], ["63", "42", "21", "7", "0"])
    axs[0].set_ylabel("lits")
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend([Patch(color="#c9c3ea"), Patch(color="#8f7fd6"), Patch(color="#f0a487"), Line2D([], [], color=INK)],
               ["patients datés", "fantômes programmés", "fantômes d'urgence", "lits finalement occupés"],
               loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(.5, -.1))
    fig.tight_layout(rect=(0, .06, 1, 1))
    sauver(fig, "fantomes_suivi")
    # 3. qualité de la prévision
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    H = [int(h) for h in F["horizons"]]
    sty = {"aucun": ("#9aa0a8", "o"), "v1": ("#E5A000", "s"), "previsions_sans_urgences": ("#8f7fd6", "^"),
           "previsions": ("#5A3FC0", "D")}
    for k, w in V.items():
        ax.plot(H, [w["rmse"][str(h)] for h in H], marker=sty[k][1], ms=4, color=sty[k][0], label=w["nom"])
    ax.set_xlabel("horizon de la prévision (jours)")
    ax.set_ylabel("erreur quadratique moyenne (lits)")
    ax.set_xticks(H)
    ax.legend(fontsize=7.5)
    sauver(fig, "fantomes_erreur")


# ------------------------------------------------------------------------- 6.2 : grille
if veut("statique") and (RES / "etude_statique.json").exists():
    S = charger_json("etude_statique")
    R = S["optimisation"]
    bmax = max(S["budgets"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    g = np.linspace(0, bmax, 161)
    for a in ALGOS:
        tr = [r["trace"] for r in R if r["algo"] == a and r["budget"] == bmax]
        M = courbe_meilleur(tr, g)
        m = np.nanmean(M, axis=0)
        a1.plot(g, m, color=C[a], lw=1.5, label=a)
        a1.fill_between(g, np.nanmin(M, axis=0), np.nanmax(M, axis=0), color=C[a], alpha=.12, lw=0)
    a1.set_xlabel("itérations")
    a1.set_ylabel("coût F de la meilleure grille")
    a1.set_ylim(3.0, 4.6)
    a1.legend(fontsize=7.5)
    a1.set_title("Évolution du coût (5 départs aléatoires)", fontsize=9, loc="left")
    B = S["budgets"]
    for i, a in enumerate(ALGOS):
        m = [st.mean(r["cout"] for r in R if r["algo"] == a and r["budget"] == b) for b in B]
        a2.plot(range(len(B)), m, marker="o", ms=3.5, color=C[a], label=a)
    a2.set_xticks(range(len(B)), [f"{b // 1000}k" for b in B])
    a2.set_xlabel("nombre d'itérations")
    a2.set_title("Coût final moyen", fontsize=9, loc="left")
    fig.tight_layout()
    sauver(fig, "cout_grille")
    # indicateurs des grilles (première date libre)
    sim = S["simulation"]
    noms, cles = ["actuelle", *ALGOS], ["actuelle", *ALGOS]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.4))
    for ax, (k, t, f) in zip(axs, [("lits_crit1", "Variance des lits, semaine", "{:.1f}"),
                                  ("delai_p90", "Délai pour 90 % (j)", "{:.0f}"),
                                  ("places_ecart_type", "Écart-type des admissions", "{:.2f}")]):
        vals, err = [], []
        for c in cles:
            xs = [v[k] for n, v in sim.items() if n == c or n.startswith(c + "#")]
            vals.append(st.mean(xs)); err.append(st.pstdev(xs) if len(xs) > 1 else 0)
        barres(ax, noms, vals, f, [C["référence"], *(C[a] for a in ALGOS)], t, err)
    fig.tight_layout()
    sauver(fig, "statique_indicateurs")


# ------------------------------------------------------------------------- 6.2 : dates
if veut("dynamique") and (RES / "etude_dynamique.json").exists():
    D = charger_json("etude_dynamique")
    lot = D["lot"]
    budgets = sorted({r["budget"] for r in lot if r["methode"] != "glouton"})
    b = max(budgets)
    noms = ["glouton", *ALGOS]
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.4))
    for ax, (k, t, f) in zip(axs, [("lits_crit1", "Variance des lits, semaine", "{:.1f}"),
                                  ("part_jours_ameliores", "Lots améliorés (%)", "{:.0f}"),
                                  ("temps_moyen_s", "Temps par lot (s)", "{:.2f}")]):
        vals, err = [], []
        for m in noms:
            xs = [r[k] * (100 if k == "part_jours_ameliores" else 1) for r in lot
                  if r["methode"] == m and (m == "glouton" or r["budget"] == b)]
            vals.append(st.mean(xs)); err.append(st.pstdev(xs))
        barres(ax, noms, vals, f, [C["glouton"], *(C[a] for a in ALGOS)], t, err)
    fig.tight_layout()
    sauver(fig, "dynamique_lot")

if veut("couts") and (RES / "final.json").exists():
    FN = charger_json("final")
    ch = {c["nom"]: c for c in FN.get("chaines", [])}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7))
    # dates par lots : gain relatif au glouton
    ax = axs[0]
    g = np.arange(0, 401, 5)
    for a in ALGOS:
        c = ch.get(f"Chaîne {a}")
        if not c or not c.get("traces_lots"):
            continue
        part = []
        for T in c["traces_lots"]:
            tr = T["trace"]
            c0, cf = tr[0][1], tr[-1][1]
            if cf >= c0 - 1e-12:
                continue
            M = courbe_meilleur([tr], g)[0]
            part.append(100 * (c0 - M) / (c0 - cf))
        ax.plot(g, np.nanmean(part, axis=0), color=C[a], lw=1.5, label=f"{a} ({len(part)} lots améliorés)")
    ax.set_xlabel("itérations")
    ax.set_ylabel("part du gain final obtenue (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Dater un lot de patients", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="lower right")
    # journée
    ax = axs[1]
    if (RES / "journee_algos.json").exists():
        J = charger_json("journee_algos")
        g = np.arange(0, 1001, 10)
        for a in ALGOS:
            rel = []
            for l in J:
                if l["algo"] == a and l.get("trace") and not l["graines"]:
                    tr = l["trace"]
                    c0, cf = tr[0][1], tr[-1][1]
                    if cf >= c0 - 1e-9:
                        continue
                    M = courbe_meilleur([tr], g)[0]
                    rel.append(100 * (c0 - M) / (c0 - cf))
            if rel:
                ax.plot(g, np.nanmean(rel, axis=0), color=C[a], lw=1.5, label=a)
        ax.set_xlabel("itérations")
        ax.set_ylabel("part du gain final obtenue (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(0, 300)
        ax.set_title("Ordonner une journée", fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    sauver(fig, "cout_dates_journee")


# ------------------------------------------------------------------------- 6.2 : journée
if veut("journee") and (RES / "gap_journee.json").exists():
    GJ = charger_json("gap_journee")
    J = charger_json("journee_algos")
    B = (100, 300, 1000)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.5))
    for a in ALGOS:
        n = [sum(abs(l[f"{a}_{b}"] - l["optimum"]) < 1e-6 * max(1, l["optimum"]) for l in GJ) for b in B]
        axs[0].plot(range(len(B)), n, marker="o", ms=3.5, color=C[a], label=a)
        t = [1000 * st.mean(l["secondes"] for l in J if l["algo"] == a and l["budget"] == b and l["graines"]) for b in B]
        axs[1].plot(range(len(B)), t, marker="o", ms=3.5, color=C[a], label=a)
    dep = sum(abs(l["depart"] - l["optimum"]) < 1e-6 * max(1, l["optimum"]) for l in GJ)
    axs[0].axhline(dep, color=MUTED, ls="--", lw=.9)
    axs[0].text(0, dep - 6, "meilleur ordre simple", fontsize=7, color=MUTED, va="top")
    axs[0].set_ylim(0, len(GJ) * 1.08)
    axs[0].set_ylabel(f"journées à l'optimum (sur {len(GJ)})")
    axs[1].set_ylabel("temps par journée (ms)")
    for ax in axs:
        ax.set_xticks(range(len(B)), [str(b) for b in B])
        ax.set_xlabel("nombre d'itérations")
    axs[0].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    sauver(fig, "journee_algos")


# ------------------------------------------------------------------------- 7.2 : chaînes
if veut("chaines") and (RES / "final.json").exists():
    FN = charger_json("final")
    ch = {c["nom"]: c for c in FN["chaines"]}
    ordre = [("Référence", "référence"), ("Chaîne tabou", "tabou"), ("Chaîne recuit", "recuit"),
             ("Chaîne génétique", "génétique"), ("Chaîne hybride", "hybride")]
    R = FN["reel"]
    noms = ["réel"] + [o[1] for o in ordre]
    cols = [C[n] for n in noms]
    fig, axs = plt.subplots(2, 3, figsize=(7.8, 4.6))
    spec = [("lits_crit1", "Variance des lits, semaine", "{:.1f}"),
            ("lits_max", "Pic de lits", "{:.0f}"),
            ("places_ecart_type", "Écart-type des admissions", "{:.2f}"),
            ("delai_median", "Délai médian (j)", "{:.0f}"),
            ("delai_p90", "Délai pour 90 % (j)", "{:.0f}"),
            ("sans_date", "Patients sans date", "{:.0f}")]
    for ax, (k, t, f) in zip(axs.flat, spec):
        vals = [R.get(k, np.nan)] + [ch[o[0]][k] for o in ordre]
        if k in ("delai_median", "delai_p90", "sans_date"):
            vals[0] = np.nan
        barres(ax, noms, vals, f, cols, t)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    fig.tight_layout()
    sauver(fig, "chaines")
    # profil par nuit de semaine : réel vs référence vs hybride
    import protocole
    _, don = protocole.charger()
    t0, t1 = don.test
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    jours = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    for nom, serie, coul in (("réel", R["lits_serie"], C["réel"]), ("référence", ch["Référence"]["lits_serie"], C["référence"]),
                             ("hybride", ch["Chaîne hybride"]["lits_serie"], C["hybride"])):
        L = np.array(serie, float)
        w = np.array([don.date(t0 + i).weekday() for i in range(len(L))])
        ax.plot(range(7), [L[w == d].mean() for d in range(7)], marker="o", ms=3.5, color=coul, label=nom)
    ax.set_xticks(range(7), jours)
    ax.set_ylabel("lits occupés (moyenne)")
    ax.set_title("Lits selon la nuit, mai–décembre 2022", fontsize=9, loc="left")
    ax.legend(fontsize=7.5)
    sauver(fig, "lits_semaine")


# ------------------------------------------------------------------------- 7.3 : ablations
if veut("ablations") and (RES / "final.json").exists():
    FN = charger_json("final")
    A = [a for a in FN["ablations"] if a["grille"] == "hybride"]
    ds = [a for a in A if a["fantomes"] == "previsions" and a["choix"] == "plus_proche"]
    ds.sort(key=lambda a: a["delta"])
    x = [a["delta"] for a in ds]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, k, t, f, c in ((axs[0], "lits_crit1", "Variance des lits, semaine", "{:.1f}", C["hybride"]),
                           (axs[1], "delai_median", "Délai médian (jours)", "{:.0f}", C["recuit"])):
        v = [a[k] for a in ds]
        ax.plot(x, v, marker="o", color=c, lw=1.8)
        for xi, vi in zip(x, v):
            ax.annotate(fr(vi, f), (xi, vi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xlabel("fenêtre de tolérance Δ (jours)")
        ax.axvline(14, color=MUTED, lw=.8, ls="--")
        ax.set_title(t, fontsize=9.5, loc="left")
        ax.margins(y=.25)
    fig.tight_layout(w_pad=2)
    sauver(fig, "ablations")


# ------------------------------------------------------------------------- 6.3 : grille hybride
if veut("hybride_grille"):
    import protocole
    G = protocole.lire_grille("hybride")["meta"]
    cands = G["candidates"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    for c in cands:
        tr = c["trace"]
        coul = "#8f7fd6" if c["w"] == 1.0 else C["hybride"]
        ls = "-" if c["origine"].startswith("recuit") else "--"
        x = [p[0] for p in tr]
        a1.plot(x, [p[1] for p in tr], color=coul, lw=.9, ls=ls, alpha=.85)
    from matplotlib.lines import Line2D
    a1.legend([Line2D([], [], color="#8f7fd6"), Line2D([], [], color=C["hybride"]),
               Line2D([], [], color=MUTED, ls="--")],
              ["recuit, poids des lits 1", "recuit, poids des lits 4", "après croisement"], fontsize=7)
    a1.set_xlabel("itérations")
    a1.set_ylabel("coût F")
    a1.set_ylim(2.8, 4.2)
    a1.set_title("Recherche des grilles candidates", fontsize=9, loc="left")
    for w, coul, m in ((1.0, "#8f7fd6", "o"), (4.0, C["hybride"], "s")):
        cs = [c for c in cands if c["w"] == w]
        a2.scatter([c["cout"] for c in cs], [c["app_crit1"] for c in cs], color=coul, marker=m, s=22,
                   label=f"poids des lits {int(w)}", zorder=3)
    r = next(c for c in cands if c["origine"] == G["retenue"])
    a2.scatter([r["cout"]], [r["app_crit1"]], s=110, facecolors="none", edgecolors=INK, lw=1.2, zorder=4)
    a2.annotate("retenue", (r["cout"], r["app_crit1"]), xytext=(8, -12), textcoords="offset points",
                fontsize=7.5, color=INK)
    a2.set_xlabel("coût F de la grille")
    a2.set_ylabel("variance des lits, semaine\n(janvier–avril, simulée)")
    a2.set_title("Choix par simulation", fontsize=9, loc="left")
    a2.legend(fontsize=7)
    fig.tight_layout()
    sauver(fig, "hybride_grille")


# ------------------------------------------------------------------------- 7.3 : cible
if veut("cible") and (RES / "etude_cibles.json").exists():
    E = charger_json("etude_cibles")
    cles = [(1.0, 1.0), (0.85, 0.70), (0.80, 0.75), (0.75, 0.70), (0.70, 0.65), (0.65, 0.60)]
    M = {}
    for c in cles:
        rs = [r for r in E if tuple(r["cible"]) == c]
        if rs:
            M[c] = {k: st.mean(r[k] for r in rs) for k in rs[0] if isinstance(rs[0][k], (int, float))}
    cles = [c for c in cles if c in M]
    lab = ["sans\ncible" if c == (1.0, 1.0) else f"{c[0]:.2f}\n{c[1]:.2f}".replace(".", ",") for c in cles]
    col = [C["hybride"] if c == (0.70, 0.65) else "#b9b2dc" for c in cles]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.6))
    barres(axs[0], lab, [M[c]["lits_crit1"] for c in cles], "{:.1f}", col, "Variance des lits, semaine")
    barres(axs[1], lab, [M[c]["lits_ratio_weekend"] for c in cles], "{:.2f}", col, "Week-end / semaine")
    barres(axs[2], lab, [M[c]["lits_ratio_vacances"] for c in cles], "{:.2f}", col, "Vacances / semaine")
    for ax in axs:
        ax.set_xlabel("cible : week-end / vacances", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    sauver(fig, "cible")


# ------------------------------------------------------------------------- 7.4 : surcharge
VF = RES
if veut("surcharge") and (VF / "surcharge.json").exists():
    S = json.loads((VF / "surcharge.json").read_text())
    niv = sorted({r["taux"] for r in S})
    x = [round(100 * t) for t in niv]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.5))
    for ax, (m, t, f) in zip(axs, [("patients_par_semaine", "Patients opérés par semaine", "{:.0f}"),
                                   ("sans_date", "Patients sans date", "{:.0f}"),
                                   ("delai_p90", "Délai pour 90 % (j)", "{:.0f}")]):
        for k, nom in (("reference", "référence"), ("hybride", "hybride")):
            v = [st.mean(r[m] for r in S if r["taux"] == t and r["chaine"] == k) for t in niv]
            ax.plot(x, v, marker="o", lw=1.8, color=C[nom], label=nom)
            ax.text(x[-1] + 1.2, v[-1], fr(v[-1], f), fontsize=7.5, color=C[nom], va="center")
        ax.set_xticks(x, [f"+{v} %" for v in x])
        ax.set_xlim(-3, x[-1] + 7)
        ax.set_title(t, fontsize=9, loc="left")
        ax.set_xlabel("patients en plus", fontsize=7.5, color=MUTED)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    sauver(fig, "surcharge")


# ------------------------------------------------------------------------- 7.5 : SMA
def _taux(r):
    R = r["resume"]
    n = max(1, R["urgences_arrivees"])
    return dict(jour=100 * R["urgences_operees_jour_meme"] / n, echec=100 * R["urgences_echec"] / n,
                dep=R["depassement_min"] / R["jours"], attente=R["attente_urgence_moy"],
                conv=R["conversions_nuit"], decales=R["patients_decales"] / R["jours"])


if veut("sma") and (RES / "sma.json").exists():
    SM = charger_json("sma")
    runs = SM["runs"]
    A = [r for r in runs if r["exp"] == "A" and r["graine"] is not None]
    pols = [("statique", "règle statique", "#c4c8ce"), ("reoptimisation", "réoptimisation", "#8f96a0"),
            ("sma", "multi-agents", C["hybride"])]
    plans = [("reference", "plan de référence"), ("hybride", "plan hybride")]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.6))
    w = 0.26
    for ax, (m, t, f) in zip(axs, [("jour", "Urgences opérées le jour même (%)", "{:.0f}"),
                                   ("echec", "Urgences sans place (%)", "{:.0f}"),
                                   ("dep", "Dépassement (min par jour)", "{:.0f}")]):
        top = 0
        for i, (p, lab, col) in enumerate(pols):
            v = [st.mean(_taux(r)[m] for r in A if r["chaine"] == c and r["politique"] == p) for c, _ in plans]
            xx = np.arange(2) + (i - 1) * w
            ax.bar(xx, v, width=w, color=col, edgecolor="white", lw=1.2, zorder=3, label=lab)
            for xi, vi in zip(xx, v):
                ax.text(xi, vi, fr(vi, f), ha="center", va="bottom", fontsize=6.8, color=INK)
            top = max(top, max(v))
        ax.set_xticks(np.arange(2), [p[1] for p in plans], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, top * 1.2)
        ax.set_title(t, fontsize=8.5, loc="left")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(.5, 1.07), fontsize=8)
    fig.tight_layout()
    sauver(fig, "sma")
    # intensité
    B = [r for r in runs if r["exp"] == "B" and r["chaine"] == "hybride"]
    if B:
        I = sorted({r["urgences_par_jour"] for r in B})
        fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5))
        for ax, (m, t) in zip(axs, [("jour", "Urgences opérées le jour même (%)"), ("echec", "Urgences sans place (%)")]):
            for p, lab, col in (pols[0], pols[2]):
                v = [st.mean(_taux(r)[m] for r in B if r["urgences_par_jour"] == u and r["politique"] == p) for u in I]
                ax.plot(I, v, marker="o", color=col if p == "sma" else MUTED, label=lab)
            ax.set_xticks(I, [fr(u, "{:.1f}") for u in I])
            ax.set_xlabel("urgences, en multiple du flux réel de 2022", fontsize=7.5)
            ax.set_title(t, fontsize=9, loc="left")
        axs[0].legend(fontsize=7.5)
        fig.tight_layout()
        sauver(fig, "sma_intensite")


# ------------------------------------------------------------------------- 7.5 : aléas déclarés
if veut("aleas") and (VF / "aleas.json").exists():
    AL = json.loads((VF / "aleas.json").read_text())
    d = {l["nom"]: l for l in AL}
    mini, comp = d["9. Tout · gestion minimale"], d["9. Tout · gestion complète"]
    spec = [("a_arbitrer", "Patients\nà arbitrer", "{:.0f}"), ("plus_30j", "Reprogrammés\nà plus de 30 j", "{:.0f}"),
            ("delai_p90", "Délai pour\n90 % (j)", "{:.0f}"), ("lits_crit1", "Variance des\nlits, semaine", "{:.1f}")]
    fig, axs = plt.subplots(1, 4, figsize=(7.8, 2.4))
    for ax, (k, t, f) in zip(axs, spec):
        barres(ax, ["gestion\nminimale", "gestion\ncomplète"], [mini[k], comp[k]], f, ["#c4c8ce", C["hybride"]], t)
    fig.tight_layout()
    sauver(fig, "aleas")


# ------------------------------------------------------------------------- 6.2 : réglage
if veut("reglage") and (RES / "reglage_parametres.json").exists():
    R = charger_json("reglage_parametres")
    RETENU = {}
    NOM = {"k": "voisins par itération k", "tenure": "durée d'interdiction",
           "T0_facteur": "température de départ\n(× dégradation typique)", "Tmin_facteur": "température finale\n(× température de départ)",
           "population": "taille de la population", "p_mutation": "probabilité de mutation",
           "descente": "itérations de descente finale"}
    plans = R["plans"]
    fig, axs = plt.subplots(2, 4, figsize=(7.8, 4.4))
    cases = [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[1, 0], axs[1, 1], axs[1, 2]]
    for ax in cases:
        ax.set_ylim(3.0, 4.15)
    for ax in cases[1:4] + cases[5:]:
        ax.tick_params(labelleft=False)
    for ax, p in zip(cases, plans):
        a, prm, vals = p["algo"], p["param"], p["valeurs"]
        m, lo, hi = [], [], []
        for v in vals:
            c = [r["cout"] for r in R["runs"] if r["algo"] == a and r["param"] == prm and r["valeur"] == v]
            m.append(np.mean(c)); lo.append(min(c)); hi.append(max(c))
        x = np.arange(len(vals))
        ax.fill_between(x, lo, hi, color=C[a], alpha=.15, lw=0)
        ax.plot(x, m, marker="o", ms=3.5, color=C[a])
        ret = RETENU.get((a, prm), p["retenue"])
        i = vals.index(ret)
        ax.scatter([i], [m[i]], s=90, facecolors="none", edgecolors=INK, lw=1.1, zorder=4)
        ax.set_xticks(x, [fr(v, "{:g}") if isinstance(v, float) else str(v) for v in vals], fontsize=7)
        ax.set_xlabel(NOM[prm], fontsize=7.5)
        ax.set_title(a, fontsize=8.5, loc="left", color=C[a])
    ax = axs[1, 3]
    if (RES / "reglage_lits.json").exists():
        RL = charger_json("reglage_lits")
        for a in ALGOS:
            rs = [r for r in RL["runs"] if r["algo"] == a]
            ax.scatter([r["cout"] for r in rs], [r["crit1"] for r in rs], s=9, color=C[a], alpha=.75, label=a)
        cc = np.corrcoef([r["cout"] for r in RL["runs"]], [r["crit1"] for r in RL["runs"]])[0, 1]
        ax.set_xlabel("coût F de la grille", fontsize=7.5)
        ax.set_ylabel("variance des lits\n(simulée, févr.–avr.)", fontsize=7.5)
        ax.set_title(f"coût et lits : r = {cc:.2f}".replace(".", ","), fontsize=8.5, loc="left")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.sharey = None
    axs[0, 0].set_ylabel("coût F final")
    axs[1, 0].set_ylabel("coût F final")
    axs[0, 0].set_ylim(3.0, 4.15)
    fig.tight_layout()
    sauver(fig, "reglage")


# ------------------------------------------------------------------------- V5 : couleur par rang
def carte_rang(ax, lignes, colonnes, M, mieux=None, fmts=None, hors_classement=(), titre_="", fs=8, tol_rel=0.01):
    """Chaque colonne est classée indépendamment : meilleure valeur en vert foncé et en gras,
    deuxième en vert clair (si au moins 5 lignes classées), moins bonne en rouge clair.
    Les lignes de `hors_classement` (référence, réel) sont grisées et non classées."""
    from matplotlib.patches import Rectangle
    A = np.array(M, dtype=float)
    n, m = A.shape
    mieux = mieux or ["bas"] * m
    fmts = fmts or ["{:.1f}"] * m
    ax.set_xlim(-.5, m - .5)
    ax.set_ylim(n - .5, -.5)
    for i in range(n):
        for j in range(m):
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#ececec" if i in hors_classement else "white",
                                   edgecolor="white", lw=1.5))
    for j in range(m):
        idx = [i for i in range(n) if i not in hors_classement and not np.isnan(A[i, j])]
        if len(idx) < 2:
            continue
        v = np.array([A[i, j] for i in idx])
        sgn = 1 if mieux[j] == "bas" else -1
        if mieux[j] is None:
            continue
        best, worst = (sgn * v).min(), (sgn * v).max()
        if abs(best - worst) <= tol_rel * max(abs(best), abs(worst), 1e-9):
            continue                      # écart négligeable : pas de classement
        tol = 1e-6 * max(1, abs(best))
        ordre = sorted(set(np.round(sgn * v, 9)))
        for i, x in zip(idx, sgn * v):
            if x <= best + tol:
                c = "#2e8b57"
            elif len(idx) >= 5 and len(ordre) > 2 and x <= ordre[1] + tol:
                c = "#bfe3cc"
            elif x >= worst - tol:
                c = "#f4c7c3"
            else:
                continue
            ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor="white", lw=1.5))
    for i in range(n):
        for j in range(m):
            v = A[i, j]
            idx = [k for k in range(n) if k not in hors_classement and not np.isnan(A[k, j])]
            sgn = 1 if mieux[j] == "bas" else -1
            vals_ = [sgn * A[k, j] for k in idx]
            est_best = (mieux[j] is not None and i in idx and len(idx) >= 2 and
                        abs(sgn * v - min(vals_)) < 1e-6 * max(1, abs(v)) and
                        max(vals_) - min(vals_) > tol_rel * max(abs(min(vals_)), abs(max(vals_)), 1e-9))
            ax.text(j, i, "—" if np.isnan(v) else fr(v, fmts[j]), ha="center", va="center", fontsize=fs,
                    color="white" if est_best else (MUTED if i in hors_classement else INK),
                    fontweight="bold" if est_best else "normal")
    ax.set_xticks(range(m), colonnes, fontsize=fs - .5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    ax.set_yticks(range(n), lignes, fontsize=fs)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if titre_:
        ax.set_title(titre_, fontsize=fs + 1.5, loc="left", pad=30, color=INK)


def legende_rang(fig, y=0.0):
    from matplotlib.patches import Patch
    fig.legend([Patch(color="#2e8b57"), Patch(color="#f4c7c3"), Patch(color="#ececec")],
               ["meilleure valeur de la colonne", "moins bonne", "référence, non classée"],
               loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, y), frameon=False)


# ------------------------------------------------------------------------- 6.2 : cartes de chaleur
def carte(ax, lignes, colonnes, M, ref=0, fmts=None, mieux=None, titre_=""):
    """M[i][j] valeurs ; couleur = gain relatif par rapport à la ligne ref (vert = mieux)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("g", ["#d9534f", "#f7f7f7", "#2e9d5b"])
    A = np.array(M, dtype=float)
    G = np.zeros_like(A)
    for j in range(A.shape[1]):
        r = A[ref, j]
        if np.isnan(r):                      # colonne sans valeur de référence : première valeur connue
            col = A[:, j][~np.isnan(A[:, j])]
            r = col[0] if len(col) else np.nan
        for i in range(A.shape[0]):
            v = A[i, j]
            if np.isnan(v) or np.isnan(r):
                G[i, j] = 0
                continue
            d = (v - r) / max(abs(r), 1e-9) if abs(r) > 1e-9 else (0 if abs(v) < 1e-9 else np.sign(v))
            G[i, j] = np.clip(-d if (mieux is None or mieux[j] == "bas") else d, -0.5, 0.5)
    ax.imshow(G, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A[i, j]
            f = (fmts or ["{:.1f}"] * A.shape[1])[j]
            ax.text(j, i, "n.d." if np.isnan(v) else fr(v, f), ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(colonnes)), colonnes, fontsize=7)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(lignes)), lignes, fontsize=7.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(titre_, fontsize=9, loc="left", pad=28)


if veut("chaleur"):
    S = charger_json("etude_statique")
    sim = S["simulation"]
    R = S["optimisation"]
    bmax = max(S["budgets"])
    cols = ["variance\nlits", "pic\nlits", "é.-t.\nadmissions", "délai\nmédian", "délai\n90 %", "sans\ndate", "coût F", "temps\n(s)"]
    lig, M = [], []
    for nom, cle in (("grille actuelle", "actuelle"), ("tabou", "tabou"), ("recuit", "recuit"), ("génétique", "génétique")):
        xs = [v for n, v in sim.items() if n == cle or n.startswith(cle + "#")]
        cf = [r["cout"] for r in R if r["algo"] == cle and r["budget"] == bmax]
        tm = [r["secondes"] for r in R if r["algo"] == cle and r["budget"] == bmax]
        lig.append(nom)
        M.append([np.mean([x[k] for x in xs]) for k in ("lits_crit1", "lits_semaine_max", "places_ecart_type",
                                                       "delai_median", "delai_p90", "sans_date")]
                 + [np.mean(cf) if cf else np.nan, np.mean(tm) if tm else np.nan])
    fig1 = (lig, cols, M)
    D = charger_json("dates_fantomes")["runs"] if (RES / "dates_fantomes.json").exists() else []
    colsd = ["variance\nlits", "pic\nlits", "é.-t.\nadmissions", "délai\nmédian", "délai\n90 %", "sans\ndate", "temps par\nlot (ms)"]
    ligd, Md = [], []
    for m, nm in (("glouton", "énumération"), ("tabou", "tabou"), ("recuit", "recuit"), ("génétique", "génétique")):
        for f in (False, True):
            rs = [r for r in D if r["methode"] == m and r["fantomes"] == f]
            if not rs:
                continue
            ligd.append(f"{nm}{' + fantômes' if f else ''}")
            Md.append([np.mean([r[k] for r in rs]) for k in ("lits_crit1", "lits_crit1_max", "places_ecart_type",
                                                             "delai_median", "delai_p90", "sans_date")]
                      + [1000 * np.mean([r["temps_lot"] for r in rs])])
    JC = charger_json("journee_criteres")["moyennes"]
    GJ = charger_json("gap_journee")
    colsj = ["coût J", "pic de\nplaces", "é.-t.\nplaces", "ambu après\n20 h / 10 j", "dépass.\nréel (min)", "optimum\n/ 144", "temps\n(ms)"]
    ligj, Mj = [], []
    opt = {"ordre simple": sum(abs(l["depart"] - l["optimum"]) < 1e-6 * l["optimum"] for l in GJ)}
    for a in ALGOS:
        opt[a] = sum(abs(l[f"{a}_1000"] - l["optimum"]) < 1e-6 * l["optimum"] for l in GJ)
    for nom in ("inscription", "ordre simple", "tabou", "recuit", "génétique"):
        x = JC[nom]
        ligj.append("ordre d'inscription" if nom == "inscription" else nom)
        Mj.append([x["cout"], x["pic_places"], x["ecart_type_places"], 10 * x["hors_uca_reel"],
                   x["depassement_reel_min"], opt.get(nom, np.nan), 1000 * x["secondes"]])
    fig, axs = plt.subplots(3, 1, figsize=(7.6, 8.6), gridspec_kw=dict(height_ratios=[4, max(len(ligd), 1), 5]))
    carte(axs[0], *fig1, fmts=["{:.1f}", "{:.0f}", "{:.2f}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.2f}", "{:.0f}"],
          titre_="Construire la grille (chaque patient à la première date libre)")
    if Md:
        carte(axs[1], ligd, colsd, Md, fmts=["{:.1f}", "{:.0f}", "{:.2f}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.0f}"],
              titre_="Dater les patients (grille actuelle)")
    carte(axs[2], ligj, colsj, Mj, fmts=["{:.0f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:.1f}", "{:.0f}", "{:.0f}"],
          mieux=["bas"] * 5 + ["haut", "bas"], titre_="Ordonner la journée (1 000 évaluations)")
    fig.tight_layout(h_pad=1.2)
    sauver(fig, "chaleur")


if veut("convergence"):
    S = charger_json("etude_statique")
    R = S["optimisation"]
    bmax = max(S["budgets"])
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.5))
    ax = axs[0]
    g = np.linspace(0, bmax, 161)
    for a in ALGOS:
        M_ = courbe_meilleur([r["trace"] for r in R if r["algo"] == a and r["budget"] == bmax], g)
        ax.plot(g / 1000, np.nanmean(M_, axis=0), color=C[a], lw=1.5, label=a)
    ax.set_ylim(3.0, 4.6)
    ax.set_xlabel("milliers d'itérations")
    ax.set_ylabel("meilleur coût trouvé")
    ax.set_title("Grille : coût F", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    FN = charger_json("final")
    ch = {c["nom"]: c for c in FN.get("chaines", [])}
    ax = axs[1]
    g = np.arange(0, 401, 5)
    for a in ALGOS:
        c = ch.get(f"Chaîne {a}")
        part = []
        for T in (c or {}).get("traces_lots", []):
            tr = T["trace"]
            c0, cf = tr[0][1], tr[-1][1]
            if cf < c0 - 1e-12:
                part.append(100 * (c0 - courbe_meilleur([tr], g)[0]) / (c0 - cf))
        if part:
            ax.plot(g, np.nanmean(part, axis=0), color=C[a], lw=1.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("itérations")
    ax.set_ylabel("part du gain final (%)")
    ax.set_title("Dates d'un lot", fontsize=9, loc="left")
    ax = axs[2]
    J = charger_json("journee_algos")
    g = np.arange(0, 301, 5)
    for a in ALGOS:
        rel = []
        for l in J:
            if l["algo"] == a and l.get("trace") and not l["graines"]:
                tr = l["trace"]
                c0, cf = tr[0][1], tr[-1][1]
                if cf < c0 - 1e-9:
                    rel.append(100 * (c0 - courbe_meilleur([tr], g)[0]) / (c0 - cf))
        if rel:
            ax.plot(g, np.nanmean(rel, axis=0), color=C[a], lw=1.5)
    ax.set_ylim(0, 105)
    ax.set_xlabel("itérations")
    ax.set_title("Journée : coût J", fontsize=9, loc="left")
    fig.tight_layout()
    sauver(fig, "convergence")


# ------------------------------------------------------------------------- 7 : escalier et combinaisons
NOM_PREV = {"connu": "séjour connu", "mediane": "médiane", "ml_nuits": "nuits par appr.", "ml": "durées et nuits par appr."}


def _moy(rs, k):
    v = [r[k] for r in rs if k in r]
    return float(np.mean(v)) if v else np.nan


if veut("escalier") and (RES / "recap_v3.json").exists():
    RC = charger_json("recap_v3")
    P_, J_ = RC["plans"], RC.get("jour", [])
    etapes = [n for n in dict.fromkeys(r["nom"] for r in P_) if not n.startswith("c ") and n != "3 + règle de date"]
    # configuration retenue pour le jour J : plus faible variance des lits avec agents
    combos = sorted({(r["fantomes"], r["prevision"]) for r in J_})
    ret = min(combos, key=lambda c: _moy([r for r in J_ if (r["fantomes"], r["prevision"]) == c and r["politique"] == "sma"],
                                         "lits_crit1")) if J_ else None
    lab = [n.split(" ", 1)[1].replace("+ fantômes", "+ règle de date et fantômes") for n in etapes]
    val = {k: [_moy([r for r in P_ if r["nom"] == n], k) for n in etapes] for k in ("lits_crit1", "lits_crit1_max", "delai_p90")}
    if ret:
        for pol, nom in (("statique", "jour J sans agents"), ("sma", "jour J avec agents")):
            rs = [r for r in J_ if (r["fantomes"], r["prevision"]) == ret and r["politique"] == pol]
            lab.append(nom)
            val["lits_crit1"].append(_moy(rs, "lits_crit1"))
            val["lits_crit1_max"].append(_moy(rs, "lits_crit1_max"))
            val["delai_p90"].append(np.nan)
    cols = [C["référence"]] + [C["hybride"]] * 2 + ["#8f7fd6"] * (len(etapes) - 3) + ["#c4c8ce", C["hybride"]][:len(lab) - len(etapes)]
    fig, axs = plt.subplots(1, 3, figsize=(7.8, 3.3), sharey=True)
    y = np.arange(len(lab))[::-1]
    for ax, (k, t, f) in zip(axs, [("lits_crit1", "Variance des lits, semaine", "{:.1f}"),
                                  ("lits_crit1_max", "Pic de lits", "{:.0f}"),
                                  ("delai_p90", "Délai pour 90 % (j)", "{:.0f}")]):
        v = val[k]
        ax.barh(y, [0 if np.isnan(x) else x for x in v], color=cols, height=.65, zorder=3)
        top = np.nanmax(v)
        for yi, x in zip(y, v):
            ax.text((0 if np.isnan(x) else x) + top * .02, yi, "—" if np.isnan(x) else fr(x, f), va="center", fontsize=7)
        ax.set_xlim(0, top * 1.2)
        ax.set_title(t, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y, lab, fontsize=7.5)
    fig.tight_layout()
    sauver(fig, "escalier")
    (OUT / "escalier_retenu.json").write_text(json.dumps(dict(retenu=ret)))

if veut("combinaisons") and (RES / "recap_v3.json").exists():
    RC = charger_json("recap_v3")
    P_, J_ = RC["plans"], RC["jour"]
    lig, M = [], []
    for fa in ("previsions",):
        for pr in ("mediane", "ml_nuits", "ml"):
            nomp = [r["nom"] for r in P_ if r["fantomes"] == fa and r["prevision"] == pr and r["grille"] == "hybride" and r["regle"] == "hybride"]
            rp = [r for r in P_ if r["nom"] in nomp[:1]]
            for pol, nm in ((None, "sans aléa"), ("statique", "aléas, sans agents"), ("sma", "aléas, avec agents")):
                if pol is None:
                    rs = rp
                    row = [_moy(rs, "lits_crit1"), _moy(rs, "lits_crit1_max"), _moy(rs, "delai_p90"), np.nan, np.nan, np.nan, np.nan]
                else:
                    rs = [r for r in J_ if r["fantomes"] == fa and r["prevision"] == pr and r["politique"] == pol]
                    row = [_moy(rs, "lits_crit1"), _moy(rs, "lits_crit1_max"), _moy(rp, "delai_p90"), _moy(rs, "urg_delai"),
                           _moy(rs, "urg_jour"), _moy(rs, "depassement_jour"), _moy(rs, "reportes")]
                retenue = (fa, pr, pol) == ("previsions", "ml_nuits", "sma")
                lig.append(f"{'► ' if retenue else ''}{NOM_PREV[pr]} · {nm}")
                M.append(row)
    cols = ["variance\nlits", "pic\nlits", "délai\n90 %", "urgences\nà temps %", "urgences\njour J %",
            "dépass.\n(min / j)", "patients\nreportés"]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    carte_rang(ax, lig, cols, M, fmts=["{:.1f}", "{:.1f}", "{:.0f}", "{:.1f}", "{:.0f}", "{:.1f}", "{:.0f}"],
               mieux=["bas", "bas", "bas", "haut", "haut", "bas", "bas"], fs=7.5)
    for t in ax.get_yticklabels():
        if t.get_text().startswith("►"):
            t.set_fontweight("bold")
            t.set_color(INK)
    for i in range(3, len(lig), 3):
        ax.axhline(i - .5, color=MUTED, lw=.6)
    fig.tight_layout(rect=(0, .04, 1, 1))
    from matplotlib.patches import Patch
    fig.legend([Patch(color="#2e8b57"), Patch(color="#bfe3cc"), Patch(color="#f4c7c3")],
               ["meilleure valeur de la colonne", "deuxième", "moins bonne"],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    sauver(fig, "combinaisons")


# ------------------------------------------------------------------------- V5 : études d'origine
def _eo(partie, nom, k):
    rs = [r for r in EO["runs"] if r["partie"] == partie and r["nom"] == nom]
    return float(np.mean([r[k] for r in rs])) if rs and k in rs[0] else np.nan


if (veut("origine") or veut("chaines_origine") or veut("fantomes_grilles")) and (RES / "etudes_origine.json").exists():
    EO = charger_json("etudes_origine")

if veut("origine") and (RES / "etudes_origine.json").exists():
    cg = ["variance\nlits", "pic\nlits", "é.-t.\nadmissions", "heures\noffertes", "heures\nopérées", "sans\ndate",
          "délai\nmédian", "délai\n90 %"]
    kg = ["lits_crit1", "lits_crit1_max", "places_ecart_type", "heures_offertes_semaine", "occupation_reelle_pct",
          "sans_date", "delai_median", "delai_p90"]
    fg = ["{:.1f}", "{:.0f}", "{:.2f}", "{:.0f}", "{:.0f} %", "{:.0f}", "{:.0f}", "{:.0f}"]
    mg = ["bas", "bas", "bas", None, "haut", "bas", "bas", "bas"]
    lg = [("grille actuelle", "actuelle"), ("tabou (Gabriel)", "tabou"), ("génétique (Killian)", "génétique"),
          ("recuit (Lucie)", "recuit")]
    MG = [[_eo("G", c, k) for k in kg] for _, c in lg]
    cd = ["variance\nlits", "pic\nlits", "é.-t.\nadmissions", "heures\nopérées", "sans\ndate", "délai\nmédian",
          "délai\n90 %", "temps par\npatient (ms)"]
    kd = ["lits_crit1", "lits_crit1_max", "places_ecart_type", "occupation_reelle_pct", "sans_date", "delai_median",
          "delai_p90", "secondes_par_patient"]
    ld = [("tabou : au plus tôt", "tabou"), ("génétique : 12 dates notées", "génétique"),
          ("recuit : première date libre", "première date libre (recuit)")]
    MD = [[_eo("D", c, k) * (1000 if k == "secondes_par_patient" else 1) for k in kd] for _, c in ld]
    JC = charger_json("journee_criteres")["moyennes"]
    GJ = charger_json("gap_journee")
    opt = {a: sum(abs(l[f"{a}_1000"] - l["optimum"]) < 1e-6 * l["optimum"] for l in GJ) for a in ALGOS}
    cj = ["coût J", "pic de\nplaces", "é.-t.\nplaces", "ambu après\n20 h / 10 j", "dépass.\nréel (min)",
          "optimum\n/ 144", "temps\n(ms)"]
    lj = [("ordre d'inscription", "inscription"), ("tabou", "tabou"), ("génétique", "génétique"), ("recuit", "recuit")]
    MJ = [[JC[c]["cout"], JC[c]["pic_places"], JC[c]["ecart_type_places"], 10 * JC[c]["hors_uca_reel"],
           JC[c]["depassement_reel_min"], opt.get(c, np.nan), 1000 * JC[c]["secondes"]] for _, c in lj]
    fig, axs = plt.subplots(3, 1, figsize=(7.4, 7.0), gridspec_kw=dict(height_ratios=[4, 3, 4]))
    carte_rang(axs[0], [a for a, _ in lg], cg, MG, mieux=mg, fmts=fg, hors_classement=(0,),
               titre_="1. La grille de chaque étude (chaque patient à la première date libre)")
    carte_rang(axs[1], [a for a, _ in ld], cd, MD, mieux=["bas", "bas", "bas", "haut", "bas", "bas", "bas", "bas"],
               fmts=["{:.1f}", "{:.0f}", "{:.2f}", "{:.0f} %", "{:.0f}", "{:.0f}", "{:.0f}", "{:.1f}"],
               titre_="2. La règle de date de chaque étude (sur la grille actuelle)")
    carte_rang(axs[2], [a for a, _ in lj], cj, MJ, mieux=["bas"] * 5 + ["haut", "bas"],
               fmts=["{:.0f}", "{:.2f}", "{:.2f}", "{:.1f}", "{:.1f}", "{:.0f}", "{:.0f}"], hors_classement=(0,),
               titre_="3. L'ordre de la journée (1 000 évaluations)")
    fig.tight_layout(h_pad=1.0, rect=(0, .04, 1, 1))
    legende_rang(fig)
    sauver(fig, "origine")

if veut("chaines_origine") and (RES / "etudes_origine.json").exists():
    FN = charger_json("final")
    R = FN["reel"]
    cc = ["variance\nlits", "pic\nlits", "é.-t.\nadmissions", "heures\nopérées", "sans\ndate", "délai\nmédian",
          "délai\n90 %", "ambu après\n20 h (total)", "dépass.\n(min / j)"]
    kc = ["lits_crit1", "lits_crit1_max", "places_ecart_type", "occupation_reelle_pct", "sans_date", "delai_median",
          "delai_p90", "hors_uca_reel", "depassement_reel_min"]
    lc = [("tabou (Gabriel)", "tabou"), ("génétique (Killian)", "génétique"), ("recuit (Lucie)", "recuit"),
          ("chaîne hybride", "hybride")]
    M = [[R.get("lits_crit1", np.nan), R.get("lits_crit1_max", R.get("lits_max", np.nan)), R.get("places_ecart_type", np.nan)]
         + [np.nan] * 6, [_eo("C", "référence", k) for k in kc]] + [[_eo("C", c, k) for k in kc] for _, c in lc]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    carte_rang(ax, ["hôpital réel", "référence"] + [a for a, _ in lc], cc, M,
               mieux=["bas", "bas", "bas", "haut", "bas", "bas", "bas", "bas", "bas"],
               fmts=["{:.1f}", "{:.0f}", "{:.2f}", "{:.0f} %", "{:.0f}", "{:.0f}", "{:.0f}", "{:.0f}", "{:.1f}"],
               hors_classement=(0, 1))
    fig.tight_layout(rect=(0, .09, 1, 1))
    legende_rang(fig)
    sauver(fig, "chaines_origine")


# ------------------------------------------------------------------------- V5 : les trois cas des fantômes
if veut("fantomes_cas") and (RES / "fantomes_cas.json").exists():
    FC = charger_json("fantomes_cas")
    choix = [("2022-09-15", "1. Les patients remplacent\nles fantômes"),
             ("2022-10-20", "2. Moins de patients que prévu :\nles fantômes s'effacent"),
             ("2022-10-17", "3. Plus de patients que prévu :\nils dépassent la prévision")]
    par_date = {n["date"]: n for n in FC["nuits"]}
    JRS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
    for ax, (d, titre) in zip(axs, choix):
        n = par_date[d]
        V = n["vues"]
        x = np.arange(len(V))
        er = np.array([v["Er"] for v in V])
        eg = np.array([v["Eg"] for v in V])
        eu = np.array([v["Eu"] for v in V])
        ax.bar(x, er, width=.62, color="#c9c3ea", label="patients datés")
        ax.bar(x, eg, bottom=er, width=.62, color="#8f7fd6", label="fantômes programmés")
        ax.bar(x, eu, bottom=er + eg, width=.62, color="#f0a487", label="fantômes d'urgence")
        for xi, a, t in zip(x, er, er + eg + eu):
            ax.text(xi, t + .3, fr(t, "{:.1f}"), ha="center", fontsize=8, color=INK, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=.6))
            if a >= 1.5:
                ax.text(xi, a / 2, f"{a:.0f}", ha="center", va="center", fontsize=8, color=INK)
        ax.axhline(n["final"], color=INK, lw=1.3, ls="--", zorder=3)
        jd = dt.date.fromisoformat(d)
        ax.set_title(f"{titre}\nnuit du {JRS[jd.weekday()]} {jd.strftime('%d/%m')} : {n['final']:.0f} lits occupés",
                     fontsize=9, loc="left", color=INK)
        ax.set_xticks(x, [f"{round(a / 7)}" for a in FC["avances"]], fontsize=8.5)
        ax.set_xlabel("semaines avant la nuit", fontsize=8.5)
        ax.set_ylim(0, 18.5)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("lits prévus pour la nuit")
    h, l = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    fig.legend(h + [Line2D([], [], color=INK, ls="--")], l + ["lits finalement occupés"], loc="lower center",
               ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1), w_pad=1.2)
    sauver(fig, "fantomes_cas")
