r"""
figures_scenarios.py — Figures des scénarios de la soutenance, lues dans resultats/scenarios_*.json
(scenarios_soutenance.py). Même style que figures_rapport.py.
    python3 chaine_finale/figures/figures_scenarios.py <dossier de sortie> [figure ...]
Figures : surcharge, aleas_global, aleas_types, echelle, mc_distributions, mc_convergence.
Bandes et barres d'erreur : ± un écart-type entre tirages.
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from protocole import RACINE

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "resultats" / "figures_scenarios"
OUT.mkdir(parents=True, exist_ok=True)
QUOI = set(sys.argv[2:])
RES = RACINE / "resultats"
INK, MUTED, GRID = "#1f2328", "#5b616b", "#e6e8eb"
GRIS, GRIS_F, VIOLET, VIOLET_C = "#a3a9b1", "#5b616b", "#5A3FC0", "#a594e6"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "font.family": "DejaVu Sans"})

# (chaîne, politique) -> (libellé, couleur, style, marqueur plein ?)
STYLE = {("reference", "statique"): ("référence · sans agents", GRIS, "--", False),
         ("reference", "sma"): ("référence · avec agents", GRIS_F, "-", True),
         ("hybride", "statique"): ("chaîne hybride · sans agents", VIOLET_C, "--", False),
         ("hybride", "sma"): ("chaîne hybride · avec agents", VIOLET, "-", True)}
CONFIGS = list(STYLE)
CHAINES = {"reference": ("référence", GRIS_F), "hybride": ("chaîne hybride", VIOLET)}


def veut(nom):
    return not QUOI or nom in QUOI


def lire(nom):
    f = RES / f"scenarios_{nom}.json"
    return json.loads(f.read_text()) if f.exists() else None


def sauver(fig, nom):
    for ax in fig.axes:
        etiquettes(ax)
    fig.savefig(OUT / f"{nom}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nom}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure", nom)


def fr(x, f="{:.1f}"):
    if f == "{:.0f}":
        return str(int(np.floor(x + 0.5)))
    return f.format(x).replace(".", ",")


URG = {"urg_delai", "urg_jour", "urg_echec", "attente"}


def moy_et(rs, k):
    v = np.array([np.nan if (k in URG and r.get("urgences", 1) == 0) else r[k] for r in rs], float)
    v = v[~np.isnan(v)]
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def courbe(ax, x, rs_par_x, k, cle, f="{:.1f}", etiquette=True, decal=0.0):
    nom, col, ls, plein = STYLE[cle] if cle in STYLE else (*CHAINES[cle], "-", True)
    m = np.array([moy_et(rs, k)[0] for rs in rs_par_x])
    s = np.array([moy_et(rs, k)[1] for rs in rs_par_x])
    ax.fill_between(x, m - s, m + s, color=col, alpha=0.10, lw=0)
    ax.plot(x, m, ls=ls, lw=1.8, color=col, marker="o", ms=4,
            mfc=col if plein else "white", mec=col, label=nom, zorder=3)
    if etiquette and not np.isnan(m[-1]):
        ax._etiq = getattr(ax, "_etiq", []) + [(x[-1], m[-1], fr(m[-1], f), col)]
    return m


def etiquettes(ax, ecart=8.5):
    """Valeurs en bout de courbe, écartées verticalement pour ne pas se chevaucher (en points)."""
    items = getattr(ax, "_etiq", [])
    if not items:
        return
    ax.figure.canvas.draw()
    tr = ax.transData
    pts = sorted(((tr.transform((x, y))[1] * 72 / ax.figure.dpi, x, y, t, c) for x, y, t, c in items))
    pos = []
    for yp, *_ in pts:
        pos.append(max(yp, pos[-1] + ecart) if pos else yp)
    exces = 0
    for (yp, x, y, t, c), p in zip(pts, pos):
        ax.annotate(t, (x, y), xytext=(5, p - yp - exces), textcoords="offset points", fontsize=7,
                    color=c, va="center")
    ax._etiq = []


def legende_configs(fig, cles=CONFIGS, y=1.02, ncol=4):
    h = [Line2D([], [], color=STYLE[c][1], ls=STYLE[c][2], lw=1.8, marker="o", ms=4,
                mfc=STYLE[c][1] if STYLE[c][3] else "white", mec=STYLE[c][1]) for c in cles]
    fig.legend(h, [STYLE[c][0] for c in cles], loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol,
               fontsize=7.5)


def titre(ax, t, sous=None):
    ax.set_title(t, fontsize=9, loc="left", color=INK)
    if sous:
        ax.text(0, 1.0, sous, transform=ax.transAxes, fontsize=6.8, color=MUTED, va="bottom")


# ------------------------------------------------------------------------------------ surcharge
S = lire("surcharge")
if veut("surcharge") and S:
    niv = sorted({r["taux"] for r in S["plans"]})
    x = np.array([round(100 * t) for t in niv])
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.0))
    # plan : dépend de la chaîne seulement
    for ax, (k, t, f) in zip(axs[0, :2], [("sans_date", "Patients sans date (sur 6 mois)", "{:.0f}"),
                                          ("delai_p90", "Délai pour 90 % des patients (j)", "{:.0f}")]):
        for c in ("reference", "hybride"):
            courbe(ax, x, [[r for r in S["plans"] if r["taux"] == t and r["chaine"] == c] for t in niv], k, c, f)
        titre(ax, t)
    ax = axs[0, 2]
    for c in ("reference", "hybride"):
        courbe(ax, x, [[r for r in S["plans"] if r["taux"] == t and r["chaine"] == c] for t in niv],
               "patients_semaine", c, "{:.0f}")
    titre(ax, "Programmés opérés / semaine")
    # jour J : quatre configurations
    J = S["jour"]
    for ax, (k, t, f) in zip(axs[1], [("lits_crit1", "Variance des lits (nuits de semaine)", "{:.1f}"),
                                      ("urg_delai", "Urgences opérées dans leur délai (%)", "{:.0f}"),
                                      ("depassement_jour", "Dépassement du bloc (min / jour)", "{:.1f}")]):
        for cle in CONFIGS:
            courbe(ax, x, [[r for r in J if r["taux"] == t and (r["chaine"], r["politique"]) == cle] for t in niv],
                   k, cle, f)
        titre(ax, t)
    for ax in axs.flat:
        ax.set_xticks(x, [f"+{v} %" for v in x])
        ax.set_xlim(x[0] - 2, x[-1] + 6)
    for ax in axs[1]:
        ax.set_xlabel("patients programmés en plus", fontsize=7.5, color=MUTED)
    h = [Line2D([], [], color=CHAINES[c][1], lw=1.8, marker="o", ms=4) for c in CHAINES]
    axs[0, 0].legend(h, [CHAINES[c][0] for c in CHAINES], fontsize=7, loc="upper left")
    fig.tight_layout(h_pad=1.6)
    legende_configs(fig, y=-0.04)
    sauver(fig, "surcharge")

# ---------------------------------------------------------------------------------------- aléas
A = lire("aleas")
if A and veut("aleas_global"):
    ks = A["k_global"]
    J = [r for r in A["jour"] if r["etiquette"].startswith("global:")]
    par = lambda cle: [[r for r in J if r["etiquette"] == f"global:{k}" and (r["chaine"], r["politique"]) == cle]
                       for k in ks]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.0))
    for ax, (k, t, f) in zip(axs.flat, [
            ("urg_delai", "Urgences opérées dans leur délai (%)", "{:.0f}"),
            ("attente", "Attente des urgences du jour (min)", "{:.0f}"),
            ("depassement_jour", "Dépassement du bloc (min / jour)", "{:.1f}"),
            ("reportes", "Patients programmés reportés", "{:.0f}"),
            ("conversions", "Ambulatoires gardés la nuit", "{:.0f}"),
            ("lits_crit1", "Variance des lits (nuits de semaine)", "{:.1f}")]):
        for cle in CONFIGS:
            courbe(ax, np.array(ks), par(cle), k, cle, f)
        titre(ax, t)
        ax.set_xticks(ks, [f"×{fr(v, '{:g}')}" for v in ks])
        ax.set_xlim(-0.15, ks[-1] + 0.45)
    fig.tight_layout(h_pad=1.6)
    fig.text(0.5, -0.005, "intensité de tous les aléas du jour : urgences, annulations, chirurgiens absents, pannes, "
             "lits fermés (×1 = niveau de base, ×0 = journée sans aléa)", ha="center", fontsize=7.5, color=MUTED)
    legende_configs(fig, y=-0.075)
    sauver(fig, "aleas_global")

if A and veut("aleas_types"):
    # un aléa à la fois ; ligne du haut : l'effet sur les urgences ; ligne du bas : sur les programmés
    TY = [("annulations", "Annulations du jour", lambda v: f"{fr(100 * v, '{:.0f}')} %", "part des patients du jour"),
          ("absences", "Chirurgien absent", lambda v: f"{fr(100 * v, '{:.0f}')} %",
           "par chirurgien et par jour"),
          ("pannes", "Panne de salle (1 à 3 h)", lambda v: f"{fr(100 * v, '{:.0f}')} %", "jours ouvrés touchés"),
          ("urgences", "Afflux d'urgences", lambda v: f"×{fr(v, '{:g}')}", "débit d'urgences (×1 = 2022)")]
    LIG = [("urg_delai", "Urgences dans le délai (%)", "{:.0f}"),
           ("depassement_jour", "Dépassement (min / j)", "{:.1f}"),
           ("reportes", "Programmés reportés", "{:.0f}")]
    fig, axs = plt.subplots(len(LIG), len(TY), figsize=(9.6, 6.4))
    for j, (typ, nom, fmt, unite) in enumerate(TY):
        vals = A["par_type"][typ][1]
        J = [r for r in A["jour"] if r["etiquette"].startswith(typ + ":")]
        x = np.arange(len(vals))
        for i, (k, t, f) in enumerate(LIG):
            ax = axs[i, j]
            for cle in CONFIGS:
                courbe(ax, x, [[r for r in J if r["etiquette"] == f"{typ}:{v}" and (r["chaine"], r["politique"]) == cle]
                               for v in vals], k, cle, f, etiquette=False)
            ax.set_xticks(x, [fmt(v) for v in vals], fontsize=7)
            if j == 0:
                ax.set_ylabel(t, fontsize=8)
            if i == len(LIG) - 1:
                ax.set_xlabel(unite, fontsize=7, color=MUTED)
        axs[0, j].set_title(nom, fontsize=9, loc="left", color=INK)
    for i in range(len(LIG) - 1):            # mêmes échelles sur une ligne (sauf les reports)
        lo = min(a.get_ylim()[0] for a in axs[i]); hi = max(a.get_ylim()[1] for a in axs[i])
        for a in axs[i]:
            a.set_ylim(lo, hi)
        for a in axs[i, 1:]:
            a.tick_params(labelleft=False)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    legende_configs(fig, y=-0.035)
    sauver(fig, "aleas_types")

# -------------------------------------------------------------------------------------- échelle
E = lire("echelle")
if E and veut("echelle"):
    P, J = E["plans"], E["jour"]
    ech = tuple(e for e in ("grande", "petite", "petite4") if any(r["echelle"] == e for r in P))
    noms_e = {"grande": "grande\n18 chir. · 42 lits",
              "petite": "petite\n9 chir. · 21 lits\nhybride : 2 salles",
              "petite4": "petite\n9 chir. · 21 lits\nhybride : 4 salles"}
    IND = [  # (source, clé, titre, format, transformation)
        ("plan", "delai_median", "Délai médian (j)", "{:.0f}"),
        ("plan", "delai_p90", "Délai pour 90 % (j)", "{:.0f}"),
        ("jour", "cv_lits", "Irrégularité des lits\n(écart-type / moyenne, %)", "{:.0f}"),
        ("jour", "urg_delai", "Urgences dans le délai (%)", "{:.0f}"),
        ("jour", "depassement_vac", "Dépassement par vacation (min)", "{:.1f}"),
        ("jour", "reportes_pct", "Programmés reportés (%)", "{:.1f}")]
    for r in P:
        r["sans_date_pct"] = 100 * r["sans_date"] / r["patients_test"]
    for r in J:
        r["cv_lits"] = 100 * np.sqrt(r["lits_crit1"]) / max(1e-9, r["lits_moy"])
    fig, axs = plt.subplots(2, 3, figsize=(9.4, 5.6))
    for ax, (src, k, t, f) in zip(axs.flat, IND):
        cles = list(CHAINES) if src == "plan" else CONFIGS
        w = 0.8 / len(cles)
        for i, cle in enumerate(cles):
            if src == "plan":
                nom, col = CHAINES[cle]; hatch = None; fc = col
                rs = [[r for r in P if r["echelle"] == e and r["chaine"] == cle] for e in ech]
            else:
                nom, col, ls, plein = STYLE[cle]
                fc = col if plein else "white"; hatch = None if plein else "////"
                rs = [[r for r in J if r["echelle"] == e and (r["chaine"], r["politique"]) == cle] for e in ech]
            m = [moy_et(x, k)[0] for x in rs]
            s = [moy_et(x, k)[1] for x in rs]
            xs = np.arange(len(ech)) + (i - (len(cles) - 1) / 2) * w
            ax.bar(xs, m, w * 0.92, color=fc, edgecolor=col, hatch=hatch, lw=1.0, yerr=s,
                   error_kw=dict(ecolor=MUTED, lw=0.7, capsize=1.5), zorder=3)
            for xi, mi, si in zip(xs, m, s):
                ax.text(xi, mi + si, fr(mi, f), ha="center", va="bottom", fontsize=5.5, color=INK)
        ax.set_xticks(range(len(ech)), [noms_e[e] for e in ech], fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        titre(ax, t)
    from matplotlib.patches import Patch
    h = [Patch(facecolor=CHAINES[c][1], edgecolor=CHAINES[c][1]) for c in CHAINES]
    axs[0, 0].legend(h, [CHAINES[c][0] for c in CHAINES], fontsize=7, loc="upper left")
    fig.tight_layout(h_pad=1.6)
    h = [Patch(facecolor=STYLE[c][1] if STYLE[c][3] else "white", edgecolor=STYLE[c][1],
               hatch=None if STYLE[c][3] else "////") for c in CONFIGS]
    fig.legend(h, [STYLE[c][0] for c in CONFIGS], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=4,
               fontsize=7.5)
    sauver(fig, "echelle")

# ---------------------------------------------------------------------------------- Monte-Carlo
M = lire("mc")
if M and veut("mc_distributions"):
    R = M["runs"]
    IND = [("lits_crit1", "Variance des lits (nuits de semaine)", "{:.1f}", "bas"),
           ("urg_delai", "Urgences dans le délai (%)", "{:.0f}", "haut"),
           ("attente", "Attente des urgences du jour (min)", "{:.0f}", "bas"),
           ("depassement_jour", "Dépassement (min / jour)", "{:.1f}", "bas"),
           ("reportes", "Programmés reportés", "{:.0f}", "bas"),
           ("conversions", "Ambulatoires gardés la nuit", "{:.0f}", "bas")]
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.2))
    for ax, (k, t, f, sens) in zip(axs.flat, IND):
        data = [[r[k] for r in R if (r["chaine"], r["politique"]) == c] for c in CONFIGS]
        pos = np.arange(len(CONFIGS))
        vp = ax.violinplot(data, pos, widths=0.8, showextrema=False)
        for b, c in zip(vp["bodies"], CONFIGS):
            b.set_facecolor(STYLE[c][1]); b.set_alpha(0.22); b.set_edgecolor("none")
        for p, d, c in zip(pos, data, CONFIGS):
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            lo, hi = np.percentile(d, [2.5, 97.5])
            ax.plot([p, p], [lo, hi], color=STYLE[c][1], lw=0.9)
            ax.plot([p, p], [q1, q3], color=STYLE[c][1], lw=4, solid_capstyle="butt")
            ax.plot(p, med, "o", ms=4, mfc="white", mec=STYLE[c][1], zorder=4)
            ax.text(p + 0.12, med, fr(med, f), fontsize=6.5, color=INK, va="center")
        ax.set_xticks(pos, ["réf.\nsans", "réf.\navec", "hybr.\nsans", "hybr.\navec"], fontsize=7)
        ax.grid(axis="x", visible=False)
        titre(ax, t)
    fig.tight_layout(h_pad=1.6)
    fig.text(0.5, -0.02, f"{M['n']} tirages Monte-Carlo : ordre des consultations, durées réelles, séjours, aléas "
             "du jour. Point : médiane ; trait épais : 50 % des tirages ; trait fin : 95 %.",
             ha="center", fontsize=7, color=MUTED)
    sauver(fig, "mc_distributions")

if M and veut("mc_convergence"):
    R = M["runs"]
    g = sorted({r["graine"] for r in R})
    fig, axs = plt.subplots(1, 3, figsize=(8.6, 2.7))
    for ax, (k, t, f) in zip(axs, [("lits_crit1", "Variance des lits — jour J", "{:.2f}"),
                                   ("urg_delai", "Urgences dans le délai (%)", "{:.1f}"),
                                   ("depassement_jour", "Dépassement (min / jour)", "{:.2f}")]):
        for c in CONFIGS:
            v = np.array([next(r[k] for r in R if r["graine"] == x and (r["chaine"], r["politique"]) == c)
                          for x in g], float)
            n = np.arange(1, len(v) + 1)
            mu = np.cumsum(v) / n
            sd = np.array([v[:i].std(ddof=1) if i > 1 else np.nan for i in n])
            ic = 1.96 * sd / np.sqrt(n)
            col, ls = STYLE[c][1], STYLE[c][2]
            ax.fill_between(n, mu - ic, mu + ic, color=col, alpha=0.12, lw=0)
            ax.plot(n, mu, color=col, ls=ls, lw=1.5)
            ax._etiq = getattr(ax, "_etiq", []) + [(n[-1], mu[-1], f"{fr(mu[-1], f)} ± {fr(ic[-1], f)}", col)]
        ax.set_xlim(1, len(g) * 1.42)
        ax.set_xlabel("nombre de tirages", fontsize=7.5, color=MUTED)
        titre(ax, t)
    fig.tight_layout()
    legende_configs(fig, y=-0.12)
    sauver(fig, "mc_convergence")
