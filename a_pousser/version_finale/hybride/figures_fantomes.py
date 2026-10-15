"""Figures « patients fantômes » à partir de resultats/fantomes.json.  python3 figures_fantomes.py <dossier>"""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from chemins import RACINE
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
d = json.load(open(RACINE / "resultats" / "fantomes.json"))
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "axes.axisbelow": True})
SANS, AVEC, GH, REEL = "#9a9890", "#5B3FC4", "#c9bdf5", "#3D7DCA"
MOIS = ["janv.", "févr.", "mars", "avril", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
def sauver(f, n):
    f.savefig(OUT / f"{n}.pdf", bbox_inches="tight"); f.savefig(OUT / f"{n}.png", dpi=170, bbox_inches="tight"); plt.close(f)
def fr(date):
    x = dt.date.fromisoformat(date); return f"{x.day} {MOIS[x.month - 1]} {x.year}"

# 1. ce que voit la règle à une date donnée
C = d["C"]
for t in C["uniformes"]["snap"]:
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, mode, titre in ((axs[0], "aucun", "Sans fantômes"), (axs[1], "uniformes", "Avec fantômes")):
        s = C[mode]["snap"][t]
        E, G, F = np.array(s["E"]), np.array(s["G"]), np.array(s["final"])
        R = E - G; x = np.arange(1, len(E) + 1)
        ax.bar(x, R, width=1, color=SANS if mode == "aucun" else AVEC, alpha=.85, label="Lits des patients déjà datés")
        if mode != "aucun":
            ax.bar(x, G, bottom=R, width=1, color=GH, label="Lits des patients fantômes")
        ax.step(x, F, where="mid", color="#111", lw=1.4, label="Lits réellement occupés ensuite")
        ax.axvspan(0.5, 7.5, color="#f1efe8", zorder=0)
        ax.text(4, 20.5, "figé", ha="center", fontsize=8, color="#777")
        ax.set_title(titre, loc="left", fontweight="bold")
        ax.set_xlabel("jours après la consultation"); ax.set_xlim(.5, len(E) + .5); ax.set_ylim(0, 22)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    axs[0].set_ylabel("lits")
    fig.suptitle(f"Courbe des lits vue par la règle le {fr(s['date'])}", x=.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(); sauver(fig, f"courbe_attendue_{s['date']}")

# 2. qualité de la prévision selon l'horizon
fig, ax = plt.subplots(figsize=(6.5, 3.6))
H = (7, 14, 28, 42, 56)
for mode, col, lab in (("aucun", SANS, "Sans fantômes"), ("uniformes", AVEC, "Avec fantômes")):
    P = C[mode]["prev"]
    err = [np.mean([abs(p["attendu"] - p["final"]) for p in P if p["h"] == h]) for h in H]
    ax.plot([h / 7 for h in H], err, marker="o", lw=2.2, color=col, label=lab)
    for h, e in zip(H, err):
        ax.text(h / 7, e + (.3 if mode == "aucun" else -.6), f"{e:.1f}", ha="center", fontsize=8, color=col)
ax.set_xlabel("horizon (semaines)"); ax.set_ylabel("erreur moyenne (lits)")
ax.set_title("Erreur de la courbe attendue vs lits finaux", loc="left", fontweight="bold")
ax.legend(frameon=False); ax.set_ylim(0, 9)
sauver(fig, "erreur_prevision")

# 3. jour choisi
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
noms = ["Ambulatoire", "Séjour court\n(1-2 nuits)", "Séjour long\n(3+ nuits)"]
ax = axs[0]; w = .36
for i, (mode, col, lab) in enumerate((("aucun", SANS, "Sans fantômes"), ("uniformes", AVEC, "Avec fantômes"))):
    ch = C[mode]["choix"]
    v = [100 * np.mean([c[2] == 0 for c in ch if c[1] == k]) for k in range(3)]
    b = ax.bar(np.arange(3) + (i - .5) * w, v, w, color=col, label=lab)
    for x, y in zip(np.arange(3) + (i - .5) * w, v):
        ax.text(x, y + 1.5, f"{y:.0f} %", ha="center", fontsize=8)
ax.set_xticks(range(3), noms); ax.set_ylim(0, 112); ax.set_ylabel("% de patients")
ax.set_title("Patients opérés à la première date possible", loc="left", fontweight="bold"); ax.legend(frameon=False, loc="upper right")
ax = axs[1]
bins = np.arange(-.5, 15.5, 1)
for mode, col, lab in (("aucun", SANS, "Sans fantômes"), ("uniformes", AVEC, "Avec fantômes")):
    off = [c[2] for c in C[mode]["choix"] if c[1] > 0]
    h, _ = np.histogram(off, bins=bins)
    ax.plot(range(15), 100 * h / len(off), marker="o", ms=4, lw=2, color=col, label=f"{lab} (moyenne {np.mean(off):.1f} j)")
ax.set_xlabel("jours après la première date possible"); ax.set_ylabel("% des hospitalisés")
ax.set_title("Report choisi pour les hospitalisés", loc="left", fontweight="bold"); ax.legend(frameon=False)
fig.tight_layout(); sauver(fig, "jour_choisi")

# 4. résultats, deux grilles
A = {(a["grille"], a["fantomes"]): a for a in d["A"]}
crit = [("lits_variance", "Variance des lits"), ("lits_max", "Pic de lits"), ("lits_p95", "Lits pour 95 % des nuits"),
        ("places_ecart_type", "Écart-type admissions / jour"), ("delai_median", "Délai médian (j)"), ("delai_p90", "Délai pour 90 % (j)")]
fig, axs = plt.subplots(1, len(crit), figsize=(15, 3.4))
for ax, (k, t) in zip(axs, crit):
    for i, g in enumerate(("actuelle", "hybride")):
        for j, (mode, col) in enumerate((("aucun", SANS), ("uniformes", AVEC))):
            a = A[(g, mode)]; x = i + (j - .5) * .38
            ax.bar(x, a[k], .36, yerr=a[k + "_sd"], color=col, capsize=2, error_kw=dict(lw=.8))
            ax.text(x, a[k] + a[k + "_sd"] + .01 * max(1, a[k]), f"{a[k]:.1f}", ha="center", fontsize=7)
    ax.set_xticks([0, 1], ["Grille\nactuelle", "Grille\nhybride"], fontsize=8); ax.set_title(t, loc="left", fontsize=9.5, fontweight="bold")
    ax.grid(axis="x", visible=False)
fig.legend(handles=[Patch(fc=SANS, label="Sans fantômes"), Patch(fc=AVEC, label="Avec fantômes")], loc="lower center",
           ncol=2, frameon=False, bbox_to_anchor=(.5, -.06))
fig.tight_layout(); sauver(fig, "resultats_grilles")

# 5. fenêtre Δ
B = d["B"]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (k, t) in zip(axs, (("lits_variance", "Variance des lits"), ("lits_max", "Pic de lits"), ("delai_median", "Délai médian (j)"))):
    for mode, col, lab in (("aucun", SANS, "Sans fantômes"), ("uniformes", AVEC, "Avec fantômes")):
        xs = [b["delta"] for b in B if b["fantomes"] == mode]; ys = [b[k] for b in B if b["fantomes"] == mode]
        ax.plot(xs, ys, marker="o", lw=2.2, color=col, label=lab)
    ax.axvline(14, color="#d0342c", ls=":", lw=1); ax.set_xlabel("fenêtre Δ (jours)"); ax.set_title(t, loc="left", fontweight="bold")
axs[0].legend(frameon=False); axs[0].text(14.4, axs[0].get_ylim()[1] * .97, "Δ retenu", color="#d0342c", fontsize=8, va="top")
fig.tight_layout(); sauver(fig, "fenetre_delta")

# 6. hebdomadaire
dates = [dt.date.fromisoformat(x) for x in C["aucun"]["dates"]]
def hebdo(L):
    sem = {}
    for x, v in zip(dates, L):
        sem.setdefault(x - dt.timedelta(days=x.weekday()), []).append(v)
    k = sorted(s for s in sem if len(sem[s]) == 7)
    return k, np.array([np.mean(sem[s]) for s in k]), np.array([max(sem[s]) for s in k]), np.array([min(sem[s]) for s in k])
fig, ax = plt.subplots(figsize=(12, 3.8))
for mode, col, lab in (("aucun", SANS, "Sans fantômes"), ("uniformes", AVEC, "Avec fantômes")):
    k, m, M, mi = hebdo(C[mode]["lits"])
    ax.fill_between(k, mi, M, color=col, alpha=.18, lw=0)
    ax.plot(k, m, color=col, lw=2.2, marker="o", ms=3.5, label=f"{lab} (moyenne)")
ax.set_ylabel("lits occupés"); ax.set_title("Occupation hebdomadaire des lits — grille hybride, mai–déc. 2022", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper left", ncol=2)
ax.set_xticks([dt.date(2022, m, 1) for m in range(5, 13)], MOIS[4:])
sauver(fig, "hebdo_lits")
print("ok")
