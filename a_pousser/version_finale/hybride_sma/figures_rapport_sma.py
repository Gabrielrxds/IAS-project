"""Figures du rapport « hybride + SMA » à partir de resultats/sma.json.
    python3 figures_rapport_sma.py <dossier de sortie>"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
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
from matplotlib.patches import Patch, Rectangle

from chemins import RACINE

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else RACINE / "resultats" / "figures_sma")
OUT.mkdir(parents=True, exist_ok=True)
d = json.load(open(RACINE / "resultats" / "sma.json"))
R = d["runs"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6, "axes.axisbelow": True,
                     "font.family": "DejaVu Sans"})
GRIS, HYB, URG = "#5f5e5a", "#5b3fc4", "#d0342c"
CL = {"reference": "#b9b8b2", "hybride": "#b7a6f0"}           # statique : clair + hachures
CF = {"reference": GRIS, "hybride": HYB}                       # SMA : plein
NOM = {"reference": "Grille actuelle", "hybride": "Hybride"}
CONF = [("reference", "statique"), ("reference", "sma"), ("hybride", "statique"), ("hybride", "sma")]
LAB = {c: f"{NOM[c[0]]}\n{'statique' if c[1] == 'statique' else 'SMA'}" for c in CONF}
COURT = {c: f"{'Actuelle' if c[0] == 'reference' else 'Hybride'}\n{'statique' if c[1] == 'statique' else 'SMA'}" for c in CONF}


def couleur(ch, pol):
    return CF[ch] if pol == "sma" else CL[ch]


def hach(pol):
    return None if pol == "sma" else "////"


def runs(exp, ch, pol, urg=1.5):
    return [r for r in R if r["exp"] in exp and r["chaine"] == ch and r["politique"] == pol
            and r["graine"] is not None and r["urgences_par_jour"] == urg]


def pct(r, k):
    return 100 * r["resume"][k] / max(1, r["resume"]["urgences_arrivees"])


def sauver(fig, nom):
    fig.savefig(OUT / f"{nom}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nom}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. journée racontée
J = d["journee"]


def hh(t):
    return f"{int(t) // 60}h{int(t) % 60:02d}"


fig, axs = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
for ax, pol in zip(axs, ("statique", "sma")):
    D = J[pol]
    salles = sorted(D["salles"], key=lambda s: (s["urgence"], s["salle"], s["debut"]))
    for y, s in enumerate(salles):
        ax.add_patch(Rectangle((s["debut"], y - .42), s["fin"] - s["debut"], .84, fc="#f1efe8",
                               ec="#c9c6bc", lw=.8, zorder=1))
        for a in s["actes"]:
            c = URG if a["urg"] else couleur("hybride", "sma")
            ax.add_patch(Rectangle((a["d"], y - .3), a["f"] - a["d"], .6, fc=c, ec="white", lw=1.2,
                                   hatch="...." if a["ambu"] and not a["urg"] else None, zorder=3))
            if a["urg"]:
                ax.text((a["d"] + a["f"]) / 2, y, "URG", ha="center", va="center", color="white",
                        fontsize=7, fontweight="bold", zorder=4)
        fin = max((a["f"] for a in s["actes"]), default=s["fin"])
        if fin > s["fin"]:
            ax.annotate(f"+{fin - s['fin']} min", (fin, y + .42), fontsize=8, color=URG,
                        ha="left", va="bottom", fontweight="bold")
    k = 0
    for a in D["aleas"]:
        if a["type"] == "urgence":
            ax.axvline(a["t"], color=URG, lw=1, ls=":", zorder=2)
            ax.text(a["t"] + 3, len(salles) - .5 + .55 * (k % 2), f"urgence {hh(a['t'])}", color=URG, fontsize=7,
                    ha="left", va="bottom")
            k += 1
        elif a["type"] == "annulation":
            ax.axvline(a["t"], color="#888", lw=.8, ls="--", zorder=2)
    ax.set_yticks(range(len(salles)), [("URGENCES" if s["urgence"] else f"salle {s['salle']} · {s['chir']}")
                                       for s in salles])
    ax.set_ylim(-.6, len(salles) + .9)
    b = D["bilan"]
    titre = {"statique": "Politique statique : règle du modèle, urgence ajoutée en fin de programme",
             "sma": "SMA : appel d'offres aux salles sur leur état réel"}[pol]
    ax.set_title(f"{titre}  —  urgences opérées le jour même : {b['urgences_operees_jour_meme']}/"
                 f"{b['urgences_arrivees']}, reportées : {b['urgences_reportees']}, échecs : "
                 f"{b['urgences_echec']}, attente moyenne {b['attente_urgence_moy']:.0f} min",
                 fontsize=9.5, loc="left")
    ax.grid(axis="y", visible=False)
axs[1].set_xlim(7 * 60 + 30, 20 * 60)
axs[1].set_xticks(range(8 * 60, 20 * 60 + 1, 60), [f"{h}h" for h in range(8, 21)])
axs[1].legend(handles=[Patch(fc=couleur("hybride", "sma"), label="patient programmé (hospitalisé)"),
                       Patch(fc=couleur("hybride", "sma"), hatch="....", ec="white", label="patient programmé (ambulatoire)"),
                       Patch(fc=URG, label="urgence"), Patch(fc="#f1efe8", ec="#c9c6bc", label="vacation (TVO)"),
                       plt.Line2D([], [], color=URG, ls=":", label="arrivée d'une urgence"),
                       plt.Line2D([], [], color="#888", ls="--", label="annulation")],
              loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
sauver(fig, "journee")

# ---------------------------------------------------------------- 2. urgences
fig, axs = plt.subplots(1, 2, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axs[0]
for i, (ch, pol) in enumerate(CONF):
    rs = runs("A", ch, pol)
    jm = np.mean([pct(r, "urgences_operees_jour_meme") for r in rs])
    le = np.mean([pct(r, "urgences_operees_lendemain") for r in rs])
    ec = np.mean([pct(r, "urgences_echec") for r in rs])
    y = 3 - i
    ax.barh(y, jm, color=couleur(ch, pol), hatch=hach(pol), ec="white", lw=2)
    ax.barh(y, le, left=jm, color="#e9c46a", ec="white", lw=2)
    ax.barh(y, ec, left=jm + le, color=URG, ec="white", lw=2, alpha=.85)
    ax.text(jm / 2, y, f"{jm:.0f} %", ha="center", va="center", color="white" if pol == "sma" else "#222",
            fontweight="bold")
    ax.text(jm + le / 2, y, f"{le:.0f} %", ha="center", va="center", fontsize=8)
    ax.text(jm + le + ec / 2, y, f"{ec:.0f} %", ha="center", va="center", color="white", fontweight="bold")
ax.set_yticks([3, 2, 1, 0], [LAB[c].replace("\n", " · ") for c in CONF])
ax.set_xlim(0, 100); ax.set_xlabel("part des urgences arrivées (%)")
ax.set_title("Devenir des urgences (1,5 par jour, moyenne sur 20 tirages)", loc="left", fontsize=10)
ax.legend(handles=[Patch(fc="#999", label="opérées le jour même"), Patch(fc="#e9c46a", label="opérées le jour ouvré suivant"),
                   Patch(fc=URG, label="sans place (échec)")], loc="upper center", bbox_to_anchor=(.5, -.2),
          ncol=3, frameon=False)
ax.grid(axis="y", visible=False)
ax = axs[1]
for i, (ch, pol) in enumerate(CONF):
    v = [r["resume"]["attente_urgence_moy"] for r in runs("A", ch, pol)]
    ax.bar(i, np.mean(v), yerr=np.std(v), color=couleur(ch, pol), hatch=hach(pol), ec="white", lw=1, capsize=3,
           width=.7)
    ax.text(i, np.mean(v) + np.std(v) + 3, f"{np.mean(v):.0f}", ha="center", fontsize=8)
ax.set_xticks(range(4), [COURT[c] for c in CONF], fontsize=8)
ax.set_title("Attente d'une urgence opérée\nle jour même (min)", loc="left", fontsize=10)
ax.grid(axis="x", visible=False)
sauver(fig, "urgences")

# ---------------------------------------------------------------- 3. bloc
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5))
ax = axs[0]
for ch, pol in CONF:
    v = np.sort(np.concatenate([r["dep_par_jour"] for r in runs("A", ch, pol)]))
    y = 1 - np.arange(1, len(v) + 1) / len(v)
    ax.step(v, 100 * y, where="post", color=couleur(ch, pol) if pol == "sma" else CF[ch],
            ls="-" if pol == "sma" else "--", lw=2, label=LAB[(ch, pol)].replace("\n", " · "))
ax.set_xlim(0, 120); ax.set_ylim(0, 32)
ax.set_xlabel("dépassement du bloc dans la journée (min)"); ax.set_ylabel("% des journées au-delà")
ax.set_title("Journées qui débordent de plus de x min", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8)
ax = axs[1]
for i, (ch, pol) in enumerate(CONF):
    v = [r["resume"]["depassement_min"] / r["resume"]["jours"] for r in runs("A", ch, pol)]
    ax.bar(i, np.mean(v), yerr=np.std(v), color=couleur(ch, pol), hatch=hach(pol), ec="white", capsize=3, width=.7)
    ax.text(i, np.mean(v) + np.std(v) + .4, f"{np.mean(v):.1f}", ha="center", fontsize=8)
for ch, x0 in (("reference", 0), ("hybride", 2)):
    s = [r for r in R if r["exp"] == "A" and r["chaine"] == ch and r["graine"] is None and r["durees"] == "reelles"][0]
    v = s["resume"]["depassement_min"] / s["resume"]["jours"]
    ax.plot([x0 - .4, x0 + 1.4], [v, v], color="#222", lw=1, ls=":")
    ax.text(x0 + .5, v + .3, "sans aléa", fontsize=7, ha="center")
ax.set_xticks(range(4), [COURT[c] for c in CONF], fontsize=8)
ax.set_title("Dépassement moyen (min / jour)", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
ax = axs[2]
for i, (ch, pol) in enumerate(CONF):
    v = [100 * r["opere_min"] / r["tvo_utilise_min"] for r in runs("A", ch, pol)]
    ax.bar(i, np.mean(v), color=couleur(ch, pol), hatch=hach(pol), ec="white", width=.7)
    ax.text(i, np.mean(v) + .8, f"{np.mean(v):.1f} %", ha="center", fontsize=8)
ax.set_ylim(40, 70)
ax.set_xticks(range(4), [COURT[c] for c in CONF], fontsize=8)
ax.set_title("Temps opéré / TVO des vacations utilisées", loc="left", fontsize=10); ax.grid(axis="x", visible=False)
fig.tight_layout()
sauver(fig, "bloc")

# ---------------------------------------------------------------- 4. lits et places
import datetime as dt
fig, axs = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, sharey=True)
for ax, ch in zip(axs, ("reference", "hybride")):
    r = [x for x in runs("A", ch, "sma") if x["graine"] == 1][0]
    s = [x for x in runs("A", ch, "statique") if x["graine"] == 1][0]
    n = len(r["lits"]); jours = np.arange(n)
    ax.plot(jours, r["lits_plan"], color="#999", lw=1, label="plan du matin (sans aléa)")
    ax.plot(jours, s["lits"], color=CL[ch], lw=1.4, ls="--", label="après aléas · statique")
    ax.plot(jours, r["lits"], color=CF[ch], lw=1.4, label="après aléas · SMA")
    ax.axhline(15, color=URG, lw=.8, ls=":")
    ax.set_title(f"{NOM[ch]} — variance des lits après aléas : statique {np.var(s['lits']):.1f}, "
                 f"SMA {np.var(r['lits']):.1f} (plan : {np.var(r['lits_plan']):.1f})", loc="left", fontsize=10)
    ax.set_ylabel("lits occupés")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
mois = [(dt.date(2022, m, 1) - dt.date(2022, 5, 1)).days for m in range(5, 13)]
axs[1].set_xticks(mois, ["mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."])
sauver(fig, "lits")

fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.4))
specs = [("lits", "lits occupés par nuit", lambda r: r["lits"]),
         ("adm", "admissions ambulatoires par jour", lambda r: r["adm"]),
         ("pic", "pic de places simultanées par jour", lambda r: [b["pic_places"] for b in r["bilans"]])]
for ax, (k, titre, f) in zip(axs, specs):
    data, cols, hs = [], [], []
    for ch, pol in CONF:
        r = [x for x in runs("A", ch, pol) if x["graine"] == 1][0]
        data.append(f(r)); cols.append(couleur(ch, pol)); hs.append(hach(pol))
    bp = ax.boxplot(data, patch_artist=True, widths=.6, showfliers=False, medianprops=dict(color="#111"))
    for b, c, h in zip(bp["boxes"], cols, hs):
        b.set_facecolor(c); b.set_hatch(h); b.set_edgecolor("#444")
    if k == "pic":
        ax.axhline(12, color=URG, ls="--", lw=1); ax.text(4.4, 12.2, "capacité 12", color=URG, fontsize=7, ha="right")
    ax.set_xticks(range(1, 5), [COURT[c] for c in CONF], fontsize=8)
    ax.set_title(titre, loc="left", fontsize=10); ax.grid(axis="x", visible=False)
fig.tight_layout()
sauver(fig, "distributions")

# ---------------------------------------------------------------- 5. intensité
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.5))
U = (0.5, 1.0, 1.5, 2.0, 3.0)
mes = [("% d'urgences sans place", lambda r: pct(r, "urgences_echec")),
       ("% d'urgences opérées le jour même", lambda r: pct(r, "urgences_operees_jour_meme")),
       ("dépassement du bloc (min / jour)", lambda r: r["resume"]["depassement_min"] / r["resume"]["jours"])]
for ax, (titre, f) in zip(axs, mes):
    for ch, pol in CONF:
        m = [np.mean([f(r) for r in runs("AB", ch, pol, u)]) for u in U]
        ax.plot(U, m, marker="o", ms=5, lw=2, color=CF[ch], ls="-" if pol == "sma" else "--",
                mfc=CF[ch] if pol == "sma" else "white", label=LAB[(ch, pol)].replace("\n", " · "))
    ax.set_xlabel("urgences par jour (moyenne)"); ax.set_title(titre, loc="left", fontsize=10)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
sauver(fig, "intensite")

# ---------------------------------------------------------------- 6. réserves
fig, ax = plt.subplots(figsize=(8.5, 4.6))
RES = [("_r0_0_0", "0 %"), ("", "5 %"), ("_r10_3_3", "10 %"), ("_r15_4_4", "15 %")]
for ch in ("reference", "hybride"):
    for pol in ("statique", "sma"):
        xs, ys = [], []
        for suf, lab in RES:
            rs = [r for r in R if r["chaine"] == ch + suf and r["politique"] == pol and r["urgences_par_jour"] == 1.5
                  and r["graine"] in range(1, 11)]
            xs.append(d["plans"][ch + suf]["delai_median"]); ys.append(np.mean([pct(r, "urgences_echec") for r in rs]))
        ax.plot(xs, ys, marker="o", ms=7, lw=2, color=CF[ch], ls="-" if pol == "sma" else "--",
                mfc=CF[ch] if pol == "sma" else "white", label=LAB[(ch, pol)].replace("\n", " · "))
        for x, y, (suf, lab) in zip(xs, ys, RES):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color=CF[ch])
ax.set_xlabel("délai médian des patients programmés (jours, consultation → opération)")
ax.set_ylabel("% d'urgences sans place")
ax.set_title("Le prix des réserves : chaque point = tampon de TVO (et lits / admissions réservés)",
             loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8)
sauver(fig, "reserves")

# ---------------------------------------------------------------- 7. radar
CRIT = [("Urgences\nle jour même", lambda r: pct(r, "urgences_operees_jour_meme"), 1),
        ("Urgences\nsans place", lambda r: pct(r, "urgences_echec"), -1),
        ("Dépassement\ndu bloc", lambda r: r["resume"]["depassement_min"], -1),
        ("Lits\n(variance)", lambda r: np.var(r["lits"]), -1),
        ("Admissions\n(variance)", lambda r: np.var(r["adm"]), -1),
        ("Places\n(jours > 12)", lambda r: r["jours_surcap_places"], -1),
        ("Horaires tenus\n(décalés > 30 min)", lambda r: r["resume"]["patients_decales"], -1)]
val = {c: [np.mean([f(r) for r in runs("A", *c)]) * s for _, f, s in CRIT] for c in CONF}
V = np.array([val[c] for c in CONF])
lo, hi = V.min(0), V.max(0)
N = (V - lo) / np.where(hi > lo, hi - lo, 1) * .9 + .1
ang = np.linspace(0, 2 * np.pi, len(CRIT), endpoint=False).tolist()
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), subplot_kw=dict(polar=True))
fig.subplots_adjust(wspace=.55)
for ax, c, n in zip(axs, CONF, N):
    v = n.tolist() + [n[0]]
    ax.fill(ang + ang[:1], v, color=couleur(*c), alpha=.45 if c[1] == "sma" else .6, hatch=hach(c[1]))
    ax.plot(ang + ang[:1], v, color=CF[c[0]], lw=1.6)
    ax.set_xticks(ang, [k for k, _, _ in CRIT], fontsize=6.5)
    ax.set_yticks([]); ax.set_ylim(0, 1)
    ax.set_title(LAB[c].replace("\n", " · "), fontsize=9.5, color=CF[c[0]], pad=14)
fig.text(.5, -.02, "1 = la meilleure des quatre configurations sur ce critère ; 0,1 = la moins bonne",
         ha="center", fontsize=8, color="#555")
sauver(fig, "radar")
print("ok", OUT)
