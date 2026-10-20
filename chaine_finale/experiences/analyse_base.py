"""Analyses de la base de données (rapport V4, partie 4 et partie apprentissage).
    python3 analyse_base.py <dossier des figures>
Les chirurgiens sont anonymisés (C1, C2, ... par volume décroissant).
Sorties : figures donnees_lits, donnees_urgences, ml_durees, ml_resultats ; resultats/analyse_base.json"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from protocole import charger, RACINE, est_urgent_diag
import calendrier
import urgences_fssa as U

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RACINE / "resultats" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#5b616b", "#e6e8eb"
HYB, GRIS, ORA, VERT = "#5A3FC0", "#9aa0a8", "#E8663D", "#1BAF7A"
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "font.family": "DejaVu Sans"})


def sauver(fig, nom):
    fig.savefig(OUT / f"{nom}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{nom}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("figure", nom)


def titre(ax, t):
    ax.set_title(t, fontsize=10.5, loc="left", color=INK, pad=6)


df, don = charger()
d = df[df["Date Inter"].notna() & df["nuits"].notna() & (df["annee"] >= 2019)].copy()
d["di"] = pd.to_datetime(d["Date Inter"])
d["ambu"] = d["nuits"] == 0
d["urgent"] = est_urgent_diag(d["CIM Diag Pr."].astype("string").fillna("").tolist())
vol = d["Praticien"].value_counts()
code = {c: f"C{i + 1}" for i, c in enumerate(vol.index)}
d["chir"] = d["Praticien"].map(code)
S = {}

# lits par nuit, 2019-2022, séjours réels
deb, fin = dt.date(2019, 1, 1), dt.date(2022, 12, 31)
n = (fin - deb).days + 1
lits = np.zeros(n)
for _, r in d[~d["ambu"]].iterrows():
    e = (r["di"].date() - deb).days - int(r["nuits_avant"])
    lits[max(0, e):min(n, e + int(r["nuits"]))] += 1
jours = [deb + dt.timedelta(i) for i in range(n)]
L = pd.Series(lits, index=pd.to_datetime(jours))
fer, ponts = calendrier.feries_et_ponts(range(2018, 2024))
typ = pd.Series([calendrier.type_nuit(j, fer, ponts) for j in jours], index=L.index)

import matplotlib.dates as mdates
from scipy.stats import poisson

jr = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]
jc = ["L", "M", "M", "J", "V", "S", "D"]
VAC = [(a, b) for a, b in calendrier._VAC]
COVID = [(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-06-07")), (pd.Timestamp("2020-10-01"), pd.Timestamp("2021-06-06"))]


def hors_covid(idx):
    m = np.ones(len(idx), bool)
    for a, b in COVID:
        m &= ~((idx >= a) & (idx <= b))
    return m


# ---------------------------------------------------------------- figure 1 : les lits à lisser
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1.45, 1]))
ax = axs[0]
L22 = L["2022"]
for a, b in VAC:
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    if b.year == 2022 or a.year == 2022:
        ax.axvspan(max(a, pd.Timestamp("2022-01-01")), min(b, pd.Timestamp("2022-12-31")), color=GRID, lw=0, zorder=0)
ax.plot(L22.index, L22.values, color=HYB, lw=.8)
we = L22.index.dayofweek >= 5
ax.scatter(L22.index[we], L22[we], s=7, color=ORA, zorder=3, label="nuit de week-end")
ax.text(pd.Timestamp("2022-08-01"), 23, "vacances\nscolaires", ha="center", va="top", fontsize=8.5, color=MUTED)
ax.set_ylabel("lits occupés")
ax.set_ylim(0, 26)
ax.legend(fontsize=8.5, loc="upper left")
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 3, 5, 7, 9, 11]))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
ax.grid(axis="x", visible=False)
titre(ax, "Lits occupés chaque nuit, 2022")
ax = axs[1]
Lh = L[hors_covid(L.index)]
w = Lh.groupby(Lh.index.dayofweek)
m = w.mean()
ax.bar(range(7), m.values, color=[HYB] * 5 + [GRIS] * 2, width=.68)
for i, v in enumerate(m.values):
    ax.text(i, v + .3, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(range(7), jc)
ax.set_ylim(0, 17)
ax.grid(axis="x", visible=False)
titre(ax, "Lits selon la nuit, 2019–2022")
fig.tight_layout(w_pad=2)
sauver(fig, "donnees_lits")
S["semaine"] = {jr[i]: float(m.values[i]) for i in range(7)}
an = d.groupby("annee").agg(sejour=("nuits", lambda x: x[x > 0].mean()), ambu=("ambu", "mean"))
S["annees"] = {int(k): dict(sejour=float(v["sejour"]), ambu=float(v["ambu"])) for k, v in an.iterrows()}
S["lits_2022"] = dict(min=float(L22.min()), max=float(L22.max()), moy=float(L22.mean()))
vac = L22[(typ["2022"] == "vacances") & (L22.index.dayofweek < 5)].mean()
hv = L22[(typ["2022"] == "semaine") & (L22.index.dayofweek < 5)].mean()
S["vacances_2022"] = dict(vacances=float(vac), hors=float(hv))

# ---------------------------------------------------------------- figure 2 : les urgences
u = d[d["urgent"]].copy()
u["classe"] = [U.classer(x)["classe_urgence"] for x in u["CIM Diag Pr."].astype(str)]
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.4))
ax = axs[0, 0]
ws = u.set_index("di").resample("W-MON").size()
ok = hors_covid(ws.index)
for a, b in COVID:
    ax.axvspan(a, b, color=GRID, lw=0, zorder=0)
ax.plot(ws.index, ws.values, color=GRIS, lw=.7)
ax.plot(ws.index, ws.where(ok).rolling(13, center=True, min_periods=6).mean(), color=ORA, lw=2)
ax.text(pd.Timestamp("2020-12-15"), 15.2, "Covid\n(exclu)", ha="center", va="top", fontsize=8.5, color=MUTED)
ax.set_ylabel("urgences par semaine")
ax.set_ylim(0, 16)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(axis="x", visible=False)
titre(ax, "Un rythme stable : environ 4,5 par semaine")
ax = axs[0, 1]
wv = ws[ok]
lam = wv.mean()
k = np.arange(0, 14)
obs = np.array([(wv == i).mean() for i in k])
ax.bar(k, 100 * obs, color="#f0a487", width=.75, label="semaines observées")
ax.plot(k, 100 * poisson.pmf(k, lam), color=INK, lw=1.4, marker="o", ms=4, label=f"loi de Poisson")
ax.set_xlabel("urgences dans la semaine")
ax.set_xticks(range(0, 14, 2))
ax.set_ylabel("% des semaines")
ax.set_ylim(0, 26)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="x", visible=False)
titre(ax, "Le nombre suit une loi de Poisson")
ax = axs[1, 0]
u22 = u[u["annee"] == 2022]
jw = u22["di"].dt.dayofweek
v1 = np.array([((jw == k) & (u22["classe"] == "U1")).sum() / 52 for k in range(7)])
v2 = np.array([((jw == k) & (u22["classe"] == "U2")).sum() / 52 for k in range(7)])
ax.bar(range(7), v1, color=ORA, width=.68, label="U1 : moins de 24 h")
ax.bar(range(7), v2, bottom=v1, color="#f0a487", width=.68, label="U2 : moins de 72 h")
ax.set_xticks(range(7), jc)
ax.set_ylabel("urgences par jour")
ax.set_ylim(0, 1.75)
ax.legend(fontsize=8.5, loc="upper right")
ax.grid(axis="x", visible=False)
titre(ax, "Selon le jour, 2022")
ax = axs[1, 1]
fer, ponts = calendrier.feries_et_ponts(range(2018, 2024))


def statut(x):
    """jour ouvré normal, jour ouvré de vacances scolaires, férié ou pont, week-end"""
    x = x.date()
    if x.weekday() >= 5:
        return "we"
    if x in fer or x in ponts:
        return "ferie"
    return "vac" if any(a <= x <= b for a, b in calendrier._VAC) else "normal"


jours_o = pd.date_range("2019-01-01", "2022-12-31")
jours_o = jours_o[hors_covid(jours_o)]
nj = pd.Series([statut(x) for x in jours_o]).value_counts()
dd = d[hors_covid(pd.DatetimeIndex(d["di"]))].copy()
dd["st"] = [statut(x) for x in dd["di"]]
prog = {s_: (~dd["urgent"] & (dd["st"] == s_)).sum() / nj[s_] for s_ in ("normal", "vac")}
urg = {s_: (dd["urgent"] & (dd["st"] == s_)).sum() / nj[s_] for s_ in ("normal", "vac")}
rp, ru = prog["vac"] / prog["normal"], urg["vac"] / urg["normal"]
baisse = [100 * (1 - rp), 100 * (1 - ru)]
ax.bar([0, 1], baisse, color=[GRIS, ORA], width=.55)
fmt = lambda v: f"{v:.2f}".replace(".", ",") if v < 2 else f"{v:.1f}".replace(".", ",")
for xi, b, dic in zip([0, 1], baisse, (prog, urg)):
    ax.text(xi, b + 1.5, f"− {b:.0f} %", ha="center", fontsize=10, fontweight="bold")
    ax.text(xi, b + 7.5, f"{fmt(dic['normal'])} → {fmt(dic['vac'])} par jour", ha="center", fontsize=8.5, color=MUTED)
ax.set_xticks([0, 1], ["programmées", "urgences"])
ax.set_ylabel("baisse par jour ouvré (%)")
ax.set_ylim(0, 50)
ax.set_xlim(-.6, 1.6)
ax.grid(axis="x", visible=False)
titre(ax, "Baisse pendant les vacances scolaires")
fig.tight_layout(h_pad=1.5, w_pad=2)
sauver(fig, "donnees_urgences")
hs = (u.set_index("di").resample("W-MON")["tros"].sum() / 60)[ok]
S["urgences"] = dict(par_semaine=float(lam), var_sur_moy=float(wv.var() / lam), semaines=int(len(wv)),
                     heures_semaine=float(hs.mean()), heures_p90=float(hs.quantile(.9)),
                     par_annee={int(y): int((u["annee"] == y).sum()) for y in range(2019, 2023)},
                     ratio_vacances_prog=float(rp), ratio_vacances_urg=float(ru))
S["urgences_2022"] = dict(n=int(len(u22)), U1=int((u22["classe"] == "U1").sum()), U2=int((u22["classe"] == "U2").sum()),
                          weekend=float((jw >= 5).mean()), ambu=float(u22["ambu"].mean()))

# ---------------------------------------------------------------- figure 3 : les durées en salle
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
ax = axs[0, 0]
t = d["tros"].dropna()
ax.hist(t.clip(0, 300), bins=np.arange(0, 305, 10), color=HYB, alpha=.85)
med, p90 = t.median(), t.quantile(.9)
for v, lab in ((med, f"médiane {med:.0f} min"), (p90, f"9 sur 10 sous {p90:.0f} min")):
    ax.axvline(v, color=INK, lw=1, ls="--")
    ax.text(v + 4, ax.get_ylim()[1] * (.92 if v == med else .72), lab.replace(" sous", "\nsous"), fontsize=8.5, color=INK, va="top")
ax.set_xlabel("minutes en salle")
ax.set_ylabel("interventions")
ax.grid(axis="x", visible=False)
titre(ax, "Durée en salle, 2019–2022")
ax = axs[0, 1]
NOM = {"Varices": "varices", "Pied Simple": "pied simple", "Arthroscopie Genou": "arthroscopie genou",
       "Main Simple": "main simple", "Prothese Totale Genou": "prothèse de genou",
       "Arthroscopie Epaule Complexe": "arthroscopie épaule", "Prothese Totale Hanche": "prothèse de hanche",
       "Rachis Lombaire Simple": "rachis lombaire", "Canal Carpien": "canal carpien"}
ty = [k for k in NOM]
ty = sorted(ty, key=lambda k: d.loc[d["Interv Type"] == k, "tros"].median())
data = [d.loc[d["Interv Type"] == k, "tros"].dropna().clip(0, 250).values for k in ty]
bp = ax.boxplot(data, vert=False, tick_labels=[NOM[k] for k in ty], patch_artist=True, showfliers=False, widths=.6)
for b in bp["boxes"]:
    b.set(facecolor="#c9c3ea", edgecolor=MUTED)
for m_ in bp["medians"]:
    m_.set(color=HYB, lw=1.8)
ax.set_xlabel("minutes en salle")
ax.grid(axis="y", visible=False)
titre(ax, "Selon le type d'intervention")
ax = axs[1, 0]
top = d[d["Interv Type"] == "Arthroscopie Genou"]
cc = top["chir"].value_counts()
cc = cc[cc >= 30].index[:6]
cc = sorted(cc, key=lambda k: top.loc[top["chir"] == k, "tros"].median())
data = [top.loc[top["chir"] == k, "tros"].clip(0, 150).values for k in cc]
bp = ax.boxplot(data, tick_labels=list(cc), patch_artist=True, showfliers=False, widths=.55)
for b in bp["boxes"]:
    b.set(facecolor="#c9c3ea", edgecolor=MUTED)
for m_ in bp["medians"]:
    m_.set(color=HYB, lw=1.8)
ax.set_ylabel("minutes en salle")
ax.set_xlabel("chirurgien")
ax.grid(axis="x", visible=False)
titre(ax, "Arthroscopie du genou, selon le chirurgien")
ax = axs[1, 1]
t22 = d[d["annee"] == 2022]
est = np.array([don.patients[p].duree_op if p in don.patients else
                (don.urgents[p].duree_op if p in don.urgents else np.nan) for p in t22.index])
err_all = t22["tros"].values - est
okk = ~np.isnan(err_all)
err = err_all[okk]
tol = np.maximum(15, 0.15 * est[okk])
ax.hist(np.clip(err, -90, 90), bins=np.arange(-90, 95, 5), color=GRIS)
ax.axvspan(-15, 15, color=HYB, alpha=.15, lw=0)
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
ax.text(0, ax.get_ylim()[1] * .98, "± 15 min", ha="center", va="top", fontsize=8.5, color=HYB)
ax.set_xlabel("durée réelle − durée prévue (min)")
ax.set_ylabel("interventions de 2022")
ax.grid(axis="x", visible=False)
titre(ax, "Erreur de la médiane de l'acte")
fig.tight_layout(h_pad=1.5, w_pad=2)
sauver(fig, "ml_durees")
bien = float((np.abs(err) <= tol).mean() * 100)
S["duree_salle"] = dict(mediane=float(med), p90=float(p90))
S["duree_mediane"] = dict(bien=bien, mae=float(np.abs(err).mean()), sous_estime_30=float((err > 30).mean() * 100))
S["ml_genou"] = {k: float(np.median(x)) for k, x in zip(cc, data)}
S["types"] = {NOM[k]: float(d.loc[d["Interv Type"] == k, "tros"].median()) for k in ty}

# ---------------------------------------------------------------- figure 4 : ce que donne l'apprentissage
ML = json.loads((RACINE / "resultats" / "ml_prevision.json").read_text())
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]
lab = {"diag": "diagnostic", "chir": "chirurgien", "acte": "acte prévu", "diag3": "famille du diag.",
       "nb_actes": "nombre d'actes", "sexe": "sexe", "age": "âge", "nb_assoc": "diag. associés"}
imp = ML["importance"]["duree"]
o = [k for k in sorted(imp, key=imp.get) if imp[k] > 0.05]
ax.barh([lab[k] for k in o], [imp[k] for k in o], color=HYB, height=.65)
for i, k in enumerate(o):
    ax.text(imp[k] + .15, i, f"{imp[k]:.1f}".replace(".", ","), va="center", fontsize=9)
ax.set_xlabel("minutes d'erreur en plus\nsi la variable est brouillée")
ax.set_xlim(0, max(imp.values()) * 1.18)
ax.grid(axis="y", visible=False)
titre(ax, "Ce qui compte pour la durée")
ax = axs[1]
Q = ML["qualite"]
noms = ["durées\nbien prévues", "nuits\nexactes", "ambulatoire\nreconnu"]
v_med = [Q["duree"]["mediane"]["bien"], Q["nuits"]["mediane"]["exact"], Q["nuits"]["mediane"]["ambu"]]
v_ml = [Q["duree"]["ml"]["bien"], Q["nuits"]["ml"]["exact"], Q["nuits"]["ml"]["ambu"]]
x = np.arange(3)
ax.bar(x - .19, v_med, width=.36, color=GRIS, label="médiane de l'acte")
ax.bar(x + .19, v_ml, width=.36, color=HYB, label="apprentissage")
for xi, a, b in zip(x, v_med, v_ml):
    ax.text(xi - .19, a + 1.5, f"{a:.0f}", ha="center", fontsize=9)
    ax.text(xi + .19, b + 1.5, f"{b:.0f}", ha="center", fontsize=9)
ax.set_xticks(x, noms)
ax.set_ylim(0, 118)
ax.set_ylabel("% des patients de 2022")
ax.legend(fontsize=8.5, loc="upper left", ncol=2)
ax.grid(axis="x", visible=False)
titre(ax, "Qualité des prévisions, test 2022")
fig.tight_layout(w_pad=2)
sauver(fig, "ml_resultats")
(RACINE / "resultats" / "analyse_base.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
print(json.dumps(S, indent=1, ensure_ascii=False))
