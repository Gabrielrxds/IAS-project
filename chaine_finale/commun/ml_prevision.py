r"""
ml_prevision.py — Prévoir, à la consultation, la durée opératoire et le séjour, et brancher ces
prévisions dans la chaîne (rapport V3).

Deux façons de prévoir, pour chaque patient de 2022, ce que la planification utilise :
  « médiane » (sans apprentissage) : durée = médiane du TROS du même acte (et chirurgien) sur
      2019-2021, marge = P90 − médiane (estimateur du protocole) ; nuits et nuits avant
      l'opération = médianes de l'acte ;
  « ML » : gradient boosting appris sur 2019-2021 (confinement exclu) avec les seules variables
      connues à la consultation : âge, sexe, diagnostic principal (code et chapitre), acte prévu,
      nombre d'actes et de diagnostics associés, chirurgien. Durée : perte absolue (médiane) et
      perte quantile 0,9 (marge) ; nuits : classification 0..7+ (espérance de la loi, arrondie) ;
      entrée la veille : classification.
Le planning est construit avec les prévisions ; les lits et les admissions sont MESURÉS avec les
séjours réels (lits_reels), et le jour J se déroule avec les durées réelles.

Sorties : resultats/ml_prevision.json (qualité des prévisions, importance des variables,
analyses de la base) ; cache/ml_prevision.pkl (prévisions par patient).
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import copy
import dataclasses
import json
import pickle

import numpy as np
import pandas as pd

from protocole import RACINE, CACHE, charger, mesurer_profils

VARS = ["age", "sexe", "diag", "diag3", "acte", "nb_actes", "nb_assoc", "chir"]
CATS = ["sexe", "diag", "diag3", "acte", "chir"]
FICHIER = CACHE / "ml_prevision.pkl"


def variables(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = (pd.to_datetime(df["Date Inter"]) - pd.to_datetime(df["Date Naissance"])).dt.days / 365.25
    X["sexe"] = df["Sexe"].astype(str)
    X["diag"] = df["CIM Diag Pr."].astype(str)
    X["diag3"] = X["diag"].str[:3]
    X["acte"] = df["CCAM 1"].astype(str)
    X["nb_actes"] = df[["CCAM 1", "CCAM 2", "CCAM 3", "CCAM 4"]].notna().sum(axis=1)
    X["nb_assoc"] = df[[f"Cim Assoc {i}" for i in range(1, 6)]].notna().sum(axis=1)
    X["chir"] = df["Praticien"].astype(str)
    return X


def _encoder(Xtr, Xte):
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1,
                         min_frequency=3, max_categories=250)
    A, B = Xtr.copy(), Xte.copy()
    A[CATS] = enc.fit_transform(Xtr[CATS])
    B[CATS] = enc.transform(Xte[CATS])
    return A[VARS].astype(float), B[VARS].astype(float)


def apprendre(df=None, don=None):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    if df is None:
        df, don = charger()
    d = df[df["Date Inter"].notna() & df["nuits"].notna() & (df["nuits"] >= 0)].copy()
    di = pd.to_datetime(d["Date Inter"])
    covid = (di >= "2020-03-15") & (di <= "2020-06-30")
    tr = d[d["annee"].isin([2019, 2020, 2021]) & ~covid]
    te = d[d["annee"] == 2022]
    Xtr, Xte = _encoder(variables(tr), variables(te))
    masque = [v in CATS for v in VARS]
    kw = dict(categorical_features=masque, max_iter=300, learning_rate=0.06, random_state=0)
    ytr = tr["tros"].clip(5, 600).values
    op50 = HistGradientBoostingRegressor(loss="absolute_error", **kw).fit(Xtr, ytr)
    op90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **kw).fit(Xtr, ytr)
    ntr = tr["nuits"].clip(0, 7).astype(int).values
    nuits = HistGradientBoostingClassifier(**kw).fit(Xtr, ntr)
    avant = HistGradientBoostingClassifier(**kw).fit(Xtr, (tr["nuits_avant"] > 0).astype(int).values)

    pred = pd.DataFrame(index=te.index)
    pred["duree_ml"] = np.clip(op50.predict(Xte), 5, None)
    pred["marge_ml"] = np.clip(op90.predict(Xte) - pred["duree_ml"], 5, None)
    pred["nuits_ml_mode"] = nuits.predict(Xte)
    proba = nuits.predict_proba(Xte)
    pred["nuits_ml_esp"] = proba @ nuits.classes_.astype(float)          # espérance (0..7)
    pred["nuits_ml"] = np.rint(pred["nuits_ml_esp"]).astype(int)
    pred["avant_ml"] = avant.predict(Xte)
    # médianes de l'acte (type d'intervention), sans apprentissage
    med_n = tr.groupby("type")["nuits"].median()
    med_a = tr.groupby("type")["nuits_avant"].median()
    pred["nuits_med"] = te["type"].map(med_n).fillna(tr["nuits"].median()).round().astype(int)
    pred["avant_med"] = te["type"].map(med_a).fillna(0).round().astype(int)

    # ---- qualité des prévisions (test 2022)
    y = te["tros"].values.astype(float)
    est = np.array([don.patients[p].duree_op if p in don.patients else
                    (don.urgents[p].duree_op if p in getattr(don, "urgents", {}) else np.nan) for p in te.index])
    ok = ~np.isnan(est)

    def bien(prev):
        tol = np.maximum(15, 0.15 * prev[ok])
        return float((np.abs(y[ok] - prev[ok]) <= tol).mean() * 100)

    def mae(prev):
        return float(np.mean(np.abs(y[ok] - prev[ok])))

    nreel = te["nuits"].clip(0, 7).astype(int).values
    Q = dict(
        duree=dict(mediane=dict(bien=bien(est), mae=mae(est)),
                   ml=dict(bien=bien(pred["duree_ml"].values), mae=mae(pred["duree_ml"].values)),
                   couverture_p90_ml=float((y <= pred["duree_ml"].values + pred["marge_ml"].values).mean() * 100)),
        nuits=dict(mediane=dict(exact=float((pred["nuits_med"].clip(0, 7).values == nreel).mean() * 100),
                                ambu=float(((pred["nuits_med"].values == 0) == (nreel == 0)).mean() * 100),
                                mae=float(np.mean(np.abs(pred["nuits_med"].values - te["nuits"].values)))),
                   ml=dict(exact=float((pred["nuits_ml"].values == nreel).mean() * 100),
                           ambu=float(((pred["nuits_ml"].values == 0) == (nreel == 0)).mean() * 100),
                           mae=float(np.mean(np.abs(pred["nuits_ml"].values - te["nuits"].values))))),
    )
    # importance des variables (permutation, test 2022)
    imp = {}
    for nom, mod, yy, sc in (("duree", op50, te["tros"].clip(5, 600).values, "neg_mean_absolute_error"),
                             ("nuits", nuits, nreel, "accuracy")):
        r = permutation_importance(mod, Xte, yy, scoring=sc, n_repeats=5, random_state=0)
        imp[nom] = {v: float(m) for v, m in zip(VARS, r.importances_mean)}

    out = dict(qualite=Q, importance=imp, n_app=int(len(tr)), n_test=int(len(te)))
    FICHIER.write_bytes(pickle.dumps(pred))
    (RACINE / "resultats" / "ml_prevision.json").write_text(json.dumps(out, indent=1, default=float))
    return pred, out


def predictions():
    if not FICHIER.exists():
        apprendre()
    return pickle.loads(FICHIER.read_bytes())


def don_prevu(don, mode: str):
    """Copie de l'étude où les patients programmés portent les PRÉVISIONS de la consultation.
    mode = "connu" (séjour réel connu, hypothèse des autres résultats), "mediane" ou "ml"."""
    if mode == "connu":
        return don
    P = predictions()
    d = copy.copy(don)
    d.patients = {}
    for pid, p in don.patients.items():
        if pid not in P.index:
            d.patients[pid] = p
            continue
        r = P.loc[pid]
        col = {"ml": "nuits_ml", "ml_mode": "nuits_ml_mode", "ml_duree": "nuits_med", "ml_nuits": "nuits_ml"}
        n = int(r[col.get(mode, "nuits_med")])
        mode_a = "ml" if mode in ("ml", "ml_mode", "ml_nuits") else "med"
        a = int(min(r["avant_ml"] if mode_a == "ml" else r["avant_med"], n))
        kw = dict(duree_sejour=n + 1, nuits_avant=a, ambulatoire=None)
        if mode in ("ml", "ml_mode", "ml_duree"):
            kw.update(duree_op=int(round(r["duree_ml"])), marge_perso=int(round(r["marge_ml"])))
        d.patients[pid] = dataclasses.replace(p, **kw)
    return d


def lits_reels(don_reel, sol):
    """Lits par nuit et admissions ambulatoires par jour avec les SÉJOURS RÉELS des patients datés."""
    n = len(sol.lits_jour)
    lits, places = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    inst = sol.inst
    for pid, vid in sol.affectation.items():
        if vid is None or pid not in don_reel.patients:
            continue
        p, j = don_reel.patients[pid], inst.vacations[vid].jour
        if p.nb_nuits == 0:
            places[j] += 1
        else:
            d = j - p.nuits_avant
            lits[max(0, d):min(n, d + p.nb_nuits)] += 1
    return lits, places


def mesurer_sejours_reels(don_reel, sol) -> dict:
    lits, places = lits_reels(don_reel, sol)
    return mesurer_profils(don_reel, lits, places)


if __name__ == "__main__":
    _, out = apprendre()
    print(json.dumps(out["qualite"], indent=1))
    print({k: {v: round(x, 4) for v, x in d.items()} for k, d in out["importance"].items()})
