r"""
prediction_sejour.py — Prédire, AU MOMENT DE LA CONSULTATION, la durée
d'hospitalisation d'un patient, sous la forme utile pour lisser les lits :
le nombre de lits qu'il occupera EN MOYENNE chaque nuit autour de son
opération.

POURQUOI UNE DISTRIBUTION ET PAS UN SEUL NOMBRE
-----------------------------------------------
Pour programmer, on additionne les patients nuit par nuit. Dire « 2 nuits »
pour un patient qui a 50 % de chances de sortir le jour même et 50 % de
rester 4 nuits fausserait le calcul des lits. Le modèle prédit donc des
PROBABILITÉS :
  p_veille        probabilité d'entrer la veille (1 nuit avant l'opération)
  p_post[k]       probabilité de rester exactement k nuits après (k = 0..7, 7 = 7 et +)
d'où le PROFIL DE LITS : lits attendus la nuit -1 (= p_veille), la nuit 0
(= P(au moins 1 nuit après)), la nuit 1 (= P(au moins 2 nuits))…
La somme des profils des patients d'un planning donne les lits attendus
chaque nuit : c'est exactement ce que le recuit simulé cherche à lisser.

VARIABLES : celles connues à la consultation (cf. preparation_donnees) —
le médecin a indiqué que le diagnostic, l'acte et l'âge sont les plus
déterminants ; on y ajoute le chirurgien, les comorbidités, l'historique du
patient et la classe d'urgence.
MODÈLE : HistGradientBoosting de scikit-learn (boosting d'arbres, cours 2-3).

    from prediction_sejour import PredicteurSejour
    p = PredicteurSejour().fit(base_apprentissage)
    pred = p.predire(nouveaux_patients)       # probabilités, espérance, médiane, P90
    profils = p.profil_lits(nouveaux_patients)  # lits attendus nuit -1, 0, 1, …
    p.sauver("modeles/predicteur_sejour.joblib")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VARIABLES_PRED = ["age", "sexe", "diag_categorie", "diag_principal", "acte_chirurgical", "acte_topographie",
                  "type_intervention_norm", "chirurgien", "classe_urgence", "nb_diag_associes",
                  "comorbidite_hypertension", "comorbidite_diabete", "comorbidite_obesite", "comorbidite_tabac",
                  "comorbidite_anticoagulant", "nb_sejours_anterieurs", "acte_sous_arthroscopie",
                  "nb_actes_chirurgicaux", "nb_actes_diagnostiques"]


class Encodeur:
    """Encodage appris sur la base d'apprentissage et réappliqué à l'identique
    aux nouveaux patients : médiane pour les nombres, modalités fréquentes +
    « autre » puis indicatrices (get_dummies) pour les catégories."""

    def __init__(self, variables, top: int = 40):
        self.variables, self.top = list(variables), top

    def fit(self, df):
        self.num, self.cat, self.med, self.modalites = [], [], {}, {}
        for v in self.variables:
            s = df[v]
            if s.dtype == bool or str(s.dtype) == "boolean":
                self.num.append(v)
                self.med[v] = 0
            elif pd.api.types.is_numeric_dtype(s):
                self.num.append(v)
                self.med[v] = float(s.median())
            else:
                self.cat.append(v)
                self.modalites[v] = list(s.astype("string").value_counts().index[:self.top])
        self.colonnes = list(self.transform(df.head(50)).columns)
        self.colonnes = list(dict.fromkeys(self.colonnes + [f"{v}_{m}" for v in self.cat
                                                             for m in self.modalites[v] + ["autre"]]))
        return self

    def transform(self, df):
        X = pd.DataFrame(index=df.index)
        for v in self.num:
            s = df[v] if v in df else pd.Series(np.nan, index=df.index)
            X[v] = pd.to_numeric(s.astype("float64") if s.dtype != object else s, errors="coerce").fillna(self.med[v])
        for v in self.cat:
            s = (df[v] if v in df else pd.Series(pd.NA, index=df.index)).astype("string")
            X[v] = s.where(s.isin(self.modalites[v]), "autre").fillna("autre")
        X = pd.get_dummies(X, columns=self.cat, dtype=np.int8)
        if hasattr(self, "colonnes"):
            X = X.reindex(columns=self.colonnes, fill_value=0)
        return X


class PredicteurSejour:

    def __init__(self, variables=None, nuits_max: int = 7, top: int = 40, random_state: int = 0,
                 max_iter: int = 300, learning_rate: float = 0.06):
        self.variables = list(variables or VARIABLES_PRED)
        self.nuits_max = nuits_max
        self.top = top
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, random_state=random_state,
                           l2_regularization=1.0, min_samples_leaf=30)

    # -- apprentissage -------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PredicteurSejour":
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.enc = Encodeur(self.variables, self.top).fit(df)
        X = self.enc.transform(df)
        post = df.nuits_apres_op.clip(upper=self.nuits_max).astype(int)
        self.clf_post = HistGradientBoostingClassifier(**self.params).fit(X, post)
        self.clf_veille = HistGradientBoostingClassifier(**self.params).fit(X, df.entree_veille.astype(int))
        # nuits moyennes au-delà de nuits_max (pour l'espérance et le profil)
        longs = df.nuits_apres_op[df.nuits_apres_op >= self.nuits_max]
        self.queue = [float((longs > s).mean()) for s in range(self.nuits_max, 31)] if len(longs) else [0.0]
        self.moyenne_longs = float(longs.mean()) if len(longs) else float(self.nuits_max)
        return self

    # -- prédiction ---------------------------------------------------------------------
    def probabilites(self, df):
        X = self.enc.transform(df)
        pp = np.zeros((len(df), self.nuits_max + 1))
        pp[:, self.clf_post.classes_] = self.clf_post.predict_proba(X)
        pv = self.clf_veille.predict_proba(X)[:, list(self.clf_veille.classes_).index(1)] \
            if 1 in self.clf_veille.classes_ else np.zeros(len(df))
        return pp, pv

    def predire(self, df) -> pd.DataFrame:
        pp, pv = self.probabilites(df)
        k = np.arange(self.nuits_max + 1, dtype=float)
        k[-1] = self.moyenne_longs
        esp_post = pp @ k
        cum = pp.cumsum(axis=1)
        med_post = (cum >= 0.5).argmax(axis=1)
        p90_post = (cum >= 0.9).argmax(axis=1)
        r = pd.DataFrame(pp, index=df.index, columns=[f"p_post_{i}" if i < self.nuits_max else f"p_post_{i}+"
                                                      for i in range(self.nuits_max + 1)])
        r.insert(0, "p_veille", pv)
        r.insert(1, "p_ambulatoire", (1 - pv) * pp[:, 0])
        r.insert(2, "nuits_esperees", pv + esp_post)
        r.insert(3, "nuits_apres_op_mediane", med_post)
        r.insert(4, "nuits_apres_op_p90", p90_post)
        r.insert(5, "nuits_totales_estimees", (pv > 0.5).astype(int) + med_post)   # valeur la plus probable (médiane)
        return r

    def profil_lits(self, df, horizon: int = 30) -> np.ndarray:
        """Matrice (patients x (1 + horizon)) : lits attendus la nuit -1
        (veille), puis les nuits 0, 1, … après l'opération."""
        pp, pv = self.probabilites(df)
        surv = 1 - pp.cumsum(axis=1)                    # P(post > s), s = 0..nuits_max
        prof = np.zeros((len(df), 1 + horizon))
        prof[:, 0] = pv
        for s in range(horizon):
            if s < self.nuits_max:
                prof[:, 1 + s] = surv[:, s]
            else:
                q = self.queue[s - self.nuits_max] if s - self.nuits_max < len(self.queue) else 0.0
                prof[:, 1 + s] = pp[:, -1] * q
        return prof

    # -- sauvegarde -----------------------------------------------------------------------
    def sauver(self, chemin) -> None:
        import joblib
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, chemin)

    @staticmethod
    def charger(chemin) -> "PredicteurSejour":
        import joblib
        return joblib.load(chemin)


# ---------------------------------------------------------------------------
# Référence simple : la distribution historique de l'acte (puis du type d'intervention)
# ---------------------------------------------------------------------------

class ProfilHistorique:
    """« Pour un nouveau patient, on prend la distribution observée dans le
    passé pour le même acte. » C'est la référence à battre."""

    def __init__(self, nuits_max: int = 7, min_effectif: int = 10):
        self.nuits_max, self.min_effectif = nuits_max, min_effectif

    def fit(self, df):
        post = df.nuits_apres_op.clip(upper=self.nuits_max)
        self.tables = {}
        for col in ("acte_chirurgical", "type_intervention_norm"):
            t = pd.crosstab(df[col], post, normalize="index").reindex(columns=range(self.nuits_max + 1), fill_value=0)
            n = df[col].value_counts()
            self.tables[col] = (t[n.reindex(t.index) >= self.min_effectif], df.groupby(col).entree_veille.mean())
        self.glob = post.value_counts(normalize=True).reindex(range(self.nuits_max + 1), fill_value=0).values
        self.glob_veille = float(df.entree_veille.mean())
        longs = df.nuits_apres_op[df.nuits_apres_op >= self.nuits_max]
        self.queue = [float((longs > s).mean()) for s in range(self.nuits_max, 31)] if len(longs) else [0.0]
        self.moyenne_longs = float(longs.mean()) if len(longs) else float(self.nuits_max)
        return self

    def probabilites(self, df):
        pp = np.tile(self.glob, (len(df), 1)).astype(float)
        pv = np.full(len(df), self.glob_veille)
        fait = np.zeros(len(df), bool)
        for col in ("acte_chirurgical", "type_intervention_norm"):
            t, veille = self.tables[col]
            cles = df[col].astype("string").values
            for i, c in enumerate(cles):
                if not fait[i] and c in t.index:
                    pp[i] = t.loc[c].values
                    pv[i] = veille.get(c, self.glob_veille)
                    fait[i] = True
        return pp, pv

    predire = PredicteurSejour.predire
    profil_lits = PredicteurSejour.profil_lits


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def evaluer(pred: pd.DataFrame, reel: pd.DataFrame) -> dict:
    """Erreurs sur le nombre de nuits (prédiction ponctuelle)."""
    y = reel.nuits_total.values
    e = pred.nuits_esperees.values
    r = pred.nuits_totales_estimees.values
    return {"MAE nuits (médiane prévue)": float(np.mean(np.abs(r - y))),
            "MAE nuits (espérance)": float(np.mean(np.abs(e - y))),
            "nuits exactes (%)": float(100 * np.mean(r == y)),
            "à ±1 nuit (%)": float(100 * np.mean(np.abs(r - y) <= 1)),
            "ambulatoire : précision (%)": float(100 * np.mean((pred.p_ambulatoire.values > 0.5) == (y == 0))),
            "biais (nuits prévues - réelles)": float(np.mean(e - y))}


def log_loss_post(modele, df) -> float:
    from sklearn.metrics import log_loss
    pp, _ = modele.probabilites(df)
    pp = np.clip(pp, 1e-6, 1)
    pp = pp / pp.sum(axis=1, keepdims=True)
    y = df.nuits_apres_op.clip(upper=modele.nuits_max).astype(int).values
    return float(log_loss(y, pp, labels=list(range(modele.nuits_max + 1))))


def lits_attendus_par_nuit(df, profils: np.ndarray, debut, fin) -> pd.Series:
    """Somme des profils des patients, opérés à leur date réelle : lits
    attendus chaque nuit entre `debut` et `fin`."""
    jours = pd.date_range(debut, fin, freq="D")
    idx = {d: i for i, d in enumerate(jours)}
    tot = np.zeros(len(jours))
    H = profils.shape[1]
    for d, prof in zip(df.date_inter, profils):
        i0 = idx.get(d)
        if i0 is None:
            continue
        for k in range(H):                       # colonne 0 = nuit -1
            j = i0 + k - 1
            if 0 <= j < len(jours):
                tot[j] += prof[k]
    return pd.Series(tot, index=jours)


def lits_reels_par_nuit(df, debut, fin) -> pd.Series:
    jours = pd.date_range(debut, fin, freq="D")
    h = df[df.nuits_total > 0]
    return pd.Series([int(((h.date_entree <= d) & (h.date_sortie > d)).sum()) for d in jours], index=jours)
