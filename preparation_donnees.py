r"""
preparation_donnees.py — Nettoie la base du bloc opératoire et la rend
directement utilisable (analyses, machine learning, simulations).

UTILISATION
-----------
    python preparation_donnees.py          # prépare tout, écrit le dossier base_preparee/

puis, dans n'importe quel code ou notebook :

    from preparation_donnees import charger_base_preparee, jeu_ml, decoupage_temporel
    df = charger_base_preparee()                         # 1 ligne = 1 intervention, ~80 colonnes propres
    X, y = jeu_ml(df, cible="ambulatoire")              # prêt pour scikit-learn
    X_train, X_test, y_train, y_test = decoupage_temporel(df, X, y, annee_test=2022)

RÈGLE CENTRALE : LA DURÉE DE SÉJOUR VIENT DES DATES
---------------------------------------------------
  nuits_total   = date de sortie − date d'entrée            (en nuits)
  nuits_avant_op = date d'intervention − date d'entrée       (entrée la veille…)
  nuits_apres_op = date de sortie − date d'intervention
  ambulatoire   <=> nuits_total == 0
La colonne « Durée séjour (1 pour ambu) » du fichier est conservée sous le nom
`duree_sejour_declaree` uniquement pour contrôle : elle contredit les dates
pour ~1 900 séjours (`duree_declaree_incoherente`).

FUITE D'INFORMATION (important pour le machine learning)
--------------------------------------------------------
Chaque variable a un « moment » où elle est connue (voir DICTIONNAIRE) :
  identifiant | consultation | anesthésie | date choisie | après l'opération | après la sortie
Pour prédire, AU MOMENT DE LA CONSULTATION, la durée de séjour ou
l'ambulatoire, il est interdit d'utiliser les variables connues après coup
(GHM, GHS, heures de bloc, dates de sortie…) : le modèle paraîtrait excellent
et serait inutilisable. `jeu_ml` ne garde que les variables autorisées.
"""

from __future__ import annotations

import unicodedata
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ICI = Path(__file__).parent
FICHIER_BRUT = ICI / "donees_bloc_anonyme_pour_centrale_2026.xlsx"
DOSSIER = ICI / "base_preparee"

# ---------------------------------------------------------------------------
# Calendrier
# ---------------------------------------------------------------------------

# Vacances scolaires, zone B (dates incluses : du premier au dernier jour sans classe).
# 2018-2020 : à vérifier sur le calendrier officiel si vous les utilisez finement.
VACANCES_ZONE_B = [
    ("2018-10-20", "2018-11-04"), ("2018-12-22", "2019-01-06"), ("2019-02-09", "2019-02-24"),
    ("2019-04-06", "2019-04-22"), ("2019-07-06", "2019-09-01"), ("2019-10-19", "2019-11-03"),
    ("2019-12-21", "2020-01-05"), ("2020-02-15", "2020-03-01"), ("2020-04-11", "2020-04-26"),
    ("2020-07-04", "2020-08-31"), ("2020-10-17", "2020-11-01"), ("2020-12-19", "2021-01-03"),
    ("2021-02-20", "2021-03-07"), ("2021-04-10", "2021-04-25"), ("2021-07-06", "2021-09-01"),
    ("2021-10-23", "2021-11-07"), ("2021-12-18", "2022-01-02"), ("2022-02-12", "2022-02-27"),
    ("2022-04-16", "2022-05-01"), ("2022-07-07", "2022-08-31"), ("2022-10-22", "2022-11-06"),
    ("2022-12-17", "2023-01-02"),
]
# Confinements COVID-19 (activité programmée fortement réduite)
COVID = [("2020-03-17", "2020-05-10"), ("2020-10-30", "2020-12-14"), ("2021-04-03", "2021-05-02")]
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
# 1re lettre d'un code CCAM : grand appareil (nomenclature CCAM)
APPAREILS_CCAM = {"A": "système nerveux", "D": "cœur", "E": "vaisseaux", "L": "rachis / tête et tronc",
                  "M": "membre supérieur", "N": "membre inférieur", "P": "ostéo-articulaire sans précision",
                  "Q": "peau et tissus mous"}


def _paques(a: int) -> date:
    b, c = a // 100, a % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * (a % 19) + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = ((a % 19) + 11 * h + 22 * l) // 451
    return date(a, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)


def feries_et_ponts(annees) -> tuple[set, set]:
    fer = set()
    for a in annees:
        p = _paques(a)
        fer |= {date(a, 1, 1), p + timedelta(1), date(a, 5, 1), date(a, 5, 8), p + timedelta(39),
                p + timedelta(50), date(a, 7, 14), date(a, 8, 15), date(a, 11, 1), date(a, 11, 11), date(a, 12, 25)}
    ponts = {f - timedelta(1) for f in fer if f.weekday() == 1} | {f + timedelta(1) for f in fer if f.weekday() == 3}
    return fer, ponts


def _dans(periodes, s: pd.Series) -> pd.Series:
    r = pd.Series(False, index=s.index)
    for a, b in periodes:
        r |= (s >= pd.Timestamp(a)) & (s <= pd.Timestamp(b))
    return r


# ---------------------------------------------------------------------------
# Outils de nettoyage
# ---------------------------------------------------------------------------

def _col(df, debut):
    for c in df.columns:
        if str(c).strip().lower().startswith(debut.lower()):
            return c
    raise KeyError(f"colonne commençant par « {debut} » introuvable")


def _minutes(x):
    """Heure -> minutes depuis minuit. 00:00 = heure inconnue (NaN)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, datetime):
        x = x.time()
    if isinstance(x, time):
        m = x.hour * 60 + x.minute
    else:
        try:
            h, mn = str(x).split(":")[:2]
            m = int(h) * 60 + int(mn)
        except ValueError:
            return np.nan
    return np.nan if m == 0 else float(m)


def normaliser_texte(s: pd.Series) -> pd.Series:
    """Minuscules, sans accents, espaces simples : « Prothèse  Totale » -> « prothese totale »."""
    def f(x):
        if not isinstance(x, str):
            return np.nan
        x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
        return " ".join(x.lower().replace("'", " ").split()) or np.nan
    return s.map(f)


def regrouper_rares(s: pd.Series, top: int | None = None, min_effectif: int | None = None,
                    reference: pd.Series | None = None, autre: str = "autre") -> pd.Series:
    """Garde les modalités fréquentes, remplace les autres par « autre ».
    `reference` : série sur laquelle on mesure les fréquences (ex. années
    d'apprentissage seulement, pour éviter toute fuite d'information)."""
    vc = (reference if reference is not None else s).value_counts()
    garder = set(vc.index[:top]) if top else set(vc.index)
    if min_effectif:
        garder &= set(vc[vc >= min_effectif].index)
    return s.where(s.isin(garder), autre).fillna("inconnu")


# ---------------------------------------------------------------------------
# 1. Chargement et préparation
# ---------------------------------------------------------------------------

def charger_brut(chemin=FICHIER_BRUT) -> pd.DataFrame:
    return pd.read_excel(chemin)


def preparer(brut: pd.DataFrame, top_types: int = 30) -> pd.DataFrame:
    """Une ligne par intervention, colonnes renommées, types corrects,
    variables dérivées et indicateurs de qualité."""
    b = brut
    df = pd.DataFrame(index=b.index)

    # --- identifiants -----------------------------------------------------------
    df["id_sejour"] = b[_col(b, "No Cas")].astype("int64")
    df["id_patient"] = b[_col(b, "ID Patient")].astype("int64")

    # --- dates --------------------------------------------------------------------
    df["date_entree"] = pd.to_datetime(b[_col(b, "Date Entr")]).dt.normalize()
    df["date_inter"] = pd.to_datetime(b[_col(b, "Date Inter")]).dt.normalize()
    df["date_sortie"] = pd.to_datetime(b[_col(b, "Date Sortie")]).dt.normalize()
    naissance = pd.to_datetime(b[_col(b, "Date Naissance")])
    df["annee"] = df.date_inter.dt.year      # année de l'INTERVENTION (la colonne « Année » du fichier diffère pour 2 lignes)

    # --- patient --------------------------------------------------------------------
    df["sexe"] = b[_col(b, "Sexe")].map({1: "H", 2: "F"}).astype("category")
    df["age"] = ((df.date_inter - naissance).dt.days / 365.25).round(1)
    df["classe_age"] = pd.cut(df.age, [0, 18, 40, 60, 75, 120], right=False,
                              labels=["<18", "18-39", "40-59", "60-74", "75+"])

    # --- durée de séjour : À PARTIR DES DATES ------------------------------------------
    df["nuits_avant_op"] = (df.date_inter - df.date_entree).dt.days
    df["nuits_apres_op"] = (df.date_sortie - df.date_inter).dt.days
    df["nuits_total"] = (df.date_sortie - df.date_entree).dt.days
    df["duree_sejour_jours"] = df.nuits_total + 1          # même convention que le fichier (1 = ambulatoire)
    df["ambulatoire"] = df.nuits_total == 0
    df["entree_veille"] = df.nuits_avant_op > 0
    df["classe_sejour"] = pd.cut(df.nuits_total, [-1, 0, 2, 5, 10, 10_000],
                                 labels=["ambulatoire", "1-2 nuits", "3-5 nuits", "6-10 nuits", "plus de 10 nuits"])
    df["duree_sejour_declaree"] = b[_col(b, "Dur")].astype("int64")
    df["ecart_declare_vs_dates"] = df.duree_sejour_declaree - df.duree_sejour_jours
    df["duree_declaree_incoherente"] = df.ecart_declare_vs_dates != 0

    # --- chirurgien et intervention ------------------------------------------------------
    df["chirurgien"] = b[_col(b, "Praticien")].fillna("INCONNU").astype(str).str.strip()
    df["chirurgien_nom"] = b[_col(b, "Nom Chir")].fillna("INCONNU").astype(str).str.strip()
    df["type_intervention"] = b[_col(b, "Interv Type")].astype("string").str.strip()
    df["type_intervention_norm"] = normaliser_texte(b[_col(b, "Interv Type")])
    df["type_intervention_groupe"] = regrouper_rares(df.type_intervention_norm, top=top_types)

    # --- diagnostics (CIM-10) ---------------------------------------------------------------
    diag = b[_col(b, "CIM Diag")].astype("string").str.strip().str.upper()
    df["diag_principal"] = diag
    df["diag_chapitre"] = diag.str[0]                       # M = ostéo-articulaire, S = traumatisme, I = circulatoire…
    df["diag_categorie"] = diag.str[:3]                      # ex. M17 = gonarthrose
    assoc = [c for c in b.columns if str(c).lower().startswith("cim assoc")]
    for k, c in enumerate(assoc, 1):
        df[f"diag_associe_{k}"] = b[c].astype("string").str.strip().str.upper()
    df["nb_diag_associes"] = b[assoc].notna().sum(axis=1)
    tous_assoc = b[assoc].astype("string")
    for nom, prefixes in {"hypertension": ("I10", "I11", "I12", "I13", "I15"), "diabete": ("E10", "E11", "E13", "E14"),
                          "obesite": ("E66",), "tabac": ("F17", "Z72.0"), "anticoagulant": ("Z92.1", "Z79.0")}.items():
        df[f"comorbidite_{nom}"] = tous_assoc.apply(lambda col: col.str.startswith(prefixes, na=False)).any(axis=1)

    # --- actes (CCAM) -------------------------------------------------------------------------
    ccam = [c for c in b.columns if str(c).upper().startswith("CCAM")]
    for k, c in enumerate(ccam, 1):
        df[f"ccam_{k}"] = b[c].astype("string").str.strip().str.upper()
    df["ccam_principal"] = df["ccam_1"]                      # 1er acte DE LA LISTE (souvent une radiographie !)
    df["ccam_appareil"] = df.ccam_1.str[0]
    df["ccam_groupe"] = df.ccam_1.str[:2]
    df["nb_actes_ccam"] = b[ccam].notna().sum(axis=1)
    # L'acte chirurgical réel : le 1er acte qui n'est ni un examen diagnostique
    # (3e lettre Q : radiographie, examen…), ni un acte sur le sang (F… :
    # transfusion), ni un supplément (Y…), ni un acte sans topographie (Z…).
    # Dans ~25 % des cas, le 1er acte listé est une radiographie et ne décrit
    # pas l'intervention.
    actes = df[[f"ccam_{k}" for k in range(1, len(ccam) + 1)]]
    est_chir = actes.apply(lambda c: c.notna() & (c.str[2] != "Q") & ~c.str[0].isin(["F", "Y", "Z"]))
    premier = est_chir.values.argmax(axis=1)
    trouve = est_chir.values.any(axis=1)
    vals = actes.values
    df["acte_chirurgical"] = pd.Series([vals[i, premier[i]] if trouve[i] else vals[i, 0] for i in range(len(df))],
                                       index=df.index, dtype="string")
    df["acte_chirurgical_famille"] = df.acte_chirurgical.str[:4]      # topographie (2) + action (1) + voie (1)
    df["acte_topographie"] = df.acte_chirurgical.str[:2]              # NF genou, NE hanche, MJ épaule, EJ veines…
    df["acte_appareil"] = df.acte_chirurgical.str[0].map(APPAREILS_CCAM).fillna(df.acte_chirurgical.str[0])
    df["acte_sous_arthroscopie"] = (df.acte_chirurgical.str[3] == "C").fillna(False).astype(bool)   # 4e lettre C : endoscopie
    df["nb_actes_chirurgicaux"] = est_chir.sum(axis=1)
    df["nb_actes_diagnostiques"] = actes.apply(lambda c: c.str[2] == "Q").sum(axis=1)
    df["premier_acte_est_diagnostique"] = (df.ccam_1.str[2] == "Q").fillna(False).astype(bool)

    # --- anesthésie -----------------------------------------------------------------------------
    df["anesthesie_type"] = b[_col(b, "Anesth Type")].astype("string").str.strip()
    df["anesthesie_locoreg"] = b[_col(b, "Anesth Loco")].astype("string").str.strip()
    df["anesthesie_generale"] = df.anesthesie_type.str.startswith("AG", na=False)
    df["bloc_locoregional"] = df.anesthesie_locoreg.notna()

    # --- groupage (codé APRÈS la sortie) ---------------------------------------------------------
    df["ghm"] = b[_col(b, "GHM")].astype("string").str.strip()
    df["ghm_cmd"] = df.ghm.str[:2]
    df["ghm_type"] = df.ghm.str[2]                            # C chirurgical, K actes non opératoires, M médical
    df["ghm_niveau"] = df.ghm.str[-1]                         # 1-4 sévérité, J ambulatoire, T très courte durée
    df["ghs"] = b[_col(b, "GHS")].astype("int64")

    # --- heures de bloc -----------------------------------------------------------------------------
    df["heure_sspi_pre_min"] = b[_col(b, "Heure entrée SSPI")].map(_minutes)
    df["heure_entree_salle_min"] = b[_col(b, "Heure d'entrée en salle")].map(_minutes)
    df["heure_incision_min"] = b[_col(b, "Heure Incision")].map(_minutes)
    df["heure_sortie_salle_min"] = b[_col(b, "Heure de sortie de salle")].map(_minutes)

    def duree(a, z, maxi):
        d = df[z] - df[a]
        return d.where((d >= 0) & (d <= maxi))
    df["duree_salle_min"] = duree("heure_entree_salle_min", "heure_sortie_salle_min", 12 * 60)   # TROS
    df["duree_avant_incision_min"] = duree("heure_entree_salle_min", "heure_incision_min", 3 * 60)
    df["duree_chirurgie_min"] = duree("heure_incision_min", "heure_sortie_salle_min", 12 * 60)
    df["attente_sspi_pre_min"] = duree("heure_sspi_pre_min", "heure_entree_salle_min", 6 * 60)
    df["passage_sspi_pre"] = df.attente_sspi_pre_min > 0
    df["creneau_horaire"] = pd.cut(df.heure_entree_salle_min, [0, 10 * 60, 13 * 60, 24 * 60], right=False,
                                   labels=["matin (avant 10h)", "fin de matinée", "après-midi"])

    # --- calendrier ------------------------------------------------------------------------------------
    di = df.date_inter
    df["jour_semaine"] = di.dt.weekday
    df["jour_semaine_nom"] = pd.Categorical(di.dt.weekday.map(dict(enumerate(JOURS))), categories=JOURS, ordered=True)
    df["mois"] = di.dt.month
    df["semaine_iso"] = di.dt.isocalendar().week.astype(int)
    df["semaine_paire"] = df.semaine_iso % 2 == 0
    df["trimestre"] = di.dt.quarter
    fer, ponts = feries_et_ponts(range(df.annee.min() - 1, df.annee.max() + 2))
    df["ferie"] = di.dt.date.isin(fer)
    df["pont"] = di.dt.date.isin(ponts)
    df["vacances_scolaires"] = _dans(VACANCES_ZONE_B, di)
    df["periode_covid"] = _dans(COVID, di)
    # nuits de week-end pendant le séjour (vendredi, samedi, dimanche soir)
    df["nuits_weekend"] = [sum(1 for k in range(n) if (e + timedelta(days=k)).weekday() >= 4)
                           for e, n in zip(df.date_entree, df.nuits_total)]
    df["sejour_couvre_weekend"] = df.nuits_weekend > 0

    # --- historique du patient --------------------------------------------------------------------------
    df = df.sort_values(["id_patient", "date_inter", "id_sejour"])
    g = df.groupby("id_patient")
    df["rang_sejour_patient"] = g.cumcount() + 1
    df["nb_sejours_anterieurs"] = df.rang_sejour_patient - 1
    df["jours_depuis_sejour_precedent"] = (df.date_inter - g.date_sortie.shift()).dt.days
    df["nb_sejours_patient_total"] = g.id_sejour.transform("size")

    # --- classification de l'urgence (urgences.py : FSSA, NCEPOD, SFAR/SOFCOT) -----------------------------
    from urgences import classer_df
    df = pd.concat([df, classer_df(df)], axis=1)

    # --- indicateurs de qualité ---------------------------------------------------------------------------
    df["heures_incompletes"] = df[["heure_entree_salle_min", "heure_sortie_salle_min"]].isna().any(axis=1)
    df["nuits_avant_op_atypiques"] = df.nuits_avant_op > 3
    df["sejour_long_atypique"] = df.nuits_total > 30

    df = df.sort_values(["date_inter", "chirurgien", "heure_entree_salle_min"]).reset_index(drop=True)
    for c in ["chirurgien", "type_intervention_groupe", "diag_chapitre", "ccam_appareil", "acte_appareil", "ghm_type",
              "ghm_niveau", "anesthesie_type", "anesthesie_locoreg"]:
        df[c] = df[c].astype("category")
    return df


# ---------------------------------------------------------------------------
# 2. Dictionnaire des variables : sens et moment où elles sont connues
# ---------------------------------------------------------------------------

_D = [
    ("id_sejour", "identifiant", "identifiant du séjour (No Cas)"),
    ("id_patient", "identifiant", "identifiant anonyme du patient"),
    ("date_entree", "après la sortie", "date d'entrée (connue une fois le séjour planifié)"),
    ("date_inter", "date choisie", "date de l'intervention"),
    ("date_sortie", "après la sortie", "date de sortie"),
    ("annee", "date choisie", "année de l'intervention"),
    ("sexe", "consultation", "H / F"),
    ("age", "consultation", "âge le jour de l'intervention (années)"),
    ("classe_age", "consultation", "tranche d'âge"),
    ("nuits_avant_op", "après la sortie", "nuits entre l'entrée et l'intervention (1 = entrée la veille)"),
    ("nuits_apres_op", "après la sortie", "nuits entre l'intervention et la sortie"),
    ("nuits_total", "après la sortie", "NUITS D'HOSPITALISATION = date de sortie - date d'entrée (cible principale)"),
    ("duree_sejour_jours", "après la sortie", "nuits_total + 1 (convention du fichier : 1 = ambulatoire)"),
    ("ambulatoire", "après la sortie", "vrai si aucune nuit (cible de classification)"),
    ("entree_veille", "après la sortie", "vrai si entrée la veille ou avant"),
    ("classe_sejour", "après la sortie", "ambulatoire / 1-2 / 3-5 / 6-10 / plus de 10 nuits"),
    ("duree_sejour_declaree", "après la sortie", "colonne du fichier, NON FIABLE, gardée pour contrôle"),
    ("ecart_declare_vs_dates", "après la sortie", "durée déclarée - durée calculée par les dates"),
    ("duree_declaree_incoherente", "après la sortie", "vrai si la colonne du fichier contredit les dates"),
    ("chirurgien", "consultation", "code du chirurgien (CL, JT…)"),
    ("chirurgien_nom", "consultation", "nom anonymisé du chirurgien"),
    ("type_intervention", "consultation", "libellé d'origine"),
    ("type_intervention_norm", "consultation", "libellé normalisé (minuscules, sans accents)"),
    ("type_intervention_groupe", "consultation", "30 types les plus fréquents, les autres = « autre »"),
    ("diag_principal", "consultation", "diagnostic principal CIM-10"),
    ("diag_chapitre", "consultation", "1re lettre CIM-10 (M ostéo-articulaire, S traumatisme, I circulatoire…)"),
    ("diag_categorie", "consultation", "3 premiers caractères CIM-10 (M17 gonarthrose…)"),
    ("nb_diag_associes", "consultation", "nombre de diagnostics associés (comorbidités ; codés à la sortie mais en général connus avant)"),
    ("comorbidite_hypertension", "consultation", "diagnostic associé I10-I15"),
    ("comorbidite_diabete", "consultation", "diagnostic associé E10-E14"),
    ("comorbidite_obesite", "consultation", "diagnostic associé E66"),
    ("comorbidite_tabac", "consultation", "diagnostic associé F17 / Z72.0"),
    ("comorbidite_anticoagulant", "consultation", "diagnostic associé Z92.1 / Z79.0"),
    ("ccam_principal", "consultation", "1er acte CCAM DE LA LISTE (une radiographie dans ~25 % des cas : préférer acte_chirurgical)"),
    ("ccam_appareil", "consultation", "1re lettre du 1er acte de la liste"),
    ("ccam_groupe", "consultation", "2 premières lettres du 1er acte de la liste"),
    ("nb_actes_ccam", "consultation", "nombre d'actes CCAM (0 à 4)"),
    ("acte_chirurgical", "consultation", "ACTE CHIRURGICAL : 1er acte qui n'est ni un examen (3e lettre Q), ni une transfusion (F), ni un supplément (Y), ni Z"),
    ("acte_chirurgical_famille", "consultation", "4 premiers caractères de l'acte : topographie + action + voie"),
    ("acte_topographie", "consultation", "2 premières lettres de l'acte (NF genou, NE hanche, MJ épaule, EJ veines…)"),
    ("acte_appareil", "consultation", "grand appareil de l'acte (membre inférieur, supérieur, rachis, vaisseaux…)"),
    ("acte_sous_arthroscopie", "consultation", "4e lettre C : acte par voie endoscopique (arthroscopie)"),
    ("nb_actes_chirurgicaux", "consultation", "nombre d'actes chirurgicaux"),
    ("nb_actes_diagnostiques", "consultation", "nombre d'examens associés (radiographies…)"),
    ("premier_acte_est_diagnostique", "consultation", "vrai si le 1er acte de la liste est un examen"),
    ("anesthesie_type", "anesthésie", "type d'anesthésie générale (décidé en consultation d'anesthésie)"),
    ("anesthesie_locoreg", "anesthésie", "bloc loco-régional"),
    ("anesthesie_generale", "anesthésie", "vrai si AG"),
    ("bloc_locoregional", "anesthésie", "vrai si bloc loco-régional"),
    ("ghm", "après la sortie", "groupe homogène de malades (calculé APRÈS la sortie : FUITE si utilisé pour prédire)"),
    ("ghm_cmd", "après la sortie", "catégorie majeure du GHM"),
    ("ghm_type", "après la sortie", "C chirurgical / K / M"),
    ("ghm_niveau", "après la sortie", "1-4 sévérité, J ambulatoire, T très courte durée : FUITE directe"),
    ("ghs", "après la sortie", "groupe homogène de séjour (tarif)"),
    ("heure_sspi_pre_min", "après l'opération", "heure d'entrée en SSPI pré-op (minutes depuis minuit)"),
    ("heure_entree_salle_min", "après l'opération", "heure d'entrée en salle (minutes)"),
    ("heure_incision_min", "après l'opération", "heure d'incision (minutes)"),
    ("heure_sortie_salle_min", "après l'opération", "heure de sortie de salle (minutes)"),
    ("duree_salle_min", "après l'opération", "TROS : sortie - entrée en salle (cible pour prévoir le bloc)"),
    ("duree_avant_incision_min", "après l'opération", "installation + anesthésie (entrée salle -> incision)"),
    ("duree_chirurgie_min", "après l'opération", "incision -> sortie de salle"),
    ("attente_sspi_pre_min", "après l'opération", "attente en SAS de pré-anesthésie"),
    ("passage_sspi_pre", "après l'opération", "vrai s'il y a eu un passage en SSPI pré-op"),
    ("creneau_horaire", "après l'opération", "matin / fin de matinée / après-midi"),
    ("jour_semaine", "date choisie", "0 = lundi … 6 = dimanche"),
    ("jour_semaine_nom", "date choisie", "jour de l'intervention"),
    ("mois", "date choisie", "mois de l'intervention"),
    ("semaine_iso", "date choisie", "numéro de semaine ISO"),
    ("semaine_paire", "date choisie", "semaine paire (grille de vacations)"),
    ("trimestre", "date choisie", "trimestre"),
    ("ferie", "date choisie", "intervention un jour férié"),
    ("pont", "date choisie", "intervention un jour de pont"),
    ("vacances_scolaires", "date choisie", "intervention pendant les vacances scolaires (zone B)"),
    ("periode_covid", "date choisie", "intervention pendant un confinement"),
    ("nuits_weekend", "après la sortie", "nuits du vendredi, samedi ou dimanche pendant le séjour"),
    ("sejour_couvre_weekend", "après la sortie", "vrai si au moins une nuit de week-end"),
    ("classe_urgence", "consultation", "U1 urgence (<24-48 h), U2 urgence différée (<72 h), P2 (<1 mois), P3 (<3 mois), P4 (>3 mois) — urgences.py"),
    ("classe_urgence_libelle", "consultation", "libellé de la classe d'urgence"),
    ("motif_urgence", "consultation", "situation clinique ayant déterminé la classe"),
    ("reference_urgence", "consultation", "source (FSSA, SFAR/SOFCOT…)"),
    ("regle_urgence", "consultation", "préfixe CIM-10 de la règle appliquée"),
    ("delai_max_jours", "consultation", "délai maximal recommandé avant l'opération (jours)"),
    ("urgence", "consultation", "vrai si classe U1 ou U2"),
    ("score_priorite", "consultation", "ordre de passage : classe d'urgence puis âge et comorbidités (plus haut = plus tôt)"),
    ("rang_sejour_patient", "consultation", "1 = premier séjour du patient dans la base"),
    ("nb_sejours_anterieurs", "consultation", "séjours précédents du patient"),
    ("jours_depuis_sejour_precedent", "consultation", "jours depuis sa dernière sortie (vide si premier séjour)"),
    ("nb_sejours_patient_total", "après la sortie", "nombre total de séjours du patient (utilise le futur : FUITE)"),
    ("heures_incompletes", "après l'opération", "heure d'entrée ou de sortie de salle inconnue (00:00)"),
    ("nuits_avant_op_atypiques", "après la sortie", "plus de 3 nuits avant l'opération"),
    ("sejour_long_atypique", "après la sortie", "plus de 30 nuits"),
]


def dictionnaire(df: pd.DataFrame | None = None) -> pd.DataFrame:
    d = pd.DataFrame(_D, columns=["variable", "connue_a", "description"])
    if df is not None:
        d["type"] = d.variable.map(lambda v: str(df[v].dtype) if v in df else "")
        d["valeurs_manquantes"] = d.variable.map(lambda v: int(df[v].isna().sum()) if v in df else None)
        d["modalites"] = d.variable.map(lambda v: int(df[v].nunique()) if v in df else None)
    return d


MOMENTS = ["identifiant", "consultation", "anesthésie", "date choisie", "après l'opération", "après la sortie"]


def variables_connues(moment: str = "consultation", inclure_date: bool = False) -> list[str]:
    """Variables utilisables pour une prédiction faite à ce moment."""
    autorises = {"consultation": {"consultation"},
                 "anesthésie": {"consultation", "anesthésie"},
                 "veille": {"consultation", "anesthésie", "date choisie"}}[moment]
    if inclure_date:
        autorises = autorises | {"date choisie"}
    return [v for v, m, _ in _D if m in autorises]


# ---------------------------------------------------------------------------
# 3. Rapport de qualité
# ---------------------------------------------------------------------------

def rapport_qualite(brut: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    lignes = [
        ("lignes", n, ""),
        ("séjours en double (id_sejour)", int(df.id_sejour.duplicated().sum()), "doit être 0"),
        ("patients distincts", int(df.id_patient.nunique()), ""),
        ("patients avec plusieurs séjours", int((df.groupby("id_patient").size() > 1).sum()), ""),
        ("période", f"{df.date_inter.min():%d/%m/%Y} → {df.date_inter.max():%d/%m/%Y}", ""),
        ("entrée après l'intervention", int((df.date_entree > df.date_inter).sum()), "doit être 0"),
        ("intervention après la sortie", int((df.date_inter > df.date_sortie).sum()), "doit être 0"),
        ("durée déclarée ≠ dates", int(df.duree_declaree_incoherente.sum()),
         f"{df.duree_declaree_incoherente.mean():.1%} des séjours : on utilise les dates"),
        ("  dont déclarés ambulatoires mais avec nuits", int(((df.duree_sejour_declaree == 1) & (df.nuits_total > 0)).sum()), ""),
        ("plus de 3 nuits avant l'opération", int(df.nuits_avant_op_atypiques.sum()), "à vérifier"),
        ("plus de 30 nuits", int(df.sejour_long_atypique.sum()), ""),
        ("heures de salle inconnues (00:00)", int(df.heures_incompletes.sum()), "durées laissées vides"),
        ("durée de salle aberrante (<0 ou >12 h)",
         int((df.heure_entree_salle_min.notna() & df.heure_sortie_salle_min.notna() & df.duree_salle_min.isna()).sum()),
         "durées laissées vides"),
        ("chirurgien manquant", int((df.chirurgien == "INCONNU").sum()), "codé INCONNU"),
        ("type d'intervention manquant", int(df.type_intervention.isna().sum()), ""),
        ("types d'intervention distincts (normalisés)", int(df.type_intervention_norm.nunique()),
         f"les 30 premiers couvrent {(df.type_intervention_groupe != 'autre').mean():.0%} des interventions"),
        ("anesthésie générale non renseignée", int(df.anesthesie_type.isna().sum()), "souvent : bloc seul"),
        ("colonne « Année » ≠ année de l'intervention",
         int((brut.set_index(_col(brut, "No Cas"))[_col(brut, "Ann")].reindex(df.id_sejour).values
              != df.annee.values).sum()), "on garde l'année de l'intervention"),
    ]
    return pd.DataFrame(lignes, columns=["contrôle", "valeur", "commentaire"])


# ---------------------------------------------------------------------------
# 4. Tables dérivées utiles pour la suite
# ---------------------------------------------------------------------------

def occupation_journaliere(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par jour : lits occupés la nuit (séjours avec date_entree <=
    jour < date_sortie), entrées, sorties, interventions, ambulatoires."""
    jours = pd.date_range(df.date_entree.min(), df.date_sortie.max(), freq="D")
    idx = {d: i for i, d in enumerate(jours)}
    lits = np.zeros(len(jours) + 1, dtype=int)
    hosp = df[df.nuits_total > 0]
    np.add.at(lits, hosp.date_entree.map(idx).values, 1)
    np.add.at(lits, hosp.date_sortie.map(idx).values, -1)
    occ = pd.DataFrame({"date": jours, "lits_occupes_nuit": np.cumsum(lits)[:-1]})
    occ["entrees"] = occ.date.map(df.groupby("date_entree").size()).fillna(0).astype(int)
    occ["sorties"] = occ.date.map(hosp.groupby("date_sortie").size()).fillna(0).astype(int)
    occ["interventions"] = occ.date.map(df.groupby("date_inter").size()).fillna(0).astype(int)
    occ["ambulatoires"] = occ.date.map(df[df.ambulatoire].groupby("date_inter").size()).fillna(0).astype(int)
    occ["jour_semaine_nom"] = pd.Categorical(occ.date.dt.weekday.map(dict(enumerate(JOURS))), categories=JOURS,
                                             ordered=True)
    occ["nuit_weekend"] = occ.date.dt.weekday >= 4
    fer, ponts = feries_et_ponts(range(occ.date.dt.year.min(), occ.date.dt.year.max() + 1))
    occ["vacances_ou_ferie"] = _dans(VACANCES_ZONE_B, occ.date) | occ.date.dt.date.isin(fer | ponts)
    occ["periode_covid"] = _dans(COVID, occ.date)
    occ["annee"] = occ.date.dt.year
    return occ


def profil_interventions(df: pd.DataFrame, min_effectif: int = 20) -> pd.DataFrame:
    g = df.groupby("type_intervention_norm")
    p = pd.DataFrame({
        "effectif": g.size(),
        "part_ambulatoire": g.ambulatoire.mean(),
        "nuits_moyenne": g.nuits_total.mean(),
        "nuits_mediane": g.nuits_total.median(),
        "nuits_p90": g.nuits_total.quantile(0.9),
        "part_entree_veille": g.entree_veille.mean(),
        "duree_salle_mediane": g.duree_salle_min.median(),
        "duree_salle_p90": g.duree_salle_min.quantile(0.9),
        "chirurgiens": g.chirurgien.nunique(),
    })
    return p[p.effectif >= min_effectif].sort_values("effectif", ascending=False).round(2)


def profil_actes(df: pd.DataFrame, min_effectif: int = 20, colonne: str = "acte_chirurgical") -> pd.DataFrame:
    """Une ligne par acte CCAM : libellé le plus fréquent dans la base,
    séjour, consommation de lits, durée opératoire et sa variabilité."""
    g = df.groupby(colonne, observed=True)
    total_lits = df.nuits_total.sum()
    lib = g.type_intervention_norm.agg(lambda s: s.mode().iloc[0] if s.notna().any() else "")
    concord = g.type_intervention_norm.agg(lambda s: s.value_counts(normalize=True).iloc[0] if s.notna().any() else 0)
    amb = df.pivot_table(index=colonne, columns="annee", values="ambulatoire", aggfunc="mean", observed=True)
    p = pd.DataFrame({
        "libelle_le_plus_frequent": lib,
        "part_de_ce_libelle": concord,
        "appareil": g.acte_appareil.agg(lambda s: s.mode().iloc[0]),
        "effectif": g.size(),
        "par_semaine_2022": df[df.annee == 2022].groupby(colonne, observed=True).size() / 52,
        "part_ambulatoire": g.ambulatoire.mean(),
        "part_ambulatoire_2019": amb.get(2019),
        "part_ambulatoire_2022": amb.get(2022),
        "nuits_moyenne": g.nuits_total.mean(),
        "nuits_mediane": g.nuits_total.median(),
        "nuits_p90": g.nuits_total.quantile(0.9),
        "part_entree_veille": g.entree_veille.mean(),
        "lits_nuits_total": g.nuits_total.sum(),
        "part_des_lits_nuits": g.nuits_total.sum() / total_lits,
        "duree_salle_mediane": g.duree_salle_min.median(),
        "duree_salle_p90": g.duree_salle_min.quantile(0.9),
        "duree_salle_cv": g.duree_salle_min.std() / g.duree_salle_min.mean(),
        "part_operes_jeudi_vendredi": g.jour_semaine.apply(lambda s: s.isin([3, 4]).mean()),
        "part_sejours_avec_weekend": g.sejour_couvre_weekend.mean(),
        "chirurgiens": g.chirurgien.nunique(),
        "age_moyen": g.age.mean(),
    })
    return p[p.effectif >= min_effectif].sort_values("effectif", ascending=False).round(3)


def profil_chirurgiens(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["chirurgien", "annee"], observed=True)
    p = pd.DataFrame({
        "interventions": g.size(),
        "interventions_par_semaine": g.size() / 52,
        "part_ambulatoire": g.ambulatoire.mean(),
        "nuits_moyenne_hospitalises": g.apply(lambda x: x.loc[~x.ambulatoire, "nuits_total"].mean()),
        "part_entree_veille": g.entree_veille.mean(),
        "duree_salle_mediane": g.duree_salle_min.median(),
        "part_operes_jeudi_vendredi": g.jour_semaine.apply(lambda s: s.isin([3, 4]).mean()),
        "types_differents": g.type_intervention_norm.nunique(),
    })
    return p.round(2).reset_index()


# ---------------------------------------------------------------------------
# 5. Jeux de données pour le machine learning
# ---------------------------------------------------------------------------

CIBLES = {
    "ambulatoire": "classification : le patient sort-il le jour même ?",
    "nuits_total": "régression : nombre de nuits d'hospitalisation",
    "nuits_apres_op": "régression : nuits après l'opération",
    "entree_veille": "classification : entre-t-il la veille ?",
    "classe_sejour": "classification multiclasse : ambulatoire / 1-2 / 3-5 / 6-10 / >10 nuits",
    "duree_salle_min": "régression : durée d'occupation de la salle (TROS)",
}

# variables retenues par défaut (connues à la consultation, peu redondantes)
VARIABLES_DEFAUT = ["sexe", "age", "chirurgien", "type_intervention_norm", "diag_chapitre", "diag_categorie",
                    "acte_chirurgical", "acte_topographie", "acte_sous_arthroscopie", "nb_actes_chirurgicaux",
                    "nb_actes_diagnostiques", "classe_urgence", "nb_diag_associes", "comorbidite_hypertension",
                    "comorbidite_diabete", "comorbidite_obesite", "comorbidite_tabac", "comorbidite_anticoagulant",
                    "nb_sejours_anterieurs"]


def jeu_ml(df: pd.DataFrame, cible: str = "ambulatoire", variables: list[str] | None = None,
           avec_anesthesie: bool = False, avec_date: bool = False, top: int = 30, annee_test: int | None = 2022,
           exclure_covid: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Matrice X (numérique, sans valeur manquante) et cible y, prêtes pour
    scikit-learn (comme dans le cours : `get_dummies` pour les catégories,
    médiane pour les valeurs manquantes).

    - Seules des variables connues à la consultation sont acceptées (sauf
      `avec_anesthesie` / `avec_date`), sinon une erreur est levée.
    - Les modalités rares (type d'intervention, diagnostic, acte…) sont
      regroupées en « autre » : fréquences mesurées AVANT `annee_test` pour
      ne pas regarder l'année de test.
    """
    if cible not in df:
        raise KeyError(cible)
    variables = list(variables or VARIABLES_DEFAUT)
    if avec_anesthesie:
        variables += ["anesthesie_generale", "bloc_locoregional", "anesthesie_type"]
    if avec_date:
        variables += ["jour_semaine", "mois", "vacances_scolaires", "ferie", "pont"]
    moments = {v: m for v, m, _ in _D}
    interdits = [v for v in variables if moments.get(v) in ("après l'opération", "après la sortie", "identifiant")]
    if interdits:
        raise ValueError(f"Variables connues trop tard (fuite d'information) : {interdits}")

    d = df if not exclure_covid else df[~df.periode_covid]
    d = d[d[cible].notna()]
    ref = d[d.annee < annee_test] if annee_test else d
    X = pd.DataFrame(index=d.index)
    for v in variables:
        s = d[v]
        if s.dtype == bool or str(s.dtype) == "boolean":
            X[v] = s.fillna(False).astype(int)
        elif pd.api.types.is_numeric_dtype(s):
            X[v] = s.fillna(ref[v].median())                         # imputation par la médiane (cours 4)
        else:
            s = s.astype("string")
            r = ref[v].astype("string")
            X[v] = regrouper_rares(s, top=top, reference=r)
    cat = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
    X = pd.get_dummies(X, columns=cat, drop_first=False, dtype=int)     # encodage binaire (cours 4)
    y = d[cible]
    if y.dtype == bool:
        y = y.astype(int)
    return X, y


def decoupage_temporel(df, X, y, annee_test: int = 2022):
    """Apprentissage sur les années précédentes, test sur `annee_test` :
    c'est la situation réelle (on prédit l'avenir à partir du passé), plus
    honnête qu'un train_test_split au hasard."""
    a = df.loc[X.index, "annee"]
    tr, te = a < annee_test, a == annee_test
    return X[tr], X[te], y[tr], y[te]


# ---------------------------------------------------------------------------
# 6. Compatibilité avec les codes du projet (recuit, simulations)
# ---------------------------------------------------------------------------

def vers_format_projet(df: pd.DataFrame) -> pd.DataFrame:
    """Colonnes attendues par donnees_reelles.py (charger_base)."""
    return pd.DataFrame({
        "chir": df.chirurgien.astype(str), "type": df.type_intervention.fillna("").astype(str),
        "entree": df.date_entree, "sortie": df.date_sortie, "inter": df.date_inter,
        "h_in": df.heure_entree_salle_min, "h_out": df.heure_sortie_salle_min, "tros": df.duree_salle_min,
        "nuits_pre": df.nuits_avant_op.clip(upper=3), "nuits_post": df.nuits_apres_op.clip(upper=30),
        "nuits": df.nuits_avant_op.clip(upper=3) + df.nuits_apres_op.clip(upper=30),
    })


# ---------------------------------------------------------------------------
# 7. Export et rechargement
# ---------------------------------------------------------------------------

def exporter(brut, df, dossier: Path = DOSSIER) -> None:
    dossier = Path(dossier)
    dossier.mkdir(exist_ok=True)
    df.to_pickle(dossier / "base_propre.pkl")                                   # rechargement rapide, types conservés
    df.to_csv(dossier / "base_propre.csv", sep=";", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    dictionnaire(df).to_csv(dossier / "dictionnaire_variables.csv", sep=";", index=False, encoding="utf-8-sig")
    rapport_qualite(brut, df).to_csv(dossier / "rapport_qualite.csv", sep=";", index=False, encoding="utf-8-sig")
    occupation_journaliere(df).to_csv(dossier / "occupation_journaliere.csv", sep=";", index=False,
                                      encoding="utf-8-sig", date_format="%Y-%m-%d")
    profil_interventions(df).to_csv(dossier / "profil_interventions.csv", sep=";", encoding="utf-8-sig")
    profil_actes(df).to_csv(dossier / "profil_actes.csv", sep=";", encoding="utf-8-sig")
    profil_chirurgiens(df).to_csv(dossier / "profil_chirurgiens.csv", sep=";", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(dossier / "base_preparee.xlsx") as w:                  # tout en un pour Excel
        df.to_excel(w, sheet_name="base", index=False)
        dictionnaire(df).to_excel(w, sheet_name="dictionnaire", index=False)
        rapport_qualite(brut, df).to_excel(w, sheet_name="qualité", index=False)
        profil_interventions(df).to_excel(w, sheet_name="interventions")
        profil_actes(df).to_excel(w, sheet_name="actes CCAM")
        profil_chirurgiens(df).to_excel(w, sheet_name="chirurgiens", index=False)


def charger_base_preparee(dossier: Path = DOSSIER, recalculer: bool = False) -> pd.DataFrame:
    """Charge la base préparée (la prépare si nécessaire)."""
    f = Path(dossier) / "base_propre.pkl"
    if recalculer or not f.exists():
        brut = charger_brut()
        df = preparer(brut)
        exporter(brut, df, dossier)
        return df
    return pd.read_pickle(f)


if __name__ == "__main__":
    import time as _t
    t = _t.perf_counter()
    brut = charger_brut()
    df = preparer(brut)
    exporter(brut, df)
    print(f"{len(df)} interventions, {df.shape[1]} colonnes, préparées en {_t.perf_counter() - t:.0f} s")
    print(rapport_qualite(brut, df).to_string(index=False))
    print(f"\nFichiers écrits dans {DOSSIER}")
