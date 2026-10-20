r"""
charger_historique.py — Construit une `Instance` à partir de l'historique du bloc
(donees_bloc_anonyme_pour_centrale_2026.xlsx), pour TESTER l'AG journalier.

Ce n'est PAS votre donnees.py : c'est un chargeur minimal, suffisant pour
rejouer des journées réelles. Deux simplifications à connaître :

  1. DURÉES ESTIMÉES SANS TRICHE. `duree_op` = médiane du TROS du même type
     d'intervention sur les années d'APPRENTISSAGE (2019-2021 par défaut) ;
     `marge_perso` = P90 - médiane. Les journées testées sont prises dans
     l'année de TEST (2022). `duree_reelle` garde le TROS observé, pour rejouer.

  2. VACATIONS RECONSTRUITES. Le fichier historique ne donne pas la salle ni
     la grille. On crée une vacation par (jour, praticien), calée sur une
     demi-journée standard selon l'activité observée :
         matin        08h00-13h00
         après-midi   13h30-18h30
         journée      08h00-18h00
     prolongée si l'activité réelle a débordé.
     AVEC `avec_grille=True` (défaut), on utilise d'abord la grille réelle
     (grille.py) : un groupe (jour, praticien) qui correspond à un créneau
     programmé de la grille va dans cette vacation ; sinon, il reçoit une
     vacation reconstruite comme ci-dessus, étiquetée « hors grille ».
     La grille est la version 2023 : ~78 % des patients 2022 tombent dessus.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict

import pandas as pd

from grille import derouler_grille
from modele import Instance, Medecin, Patient, Vacation

MATIN = (8 * 60, 13 * 60)
APREM = (13 * 60 + 30, 18 * 60 + 30)
JOURNEE = (8 * 60, 18 * 60)


def _minutes(t) -> int | None:
    if isinstance(t, (dt.time, dt.datetime)):
        return t.hour * 60 + t.minute
    return None


def lire_historique(chemin: str) -> pd.DataFrame:
    df = pd.read_excel(chemin)
    cols = list(df.columns)

    # noms de colonnes à l'encodage abîmé : on les retrouve par motif(s).
    # Chaque recherche doit désigner UNE SEULE colonne, sinon on s'arrête.
    def col(*motifs):
        trouvees = [c for c in cols if all(m in c for m in motifs)]
        if len(trouvees) != 1:
            raise KeyError(f"motifs {motifs} : {len(trouvees)} colonnes trouvées au lieu d'une")
        return trouvees[0]

    # CORRECTION TROS. L'intitulé de la colonne SSPI (« Heure entrée SSPI
    # avant intervention ... l'heure d'entrée en salle d'opération ... »)
    # contient lui aussi « entrée en salle », et il est placé AVANT la colonne
    # calimed : l'ancien `next(... if "entrée en salle" in c)` le prenait, et
    # le TROS était mesuré depuis l'entrée en SSPI. On exige « calimed ».
    df = df.rename(columns={
        col("entrée en salle", "calimed"): "entree",
        col("sortie de salle", "calimed"): "sortie",
        col("Ann"): "annee",
        col("Date Entr"): "date_entree",
        col("Date Sortie"): "date_sortie",
    })
    # NUITS calculées avec les DATES, jamais avec la colonne « durée de
    # séjour (1 pour ambu) » : 1 884 séjours y sont codés 1 alors que le
    # patient a passé une ou plusieurs nuits.
    entree = pd.to_datetime(df["date_entree"]).dt.normalize()
    df["nuits"] = (pd.to_datetime(df["date_sortie"]).dt.normalize() - entree).dt.days
    df["nuits_avant"] = (pd.to_datetime(df["Date Inter"]).dt.normalize() - entree).dt.days
    df["nuits_avant"] = df["nuits_avant"].clip(lower=0)
    df["nuits_avant"] = df[["nuits_avant", "nuits"]].min(axis=1)
    df["e"] = df["entree"].map(_minutes)
    df["s"] = df["sortie"].map(_minutes)
    df["tros"] = df["s"] - df["e"]
    df = df[(df["e"] > 0) & (df["tros"] > 0)].copy()
    df["Praticien"] = df["Praticien"].fillna("?").astype(str)
    df["type"] = df["Interv Type"].fillna(df["CCAM 1"]).fillna("?").astype(str)
    return df


def estimateurs(app: pd.DataFrame):
    """Médiane et P90 du TROS par type d'acte, avec repli par praticien."""
    par_type = app.groupby("type")["tros"].agg(
        med="median", p90=lambda s: s.quantile(0.9), n="size")
    par_type = par_type[par_type["n"] >= 5]
    par_chir = app.groupby("Praticien")["tros"].agg(
        med="median", p90=lambda s: s.quantile(0.9))
    glob = (app["tros"].median(), app["tros"].quantile(0.9))

    def estimer(type_, chir) -> tuple[int, int]:
        if type_ in par_type.index:
            m, q = par_type.loc[type_, ["med", "p90"]]
        elif chir in par_chir.index:
            m, q = par_chir.loc[chir, ["med", "p90"]]
        else:
            m, q = glob
        return int(round(m)), int(min(60, max(5, round(q - m))))
    return estimer


def _creneau(e_min: int, s_max: int) -> tuple[int, int]:
    """Demi-journée standard ; si l'activité réelle a débordé, on prolonge la
    vacation jusqu'à la demi-heure qui suit la dernière sortie (sinon on
    fabriquerait des dépassements qu'aucune séquence ne peut résorber)."""
    if e_min >= 12 * 60 + 30:
        d, f = APREM
    elif s_max <= 13 * 60 + 30:
        d, f = MATIN
    else:
        d, f = JOURNEE
    return d, max(f, -(-s_max // 30) * 30)


def construire_instance(chemin: str, annees_app=(2019, 2020, 2021),
                        annee_test: int = 2022, avec_grille: bool = True):
    """Renvoie (instance, historique) ; historique[jour][vid] = patients dans
    l'ordre RÉEL d'entrée en salle — c'est la référence à battre."""
    df = lire_historique(chemin)
    estimer = estimateurs(df[df["annee"].isin(annees_app)])
    test = df[df["annee"] == annee_test].sort_values(["Date Inter", "e"])

    d0 = test["Date Inter"].min().date()
    jour_zero = d0 - dt.timedelta(days=d0.weekday())       # lundi
    nb_jours = (test["Date Inter"].max().date() - jour_zero).days + 1
    inst = Instance(jour_zero=jour_zero, nb_jours=nb_jours)

    chirs = {c: i for i, c in enumerate(sorted(test["Praticien"].unique()))}
    for c, i in chirs.items():
        inst.ajouter_medecin(Medecin(i, nom=c))

    # 1. vacations de la grille (salles 2 à 5)
    grille: dict[tuple[int, int], list[int]] = defaultdict(list)
    vid = 0
    if avec_grille:
        for v in derouler_grille(jour_zero, nb_jours, chirs):
            inst.ajouter_vacation(v)
            grille[(v.jour, v.med_id)].append(v.id)
        vid = len(inst.vacations)

    # 2. patients ; vacation reconstruite (salles 100+) si hors grille
    historique: dict[int, dict[int, list[int]]] = defaultdict(dict)
    salle_hors_grille: dict[int, int] = defaultdict(lambda: 100)
    for (date, chir), g in test.groupby(["Date Inter", "Praticien"]):
        j = (date.date() - jour_zero).days
        cibles = grille.get((j, chirs[chir]))
        if cibles:
            cible = min(cibles, key=lambda x: inst.vacations[x].debut)
        else:
            debut, fin = _creneau(int(g["e"].min()), int(g["s"].max()))
            cible = vid
            inst.ajouter_vacation(Vacation(
                vid, bloc_id=salle_hors_grille[j], med_id=chirs[chir], jour=j,
                debut=debut, fin=fin, etiquette=f"{chir} {date:%d/%m} hors grille"))
            salle_hors_grille[j] += 1
            vid += 1
        historique[j].setdefault(cible, [])
        for pid, r in g.iterrows():
            duree, marge = estimer(r["type"], chir)
            nuits = int(r["nuits"])
            inst.ajouter_patient(Patient(
                id=int(pid), med_id=chirs[chir], duree_op=duree,
                marge_perso=marge, duree_sejour=nuits + 1,     # ambulatoire déduit
                nuits_avant=int(r["nuits_avant"]),
                jour_demande=max(0, j - 30), type_interv=r["type"],
                duree_reelle=int(r["tros"])))
            historique[j][cible].append(int(pid))   # g est trié par heure d'entrée
    return inst.indexer(), dict(historique)
