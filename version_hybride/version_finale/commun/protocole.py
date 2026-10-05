r"""
protocole.py — LE protocole commun aux trois méthodes (tabou, recuit, génétique).

Toutes les chaînes sont jugées ici, avec le MÊME modèle (modele.py du groupe,
version de Killian : nuits calculées par les dates, nuits avant l'opération,
réserves d'urgence, contrôles incrémentaux), les MÊMES patients et les MÊMES
indicateurs. Seules changent la grille de vacations, la règle qui donne la
date à la consultation, et l'ordonnancement de la journée.

  Données      donees bloc anonyme pour centrale 2026.xlsx
  Patients     opérés en 2022, tous les praticiens de la grille (SM compris)
  Durées       médiane du TROS par acte (et praticien) apprise sur 2019-2021,
               marge = P90 - médiane                    (preparer_donnees)
  Consultation 30 jours avant la date réelle d'opération, délai minimum 7 j
  Apprentissage 1er janvier -> 30 avril 2022  (les grilles n'apprennent que là)
  Test         1er mai -> 31 décembre 2022    (toutes les mesures)
  Marges       cumul QUADRATIQUE dans une vacation (option du modèle du groupe) :
               sommer les P90 suppose que tous les actes dérapent ensemble
  Capacités    42 lits, 18 admissions ambulatoires / jour, réserves urgences
               2 lits, 2 places, 5 % du TVO           (Reglages du groupe)

Une GRILLE est échangée entre méthodes sous une forme neutre : la liste des
vacations DATÉES (date, salle, début, fin, praticien) sur tout l'horizon.
Chaque méthode déroule son propre cycle ; le juge ne voit que des dates.
"""
from __future__ import annotations

import json
import pickle
import sys
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chemins import RACINE, AG, DONNEES  # noqa: E402
if str(AG) not in sys.path:
    sys.path.insert(0, str(AG))

from analyse_couts import (Reglages, preparer_donnees, construire_instance,   # noqa: E402
                           vacations_grille_actuelle, profils_reels)
from charger_historique import lire_historique                                  # noqa: E402
from modele import Vacation, Solution                                           # noqa: E402

CACHE = RACINE / "cache"
CACHE.mkdir(exist_ok=True)
FIN_APPRENTISSAGE = "2022-04-30"


def charger(exclus=()):
    """(df historique complet, Donnees de l'étude). Mis en cache."""
    f = CACHE / f"donnees_{'_'.join(exclus) or 'tous'}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    df = lire_historique(str(DONNEES))
    don = preparer_donnees(df, 2022, (2019, 2020, 2021), FIN_APPRENTISSAGE,
                           praticiens_exclus=tuple(exclus))
    f.write_bytes(pickle.dumps((df, don)))
    return df, don


# ---------------------------------------------------------------------------
# Grilles : format neutre <-> vacations du modèle
# ---------------------------------------------------------------------------

def vacations_vers_neutre(don, vacations) -> list[list]:
    return [[don.date(v.jour).isoformat(), int(v.bloc_id), int(v.debut), int(v.fin),
             don.med_codes[v.med_id]] for v in vacations]


def neutre_vers_vacations(don, lignes) -> list[Vacation]:
    """Vacations datées -> objets Vacation sur l'horizon de `don`.
    Jours fériés et praticiens absents de l'étude ignorés."""
    code_vers_id = {c: m for m, c in don.med_codes.items()}
    vacs, vid = [], 0
    for d, salle, debut, fin, code in sorted(lignes):
        j = (dt.date.fromisoformat(d) - don.jour_zero).days
        if not 0 <= j < don.nb_jours or code not in code_vers_id:
            continue
        if dt.date.fromisoformat(d) in don.feries:
            continue
        vacs.append(Vacation(vid, bloc_id=int(salle), med_id=code_vers_id[code], jour=j,
                             debut=int(debut), fin=int(fin), etiquette=f"{code} S{salle} {d}"))
        vid += 1
    return vacs


def sauver_grille(nom: str, lignes, meta: dict | None = None) -> Path:
    f = RACINE / "grilles" / f"{nom}.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps({"nom": nom, "meta": meta or {}, "vacations": lignes}))
    return f


def lire_grille(nom: str) -> dict:
    return json.loads((RACINE / "grilles" / f"{nom}.json").read_text())


def grille_actuelle_neutre(don) -> list[list]:
    return vacations_vers_neutre(don, vacations_grille_actuelle(don))


CUMUL_MARGES = "quadratique"     # marge d'une vacation = sqrt(Σ marges²) : aléas indépendants


def instance(don, lignes, reg: Reglages | None = None):
    reg = reg or Reglages()
    inst = construire_instance(don, neutre_vers_vacations(don, lignes), reg)
    inst.cumul_marges = CUMUL_MARGES
    return inst


# ---------------------------------------------------------------------------
# Indicateurs communs
# ---------------------------------------------------------------------------

def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"moyenne": float(x.mean()), "mediane": float(np.median(x)),
            "variance": float(x.var()), "ecart_type": float(x.std()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max()), "min": float(x.min())}


def mesurer_profils(don, lits, places) -> dict:
    """Lits : TOUTES les nuits de la période de test (les lits tournent 7 j/7).
    Places : admissions ambulatoires des jours ouvrés de la période de test."""
    t0, t1 = don.test
    nuits = range(t0, t1 + 1)
    ouvres = don.jours_ouvres(t0, t1)
    L = [lits[j] for j in nuits]
    # semaine (lun-jeu soir) vs week-end (ven, sam, dim soir)
    sem = [lits[j] for j in nuits if don.date(j).weekday() < 4]
    we = [lits[j] for j in nuits if don.date(j).weekday() >= 4]
    d = {f"lits_{k}": v for k, v in _stats(L).items()}
    d.update({f"lits_ouvres_{k}": v for k, v in _stats([lits[j] for j in ouvres]).items()
              if k in ("moyenne", "ecart_type", "variance")})
    d["lits_variation_nuit"] = float(np.mean(np.abs(np.diff(L))))
    d["lits_semaine_moy"], d["lits_weekend_moy"] = float(np.mean(sem)), float(np.mean(we))
    d.update({f"places_{k}": v for k, v in _stats([places[j] for j in ouvres]).items()
              if k in ("moyenne", "ecart_type", "max")})
    d["lits_serie"] = [int(x) for x in L]
    return d


def mesurer(don, sol: Solution, nom: str = "", reg: Reglages | None = None,
            journees=None) -> dict:
    """Indicateurs communs d'une solution du modèle du groupe."""
    inst = sol.inst
    t0, t1 = don.test
    d = {"nom": nom}
    d.update(mesurer_profils(don, sol.lits_jour, sol.places_jour))
    cohorte = [p for p, j in don.jour_reel.items() if t0 <= j <= t1]
    places = [p for p in cohorte if sol.affectation[p] is not None]
    delais = np.array([inst.vacations[sol.affectation[p]].jour - inst.patients[p].jour_demande
                       for p in places])
    decal = np.array([inst.vacations[sol.affectation[p]].jour - don.jour_reel[p] for p in places])
    vacs = [v for v, x in inst.vacations.items() if not x.urgence and t0 <= x.jour <= t1]
    util = [v for v in vacs if sol.nb[v] > 0]
    tvo = sum(inst.vacations[v].tvo for v in vacs)
    tvo_u = sum(inst.vacations[v].tvo for v in util)
    # remplissage RÉEL : durées réelles + TIS, quel que soit l'estimateur de la méthode
    reel = {v: 0 for v in util}
    for p in places:
        v = sol.affectation[p]
        if v in reel:
            reel[v] += inst.patients[p].duree_reelle
    for v in util:
        reel[v] += inst.tis * (sol.nb[v] - 1)
    d.update({
        "patients_test": len(cohorte),
        "sans_date": len(cohorte) - len(places),
        "delai_moyen": float(delais.mean()), "delai_median": float(np.median(delais)),
        "delai_p90": float(np.percentile(delais, 90)),
        "decalage_moyen": float(decal.mean()),
        "heures_offertes_semaine": tvo / 60 / ((t1 - t0 + 1) / 7),
        "vacations_inutilisees_pct": 100 * (1 - len(util) / len(vacs)) if vacs else 0.0,
        "remplissage_estime_pct": 100 * float(np.mean([sol.taux_remplissage(v) for v in util])),
        "occupation_reelle_pct": 100 * sum(reel.values()) / tvo_u if tvo_u else 0.0,
        "patients_par_semaine": len(places) / ((t1 - t0 + 1) / 7),
        "depassements_h": sol.depassement_total() / 60,
        "conflits": len(sol.verifier()),
    })
    if journees:
        for k in journees[0]:
            if isinstance(journees[0][k], (int, float)) and k != "jour":
                d[f"jour_{k}"] = float(np.mean([r[k] for r in journees]))
    return d


def mesurer_reel(don) -> dict:
    lits, places = profils_reels(don)
    d = {"nom": "Hôpital réel (2022)"}
    d.update(mesurer_profils(don, lits, places))
    return d


@dataclass
class Resultat:
    nom: str
    mesures: dict
    sol: object = None


def mesurer_dates(don, dates: dict, nom: str) -> dict:
    """Indicateurs communs pour une méthode qui ne rend que des DATES (chaîne recuit
    hors ligne). Les patients de l'étude absents de `dates` restent à leur date réelle."""
    lits = np.zeros(don.nb_jours, dtype=int)
    places = np.zeros(don.nb_jours, dtype=int)
    t0, t1 = don.test
    sans, absents, delais, decal = 0, 0, [], []
    for pid, p in don.patients.items():
        j = dates.get(pid, don.jour_reel[pid]) if pid in dates else don.jour_reel[pid]
        absents += pid not in dates
        if j is None:
            if t0 <= don.jour_reel[pid] <= t1:
                sans += 1
            continue
        if t0 <= don.jour_reel[pid] <= t1:
            delais.append(j - p.jour_demande); decal.append(j - don.jour_reel[pid])
        if p.ambulatoire:
            if 0 <= j < don.nb_jours:
                places[j] += 1
        else:
            debut = j - p.nuits_avant
            lits[max(0, debut):max(0, min(debut + p.nb_nuits, don.nb_jours))] += 1
    d = {"nom": nom}
    d.update(mesurer_profils(don, lits, places))
    d.update({"patients_test": sum(1 for p, j in don.jour_reel.items() if t0 <= j <= t1),
              "sans_date": sans, "absents_de_la_methode": absents,
              "delai_moyen": float(np.mean(delais)), "delai_median": float(np.median(delais)),
              "delai_p90": float(np.percentile(delais, 90)), "decalage_moyen": float(np.mean(decal)),
              "patients_par_semaine": len(delais) / ((t1 - t0 + 1) / 7)})
    return d
