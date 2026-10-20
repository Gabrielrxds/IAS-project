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
CIBLE = (0.70, 0.65)        # (f_we, f_vac) de la cible de lits ; (1, 1) = variance pure


# Date de consultation (absente de la base). "calee" : délai propre à chaque chirurgien, calé
# pour que la programmation au premier créneau libre sur la grille réelle redonne au mieux les vraies
# dates d'opération (calage_consultation.py) ; un entier : délai fixe en jours (ancien protocole : 30).
CONSULTATION = 30            # "calee" : délai calé par chirurgien (calage_consultation.py), en sensibilité


def delais_consultation(don, mode=None):
    """Délai consultation -> opération réelle de chaque patient (jours, entre 7 et 182)."""
    mode = CONSULTATION if mode is None else mode
    if isinstance(mode, int):
        return {p: mode for p in don.patients}
    cal = json.loads((RACINE / "resultats" / "calage_consultation.json").read_text())
    rng = np.random.default_rng(12345)
    pids = sorted(don.patients)
    z = dict(zip(pids, rng.standard_normal(len(pids))))
    anon = {c: f"C{i + 1}" for i, c in enumerate(sorted(set(don.med_codes.values())))}
    L = {c: cal["delai_type"].get(a, cal["delai_type"].get(c)) for c, a in anon.items()}
    return {p: int(np.clip(round(L[don.med_codes[don.patients[p].med_id]] * np.exp(cal["sigma"] * z[p])),
                           7, 182)) for p in pids}


def charger(exclus=(), consultation=None):
    """(df historique complet, Donnees de l'étude). Mis en cache. Les dates de consultation
    sont posées à chaque chargement selon `CONSULTATION`."""
    f = CACHE / f"donnees_{'_'.join(exclus) or 'tous'}.pkl"
    if f.exists():
        df, don = pickle.loads(f.read_bytes())
    else:
        df = lire_historique(str(DONNEES))
        don = preparer_donnees(df, 2022, (2019, 2020, 2021), FIN_APPRENTISSAGE,
                               praticiens_exclus=tuple(exclus))
        f.write_bytes(pickle.dumps((df, don)))
    mode = CONSULTATION if consultation is None else consultation
    if mode != "brut":
        for p, d in delais_consultation(don, mode).items():
            don.patients[p].jour_demande = max(0, don.jour_reel[p] - d)
    if URGENCES_SEPAREES:
        separer_urgences(df, don)
    else:
        don.urgents, don.jour_urgent = {}, {}
    return df, don


# Urgences (V3). Les urgences réelles (classes U1 et U2 du diagnostic principal, FSSA) ne sont
# pas datées par la règle de consultation : elles sont opérées à leur date réelle et occupent
# leurs lits. Elles sortent du flux programmé (don.patients) et forment un « fond » ajouté aux
# lits et aux admissions mesurés. La règle hybride les anticipe par des fantômes d'urgence.
URGENCES_SEPAREES = True


def est_urgent_diag(codes) -> "np.ndarray":
    import urgences_fssa as U
    return np.array([U.classer(c)["classe_urgence"] in ("U1", "U2") for c in codes])


def separer_urgences(df, don) -> None:
    pids = sorted(don.patients)
    diag = df.loc[pids, "CIM Diag Pr."].astype("string").fillna("").tolist()
    urg = est_urgent_diag(diag)
    don.urgents = {p: don.patients.pop(p) for p, u in zip(pids, urg) if u}
    don.jour_urgent = {p: don.jour_reel.pop(p) for p in don.urgents}


def fond_urgences(don):
    """(lits par nuit, admissions ambulatoires par jour) des urgences réelles, à leur date."""
    lits = np.zeros(don.nb_jours, dtype=int)
    places = np.zeros(don.nb_jours, dtype=int)
    for p, pat in getattr(don, "urgents", {}).items():
        j = don.jour_urgent[p]
        if pat.ambulatoire:
            places[j] += 1
        else:
            d = j - pat.nuits_avant
            lits[max(0, d):min(d + pat.nb_nuits, don.nb_jours)] += 1
    return lits, places


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


# Anonymisation des grilles sur disque : les fichiers de grilles/ ne contiennent jamais le code
# d'un chirurgien de la base, seulement C1..C18 (ordre alphabétique des codes, comme dans les
# figures du rapport). La traduction se fait ici, à la lecture et à l'écriture : le reste du code
# travaille avec les codes de la base, qu'il lit dans le fichier de données.
_ANONYMES: dict | None = None


def codes_anonymes() -> dict:
    """{code de la base: 'C1'..'C18'} pour les chirurgiens de l'étude."""
    global _ANONYMES
    if _ANONYMES is None:
        _, don = charger(consultation="brut")
        _ANONYMES = {c: f"C{i + 1}" for i, c in enumerate(sorted(set(don.med_codes.values())))}
    return _ANONYMES


def _anonymiser(x, a: dict):
    if isinstance(x, str):
        return a.get(x, x)
    if isinstance(x, list):
        return [_anonymiser(v, a) for v in x]
    if isinstance(x, dict):
        return {a.get(k, k): _anonymiser(v, a) for k, v in x.items()}
    return x


def sauver_grille(nom: str, lignes, meta: dict | None = None) -> Path:
    a = codes_anonymes()
    f = RACINE / "grilles" / f"{nom}.json"
    f.parent.mkdir(exist_ok=True)
    lignes = [[d, s, deb, fin, a.get(c, c)] for d, s, deb, fin, c in lignes]
    f.write_text(json.dumps({"nom": nom, "meta": _anonymiser(meta or {}, a), "vacations": lignes}))
    return f


def lire_grille(nom: str) -> dict:
    """Grille enregistrée, codes C1..C18 retraduits en codes de la base."""
    g = json.loads((RACINE / "grilles" / f"{nom}.json").read_text())
    inv = {v: k for k, v in codes_anonymes().items()}
    g["vacations"] = [[d, s, deb, fin, inv.get(c, c)] for d, s, deb, fin, c in g["vacations"]]
    return g


def grille_actuelle_neutre(don) -> list[list]:
    return vacations_vers_neutre(don, vacations_grille_actuelle(don))


PLACES_SIMULTANEES = None     # places ambulatoires simultanées : pas de consigne, pas de plafond
CAPACITE_VACANCES = True     # capacité post-op réduite en vacances (calendrier.py)
CUMUL_MARGES = "quadratique"     # marge d'une vacation = sqrt(Σ marges²) : aléas indépendants


def instance(don, lignes, reg: Reglages | None = None):
    reg = reg or Reglages()
    inst = construire_instance(don, neutre_vers_vacations(don, lignes), reg)
    inst.cumul_marges = CUMUL_MARGES
    inst.tis_meme_acte = inst.tis        # 15 min entre TOUS les patients, même acte identique
    if PLACES_SIMULTANEES is None:       # aucune capacité de places simultanées n'a été donnée :
        inst.capacite_places = 10 ** 6   # pas de contrainte (le pic reste minimisé dans le coût J)
    else:
        inst.capacite_places = PLACES_SIMULTANEES
    if CAPACITE_VACANCES:                # moins de personnel post-op en vacances scolaires
        import calendrier
        inst.capacite_lits_nuit = calendrier.capacites_calendrier(don.jour_zero, inst.nb_jours,
                                                                  inst.capacite_lits)
    return inst


# ---------------------------------------------------------------------------
# Indicateurs communs
# ---------------------------------------------------------------------------

def _stats(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"moyenne": float(x.mean()), "mediane": float(np.median(x)),
            "variance": float(x.var()), "ecart_type": float(x.std()),
            "p95": float(np.percentile(x, 95)), "max": float(x.max()), "min": float(x.min())}


def mesurer_profils(don, lits, places, avec_urgences: bool = True) -> dict:
    """Lits : TOUTES les nuits de la période de test (les lits tournent 7 j/7).
    Places : admissions ambulatoires des jours ouvrés de la période de test.
    V3 : les urgences réelles (fond) s'ajoutent aux lits et aux admissions programmés."""
    if avec_urgences and getattr(don, "urgents", None):
        fl, fp = fond_urgences(don)
        lits = np.asarray(lits)[:len(fl)] + fl[:len(lits)]
        places = np.asarray(places)[:len(fp)] + fp[:len(places)]
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
    # cible nuit par nuit (calendrier.py) : écart à la cible, et niveaux par type de nuit
    import calendrier
    f = calendrier.profil(don.date(t0), t1 - t0 + 1, *CIBLE)
    d["lits_ecart_cible"] = calendrier.ecart_cible(L, f)
    fer, ponts = calendrier.feries_et_ponts(range(don.date(t0).year - 1, don.date(t1).year + 2))
    types = [calendrier.type_nuit(don.date(j), fer, ponts) for j in nuits]
    moy = {k: float(np.mean([x for x, ty in zip(L, types) if ty == k] or [np.nan]))
           for k in ("semaine", "week-end", "vacances")}
    d["lits_nuits_semaine_moy"] = moy["semaine"]
    d["lits_ratio_weekend"] = moy["week-end"] / moy["semaine"]
    d["lits_ratio_vacances"] = moy["vacances"] / moy["semaine"]
    sem = [x for x, ty in zip(L, types) if ty == "semaine"]
    d["lits_semaine_variance"] = float(np.var(sem))
    d["lits_semaine_max"] = float(np.max(sem))
    # critère n°1 (V3) : nuits qui suivent un lundi..vendredi, vacances comprises
    sv = [x for j, x in zip(nuits, L) if don.date(j).weekday() < 5]
    d["lits_crit1"] = float(np.var(sv))
    d["lits_crit1_max"] = float(np.max(sv))
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
