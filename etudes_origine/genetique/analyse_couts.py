r"""
analyse_couts.py — Analyse de sensibilité aux coefficients de la fonction coût.

CHAÎNE COMPLÈTE (pour une configuration de poids de `CoutTotal`)
----------------------------------------------------------------
  1. GRILLE      l'AG de genetique_grille.py construit une nouvelle grille de
                 vacations à partir de la période d'APPRENTISSAGE (jan.-avr.
                 2022), avec les poids de la configuration.
  2. INSERTION   tous les patients de 2022 sont insérés UN PAR UN, dans l'ordre
                 de leur consultation (modèle V1 : la date est donnée dès la
                 consultation et ne bouge plus). Pour chaque patient :
                   - on essaie ses vacations au plus tôt, on garde les
                     `nb_candidats` premières réalisables (aucun dépassement
                     marges P90 comprises, lits et places dans les capacités
                     hors réserves) ;
                   - on note chacune avec `CoutTotal` (mêmes poids) ;
                   - on propose les deux meilleures, le patient prend la plus
                     tôt. Rien de réalisable : patient « sans date ».
  3. JOURNÉE     dès qu'une journée est FIGÉE (plus aucune consultation ne
                 peut y ajouter de patient, soit 7 jours avant), l'AG
                 journalier fixe l'ordre de passage, avec des poids dérivés de
                 la même configuration (cf. `Config.poids_journalier`).
  4. MESURE      sur la période de TEST (mai-déc. 2022) : lits, places,
                 remplissage, délai, patients sans date, dépassements.

L'hôpital réel ne peut être comparé que sur les lits et les places (on n'a
pas son planning 2022). Il est calculé sur EXACTEMENT les mêmes patients.

NORMALISATION. `CoutTotal` est calibré une fois pour toutes sur la grille
ACTUELLE avec la règle « premier créneau libre » (ce que ferait une
secrétaire sans outil). « Poids ×1 » veut donc dire « un terme pèse autant
que dans la pratique actuelle ».
"""

from __future__ import annotations

import datetime as dt
import random
import time
from collections import defaultdict
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from charger_historique import estimateurs
from genetique_grille import (AGGrille, ParametresAGGrille, PoidsGrille, ProblemeGrille,
                              ResultatGrille, jours_feries, profils_chirurgiens)
from genetique_jour import (AGJournee, JourneeInfaisable, ParametresAG, Poids,
                            rejouer_reel, sequences_du_jour)
from grille import GRILLE, derouler_grille
from modele import (CoutTotal, Instance, Medecin, Patient, PoidsPlanning, PoidsRessources,
                    Solution, ajouter_vacations_urgence, deriver_creneaux)

CODES_GRILLE = sorted({c.code for c in GRILLE if c.type == "programme"})
COEFS = ("lits", "places", "remplissage", "delai")
NOMS_COEFS = {"lits": "lits", "places": "places", "remplissage": "remplissage", "delai": "délai"}


# ---------------------------------------------------------------------------
# 1. Contrôles de la base
# ---------------------------------------------------------------------------

def _colonne(df: pd.DataFrame, motif: str) -> str:
    return next(c for c in df.columns if motif in c)


def audit_sejour(df: pd.DataFrame, annee: int = 2022) -> pd.DataFrame:
    """Compare la colonne « durée de séjour (1 pour ambu) » à la durée
    calculée avec les dates (nuits + 1). C'est la seconde qui est utilisée."""
    d = df[df["annee"] == annee]
    colonne = d[_colonne(d, "1 pour ambu")]
    dates = d["nuits"] + 1
    cat = np.select(
        [colonne == dates,
         (colonne == 1) & (dates > 1),
         (colonne > 1) & (dates == 1)],
        ["identique",
         "colonne = ambulatoire, dates = au moins une nuit",
         "colonne = au moins une nuit, dates = ambulatoire"],
        default="différente, même statut ambulatoire / hospitalisé")
    t = pd.Series(cat).value_counts().rename("lignes").to_frame()
    t["part (%)"] = 100 * t["lignes"] / t["lignes"].sum()
    return t


def audit_praticiens(df: pd.DataFrame, annee: int = 2022, codes=CODES_GRILLE) -> None:
    """ARRÊT si un praticien de l'année n'a aucun créneau dans la grille
    (on ne sait pas comment le gérer : c'est au groupe de décider)."""
    d = df[df["annee"] == annee]
    sans = int((d["Praticien"] == "?").sum())
    absents = sorted(set(d["Praticien"]) - set(codes) - {"?"})
    if absents:
        n = int(d["Praticien"].isin(absents).sum())
        raise RuntimeError(
            f"ARRÊT : praticien(s) présent(s) en {annee} mais absent(s) de la grille : "
            f"{absents} ({n} lignes). Décider avec le groupe comment les traiter "
            f"avant de continuer.")
    print(f"Tous les praticiens nommés de {annee} sont dans la grille. "
          f"{sans} ligne(s) sans praticien seront écartées.")


# ---------------------------------------------------------------------------
# 2. Données de l'étude
# ---------------------------------------------------------------------------

@dataclass
class Donnees:
    jour_zero: dt.date
    nb_jours: int
    fin_annee: int                         # indice du 31 décembre
    app: tuple[int, int]                   # période d'apprentissage (indices, inclus)
    test: tuple[int, int]                  # période de test
    med_codes: dict[int, str]
    patients: dict[int, Patient]
    jour_reel: dict[int, int]              # jour réel d'opération
    feries: set
    exclus: tuple
    fenetre_jours: int

    def date(self, j: int) -> dt.date:
        return self.jour_zero + dt.timedelta(days=j)

    def jour(self, date) -> int:
        return (pd.Timestamp(date).date() - self.jour_zero).days

    def periode(self, nom: str) -> tuple[int, int]:
        return self.test if nom == "test" else self.app

    def jours_ouvres(self, t0: int, t1: int) -> list[int]:
        """Jours de semaine non fériés : ceux où l'on MESURE (comme le modèle)."""
        return [j for j in range(t0, t1 + 1)
                if self.date(j).weekday() < 5 and self.date(j) not in self.feries]

    def patients_de(self, t0: int, t1: int) -> list[Patient]:
        return [self.patients[p] for p, j in self.jour_reel.items() if t0 <= j <= t1]


def preparer_donnees(df: pd.DataFrame, annee: int = 2022, annees_app=(2019, 2020, 2021),
                     fin_apprentissage: str = "2022-04-30", praticiens_exclus=("SM",),
                     delai_consultation: int = 30, fenetre_jours: int = 7,
                     marge_horizon: int = 182) -> Donnees:
    """Patients de l'année `annee` (sauf lignes sans praticien et praticiens
    exclus), avec :
      - durée estimée = médiane du TROS du même type d'acte sur `annees_app`,
        marge = P90 − médiane (charger_historique.estimateurs) ;
      - nuits = Date Sortie − Date Entrée, nuits_avant = Date Inter − Date
        Entrée (jamais la colonne « durée de séjour ») ;
      - consultation `delai_consultation` jours avant l'opération réelle,
        délai minimum `fenetre_jours`.
    L'horizon va jusqu'au 31/12 + `marge_horizon` jours, pour que les patients
    de fin d'année puissent recevoir une date."""
    estimer = estimateurs(df[df["annee"].isin(annees_app)])
    d = df[(df["annee"] == annee) & (df["Praticien"] != "?")
           & ~df["Praticien"].isin(praticiens_exclus)]
    d = d[d["Date Inter"].notna() & d["nuits"].notna() & (d["nuits"] >= 0)]

    debut = dt.date(annee, 1, 1)
    jour_zero = debut - dt.timedelta(days=debut.weekday())          # lundi
    fin_annee = (dt.date(annee, 12, 31) - jour_zero).days
    fin_app = (pd.Timestamp(fin_apprentissage).date() - jour_zero).days
    codes = sorted(d["Praticien"].unique())
    med = {c: i for i, c in enumerate(codes)}

    patients, jour_reel = {}, {}
    for pid, r in d.iterrows():
        j = (r["Date Inter"].date() - jour_zero).days
        duree, marge = estimer(r["type"], r["Praticien"])
        nuits = int(r["nuits"])
        patients[int(pid)] = Patient(
            id=int(pid), med_id=med[r["Praticien"]], duree_op=duree, marge_perso=marge,
            duree_sejour=nuits + 1, nuits_avant=int(min(r["nuits_avant"], nuits)),
            jour_demande=max(0, j - delai_consultation), fenetre_jours=fenetre_jours,
            type_interv=r["type"], duree_reelle=int(r["tros"]))
        jour_reel[int(pid)] = j

    return Donnees(jour_zero=jour_zero, nb_jours=fin_annee + 1 + marge_horizon,
                   fin_annee=fin_annee, app=((debut - jour_zero).days, fin_app),
                   test=(fin_app + 1, fin_annee), med_codes={i: c for c, i in med.items()},
                   patients=patients, jour_reel=jour_reel,
                   feries=jours_feries(range(annee, annee + 2)),
                   exclus=tuple(praticiens_exclus), fenetre_jours=fenetre_jours)


def profils_reels(don: Donnees) -> tuple[np.ndarray, np.ndarray]:
    """Lits occupés chaque nuit et admissions ambulatoires chaque jour, dans
    l'hôpital RÉEL (dates réelles), pour les patients de l'étude. Même
    convention que `Instance.nuits_occupees`."""
    lits = np.zeros(don.nb_jours, dtype=int)
    places = np.zeros(don.nb_jours, dtype=int)
    for pid, p in don.patients.items():
        j = don.jour_reel[pid]
        if p.ambulatoire:
            places[j] += 1
        else:
            debut = j - p.nuits_avant
            lits[max(0, debut):min(debut + p.nb_nuits, don.nb_jours)] += 1
    return lits, places


# ---------------------------------------------------------------------------
# 3. Réglages, configurations de poids
# ---------------------------------------------------------------------------

@dataclass
class Reglages:
    capacite_lits: int = 42              # lits (contrainte dure)
    capacite_places_jour: int = 18       # admissions ambulatoires / jour (dure)
    capacite_places: int = 12            # places simultanées (AG journalier)
    tampon_urgence: float = 0.05         # part du TVO réservée aux urgences
    reserve_lits: int = 2                # lits réservés aux urgences
    reserve_places: int = 2              # admissions réservées aux urgences
    nb_candidats: int = 12               # vacations réalisables notées par patient
    fenetre_cout_jours: int = 91         # jours mesurés par le coût lors d'une insertion
    cycle_semaines: int = 4              # longueur du cycle de la nouvelle grille
    salles: tuple = (1, 2, 3, 4, 5)      # salles que la nouvelle grille peut ouvrir
    taux_cible: float = 0.9              # remplissage visé pour dimensionner les quotas
    poids_salles: float = 1.0            # coût d'une salle-jour ouverte (grille)


def _fmt(x: float) -> str:
    return {0.25: "¼", 0.5: "½"}.get(x, f"{x:g}")


@dataclass(frozen=True)
class Config:
    """Poids de `CoutTotal`. Les mêmes servent à la grille, à l'insertion et
    (pour ce qui la concerne) à l'optimisation journalière."""
    lits: float = 1.0
    places: float = 1.0
    remplissage: float = 1.0
    delai: float = 1.0

    @property
    def coef_varie(self) -> str | None:
        diff = [k for k in COEFS if getattr(self, k) != 1.0]
        return diff[0] if len(diff) == 1 else None

    @property
    def multiplicateur(self) -> float:
        k = self.coef_varie
        return getattr(self, k) if k else 1.0

    @property
    def nom(self) -> str:
        diff = [(k, getattr(self, k)) for k in COEFS if getattr(self, k) != 1.0]
        if not diff:
            return "Référence (tous les poids ×1)"
        return ", ".join(f"Poids {NOMS_COEFS[k]} ×{_fmt(v)}" for k, v in diff)

    def cout(self, ref: dict) -> CoutTotal:
        return CoutTotal(PoidsRessources(places=self.places, lits=self.lits),
                         PoidsPlanning(remplissage=self.remplissage, delai=self.delai),
                         ref=dict(ref))

    def poids_grille(self, salles: float = 1.0) -> PoidsGrille:
        return PoidsGrille(self.lits, self.places, self.remplissage, self.delai, salles)

    def poids_journalier(self, base: Poids | None = None) -> Poids:
        """Traduction pour l'AG journalier, qui ne décide que de l'ORDRE dans
        une journée déjà figée :
          - places      -> pic et écart-type (dans la journée) des places ;
          - remplissage -> temps perdu en fin de vacation et dispersion du
                           remplissage entre vacations d'un même chirurgien ;
          - lits, délai -> aucun levier : le jour d'opération est déjà fixé.
        Les contraintes (dépassement interdit, sorties hors UCA, capacité des
        places, robustesse P90) gardent leurs poids."""
        b = base or Poids()
        return replace(b, pic_places=b.pic_places * self.places,
                       ecart_type_places=b.ecart_type_places * self.places,
                       creux=b.creux * self.remplissage,
                       dispersion=b.dispersion * self.remplissage)


def configs_un_facteur(coefs=COEFS, multiplicateurs=(0, 0.25, 4, 16)) -> list[Config]:
    """La référence (tout à ×1) puis, pour chaque coefficient, les autres
    multiplicateurs, les autres coefficients restant à ×1."""
    return [Config()] + [Config(**{c: float(m)}) for c in coefs
                         for m in multiplicateurs if m != 1]


# ---------------------------------------------------------------------------
# 4. Instance, insertion patient par patient
# ---------------------------------------------------------------------------

def construire_instance(don: Donnees, vacations, reg: Reglages) -> Instance:
    inst = Instance(jour_zero=don.jour_zero, nb_jours=don.nb_jours)
    inst.capacite_lits = reg.capacite_lits
    inst.capacite_places_jour = reg.capacite_places_jour
    inst.capacite_places = reg.capacite_places
    inst.tampon_urgence = reg.tampon_urgence
    inst.reserve_lits = reg.reserve_lits
    inst.reserve_places = reg.reserve_places
    for m, c in don.med_codes.items():
        inst.ajouter_medecin(Medecin(m, nom=c))
    for v in vacations:
        inst.ajouter_vacation(v)
    for p in don.patients.values():
        inst.ajouter_patient(p)
    inst.indexer()
    ajouter_vacations_urgence(inst, don.feries)     # créneaux URGENCES (réindexe)
    return inst


def vacations_grille_actuelle(don: Donnees):
    return derouler_grille(don.jour_zero, don.nb_jours,
                           {c: m for m, c in don.med_codes.items()}, feries=don.feries)


def consulter(sol: Solution, pid: int, cout: CoutTotal | None, nb_candidats: int):
    """Une consultation : candidates réalisables au plus tôt, notées avec
    `cout` ; les deux meilleures sont proposées, le patient prend la plus
    tôt. `cout=None` : premier créneau libre. Décision DÉFINITIVE."""
    inst = sol.inst
    cands = []
    for vid in inst.vacations_possibles(pid):
        sol.affecter(pid, vid)
        if not sol.viole(pid, vid):
            cands.append((cout(sol) if cout is not None else 0.0, inst.vacations[vid].jour, vid))
        sol.affecter(pid, None)
        if len(cands) >= nb_candidats:
            break
    if not cands:
        sol.hors_horizon.add(pid)
        return None
    cands.sort()
    vid = min(cands[:2], key=lambda c: c[1])[2]
    sol.affecter(pid, vid)
    return vid


def simuler(inst: Instance, cout: CoutTotal | None, reg: Reglages, graine: int = 0,
            apres_jour=None, delai_fige: int = 7) -> Solution:
    """Rejoue le flux des consultations, jour après jour. Après les
    consultations du jour t, la journée t + `delai_fige` ne peut plus recevoir
    de patient : `apres_jour(sol, t + delai_fige)` est alors appelé (c'est là
    que tourne l'AG journalier)."""
    sol = Solution(inst)
    rng = random.Random(graine)
    par_jour = defaultdict(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    nb_cand = reg.nb_candidats if cout is not None else 1
    for t in range(inst.nb_jours):
        pids = par_jour.get(t, [])
        if pids:
            rng.shuffle(pids)
            sol.jour_courant = t
            if cout is not None:
                sol.definir_fenetre(t, min(inst.nb_jours - 1, t + reg.fenetre_cout_jours))
            for pid in pids:
                consulter(sol, pid, cout, nb_cand)
        if apres_jour is not None and t + delai_fige < inst.nb_jours:
            apres_jour(sol, t + delai_fige)
    sol.jour_courant = None
    sol.definir_fenetre(0, inst.nb_jours - 1)
    return sol


def calibrer_cout(don: Donnees, reg: Reglages):
    """Grille actuelle + premier créneau libre : sert de RÉFÉRENCE (courbe
    « Grille actuelle ») et de normalisation de `CoutTotal`.
    Renvoie (instance, solution, ref)."""
    inst = construire_instance(don, vacations_grille_actuelle(don), reg)
    sol = simuler(inst, None, reg)
    sol.definir_fenetre(0, don.fin_annee)
    ref = CoutTotal().calibrer(sol).ref
    sol.definir_fenetre(0, inst.nb_jours - 1)
    return inst, sol, ref


def probleme_grille(don: Donnees, reg: Reglages) -> ProblemeGrille:
    """Demande de chaque chirurgien apprise sur la période d'apprentissage."""
    t0, t1 = don.app
    nb_semaines = (t1 - t0 + 1) / 7
    tis = Instance().tis
    profils = profils_chirurgiens(don.patients_de(t0, t1), don.med_codes, nb_semaines, tis)
    prob = ProblemeGrille(profils, C=reg.cycle_semaines, salles=reg.salles,
                          taux_cible=reg.taux_cible,
                          capacites=(reg.capacite_lits - reg.reserve_lits,
                                     reg.capacite_places_jour - reg.reserve_places))
    prob.calibrer()
    return prob


# ---------------------------------------------------------------------------
# 5. Optimisation journalière
# ---------------------------------------------------------------------------

def metriques_planning(inst: Instance, pj) -> dict:
    """Mesures d'un planning de journée, avec les durées estimées puis en
    rejouant les durées RÉELLES (même ordre)."""
    reel = rejouer_reel(inst, pj)
    deborde = sum(1 for vid, seq in reel.sequences.items()
                  if seq and reel.creneaux[seq[-1]].fin > inst.vacations[vid].fin)
    # temps perdu des SEULES vacations utilisées : l'AG journalier peut garder
    # une vacation vide du chirurgien dans ses séquences, `pj.creux_total`
    # compterait alors tout son TVO et fausserait la comparaison avec « Sans AG »
    creux = sum(max(0, inst.vacations[vid].fin - pj.creneaux[seq[-1]].fin)
                for vid, seq in pj.sequences.items() if seq)
    return {"pic_places": pj.pic_places,
            "ecart_type_places": pj.places_variance ** 0.5,
            "creux_min": creux,
            "hors_uca": pj.hors_uca,
            "depassement_min": pj.depassement,
            "depassement_reel_min": reel.depassement,
            "hors_uca_reel": reel.hors_uca,
            "vacations": sum(1 for s in pj.sequences.values() if s),
            "vacations_debordent_reel": deborde}


def optimiser_jour(sol: Solution, jour: int, poids: Poids, params: ParametresAG) -> dict | None:
    """AG journalier sur une journée figée. Référence : l'ordre d'inscription
    (ordre des consultations), sans optimisation."""
    inst = sol.inst
    seq = sequences_du_jour(sol, jour)
    if not seq:
        return None
    seq = {v: sorted(s, key=lambda p: (inst.patients[p].jour_demande, p)) for v, s in seq.items()}
    base = metriques_planning(inst, deriver_creneaux(inst, jour, seq))
    t = time.perf_counter()
    try:
        res = AGJournee(inst, jour, seq, poids=poids,
                        params=replace(params, graine=jour)).resoudre()
        opt, infaisable = metriques_planning(inst, res.planning), 0
    except JourneeInfaisable:
        opt, infaisable = dict(base), 1
    ligne = {"jour": jour, "date": inst.date_du_jour(jour),
             "patients": sum(len(s) for s in seq.values()), "infaisable": infaisable,
             "duree_s": time.perf_counter() - t}
    ligne.update({f"sans_ag_{k}": v for k, v in base.items()})
    ligne.update(opt)
    return ligne


def etude_journaliere(sol: Solution, jours, configs, params: ParametresAG,
                      verbeux: bool = True) -> pd.DataFrame:
    """Même solution globale (journées figées), plusieurs jeux de poids pour
    l'AG journalier. Une ligne par (configuration, jour)."""
    lignes = []
    for i, c in enumerate(configs):
        t0 = time.perf_counter()
        poids = c.poids_journalier()
        for j in jours:
            r = optimiser_jour(sol, j, poids, params)
            if r:
                r["configuration"] = c.nom
                r["coef"] = c.coef_varie or "référence"
                r["mult"] = c.multiplicateur
                lignes.append(r)
        if verbeux:
            print(f"[{i + 1}/{len(configs)}] {c.nom:<40} {time.perf_counter() - t0:6.1f} s")
    return pd.DataFrame(lignes)


INDIC_JOUR = {
    "pic_places": ("Pic de places occupées en même temps (moyenne par jour)", "mean"),
    "ecart_type_places": ("Écart-type des places dans la journée", "mean"),
    "creux_min": ("Temps perdu en fin de vacation (min par jour)", "mean"),
    "hors_uca": ("Sorties ambulatoires après fermeture de l'UCA (total)", "sum"),
    "depassement_min": ("Dépassement prévu (min, total)", "sum"),
    "depassement_reel_min": ("Dépassement avec durées réelles (min par jour)", "mean"),
    "vacations_debordent_reel": ("Vacations qui débordent avec durées réelles (total)", "sum"),
}


def resume_journalier(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par configuration + « Sans AG (ordre d'inscription) »."""
    def agreger(d, prefixe=""):
        return {nom: getattr(d[prefixe + k], f)() for k, (nom, f) in INDIC_JOUR.items()}
    lignes = {"Sans AG (ordre d'inscription)":
              agreger(df[df["configuration"] == df["configuration"].iloc[0]], "sans_ag_")}
    for nom, d in df.groupby("configuration", sort=False):
        r = agreger(d)
        r["Journées infaisables"] = int(d["infaisable"].sum())
        r["Temps de calcul (s)"] = d["duree_s"].sum()
        lignes[nom] = r
    return pd.DataFrame(lignes).T


# ---------------------------------------------------------------------------
# 6. Chaîne globale
# ---------------------------------------------------------------------------

@dataclass
class ResultatGlobal:
    config: Config
    grille: ResultatGrille | None
    inst: Instance
    sol: Solution
    journalier: pd.DataFrame | None
    duree_s: float


def executer_global(don: Donnees, config: Config, prob: ProblemeGrille, ref: dict,
                    reg: Reglages, params_grille: ParametresAGGrille,
                    params_jour: ParametresAG | None = None, jours_journalier=None,
                    graine: int = 0) -> ResultatGlobal:
    """Grille (AG) -> insertion patient par patient -> AG journalier sur les
    journées figées de `jours_journalier` (aucune si `params_jour` est None)."""
    t0 = time.perf_counter()
    rg = AGGrille(prob, config.poids_grille(reg.poids_salles), params_grille).resoudre()
    inst = construire_instance(don, prob.derouler(rg.X, don.jour_zero, don.nb_jours,
                                                  don.feries), reg)
    lignes = []
    cibles = set(jours_journalier or ()) if params_jour is not None else set()
    poids = config.poids_journalier()

    def journee_figee(sol, j):
        if j in cibles:
            r = optimiser_jour(sol, j, poids, params_jour)
            if r:
                lignes.append(r)
    sol = simuler(inst, config.cout(ref), reg, graine,
                  apres_jour=journee_figee if cibles else None,
                  delai_fige=don.fenetre_jours)
    return ResultatGlobal(config, rg, inst, sol, pd.DataFrame(lignes) if lignes else None,
                          time.perf_counter() - t0)


def etude_globale(don, configs, prob, ref, reg, params_grille, params_jour=None,
                  jours_journalier=None, verbeux=True) -> list[ResultatGlobal]:
    res = []
    for i, c in enumerate(configs):
        r = executer_global(don, c, prob, ref, reg, params_grille, params_jour,
                            jours_journalier)
        res.append(r)
        if verbeux:
            print(f"[{i + 1}/{len(configs)}] {c.nom:<40} {r.duree_s:6.1f} s "
                  f"({r.grille.generations} générations de grille)")
    return res


# ---------------------------------------------------------------------------
# 7. Indicateurs
# ---------------------------------------------------------------------------

def stats_profil(lits, places, jours, reg: Reglages) -> dict:
    l = np.array([lits[j] for j in jours], dtype=float)
    p = np.array([places[j] for j in jours], dtype=float)
    return {"Lits occupés : moyenne": l.mean(),
            "Lits occupés : écart-type": l.std(),
            "Lits occupés : pic": l.max(),
            "Nuits au-delà de la capacité de lits": int((l > reg.capacite_lits).sum()),
            "Places ambulatoires / jour : moyenne": p.mean(),
            "Places ambulatoires / jour : écart-type": p.std(),
            "Places ambulatoires / jour : pic": p.max(),
            "Jours au-delà de la capacité de places": int((p > reg.capacite_places_jour).sum())}


def indicateurs_reel(don: Donnees, lits, places, reg: Reglages, periode: str = "test") -> dict:
    t0, t1 = don.periode(periode)
    return stats_profil(lits, places, don.jours_ouvres(t0, t1), reg)


def indicateurs_solution(don: Donnees, sol: Solution, reg: Reglages,
                         periode: str = "test") -> dict:
    inst = sol.inst
    t0, t1 = don.periode(periode)
    d = stats_profil(sol.lits_jour, sol.places_jour, don.jours_ouvres(t0, t1), reg)

    vacs = [vid for vid, v in inst.vacations.items() if not v.urgence and t0 <= v.jour <= t1]
    utilisees = [vid for vid in vacs if sol.nb[vid] > 0]
    taux = np.array([sol.taux_remplissage(vid) for vid in utilisees]) if utilisees else np.zeros(1)
    tvo = sum(inst.capacite_programme(v) for v in utilisees)
    perdu = sum(max(0, inst.capacite_programme(v) - sol.charge(v)) for v in utilisees)

    cohorte = [pid for pid, j in don.jour_reel.items() if t0 <= j <= t1]
    places_ = [pid for pid in cohorte if sol.affectation[pid] is not None]
    jour_sim = {pid: inst.vacations[sol.affectation[pid]].jour for pid in places_}
    d.update({
        "Remplissage des vacations utilisées (%)": 100 * taux.mean(),
        "Écart-type du remplissage (points)": 100 * taux.std(),
        "Temps perdu dans les vacations utilisées (%)": 100 * perdu / tvo if tvo else 0.0,
        "Vacations inutilisées (%)": 100 * (1 - len(utilisees) / len(vacs)) if vacs else 0.0,
        "Salles-jours utilisées": len({(inst.vacations[v].jour, inst.vacations[v].bloc_id)
                                       for v in utilisees}),
        "Délai consultation → opération (j)": float(np.mean(
            [jour_sim[p] - inst.patients[p].jour_demande for p in places_])) if places_ else np.nan,
        "Décalage / date réelle (j)": float(np.mean(
            [jour_sim[p] - don.jour_reel[p] for p in places_])) if places_ else np.nan,
        "Patients sans date": len(cohorte) - len(places_),
        "Dépassements de vacation (h)": sol.depassement_total() / 60,
        "Conflits (règles dures)": len(sol.verifier()),
    })
    return d


def _ajouter_journalier(d: dict, df: pd.DataFrame | None) -> dict:
    if df is not None and len(df):
        d["Pic de places simultanées (moyenne par jour)"] = df["pic_places"].mean()
        d["Temps perdu en fin de vacation (min par jour)"] = df["creux_min"].mean()
        d["Dépassement avec durées réelles (min par jour)"] = df["depassement_reel_min"].mean()
        d["Journées infaisables"] = int(df["infaisable"].sum())
    return d


def tableau_global(don, resultats, lits_reel, places_reel, reg, base=None,
                   periode: str = "test") -> pd.DataFrame:
    """Une ligne par planning : hôpital réel, grille actuelle (si `base`),
    puis chaque configuration. Colonnes « coef » et « mult » pour les graphes."""
    lignes = {"Hôpital réel": indicateurs_reel(don, lits_reel, places_reel, reg, periode)}
    lignes["Hôpital réel"].update(coef=np.nan, mult=np.nan)
    if base is not None:
        lignes["Grille actuelle, premier créneau libre"] = {
            **indicateurs_solution(don, base, reg, periode), "coef": np.nan, "mult": np.nan}
    for r in resultats:
        d = indicateurs_solution(don, r.sol, reg, periode)
        d = _ajouter_journalier(d, r.journalier)
        d.update(coef=r.config.coef_varie or "référence", mult=r.config.multiplicateur)
        lignes[r.config.nom] = d
    return pd.DataFrame(lignes).T


def gains_vs_reel(tab: pd.DataFrame) -> pd.DataFrame:
    """Variation (%) par rapport à l'hôpital réel ; négatif = mieux (moins de
    dispersion, pic plus bas)."""
    cols = ["Lits occupés : écart-type", "Lits occupés : pic",
            "Places ambulatoires / jour : écart-type", "Places ambulatoires / jour : pic"]
    reel = tab.loc["Hôpital réel", cols].astype(float)
    g = (tab[cols].astype(float) - reel) / reel * 100
    return g.drop(index="Hôpital réel").rename(columns=lambda c: c + " (% vs réel)")


# ---------------------------------------------------------------------------
# 8. Courbes
# ---------------------------------------------------------------------------

COULEURS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#17becf"]


def _serie(don, valeurs, jours):
    return [don.date(j) for j in jours], [valeurs[j] for j in jours]


def tracer_profils(don: Donnees, reel, courbes: dict, quoi: str = "lits",
                   titre: str = "", periode: str = "test", zoom_semaines: int = 4,
                   debut_zoom=None, grille_actuelle=None):
    """Deux graphiques : toute la période, puis un zoom de quelques semaines.
    `reel` : profil de l'hôpital réel ; `courbes` : {nom: profil}."""
    t0, t1 = don.periode(periode)
    if quoi == "lits":
        jours = list(range(t0, t1 + 1))
        ylabel = "Lits occupés (chaque nuit)"
    else:
        jours = don.jours_ouvres(t0, t1)
        ylabel = "Patients ambulatoires admis (chaque jour ouvré)"
    z0 = don.jour(debut_zoom) if debut_zoom else t0 + 28 + (-(t0 + 28)) % 7
    zjours = [j for j in jours if z0 <= j < z0 + 7 * zoom_semaines]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.3, 1]})
    for ax, js, lw in ((axes[0], jours, 1.0), (axes[1], zjours, 1.8)):
        ax.plot(*_serie(don, reel, js), color="black", lw=lw + 1.2, label="Hôpital réel")
        if grille_actuelle is not None:
            ax.plot(*_serie(don, grille_actuelle, js), color="grey", lw=lw, ls=":",
                    label="Grille actuelle, premier créneau libre")
        for i, (nom, prof) in enumerate(courbes.items()):
            ax.plot(*_serie(don, prof, js), color=COULEURS[i % len(COULEURS)], lw=lw,
                    alpha=0.85, marker="o" if ax is axes[1] and quoi != "lits" else None,
                    ms=3, label=nom)
        ax.set_ylabel(ylabel)
    axes[0].set_title(titre or ylabel)
    axes[1].set_title(f"Zoom sur {zoom_semaines} semaines")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    plt.show()


def tracer_famille(don, resultats, reel, coef: str, quoi: str = "lits", base=None, **kw):
    """Profils (lits ou places) des configurations où seul `coef` varie."""
    fam = sorted([r for r in resultats if r.config.coef_varie in (coef, None)],
                 key=lambda r: r.config.multiplicateur)
    attr = "lits_jour" if quoi == "lits" else "places_jour"
    courbes = {f"Poids {NOMS_COEFS[coef]} ×{_fmt(r.config.multiplicateur)}"
               + (" (référence)" if r.config.coef_varie is None else ""): getattr(r.sol, attr)
               for r in fam}
    quoi_txt = "Lits occupés chaque nuit" if quoi == "lits" else "Admissions ambulatoires par jour"
    tracer_profils(don, reel, courbes, quoi,
                   titre=f"{quoi_txt} — effet du poids « {NOMS_COEFS[coef]} »",
                   grille_actuelle=getattr(base, attr) if base is not None else None, **kw)


def tracer_sensibilite(tab: pd.DataFrame, indicateurs, coefs=COEFS, ncols: int = 3):
    """Un petit graphique par indicateur : en abscisse le multiplicateur du
    poids, une courbe par coefficient varié. Trait noir : hôpital réel ;
    pointillé gris : grille actuelle."""
    cfg = tab[tab["mult"].notna()]
    n = len(indicateurs)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    mults = sorted(cfg["mult"].astype(float).unique())
    pos = {m: i for i, m in enumerate(mults)}
    for ax, ind in zip(axes.ravel(), indicateurs):
        for i, c in enumerate(coefs):
            d = cfg[(cfg["coef"] == c) | (cfg["coef"] == "référence")]
            d = d.assign(m=d["mult"].astype(float)).sort_values("m")
            if len(d) > 1:
                ax.plot([pos[m] for m in d["m"]], d[ind].astype(float), marker="o",
                        color=COULEURS[i], label=f"Poids {NOMS_COEFS[c]}")
        if "Hôpital réel" in tab.index and pd.notna(tab.loc["Hôpital réel", ind]):
            ax.axhline(float(tab.loc["Hôpital réel", ind]), color="black", lw=1.5,
                       label="Hôpital réel")
        nom_base = "Grille actuelle, premier créneau libre"
        if nom_base in tab.index and pd.notna(tab.loc[nom_base, ind]):
            ax.axhline(float(tab.loc[nom_base, ind]), color="grey", ls=":", lw=1.5,
                       label="Grille actuelle")
        ax.set_xticks(range(len(mults)), [f"×{_fmt(m)}" for m in mults])
        ax.set_title(ind, fontsize=10)
        ax.set_xlabel("multiplicateur du poids")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def tracer_journalier(resume: pd.DataFrame, colonnes=None, ncols: int = 2):
    """Barres horizontales : une barre par configuration (et « Sans AG »)."""
    colonnes = colonnes or [v[0] for v in INDIC_JOUR.values()]
    n = len(colonnes)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 0.35 * len(resume) * nrows + 1.5 * nrows),
                             squeeze=False)
    for ax, col in zip(axes.ravel(), colonnes):
        v = resume[col].astype(float)
        couleurs = ["grey" if i.startswith("Sans AG") else "#1f77b4" if i.startswith("Référence")
                    else "#9ecae1" for i in v.index]
        ax.barh(range(len(v)), v.values, color=couleurs)
        ax.set_yticks(range(len(v)), v.index)
        ax.invert_yaxis()
        ax.set_title(col, fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def tracer_pic_places_jours(df: pd.DataFrame, configurations=None):
    """Pic de places occupées en même temps, jour par jour."""
    configurations = configurations or list(df["configuration"].unique())
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ref = df[df["configuration"] == configurations[0]].sort_values("date")
    ax.plot(ref["date"], ref["sans_ag_pic_places"], color="grey", lw=2,
            label="Sans AG (ordre d'inscription)")
    for i, c in enumerate(configurations):
        d = df[df["configuration"] == c].sort_values("date")
        ax.plot(d["date"], d["pic_places"], color=COULEURS[i % len(COULEURS)], lw=1.2,
                marker="o", ms=3, label=c)
    ax.set_ylabel("Places occupées en même temps (pic du jour)")
    ax.set_title("Pic de places ambulatoires occupées en même temps, chaque jour")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    plt.show()


def tracer_convergence_grilles(resultats):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, r in enumerate(resultats):
        g = [x[0] for x in r.grille.convergence]
        c = [x[1] for x in r.grille.convergence]
        ax.plot(g, c, color=COULEURS[i % len(COULEURS)] if i < len(COULEURS) else None,
                lw=1, label=r.config.nom)
    ax.set_xlabel("génération")
    ax.set_ylabel("coût de la meilleure grille")
    ax.set_title("Convergence de l'AG de grille (une courbe par configuration)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    fig.tight_layout()
    plt.show()
