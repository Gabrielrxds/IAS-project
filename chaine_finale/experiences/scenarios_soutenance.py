r"""
scenarios_soutenance.py — Les scénarios de la soutenance, sur la chaîne retenue (nuits prévues par
apprentissage, fantômes, journée tabou), comparée à la référence, chacune SANS puis AVEC agents
au jour J (mêmes aléas pour les quatre configurations d'un même tirage).

  base      la configuration retenue (tableau 12 du rapport) ; redonne resultats/recap_v3.json
  surcharge +0 / +10 / +20 / +30 % de patients programmés (copies de patients réels placées au
            hasard, comme experiences_surcharge.py), heures de bloc constantes ; plan + jour J
  aleas     (a) tous les aléas du jour multipliés par k = 0 ... 3 ;
            (b) un aléa à la fois : annulations, chirurgien absent, panne de salle, lits fermés,
            urgences (les autres à leur niveau de base)
  echelle   petite échelle : 9 chirurgiens sur 18 (50 % des patients, 48 % des heures de la
            grille actuelle), moitié des lits (21), des admissions ambulatoires (9), des réserves,
            urgences de ces chirurgiens seulement ; grille hybride RECONSTRUITE par la même méthode
            sur 2 salles (« petite ») ou sur les 4 salles actuelles (« petite4 », variante) ; la
            référence garde la grille actuelle de ces chirurgiens ; contre la grande échelle
  mc        Monte-Carlo : chaque tirage refait l'ordre des consultations, tire les durées réelles
            (médiane de l'acte + écart d'un patient du même acte), les séjours (nuits d'un patient
            hospitalisé du même acte) et les aléas du jour (urgences rééchantillonnées)

    python3 chaine_finale/experiences/scenarios_soutenance.py base|surcharge|aleas|echelle|mc [N]
    puis : python3 chaine_finale/figures/figures_scenarios.py   (figures dans resultats/figures_scenarios/)
Durées sur 2 cœurs : base 2 min, surcharge 1 min, aléas 5 min, échelle 5 min, Monte-Carlo (100) 16 min.
Sortie : resultats/scenarios_<partie>.json   (aucune donnée de patient, chirurgiens anonymisés)
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
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

import calendrier
from protocole import (charger, instance, grille_actuelle_neutre, lire_grille, mesurer, mesurer_profils,
                       vacations_vers_neutre, Reglages, RACINE)
from politiques import PremierCreneau, simuler
from ml_prevision import don_prevu, lits_reels

PREV = "ml_nuits"                       # prévision retenue (rapport, section 7.4)
CONFIGS = (("reference", "statique"), ("reference", "sma"), ("hybride", "statique"), ("hybride", "sma"))
BASE_ALEAS = dict(p_annulation=0.04, p_absence=0.01, p_indispo=0.05, p_lits=0.15)   # GenerateurV3
CACHE = RACINE / "cache"
CACHE.mkdir(exist_ok=True)

LITS_ARMES_GRANDE = calendrier.LITS_ARMES


# ---------------------------------------------------------------------------------------------
# Données et réglages selon l'échelle
# ---------------------------------------------------------------------------------------------

def moities():
    """(codes gardés, codes exclus) pour la petite échelle : 9 chirurgiens sur 18, choisis pour
    garder à la fois la moitié des patients de 2022 et la moitié des heures de la grille actuelle."""
    import datetime as dt
    from itertools import combinations
    from grille import creneaux_du
    from hybride_grille import SALLES_ACTUELLES
    _, don = charger()
    n = defaultdict(int)
    for p in list(don.patients.values()) + list(don.urgents.values()):
        n[don.med_codes[p.med_id]] += 1
    h = defaultdict(float)
    for w in range(1, 5):
        for d in range(5):
            for c in creneaux_du(dt.date.fromisocalendar(2022, w, d + 1)):
                if c.salle in SALLES_ACTUELLES:
                    h[c.code] += (c.fin - c.debut) / 60
    codes = sorted(n)
    N, H = sum(n.values()), sum(h[c] for c in codes)
    best = min(combinations(codes, len(codes) // 2),
               key=lambda g: (abs(sum(n[c] for c in g) / N - 0.5) + abs(sum(h[c] for c in g) / H - 0.5), g))
    garde = list(best)
    return garde, [c for c in codes if c not in garde]


def contexte_echelle(echelle: str):
    """(df, don réel, don prévu, réglages) ; la petite échelle règle aussi les modules partagés."""
    import regle_hybride as RH
    if echelle == "grande":
        calendrier.LITS_ARMES = LITS_ARMES_GRANDE
        df, don = charger()
        RH._CACHE["df"] = df
        return df, don, don_prevu(don, PREV), Reglages()
    garde, exclus = moities()          # « petite » et sa variante « petite4 » (grille hybride sur 4 salles)
    df, don = charger(exclus=tuple(sorted(exclus)))
    df = df[df["Praticien"].isin(garde)]
    RH._CACHE["df"] = df                       # fantômes d'urgence appris sur ces chirurgiens seulement
    calendrier.LITS_ARMES = round(LITS_ARMES_GRANDE / 2)
    reg = dataclasses.replace(Reglages(), capacite_lits=21, capacite_places_jour=9,
                              reserve_lits=1, reserve_places=1)
    return df, don, don_prevu(don, PREV), reg


# ---------------------------------------------------------------------------------------------
# Surcharge : mêmes copies que experiences_surcharge.py, dans les données réelles ET prévues
# ---------------------------------------------------------------------------------------------

def surcharger(don, dp, taux, graine):
    if taux == 0:
        return don, dp
    rng = random.Random(1000 + graine)
    d, e = copy.copy(don), copy.copy(dp)
    d.patients, e.patients = dict(don.patients), dict(dp.patients)
    d.jour_reel = dict(don.jour_reel)
    e.jour_reel = d.jour_reel
    base = sorted(don.jour_reel)
    nid = max(don.patients) + 1
    for _ in range(round(taux * len(base))):
        p0 = rng.choice(base)
        dec = rng.randint(-3, 3)
        j_dem = min(max(0, don.patients[p0].jour_demande + dec), don.nb_jours - 1)
        d.patients[nid] = dataclasses.replace(don.patients[p0], id=nid, jour_demande=j_dem)
        e.patients[nid] = dataclasses.replace(dp.patients[p0], id=nid, jour_demande=j_dem)
        d.jour_reel[nid] = min(max(0, don.jour_reel[p0] + dec), don.nb_jours - 1)
        nid += 1
    return d, e


# ---------------------------------------------------------------------------------------------
# Plan du matin
# ---------------------------------------------------------------------------------------------

def grille(chaine, dp, echelle):
    if chaine == "reference":
        return grille_actuelle_neutre(dp)
    if echelle == "grande":
        return lire_grille("hybride")["vacations"]
    f = CACHE / f"scenarios_grille_{echelle}.pkl"
    return pickle.loads(f.read_bytes())


def construire_matin(chaine, df, don, dp, reg, echelle="grande", graine=0):
    """Solution datée (consultations dans un ordre tiré par `graine`) et ordre de chaque journée."""
    import algos_journee as AJ
    from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
    from sma_aleas import optimiseur_glouton
    from regle_hybride import RegleHybride
    t0, t1 = dp.test
    JT = set(dp.jours_ouvres(t0, t1))
    plans = {}

    def ordre(sol, j):
        if j not in JT:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        if chaine == "reference":
            plans[j] = optimiseur_glouton(sol.inst, j, seq)
            return
        try:
            plans[j] = AJ.ALGOS["tabou"](sol.inst, j, seq, Poids(), 1000, 0, graines=True).sequences
        except JourneeInfaisable:
            plans[j] = optimiseur_glouton(sol.inst, j, seq)

    inst = instance(dp, grille(chaine, dp, echelle), reg)
    regle = PremierCreneau() if chaine == "reference" else RegleHybride(dp, delta=14, fantomes="previsions")
    t = time.time()
    sol = simuler(inst, regle, graine=graine, apres_figer=ordre)
    return {"sol": sol, "plans": plans, "plans_b": {}, "jours": sorted(JT), "secondes": time.time() - t}


def mesures_plan(don, dp, sol):
    m = mesurer(dp, sol)
    lits, places = lits_reels(don, sol)
    mr = mesurer_profils(don, lits, places)
    return dict(sans_date=m["sans_date"], patients_test=m["patients_test"], delai_median=m["delai_median"],
                delai_p90=m["delai_p90"], patients_semaine=m["patients_par_semaine"],
                occupation_pct=m["occupation_reelle_pct"], heures_semaine=m["heures_offertes_semaine"],
                plan_lits_crit1=mr["lits_crit1"], plan_lits_max=mr["lits_max"],
                plan_lits_moy=mr["lits_moyenne"], plan_places_et=mr["places_ecart_type"])


# ---------------------------------------------------------------------------------------------
# Jour de l'opération
# ---------------------------------------------------------------------------------------------

def jour_j(m, df, don, politique, graine, intensite=None, aleas=None, don_mesure=None):
    import sma_v3 as V
    inst = m["sol"].inst
    jours = [j for j in inst.jours_ouvres if min(m["jours"]) <= j <= max(m["jours"])]
    ctx = V.contexte(df, don, inst)
    kw = dict(BASE_ALEAS, **(aleas or {}))
    kw = {k: min(1.0, v) for k, v in kw.items()}
    gen = V.GenerateurV3(V.reservoir_urgences(don, *don.test), ctx.classes_reelles, intensite=intensite,
                         graine=graine, **kw)
    out = V.simuler(m, gen, politique, jours, ctx, don_reel=don_mesure or don)
    mr = mesurer_profils(don, np.array(out["lits"]), np.array(out["places"]), avec_urgences=False)
    R = out["resume"]
    n = max(1, R["urgences_arrivees"])
    sal = [x for x in out["salles"] if not x["urgence"]]
    nprog = max(1, R["programmes"])
    return dict(lits_crit1=mr["lits_crit1"], lits_max=mr["lits_max"], lits_moy=mr["lits_moyenne"],
                lits_crit1_max=mr["lits_crit1_max"], places_et=mr["places_ecart_type"],
                urgences=R["urgences_arrivees"], urg_delai=100 * R["urgences_dans_delai"] / n,
                urg_jour=100 * R["urgences_operees_jour_meme"] / n, urg_echec=100 * R["urgences_echec"] / n,
                attente=R["attente_urgence_moy"], depassement_jour=R["depassement_min"] / R["jours"],
                depassement_vac=R["depassement_min"] / max(1, len(sal)),
                salles_30=100 * float(np.mean([x["depassement"] > 30 for x in sal])) if sal else 0.0,
                reportes=R["reportes"], reportes_pct=100 * R["reportes"] / nprog,
                reportes_jours=R["reportes_jours"] / max(1, R["reportes"]), a_arbitrer=R["a_arbitrer"],
                conversions=R["conversions_nuit"], annulations=R["annulations"], absences=R["absences"],
                indispos=R.get("indisponibilites", 0), relogements=R["relogements"],
                creneau_libre=R["urgences_par_creneau_libre"], programmes=R["programmes"],
                deficit_lits=R["deficit_lits"], jours=R["jours"], secondes=out["secondes"])


# ---------------------------------------------------------------------------------------------
# Matins mis en cache (une construction par chaîne / échelle / surcharge / ordre)
# ---------------------------------------------------------------------------------------------

def matin_cache(chaine, echelle="grande", taux=0.0, graine=0):
    f = CACHE / f"scen_matin_{chaine}_{echelle}_{round(100 * taux)}_{graine}.pkl"
    df, don, dp, reg = contexte_echelle(echelle)
    don, dp = surcharger(don, dp, taux, 0)
    if f.exists():
        m = pickle.loads(f.read_bytes())
    else:
        m = construire_matin(chaine, df, don, dp, reg, echelle, graine)
        m["plan"] = mesures_plan(don, dp, m["sol"])
        f.write_bytes(pickle.dumps(m))
    return m, df, don, dp


def _matin(args):
    chaine, echelle, taux, graine = args
    m, *_ = matin_cache(chaine, echelle, taux, graine)
    return dict(chaine=chaine, echelle=echelle, taux=taux, graine=graine, secondes=m["secondes"], **m["plan"])


def _jour(args):
    chaine, pol, echelle, taux, graine_plan, graine, intensite, aleas, etiquette = args
    m, df, don, _ = matin_cache(chaine, echelle, taux, graine_plan)
    r = jour_j(m, df, don, pol, graine, intensite, aleas)
    return dict(chaine=chaine, politique=pol, echelle=echelle, taux=taux, graine_plan=graine_plan,
                graine=graine, intensite=intensite, aleas=aleas, etiquette=etiquette, **r)


def lancer(taches, f, nproc=2, chunksize=1):
    out, t = [], time.time()
    with Pool(nproc) as P:
        for k, r in enumerate(P.imap_unordered(f, taches, chunksize=chunksize), 1):
            out.append(r)
            if k % 20 == 0 or k == len(taches):
                print(f"   {k}/{len(taches)}  {time.time() - t:.0f} s", flush=True)
    return out


def ecrire(nom, obj):
    f = RACINE / "resultats" / f"scenarios_{nom}.json"
    f.write_text(json.dumps(obj, default=float, indent=0))
    print("écrit", f)


# ---------------------------------------------------------------------------------------------
# Parties
# ---------------------------------------------------------------------------------------------

GRAINES_J = tuple(range(1, 11))


def partie_base():
    plans = lancer([(c, "grande", 0.0, 0) for c in ("reference", "hybride")], _matin)
    jours = lancer([(c, p, "grande", 0.0, 0, g, None, None, "base") for c, p in CONFIGS for g in GRAINES_J],
                   _jour, chunksize=2)
    ecrire("base", dict(plans=plans, jour=jours))
    for c, p in CONFIGS:
        rs = [r for r in jours if r["chaine"] == c and r["politique"] == p]
        print(c, p, {k: round(float(np.mean([r[k] for r in rs])), 2) for k in
                     ("lits_crit1", "lits_max", "urg_delai", "urg_jour", "attente", "depassement_jour",
                      "reportes", "conversions")})


NIVEAUX_SURCHARGE = (0.0, 0.10, 0.20, 0.30)


def partie_surcharge():
    plans = lancer([(c, "grande", t, 0) for t in NIVEAUX_SURCHARGE for c in ("reference", "hybride")], _matin)
    jours = lancer([(c, p, "grande", t, 0, g, None, None, "surcharge") for t in NIVEAUX_SURCHARGE
                    for c, p in CONFIGS for g in GRAINES_J], _jour, chunksize=2)
    ecrire("surcharge", dict(plans=plans, jour=jours))


K_GLOBAL = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
PAR_TYPE = {
    "annulations": ("p_annulation", (0.0, 0.04, 0.08, 0.12, 0.16, 0.24)),
    "absences": ("p_absence", (0.0, 0.01, 0.02, 0.04, 0.06, 0.10)),
    "pannes": ("p_indispo", (0.0, 0.05, 0.10, 0.20, 0.35, 0.50)),
    "lits": ("p_lits", (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)),
    "urgences": ("intensite", (0.5, 1.0, 1.5, 2.0, 3.0)),
}
GRAINES_ALEAS = tuple(range(1, 9))


def partie_aleas():
    lancer([(c, "grande", 0.0, 0) for c in ("reference", "hybride")], _matin)
    taches = []
    for k in K_GLOBAL:
        al = {a: v * k for a, v in BASE_ALEAS.items()}
        taches += [(c, p, "grande", 0.0, 0, g, k, al, f"global:{k}") for c, p in CONFIGS for g in GRAINES_ALEAS]
    for typ, (param, vals) in PAR_TYPE.items():
        for v in vals:
            if param == "intensite":
                taches += [(c, p, "grande", 0.0, 0, g, v, None, f"{typ}:{v}") for c, p in CONFIGS for g in GRAINES_ALEAS]
            else:
                taches += [(c, p, "grande", 0.0, 0, g, 1.0, {param: v}, f"{typ}:{v}") for c, p in CONFIGS
                           for g in GRAINES_ALEAS]
    ecrire("aleas", dict(jour=lancer(taches, _jour, chunksize=4), base=BASE_ALEAS,
                         par_type={k: list(v) for k, v in PAR_TYPE.items()}, k_global=list(K_GLOBAL)))


def construire_grille_petite(echelle="petite"):
    """Grille hybride reconstruite pour la petite échelle : même méthode (chaine_hybride.py),
    budget d'heures = heures actuelles des chirurgiens gardés ; 2 salles au lieu de 4
    (« petite »), ou les 4 salles actuelles (« petite4 », variante)."""
    f = CACHE / f"scenarios_grille_{echelle}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    import datetime as dt
    import hybride_grille as HG
    import chaine_hybride as CH
    from grille import creneaux_du
    df, don, dp, reg = contexte_echelle("petite")
    garde, _ = moities()
    salles = (2, 3) if echelle == "petite" else HG.SALLES_ACTUELLES
    orig_demi = HG.demi_journees_actuelles

    def heures_gardees(salles_=HG.SALLES_ACTUELLES, annee=2022):
        h = 0.0
        for w in range(1, 5):
            for d in range(5):
                for c in creneaux_du(dt.date.fromisocalendar(annee, w, d + 1)):
                    if c.salle in HG.SALLES_ACTUELLES and c.code in garde:
                        h += (c.fin - c.debut) / 60
        return h
    HG.heures_programmees_actuelles = heures_gardees
    HG.demi_journees_actuelles = lambda codes, s=None, annee=2022: orig_demi(codes, HG.SALLES_ACTUELLES, annee)
    reg.cycle_semaines = 2
    t = time.time()
    prob = HG.probleme_hybride(don, reg, part_min=0.75, part_max=1.5, f_we=calendrier.F_WE, salles=salles)
    print(f"petite échelle : budget {prob.budget} demi-journées / cycle sur {len(salles)} salles "
          f"({heures_gardees():.0f} h / 4 sem.)", flush=True)
    cands = CH.candidates(prob, 15000, 4, verbeux=False)
    dapp = CH.don_apprentissage(don)
    for c in cands:
        mm, _ = CH.evaluer_apprentissage(dapp, prob, c["X"], reg, (calendrier.F_WE, calendrier.F_VAC))
        c.update(app_crit1=mm["lits_crit1"], app_sans=mm["sans_date"])
    mini = min(c["app_sans"] for c in cands)
    ret = min((c for c in cands if c["app_sans"] <= mini + 2), key=lambda c: c["app_crit1"])
    vacs = prob.derouler(ret["X"], don.jour_zero, don.nb_jours, don.feries)
    lignes = vacations_vers_neutre(don, vacs)
    f.write_bytes(pickle.dumps(lignes))
    print(f"grille petite échelle : {ret['origine']} ({time.time() - t:.0f} s)", flush=True)
    return lignes


ECHELLES = ("grande", "petite", "petite4")


def partie_echelle():
    construire_grille_petite("petite")
    construire_grille_petite("petite4")
    plans = lancer([(c, e, 0.0, g) for e in ECHELLES for c in ("reference", "hybride")
                    for g in (0, 1, 2)], _matin)
    jours = lancer([(c, p, e, 0.0, 0, g, None, None, "echelle") for e in ECHELLES
                    for c, p in CONFIGS for g in GRAINES_J], _jour, chunksize=2)
    from protocole import codes_anonymes
    a = codes_anonymes()
    garde, exclus = moities()
    ecrire("echelle", dict(plans=plans, jour=jours, garde=[a[c] for c in garde], exclus=[a[c] for c in exclus]))


# ------------------------------------------------------------------------------------ Monte-Carlo

def _pools(don, dp):
    """Écarts réels − estimés des durées et séjours des hospitalisés, par acte (puis chirurgien)."""
    ecarts, sej = defaultdict(list), defaultdict(list)
    for pid, p in dp.patients.items():
        r = don.patients[pid]
        if r.duree_reelle > 0:
            e = r.duree_reelle - p.duree_op
            ecarts[("a", p.type_interv)].append(e); ecarts[("m", p.med_id)].append(e); ecarts["*"].append(e)
        if r.nb_nuits > 0:
            s = (r.nb_nuits, r.nuits_avant)
            sej[("a", r.type_interv)].append(s); sej[("m", r.med_id)].append(s); sej["*"].append(s)
    return ecarts, sej


def _tirer(pools, p, rng, mini=8):
    for k in (("a", p.type_interv), ("m", p.med_id), "*"):
        if len(pools.get(k, ())) >= mini:
            return pools[k][rng.randrange(len(pools[k]))]


def tirage_realite(don, dp, graine):
    """(don réel tiré, {pid: durée réelle tirée}) : la réalité d'un tirage Monte-Carlo."""
    ecarts, sej = _pools(don, dp)
    rng = random.Random(77_000 + graine)
    d = copy.copy(don)
    d.patients = dict(don.patients)
    durees = {}
    for pid in sorted(dp.patients):
        p, r = dp.patients[pid], don.patients[pid]
        durees[pid] = max(5, int(p.duree_op + _tirer(ecarts, p, rng)))
        if r.nb_nuits > 0:
            n, a = _tirer(sej, r, rng)
            d.patients[pid] = dataclasses.replace(r, duree_sejour=n + 1, nuits_avant=min(a, n), ambulatoire=None)
    return d, durees


def _mc(args):
    g, = args
    out = []
    df, don, dp, reg = contexte_echelle("grande")
    dr, durees = tirage_realite(don, dp, g)
    for chaine in ("reference", "hybride"):
        f = CACHE / f"scen_mc_{chaine}_{g}.pkl"
        if f.exists():
            m = pickle.loads(f.read_bytes())
        else:
            m = construire_matin(chaine, df, don, dp, reg, "grande", g)
            m["plan"] = mesures_plan(don, dp, m["sol"])
            f.write_bytes(pickle.dumps(m))
        from sma_aleas import copie_de_travail
        sol = copie_de_travail(m["sol"])
        for pid, x in durees.items():
            if pid in sol.inst.patients:
                sol.inst.patients[pid] = dataclasses.replace(sol.inst.patients[pid], duree_reelle=x)
        mt = dict(m, sol=sol)
        lits, places = lits_reels(dr, m["sol"])
        mp = mesurer_profils(dr, lits, places)
        for pol in ("statique", "sma"):
            r = jour_j(mt, df, don, pol, g, 1.0, None, don_mesure=dr)
            out.append(dict(graine=g, chaine=chaine, politique=pol, delai_median=m["plan"]["delai_median"],
                            delai_p90=m["plan"]["delai_p90"], sans_date=m["plan"]["sans_date"],
                            plan_lits_crit1=mp["lits_crit1"], plan_lits_max=mp["lits_max"], **r))
    return out


def partie_mc(n=100):
    res = [r for rs in lancer([(g,) for g in range(n)], _mc) for r in rs]
    ecrire("mc", dict(runs=res, n=n))


if __name__ == "__main__":
    partie = sys.argv[1]
    t = time.time()
    if partie == "mc":
        partie_mc(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    else:
        {"base": partie_base, "surcharge": partie_surcharge, "aleas": partie_aleas,
         "echelle": partie_echelle}[partie]()
    print(f"fin {partie} ({time.time() - t:.0f} s)")
