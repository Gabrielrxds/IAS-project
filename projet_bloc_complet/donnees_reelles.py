r"""
donnees_reelles.py — Construit une instance à partir de la VRAIE base de
l'hôpital (donees_bloc_anonyme_pour_centrale_2026.xlsx), au format du
modele.py du groupe (Patient, Vacation, Instance, Solution).

PRINCIPE : « REJOUER » UNE VRAIE PÉRIODE
----------------------------------------
- Patients : ceux réellement opérés pendant la période (par défaut le 1er
  semestre 2022), avec leur chirurgien, leur type d'intervention, leurs nuits.
- Vacations : celles réellement utilisées. Chaque couple (chirurgien, jour où
  il a opéré) devient une vacation. La base ne donne pas la salle : ce n'est
  pas utile au niveau global (qui opère quel jour).
- Planning réel : chaque patient dans la vacation de son vrai jour. C'est la
  RÉFÉRENCE : ce que l'hôpital a réellement fait.
- Liberté laissée à l'algorithme : la base ne contient pas les dates de
  consultation. Hypothèse : chaque patient peut être déplacé de ± `fenetre`
  jours (30 par défaut) autour de sa vraie date, dans une vacation de SON
  chirurgien.
- Bords : les patients opérés dans les 30 jours avant / après la période sont
  chargés aussi, FIGÉS à leur vraie date, pour que les lits du début et de la
  fin de période soient réalistes.

PRÉPARATION DES DONNÉES
-----------------------
- Nuits = dates (sortie - entrée), et non la colonne « durée de séjour »,
  fausse pour ~1 900 séjours. Les nuits avant l'opération (entrée la veille)
  sont comptées. Le modèle du groupe compte les nuits à partir du jour
  opératoire : une nuit de veille est donc décalée d'un jour, mais le nombre
  total de lits-nuits est juste.
- Ambulatoire  <=>  zéro nuit (un seul critère, pas de contradiction possible).
- Durée opératoire prévue = médiane du type d'intervention ; marge_perso =
  P90 - médiane (comme dans le modèle du groupe). Heures 00:00 = inconnues.
- Chaque vacation a une durée qui contient au moins le planning réel : le
  planning de référence est donc réalisable dans le modèle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from modele import Instance, Medecin, Patient, Solution, Vacation

FICHIER = Path(__file__).with_name("donees_bloc_anonyme_pour_centrale_2026.xlsx")

# Vacances scolaires zone B (Lille) : dates incluses
VACANCES_ZONE_B = [
    ("2020-10-17", "2020-11-01"), ("2020-12-19", "2021-01-03"), ("2021-02-20", "2021-03-07"),
    ("2021-04-10", "2021-04-25"), ("2021-07-06", "2021-09-01"),
    ("2021-10-23", "2021-11-07"), ("2021-12-18", "2022-01-02"), ("2022-02-12", "2022-02-27"),
    ("2022-04-16", "2022-05-01"), ("2022-07-07", "2022-08-31"), ("2022-10-22", "2022-11-06"),
    ("2022-12-17", "2023-01-02"),
]


def _col(df, debut):
    for c in df.columns:
        if str(c).strip().lower().startswith(debut.lower()):
            return c
    raise KeyError(debut)


def _minutes(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, datetime):
        x = x.time()
    if isinstance(x, time):
        m = x.hour * 60 + x.minute
    else:
        try:
            h, mn = str(x).split(":")[:2]
            m = int(h) * 60 + int(mn)
        except ValueError:
            return None
    return None if m == 0 else m


def charger_base(chemin=FICHIER) -> pd.DataFrame:
    b = pd.read_excel(chemin)
    df = pd.DataFrame({
        "chir": b[_col(b, "Praticien")].fillna("?").astype(str).str.strip(),
        "type": b[_col(b, "Interv Type")].fillna("").astype(str).str.strip(),
        "entree": pd.to_datetime(b[_col(b, "Date Entr")]),
        "sortie": pd.to_datetime(b[_col(b, "Date Sortie")]),
        "inter": pd.to_datetime(b[_col(b, "Date Inter")]),
    })
    h_in = b[_col(b, "Heure d'entrée en salle")].map(_minutes)
    h_out = b[_col(b, "Heure de sortie de salle")].map(_minutes)
    df["h_in"], df["h_out"] = h_in, h_out
    tros = h_out - h_in
    df["tros"] = tros.where((tros > 0) & (tros < 12 * 60))
    df["nuits_pre"] = (df.inter - df.entree).dt.days.clip(lower=0, upper=3)
    df["nuits_post"] = (df.sortie - df.inter).dt.days.clip(lower=0, upper=30)
    df["nuits"] = df.nuits_pre + df.nuits_post
    return df.dropna(subset=["inter"]).reset_index(drop=True)


def feries(annee: int) -> set[date]:
    a, b, c = annee % 19, annee // 100, annee % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    p = date(annee, (h + l - 7 * m + 114) // 31, (h + l - 7 * m + 114) % 31 + 1)
    return {date(annee, 1, 1), p + timedelta(1), date(annee, 5, 1), date(annee, 5, 8), p + timedelta(39),
            p + timedelta(50), date(annee, 7, 14), date(annee, 8, 15), date(annee, 11, 1),
            date(annee, 11, 11), date(annee, 12, 25)}


@dataclass
class Rejeu:
    """Instance + planning réel + informations nécessaires aux études."""
    inst: Instance
    reel: Solution                        # le planning réellement appliqué
    mobiles: list[int]                    # patients que l'algorithme peut déplacer
    jour_max: dict[int, int]              # borne haute de chaque patient mobile
    jour_reel: dict[int, int]
    eval_debut: int                       # nuits évaluées : [eval_debut, eval_fin)
    eval_fin: int
    periode_reduite: set[int] = field(default_factory=set)   # vacances scolaires, fériés, ponts
    codes: dict[int, str] = field(default_factory=dict)      # med_id -> code chirurgien


def construire_rejeu(debut: str = "2022-01-03", semaines: int = 25, fenetre: int = 30,
                     base: pd.DataFrame | None = None, marge_bord: int = 30) -> Rejeu:
    df = base if base is not None else charger_base()
    d0 = date.fromisoformat(debut)
    d0 -= timedelta(days=d0.weekday())                     # un lundi
    jour_zero = d0 - timedelta(days=marge_bord)
    nb_jours = marge_bord + 7 * semaines + marge_bord
    fin = jour_zero + timedelta(days=nb_jours)

    # durées et marges par type (sur toute la base)
    ok = df.dropna(subset=["tros"])
    stats = {t: (g.tros.median(), g.tros.quantile(0.9)) for t, g in ok.groupby("type") if len(g) >= 5}
    glob = (ok.tros.median(), ok.tros.quantile(0.9))

    sel = df[(df.inter.dt.date >= jour_zero) & (df.inter.dt.date < fin)].copy()
    sel["jour"] = (sel.inter.dt.date - jour_zero).apply(lambda x: x.days)
    codes = sorted(sel.chir.unique())
    num = {c: i for i, c in enumerate(codes)}

    inst = Instance(jour_zero=jour_zero, nb_jours=nb_jours)
    for c in codes:
        inst.ajouter_medecin(Medecin(num[c], nom=c))

    # patients
    jour_reel, jour_max, mobiles = {}, {}, []
    eval_debut, eval_fin = marge_bord, marge_bord + 7 * semaines
    for pid, r in enumerate(sel.itertuples()):
        med, p90 = stats.get(r.type, glob)
        mobile = eval_debut <= r.jour < eval_fin
        jmin = max(0, r.jour - fenetre) if mobile else r.jour
        inst.ajouter_patient(Patient(
            id=pid, med_id=num[r.chir], duree_op=int(round(med)), marge_perso=int(round(max(0, p90 - med))),
            duree_sejour=int(r.nuits) + 1, ambulatoire=(r.nuits == 0),
            jour_demande=jmin - 7, fenetre_jours=7, type_interv=r.type,
            duree_reelle=int(r.tros) if r.tros == r.tros else 0,
            delai_max=(min(nb_jours - 1, r.jour + fenetre) - (jmin - 7)) if mobile else 7))
        jour_reel[pid] = r.jour
        jour_max[pid] = min(nb_jours - 1, r.jour + fenetre) if mobile else r.jour
        if mobile:
            mobiles.append(pid)

    # vacations réelles : (chirurgien, jour) ; durée >= planning réel
    vid = 0
    vac_de: dict[tuple[int, int], int] = {}
    for (c, j), g in sel.groupby(["chir", "jour"]):
        pats = [inst.patients[pid] for pid in g.index.map(lambda i: sel.index.get_loc(i))]
        besoin = sum(p.duree_op + p.marge_perso for p in pats) + inst.tis * (len(pats) - 1)
        h0 = g.h_in.dropna().min()
        h1 = g.h_out.dropna().max()
        debut_v = int(min(8 * 60, (h0 // 30) * 30)) if h0 == h0 else 8 * 60
        fin_obs = int(math.ceil(h1 / 30) * 30) if h1 == h1 else debut_v
        fin_v = max(debut_v + 4 * 60, fin_obs, debut_v + int(math.ceil(besoin / 30) * 30))
        inst.ajouter_vacation(Vacation(vid, bloc_id=0, med_id=num[c], jour=int(j), debut=debut_v, fin=fin_v,
                                       etiquette=f"{c} {jour_zero + timedelta(days=int(j))}"))
        vac_de[(num[c], int(j))] = vid
        vid += 1
    inst.indexer()

    # capacités : au moins ce que l'hôpital a réellement utilisé
    reel = Solution(inst)
    for pid in inst.patients:
        reel.affecter(pid, vac_de[(inst.patients[pid].med_id, jour_reel[pid])])
    inst.capacite_lits = max(inst.capacite_lits, max(reel.lits_jour))
    inst.capacite_places_jour = max(inst.capacite_places_jour, max(reel.places_jour))

    return Rejeu(inst, reel, mobiles, jour_max, jour_reel, eval_debut, eval_fin,
                 periodes_reduites(jour_zero, nb_jours), {num[c]: c for c in codes})


def jours_feries_et_ponts(debut: date, fin: date) -> set[date]:
    fer = set().union(*(feries(a) for a in range(debut.year, fin.year + 1)))
    for f in list(fer):
        if f.weekday() == 1:
            fer.add(f - timedelta(1))
        elif f.weekday() == 3:
            fer.add(f + timedelta(1))
    return fer


def periodes_reduites(jour_zero: date, nb_jours: int) -> set[int]:
    """Index des jours de vacances scolaires (zone B), fériés et ponts."""
    fer = jours_feries_et_ponts(jour_zero, jour_zero + timedelta(days=nb_jours))
    red = set()
    for j in range(nb_jours):
        d = jour_zero + timedelta(days=j)
        if d in fer or any(date.fromisoformat(a) <= d <= date.fromisoformat(b) for a, b in VACANCES_ZONE_B):
            red.add(j)
    return red


def construire_rejeu_grille(grille, base: pd.DataFrame, debut: str = "2022-01-03", semaines: int = 52,
                            fenetre: int = 30, marge_bord: int = 30, facteur_marge: float = 0.5) -> Rejeu:
    """Rejoue les VRAIS patients d'une période dans une GRILLE de vacations
    (grille_vacations.Grille) au lieu des vacations réellement utilisées.

    Chaque patient doit être opéré par son chirurgien, dans un créneau de la
    grille, à ± `fenetre` jours de sa vraie date. Le planning de départ est
    construit « comme à l'hôpital » : chaque patient, par ordre de date, dans
    le créneau réalisable le plus proche de sa vraie date. Un patient qui ne
    tient nulle part reste sans créneau (compté dans les indicateurs).

    facteur_marge : la marge du modèle (P90 - médiane, ADDITIONNÉE patient par
    patient) suppose que tous les actes d'une vacation dérapent en même temps.
    Pour rejouer une année réelle on n'en garde que la moitié par défaut ;
    avec 1.0, la grille actuelle ne peut plus absorber l'activité 2022."""
    d0 = date.fromisoformat(debut)
    d0 -= timedelta(days=d0.weekday())
    jour_zero = d0 - timedelta(days=marge_bord)
    nb_jours = marge_bord + 7 * semaines + marge_bord
    fin = jour_zero + timedelta(days=nb_jours)
    chirs = sorted(grille.heures_par_semaine())
    num = {c: i for i, c in enumerate(chirs)}
    ok = base.dropna(subset=["tros"])
    stats = {t: (g.tros.median(), g.tros.quantile(0.9)) for t, g in ok.groupby("type") if len(g) >= 5}
    glob = (ok.tros.median(), ok.tros.quantile(0.9))

    inst = Instance(jour_zero=jour_zero, nb_jours=nb_jours)
    for c in chirs:
        inst.ajouter_medecin(Medecin(num[c], nom=c))
    fer = jours_feries_et_ponts(jour_zero, fin)
    fer_seuls = set().union(*(feries(a) for a in range(jour_zero.year, fin.year + 1)))
    vid = 0
    for j in range(nb_jours):
        d = jour_zero + timedelta(days=j)
        if d in fer_seuls:
            continue
        for c, salle, deb, f in grille.creneaux_du(d):
            inst.ajouter_vacation(Vacation(vid, bloc_id=salle, med_id=num[c], jour=j, debut=deb, fin=f,
                                           etiquette=f"{c} S{salle} {d}"))
            vid += 1

    sel = base[(base.inter.dt.date >= jour_zero) & (base.inter.dt.date < fin) & base.chir.isin(chirs)].copy()
    sel["jour"] = (sel.inter.dt.date - jour_zero).apply(lambda x: x.days)
    sel = sel.sort_values("jour")
    jour_reel, jour_max, mobiles = {}, {}, []
    for pid, r in enumerate(sel.itertuples()):
        med, p90 = stats.get(r.type, glob)
        jmin = max(0, r.jour - fenetre)
        jmx = min(nb_jours - 1, r.jour + fenetre)
        inst.ajouter_patient(Patient(
            id=pid, med_id=num[r.chir], duree_op=int(round(med)),
            marge_perso=int(round(facteur_marge * max(0, p90 - med))),
            duree_sejour=int(r.nuits) + 1, ambulatoire=(r.nuits == 0), jour_demande=jmin - 7, fenetre_jours=7,
            type_interv=r.type, delai_max=jmx - (jmin - 7)))
        jour_reel[pid], jour_max[pid] = r.jour, jmx
        mobiles.append(pid)
    inst.indexer()

    # planning de départ : créneau réalisable le plus proche de la vraie date
    sol = Solution(inst)
    for pid in mobiles:
        p = inst.patients[pid]
        cands = [v for v in inst.vacations_possibles(pid) if inst.vacations[v].jour <= jour_max[pid]]
        cands.sort(key=lambda v: abs(inst.vacations[v].jour - jour_reel[pid]))
        for v in cands:
            sol.affecter(pid, v)
            j = inst.vacations[v].jour
            nuits = range(j, min(j + p.nb_nuits, nb_jours))
            if (sol.depassement(v) == 0 and all(sol.lits_jour[k] <= inst.capacite_lits for k in nuits)
                    and sol.places_jour[j] <= inst.capacite_places_jour):
                break
            sol.affecter(pid, None)
    return Rejeu(inst, sol, mobiles, jour_max, jour_reel, marge_bord, marge_bord + 7 * semaines,
                 periodes_reduites(jour_zero, nb_jours), {num[c]: c for c in chirs})
