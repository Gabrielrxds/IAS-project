r"""
scenarios_aleas.py — Générateurs d'aléas pour la simulation (une graine = un scénario rejouable).

Les fréquences sont des HYPOTHÈSES de travail, sauf les congés (déduits des semaines sans
activité de 2022) : la base ne contient ni pannes, ni fermetures de lits, ni annulations. Elles
sont réunies dans FREQUENCES pour être changées facilement (cf. GESTION_ALEAS.md §4).

Chaque indisponibilité a une durée ESTIMÉE (saisie par le déclarant) et une durée RÉELLE
= estimée × facteur aléatoire. Réelle plus longue : un aléa de PROLONGATION est déclaré à la fin
estimée. Réelle plus courte d'au moins un jour : une LEVÉE est déclarée le jour de la fin réelle.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import random
from collections import Counter, defaultdict

from gestion_aleas import (JOUR, IndispoMedecin, IndispoSalle, IndispoMateriel, FermetureLits, Levee)

FREQUENCES = {
    "maladie": 0.01,              # par journée opératoire d'un chirurgien
    "medecin_temporaire": 0.02,   # par journée opératoire d'un chirurgien (retard, urgence ailleurs…)
    "panne_salle": 0.005,         # par salle et par jour ouvré
    "maintenance_par_an": 2,      # par salle
    "materiel_par_mois": 1,
    "lits_par_mois": 0.5,
}


def _jours_ouvres(inst):
    return sorted({v.jour for v in inst.vacations.values() if not v.urgence})


def _avec_duree_reelle(rng, a, facteur=(0.5, 2.0)):
    """Renvoie [a] + éventuelle prolongation ou levée, selon la durée réelle tirée."""
    est = a.fin - a.debut
    reel = round(est * rng.uniform(*facteur))
    out = [a]
    if reel > est + 15:
        fin_est_jour = (a.fin - 1) // JOUR
        prol = type(a)(**{**a.__dict__, "id": -1, "declaration": fin_est_jour, "debut": a.fin,
                          "fin": a.debut + reel, "raison": "prolongation", "duree_estimee": f"{reel - est} min"})
        out.append(prol)
    elif est >= JOUR and est - reel >= JOUR:
        out.append(Levee(declaration=max(a.declaration, (a.debut + reel) // JOUR), alea=a))
    return out


def medecin_absences(don, inst, preavis=42, taux_maladie=None, graine=0, conges=True):
    """Congés (semaines sans activité en 2022) et arrêts maladie, au format IndispoMedecin."""
    from absences import absences_reelles, absences_maladie
    taux = FREQUENCES["maladie"] if taux_maladie is None else taux_maladie
    out = []
    if conges:
        for a in absences_reelles(don, preavis):
            out.append(IndispoMedecin(a.declaration, "conge", duree_estimee=f"{a.fin - a.debut + 1} j",
                                      debut=a.debut * JOUR, fin=(a.fin + 1) * JOUR, med_id=a.med_id))
    if taux:
        for a in absences_maladie(inst, taux, graine):
            out.append(IndispoMedecin(a.declaration, "maladie", duree_estimee=f"{a.fin - a.debut + 1} j",
                                      debut=a.debut * JOUR, fin=(a.fin + 1) * JOUR, med_id=a.med_id))
    return out


def medecin_temporaires(inst, taux=None, graine=0):
    """Indisponibilités de quelques heures, déclarées le jour même ou la veille : retard, urgence
    dans un autre service, raison personnelle. Durée estimée 1 à 4 h."""
    taux = FREQUENCES["medecin_temporaire"] if taux is None else taux
    rng = random.Random(100 + graine)
    debut_jour = defaultdict(lambda: 24 * 60)
    for v in inst.vacations.values():
        if not v.urgence:
            debut_jour[(v.med_id, v.jour)] = min(debut_jour[(v.med_id, v.jour)], v.debut)
    out = []
    for (m, j), h in sorted(debut_jour.items()):
        if rng.random() >= taux:
            continue
        motif = rng.choice(("urgence_autre_service", "retard", "personnel", "autre"))
        est = rng.choice((60, 90, 120, 180, 240))
        h0 = h + rng.choice((0, 0, 30, 60))
        a = IndispoMedecin(j - rng.choice((0, 0, 1)), motif, raison=f"{motif} (simulé)", duree_estimee=f"{est} min",
                           debut=j * JOUR + h0, fin=j * JOUR + h0 + est, med_id=m)
        out += _avec_duree_reelle(rng, a, (0.6, 1.6))
    return out


def salles(inst, taux_panne=None, maintenance_par_an=None, graine=0, salles=(2, 3, 4, 5)):
    """Pannes (déclarées le jour même, quelques heures à quelques jours) et maintenances
    programmées (1 à 2 jours, déclarées 21 jours avant)."""
    taux = FREQUENCES["panne_salle"] if taux_panne is None else taux_panne
    nmaint = FREQUENCES["maintenance_par_an"] if maintenance_par_an is None else maintenance_par_an
    rng = random.Random(200 + graine)
    jours = _jours_ouvres(inst)
    out = []
    for j in jours:
        for s in salles:
            if rng.random() >= taux:
                continue
            motif = rng.choice(("panne_materiel", "panne_materiel", "sterilisation", "degat_locaux", "personnel_manquant"))
            h = rng.randint(8 * 60, 14 * 60)
            if rng.random() < 0.7:
                est = rng.choice((120, 180, 240))
                a = IndispoSalle(j, motif, duree_estimee=f"{est} min", debut=j * JOUR + h, fin=j * JOUR + h + est, salle=s)
            else:
                d = rng.choice((1, 2, 3))
                a = IndispoSalle(j, motif, duree_estimee=f"{d} j", debut=j * JOUR + h, fin=(j + d) * JOUR, salle=s)
            out += _avec_duree_reelle(rng, a)
    n = round(nmaint * len(jours) / 250)
    for s in salles:
        for _ in range(n):
            j = rng.choice(jours[21:]) if len(jours) > 21 else rng.choice(jours)
            d = rng.choice((1, 2))
            out.append(IndispoSalle(max(0, j - 21), "maintenance", duree_estimee=f"{d} j",
                                    debut=j * JOUR, fin=(j + d) * JOUR, salle=s))
    return out


def materiel(inst, par_mois=None, graine=0, top=20):
    """Un matériel manque pour un type d'intervention fréquent, 1 à 5 jours, déclaré 0 à 3 jours
    avant."""
    par_mois = FREQUENCES["materiel_par_mois"] if par_mois is None else par_mois
    rng = random.Random(300 + graine)
    types = [t for t, _ in Counter(p.type_interv for p in inst.patients.values() if p.type_interv).most_common(top)]
    jours = _jours_ouvres(inst)
    out = []
    for _ in range(round(par_mois * inst.nb_jours / 30)):
        j = rng.choice(jours)
        d = rng.randint(1, 5)
        t = rng.choice(types)
        out.append(IndispoMateriel(max(0, j - rng.randint(0, 3)),
                                   rng.choice(("ancillaire_indisponible", "rupture_implant", "colonne_endoscopie")),
                                   raison=t, duree_estimee=f"{d} j", debut=j * JOUR, fin=(j + d) * JOUR,
                                   types=frozenset({t})))
    return out


def lits(inst, par_mois=None, graine=0):
    """Fermeture de 2 à 6 lits pendant 7 à 21 nuits, déclarée 2 à 10 jours avant."""
    par_mois = FREQUENCES["lits_par_mois"] if par_mois is None else par_mois
    rng = random.Random(400 + graine)
    out = []
    for _ in range(round(par_mois * inst.nb_jours / 30)):
        j = rng.randint(10, inst.nb_jours - 1)
        d = rng.randint(7, 21)
        out.append(FermetureLits(j - rng.randint(2, 10), rng.choice(("epidemie", "travaux", "manque_personnel")),
                                 duree_estimee=f"{d} j", debut=j * JOUR, fin=(j + d) * JOUR, nb=rng.randint(2, 6)))
    return out
