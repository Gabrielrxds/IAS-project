r"""
rapport_html.py — Génère le rapport HTML « Nouvelle grille et fonction coût » à
partir des résultats de etude_couts.py.

Les graphiques et les tableaux sont recalculés à partir des données. Le TEXTE
du rapport (verdict, commentaires) est écrit dans `rapport_gabarit.html` pour
l'exécution complète de référence (mode complet, graine 0) : si vous changez
les réglages, relisez-le.

En plus des résultats de l'étude, le rapport utilise deux simulations de
contrôle (même graine) pour séparer l'effet de la grille de celui de la règle
d'insertion :
    grille actuelle   + CoutTotal ×1
    nouvelle grille   + premier créneau libre
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyse_couts import Config, construire_instance, simuler, vacations_grille_actuelle

GABARIT = Path(__file__).with_name("rapport_gabarit.html")


def _st(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"moyenne": x.mean(), "mediane": float(np.median(x)), "variance": x.var(),
            "ecart_type": x.std(), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "min": x.min(), "max": x.max()}


def _conv(o):
    """Rend l'objet sérialisable en JSON (numpy, NaN, clés non textuelles)."""
    if isinstance(o, dict):
        return {str(k): _conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_conv(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    return o


class _Mesures:
    """Mesures sur la période de test (toutes les nuits pour les lits, jours
    ouvrés pour les places)."""

    def __init__(self, don):
        self.don = don
        self.t0, self.t1 = don.test
        self.jours = list(range(self.t0, self.t1 + 1))
        self.ouvres = don.jours_ouvres(self.t0, self.t1)
        self.ouvres_an = set(don.jours_ouvres(0, don.fin_annee))
        self.nb_sem = (don.fin_annee + 1) // 7

    def lits(self, lits) -> dict:
        lits, don = np.asarray(lits), self.don
        v = lits[self.jours]
        jds = {d: float(np.mean([lits[j] for j in self.jours if don.date(j).weekday() == d]))
               for d in range(7)}
        return {"stats": _st(v), "stats_ouvres": _st(lits[self.ouvres]), "jds": jds,
                "hist": np.bincount(v.astype(int), minlength=31)[:31].tolist(),
                "sem": [float(lits[w * 7:(w + 1) * 7].mean()) for w in range(self.nb_sem)]}

    def places(self, pl) -> dict:
        pl, don = np.asarray(pl), self.don
        jds = {d: float(np.mean([pl[j] for j in self.ouvres if don.date(j).weekday() == d]))
               for d in range(5)}
        return {"stats": _st(pl[self.ouvres]), "jds": jds,
                "hist": np.bincount(pl[self.ouvres].astype(int), minlength=21)[:21].tolist()}

    def planning(self, sol) -> dict:
        """Remplissage, patients par semaine, délai et patients sans date."""
        don, inst, t0, t1 = self.don, sol.inst, self.t0, self.t1
        vacs = [v for v, x in inst.vacations.items() if not x.urgence and t0 <= x.jour <= t1]
        sem = defaultdict(lambda: {"tvo": 0, "charge": 0, "pat": 0, "ambu": 0})
        taux = []
        for v in vacs:
            x = inst.vacations[v]
            sem[x.jour // 7]["tvo"] += x.tvo
            if sol.nb[v]:
                sem[x.jour // 7]["charge"] += min(sol.charge(v), inst.capacite_programme(v))
                taux.append(sol.taux_remplissage(v))
        cohorte = [p for p, j in don.jour_reel.items() if t0 <= j <= t1]
        delais, decal, sans = [], [], Counter()
        for p in cohorte:
            v = sol.affectation[p]
            if v is None:
                sans[don.med_codes[inst.patients[p].med_id]] += 1
                continue
            delais.append(inst.vacations[v].jour - inst.patients[p].jour_demande)
            decal.append(inst.vacations[v].jour - don.jour_reel[p])
        for p, v in sol.affectation.items():
            if v is not None and t0 <= inst.vacations[v].jour <= t1:
                w = inst.vacations[v].jour // 7
                sem[w]["pat"] += 1
                sem[w]["ambu"] += inst.patients[p].ambulatoire
        ws = sorted(w for w in sem if w * 7 >= t0 and w * 7 + 6 <= t1)   # semaines complètes
        taux = np.array(taux)
        salles_j = len({(inst.vacations[v].jour, inst.vacations[v].bloc_id) for v in vacs if sol.nb[v]})
        nsem = len({inst.vacations[v].jour // 7 for v in vacs})
        return {
            "patients_semaine": _st([sem[w]["pat"] for w in ws]),
            "ambu_semaine": float(np.mean([sem[w]["ambu"] for w in ws])),
            "hosp_semaine": float(np.mean([sem[w]["pat"] - sem[w]["ambu"] for w in ws])),
            "heures_offertes": float(np.mean([sem[w]["tvo"] for w in ws]) / 60),
            "heures_engagees": float(np.mean([sem[w]["charge"] for w in ws]) / 60),
            "remplissage_semaine": _st([sem[w]["charge"] / sem[w]["tvo"] for w in ws if sem[w]["tvo"]]),
            "vacations": len(vacs), "vacations_vides": len(vacs) - len(taux),
            "sup85": float((taux > .85).mean()) if len(taux) else 0.0,
            "sous50": float((taux < .5).mean()) if len(taux) else 0.0,
            "salles_jours_semaine": salles_j / max(1, nsem),
            "dates": len(delais), "sans_date": sum(sans.values()), "sans_par": dict(sans),
            "delai_median": float(np.median(delais)), "delai_p90": float(np.percentile(delais, 90)),
            "delai_moyen": float(np.mean(delais)), "decalage": float(np.mean(decal)),
            "depassement_h": sol.depassement_total() / 60, "conflits": len(sol.verifier())}

    def nuits_par_jour(self, affectations) -> dict:
        """Part des nuits d'hospitalisation générées par les opérations de
        chaque jour de semaine. `affectations` : [(jour, patient)]."""
        c = Counter()
        for j, p in affectations:
            if self.t0 <= j <= self.t1:
                c[self.don.date(j).weekday()] += p.nb_nuits
        tot = sum(c.values()) or 1
        return {d: c[d] / tot for d in range(7)}

    def complet(self, sol) -> dict:
        inst = sol.inst
        aff = [(inst.vacations[v].jour, inst.patients[p]) for p, v in sol.affectation.items() if v is not None]
        return {"lits": self.lits(sol.lits_jour), "places": self.places(sol.places_jour),
                "planning": self.planning(sol), "nuits_jour": self.nuits_par_jour(aff)}


def _heures_jours(don, inst):
    """Heures offertes par semaine et jours de la semaine, par médecin."""
    h, j = defaultdict(float), defaultdict(set)
    for v in inst.vacations.values():
        if not v.urgence and v.jour <= don.fin_annee:
            m = don.med_codes[v.med_id]
            h[m] += v.tvo / 60
            j[m].add(don.date(v.jour).weekday())
    nsem = len(don.jours_ouvres(0, don.fin_annee)) / 5
    return {m: h[m] / nsem for m in don.med_codes.values()}, {m: sorted(s) for m, s in j.items()}


def donnees_rapport(etat: dict) -> dict:
    """`etat` : don, REG, REF, PROB, LITS_REEL, PLACES_REEL, sol_actuelle,
    RESULTATS (le premier est la référence), RESUME_JOUR, DF_JOUR, MULTS."""
    don, REG, PROB, RES = etat["don"], etat["REG"], etat["PROB"], etat["RESULTATS"]
    M = _Mesures(don)
    base, ref = etat["sol_actuelle"], RES[0]
    par_nom = {r.config.nom: r for r in RES}

    # simulations de contrôle (grille / règle d'insertion)
    inst_b = construire_instance(don, vacations_grille_actuelle(don), REG)
    sol_b = simuler(inst_b, Config().cout(etat["REF"]), REG)
    inst_c = construire_instance(don, PROB.derouler(ref.grille.X, don.jour_zero, don.nb_jours,
                                                    don.feries), REG)
    sol_c = simuler(inst_c, None, REG)

    reel_aff = [(j, don.patients[p]) for p, j in don.jour_reel.items()]
    P = {"reel": {"lits": M.lits(etat["LITS_REEL"]), "places": M.places(etat["PLACES_REEL"]),
                  "nuits_jour": M.nuits_par_jour(reel_aff)},
         "actuelle": M.complet(base), "ag": M.complet(ref.sol),
         "actuelle_cout": M.complet(sol_b), "ag_premier": M.complet(sol_c)}
    if "Poids lits ×0" in par_nom:
        P["sans_lits"] = M.complet(par_nom["Poids lits ×0"].sol)

    VAR = []
    for r in RES:
        lits, pl = np.asarray(r.sol.lits_jour), np.asarray(r.sol.places_jour)
        pg = M.planning(r.sol)
        d = {"nom": r.config.nom, "coef": r.config.coef_varie or "reference",
             "mult": r.config.multiplicateur, "lits": _st(lits[M.jours]),
             "lits_ouvres": _st(lits[M.ouvres]), "places": _st(pl[M.ouvres]),
             "remplissage": pg["remplissage_semaine"]["moyenne"], "sans_date": pg["sans_date"],
             "delai_median": pg["delai_median"], "delai_p90": pg["delai_p90"],
             "salles_jours": pg["salles_jours_semaine"], "depassement_h": pg["depassement_h"],
             "conflits": pg["conflits"], "termes": r.grille.termes_normalises,
             "generations": r.grille.generations, "duree": r.duree_s,
             "convergence": [[g, c] for g, c, _ in r.grille.convergence]}
        if r.journalier is not None:
            d["pic_simultane"] = float(r.journalier["pic_places"].mean())
            d["creux_jour"] = float(r.journalier["creux_min"].mean())
            d["dep_reel"] = float(r.journalier["depassement_reel_min"].mean())
        VAR.append(d)

    X = ref.grille.X
    G = {"C": PROB.C, "salles": PROB.salles, "codes": [p.code for p in PROB.profils],
         "X": [[[[int(X[k, d, r, h]) for h in range(2)] for r in range(PROB.R)]
                for d in range(5)] for k in range(PROB.C)]}
    h_act, j_act = _heures_jours(don, base.inst)
    h_new, j_new = _heures_jours(don, ref.inst)
    tis = base.inst.tis
    besoin_test = defaultdict(float)
    for p in don.patients_de(M.t0, M.t1):
        besoin_test[don.med_codes[p.med_id]] += (p.duree_op + p.marge_perso + tis) / 60
    nsem_test = (M.t1 - M.t0 + 1) / 7
    BES = [{"m": p.code, "patients_app": p.nb_patients, "h_app": p.minutes_semaine / 60,
            "h_test": besoin_test[p.code] / nsem_test, "quota": int(PROB.quotas[s]),
            "h_act": h_act.get(p.code, 0), "h_new": h_new.get(p.code, 0),
            "j_act": j_act.get(p.code, []), "j_new": j_new.get(p.code, [])}
           for s, p in enumerate(PROB.profils)]

    RJ = etat["RESUME_JOUR"]
    JOUR = {"lignes": [{"nom": i, **{c: (None if isinstance(v, float) and np.isnan(v) else float(v))
                                     for c, v in RJ.loc[i].items()}} for i in RJ.index],
            "nb_jours": int(etat["DF_JOUR"]["jour"].nunique()),
            "patients_jour": float(etat["DF_JOUR"].groupby("jour")["patients"].first().mean())}

    meta = {"test": [str(don.date(M.t0)), str(don.date(M.t1))],
            "app": [str(don.date(don.app[0])), str(don.date(don.app[1]))],
            "sem_test": [M.t0 // 7, M.t1 // 7], "sem_app": [don.app[0] // 7, don.app[1] // 7],
            "nb_sem": M.nb_sem, "patients": len(don.patients),
            "patients_test": len(don.patients_de(M.t0, M.t1)), "exclus": list(don.exclus),
            "cases_libres": int((~PROB.interdit).sum()), "cases_total": int(PROB.interdit.size),
            "heures_cycle_new": float(sum(h_new.values()) * PROB.C),
            "reg": vars(REG) | {"salles": list(REG.salles)}, "ref_cout": etat["REF"],
            "mults": list(etat["MULTS"]), "alertes": PROB.alertes}
    return _conv({"P": P, "VAR": VAR, "G": G, "BES": BES, "JOUR": JOUR, "meta": meta})


def generer_rapport(etat: dict, chemin: str = "Rapport_grille_AG.html") -> str:
    data = json.dumps(donnees_rapport(etat), ensure_ascii=False)
    if "</script" in data:
        raise ValueError("données incompatibles avec une balise <script>")
    html = GABARIT.read_text(encoding="utf-8").replace("__DATA__", data)
    Path(chemin).write_text(html, encoding="utf-8")
    return chemin
