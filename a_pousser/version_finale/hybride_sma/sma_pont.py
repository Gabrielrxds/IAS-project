r"""
sma_pont.py — Branche le SMA de Kyllian (sma_aleas.py, copié tel quel) sur le protocole commun.

Le fichier de Kyllian n'est pas modifié. Trois corrections sont faites ici, par sous-classes :

  1. MARGES. Le protocole cumule les marges P90 de façon quadratique (√Σ marge²), comme
     `Solution.charge`. Les agents salle additionnaient toujours les marges : avec un planning
     rempli selon le protocole, ils voyaient presque toutes les salles « en retard » et
     refusaient les urgences. `SalleQ.projeter` suit `inst.cumul_marges`.
  2. LITS FERMÉS. Kyllian les gardait dans un registre que seul le SMA lisait : la règle
     statique pouvait placer une urgence dans un lit fermé. Ici la fermeture baisse la
     capacité de la nuit dans l'instance de travail (`capacite_lits_nuit`) : les trois
     politiques la voient.
  3. MESURES. Une simulation renvoie, en plus des bilans de Kyllian, les séries de lits et
     d'admissions après aléas et l'occupation réelle de chaque salle.

Le reste (agents, Contract Net, générateur d'aléas, politiques) est celui de Kyllian.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import time
from math import sqrt

import numpy as np

import protocole  # noqa: F401  (chemins vers le modèle du groupe)
import sma_aleas as S


class SalleQ(S.AgentSalle):
    """Agent salle dont la projection P90 cumule les marges comme le modèle."""

    def projeter(self, t, file=None, p90=False):
        inst = self.env.inst
        if not p90 or getattr(inst, "cumul_marges", "somme") != "quadratique":
            return super().projeter(t, file, p90)
        file = self.file if file is None else file
        m2 = 0.0
        if self.en_cours:
            pid, deb = self.en_cours
            p = self.env.patient(pid)
            e1, e2 = deb + p.duree_op, deb + p.duree_op + p.marge_perso
            if t < e1:
                pos, m2 = e1, float(p.marge_perso) ** 2
            elif t < e2:
                pos, m2 = t, float(e2 - t) ** 2
            else:
                pos = t + self.env.P.controle_toutes
            a_prec = True
        elif self.realises:
            pos, a_prec = self.libre_a, True
        else:
            pos, a_prec = self.v.debut, False
        debuts = {}
        for pid in file:
            p = self.env.patient(pid)
            if a_prec:
                pos += inst.tis                      # TIS plein, comme Solution.charge
            s90 = pos + sqrt(m2)
            if s90 <= t:                             # la salle attend : le risque passé est soldé
                pos, m2, s90 = t, 0.0, t
            debuts[pid] = int(round(s90))
            pos += p.duree_op
            m2 += float(p.marge_perso) ** 2
            a_prec = True
        return int(round(pos + sqrt(m2))), debuts


class LitsC(S.AgentLits):
    """Lits fermés = capacité de la nuit baissée dans l'instance de travail."""

    def libres(self, j):
        return self.env.inst.cap_lits(j) - self.occupes(j)

    def sur_inform(self, m):
        if m.contenu.get("sujet") != "indisponibilite":
            return None
        cap = self.env.inst.capacite_lits_nuit
        for j in m.contenu["nuits"]:
            cap[j] = max(0, cap[j] - m.contenu["nb"])
            self.env.registre.indisponibles[j] += m.contenu["nb"]     # pour mémoire
        for j in m.contenu["nuits"]:
            if self.libres(j) < 0:
                self.envoyer(self.env.reg, "INFORM", sujet="saturation_lits", nuit=j,
                             deficit=-self.libres(j),
                             resume=f"saturation nuit J{j} : {-self.libres(j)} lit(s) manquant(s)")
        return None


# Journee construit ses agents par les noms du module : on y met les versions corrigées.
S.AgentSalle = SalleQ
S.AgentLits = LitsC


def simuler(matin: dict, generateur, politique: str, optimiseur, avec_plans_b: bool,
            jours, durees: str = "reelles", garder: set | None = None):
    """Journées consécutives sur une copie de la Solution du matin.
    Renvoie (mesures, journées gardées)."""
    P = S.ParametresSMA(politique=politique)
    work = S.copie_de_travail(matin["sol"])
    inst = work.inst
    inst.capacite_lits_nuit = [inst.capacite_lits] * inst.nb_jours
    registre = S.RegistreLits()
    reprises, bilans, gardees, salles = [], [], {}, []
    t0 = time.perf_counter()
    for j in jours:
        seq = S.plan_du_jour(work, j, matin["plans"].get(j))
        aleas = generateur.aleas_du_jour(inst, j, seq) if generateur else []
        aleas += [S.Alea(P.arrivee_reprises, "urgence", patient=p) for p in reprises]
        pb = matin["plans_b"].get(j) if avec_plans_b else None
        J = S.Journee(work, j, seq, aleas, P, optimiseur, pb, registre, durees, graine=j)
        b = J.executer()
        bilans.append(b)
        for s in J.salles.values():
            if s.realises and not s.v.urgence:
                deb = min(d for _, d, _ in s.realises)
                fin = max(f for _, _, f in s.realises)
                salles.append(dict(jour=j, tvo=s.v.tvo, opere=sum(f - d for _, d, f in s.realises),
                                   depassement=max(0, fin - s.v.fin), n=len(s.realises)))
            elif s.realises and s.v.urgence:
                salles.append(dict(jour=j, tvo=s.v.tvo, opere=sum(f - d for _, d, f in s.realises),
                                   depassement=max(0, max(f for _, _, f in s.realises) - s.v.fin),
                                   n=len(s.realises), urgence=True))
        suivant = inst.jour_ouvre_suivant(j)
        reprises = J.a_reprendre
        if reprises and suivant not in jours:
            for p in reprises:
                J.echouer(p.id)
            b.urgences_echec += len(reprises)
            b.urgences_reportees -= len(reprises)
            reprises = []
        if garder and j in garder:
            gardees[j] = J
    nuits = range(min(jours), max(jours) + 1)
    lits = [int(work.lits_jour[n] + registre.conversions[n]) for n in nuits]
    cap = [int(inst.cap_lits(n)) for n in nuits]
    mes = {"bilans": [b.en_dict() for b in bilans], "resume": S.resumer(bilans),
           "lits": lits, "cap_lits": cap, "lits_plan": [int(matin["sol"].lits_jour[n]) for n in nuits],
           "adm": [int(work.places_jour[j]) for j in jours],
           "adm_plan": [int(matin["sol"].places_jour[j]) for j in jours],
           "salles": salles, "secondes": time.perf_counter() - t0,
           "nuits_deficit": int(sum(l > c for l, c in zip(lits, cap)))}
    return mes, gardees


def optimiseur_tabou(budget: int = 300):
    """Tabou de journée du banc (même coût que l'AG journalier), borné, pour l'escalade."""
    import algos_journee as AJ
    from genetique_jour import Poids

    def f(inst, jour, seq):
        return AJ.tabou(inst, jour, seq, Poids(), budget, 0, graines=True)
    return S.adapter(f, f"tabou journée ({budget} évaluations)")


def stats_lits(x):
    x = np.asarray(x, float)
    return dict(moy=x.mean(), var=x.var(), p95=float(np.percentile(x, 95)), pic=x.max(),
                n15=int((x >= 15).sum()))
