# modele.py — VERSION ALLÉGÉE du modele.py du groupe.
# Mêmes classes (Patient, Medecin, Vacation, Instance, Solution) et mêmes
# fonctions coût (cout_lits, cout_places, cout_remplissage, cout_delai,
# CoutTotal), avec exactement les mêmes formules : c'est tout ce dont les
# études du recuit ont besoin. Les parties non utilisées ici (urgences,
# planning d'une journée, affichage) ont été retirées.
# Si vous préférez le fichier complet du groupe, remplacez simplement ce
# fichier par le vôtre : les codes fonctionnent avec les deux.
from __future__ import annotations
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

PROGRAMME = ""; SEMI_URGENCE = "semi"; URGENCE_JOUR = "jour"; MED_URGENCE = -1

@dataclass(slots=True)
class Patient:
    id: int
    med_id: int
    duree_op: int = 60
    duree_sejour: int = 1
    ambulatoire: bool = True
    marge_perso: int = 10
    jour_demande: int = 0
    fenetre_jours: int = 7
    priorite: float = 1.0
    type_interv: str = ""
    diag: str = ""
    sexe: str = "F"
    age: int = 50
    gestes: tuple = ()
    duree_reelle: int = 0
    urgence: str = PROGRAMME
    delai_max: int | None = None
    @property
    def est_urgent(self): return self.urgence != PROGRAMME
    @property
    def jour_min(self): return self.jour_demande + self.fenetre_jours
    @property
    def jour_max(self): return None if self.delai_max is None else self.jour_demande + self.delai_max
    @property
    def nb_nuits(self): return max(0, self.duree_sejour - 1)

@dataclass(slots=True)
class Medecin:
    id: int
    nom: str = ""
    specialite: str = ""

@dataclass(slots=True)
class Vacation:
    id: int
    bloc_id: int
    med_id: int
    jour: int
    debut: int
    fin: int
    etiquette: str = ""
    urgence: bool = False
    @property
    def tvo(self): return self.fin - self.debut

@dataclass
class Instance:
    patients: dict = field(default_factory=dict)
    medecins: dict = field(default_factory=dict)
    vacations: dict = field(default_factory=dict)
    capacite_lits: int = 42
    capacite_places: int = 12
    capacite_places_jour: int = 18
    tis: int = 15
    tis_meme_acte: int = 10
    fermeture_uca: int = 20 * 60
    surveillance_ambu: int = 180
    heure_sortie_hospit: int = 10 * 60
    tampon_urgence: float = 0.0
    reserve_lits: int = 0
    reserve_places: int = 0
    jour_zero: date = date(2026, 1, 5)
    nb_jours: int = 182
    vacations_du_medecin: dict = field(default_factory=dict)
    vacations_du_jour: dict = field(default_factory=dict)
    jours_ouvres: list = field(default_factory=list)
    nb_vacations_medecin: dict = field(default_factory=dict)
    vacations_urgence_du_jour: dict = field(default_factory=dict)
    def ajouter_patient(self, p): self.patients[p.id] = p
    def ajouter_medecin(self, m): self.medecins[m.id] = m
    def ajouter_vacation(self, v): self.vacations[v.id] = v
    def indexer(self):
        self.vacations_du_medecin = defaultdict(list); self.vacations_du_jour = defaultdict(list)
        self.vacations_urgence_du_jour = defaultdict(list)
        for v in self.vacations.values():
            self.vacations_du_jour[v.jour].append(v.id)
            if v.urgence: self.vacations_urgence_du_jour[v.jour].append(v.id)
            else: self.vacations_du_medecin[v.med_id].append(v.id)
        for l in self.vacations_du_medecin.values(): l.sort(key=lambda vid: self.vacations[vid].jour)
        self.jours_ouvres = sorted(self.vacations_du_jour)
        self.nb_vacations_medecin = {m: len(v) for m, v in self.vacations_du_medecin.items()}
        return self
    def capacite_utile(self, vid): return self.vacations[vid].tvo
    def capacite_programme(self, vid):
        v = self.vacations[vid]
        if v.urgence: return 0
        return v.tvo - round(self.tampon_urgence * v.tvo)
    def date_du_jour(self, j): return self.jour_zero + timedelta(days=j)
    def est_weekend(self, j): return self.date_du_jour(j).weekday() >= 5
    def vacations_possibles(self, pid, fenetre=None):
        p = self.patients[pid]
        jmin = p.jour_demande + (p.fenetre_jours if fenetre is None else fenetre)
        return [vid for vid in self.vacations_du_medecin.get(p.med_id, ()) if self.vacations[vid].jour >= jmin]

class Solution:
    def __init__(self, inst):
        self.inst = inst
        self.affectation = {pid: None for pid in inst.patients}
        self.nb = {v: 0 for v in inst.vacations}; self.somme_durees = {v: 0 for v in inst.vacations}
        self.somme_marges = {v: 0 for v in inst.vacations}
        self.places_jour = [0] * inst.nb_jours; self.lits_jour = [0] * inst.nb_jours
        self.s_places = self.s_places2 = 0; self.s_lits = self.s_lits2 = 0
        self.nb_urg = {v: 0 for v in inst.vacations}; self.somme_urg = {v: 0 for v in inst.vacations}
        self.places_urg_jour = [0] * inst.nb_jours; self.lits_urg_jour = [0] * inst.nb_jours
        self.s_taux = {m: 0.0 for m in inst.vacations_du_medecin}; self.s_taux2 = {m: 0.0 for m in inst.vacations_du_medecin}
        self.sans_date = set(inst.patients); self.hors_horizon = set()
        self._jours_places = inst.jours_ouvres or list(range(inst.nb_jours)); self._n_jours_lits = inst.nb_jours
    def charge(self, vid):
        n = self.nb[vid]
        if n == 0: return 0
        return self.somme_durees[vid] + self.somme_marges[vid] + self.inst.tis * (n - 1)
    def charge_programme(self, vid):
        n = self.nb[vid] - self.nb_urg[vid]
        if n <= 0: return 0
        return self.somme_durees[vid] + self.somme_marges[vid] - self.somme_urg[vid] + self.inst.tis * (n - 1)
    def depassement_programme(self, vid): return max(0, self.charge_programme(vid) - self.inst.capacite_programme(vid))
    def creux(self, vid): return max(0, self.inst.capacite_utile(vid) - self.charge(vid))
    def depassement(self, vid): return max(0, self.charge(vid) - self.inst.capacite_utile(vid))
    def taux_remplissage(self, vid):
        cap = self.inst.capacite_programme(vid) or self.inst.vacations[vid].tvo
        return self.charge(vid) / cap if cap else 0.0
    def patients_de(self, vid): return [p for p, v in self.affectation.items() if v == vid]
    def _appliquer_patient(self, pid, vid, signe):
        if vid is None: return
        inst = self.inst; p = inst.patients[pid]; v = inst.vacations[vid]
        self.nb[vid] += signe; self.somme_durees[vid] += signe * p.duree_op; self.somme_marges[vid] += signe * p.marge_perso
        if p.urgence != PROGRAMME:
            self.nb_urg[vid] += signe; self.somme_urg[vid] += signe * (p.duree_op + p.marge_perso)
        if p.ambulatoire:
            j = v.jour; a = self.places_jour[j]; b = a + signe; self.places_jour[j] = b
            self.s_places += b - a; self.s_places2 += b * b - a * a
        else:
            for j in range(v.jour, min(v.jour + p.nb_nuits, inst.nb_jours)):
                a = self.lits_jour[j]; b = a + signe; self.lits_jour[j] = b
                self.s_lits += b - a; self.s_lits2 += b * b - a * a
    def affecter(self, pid, vid):
        ancien = self.affectation[pid]
        if ancien == vid: return
        med = self.inst.patients[pid].med_id
        avant = [(v, self.taux_remplissage(v)) for v in (ancien, vid) if v is not None and not self.inst.vacations[v].urgence]
        self._appliquer_patient(pid, ancien, -1); self._appliquer_patient(pid, vid, +1)
        self.affectation[pid] = vid
        for v, t0 in avant:
            t1 = self.taux_remplissage(v); self.s_taux[med] += t1 - t0; self.s_taux2[med] += t1 * t1 - t0 * t0
        if vid is None: self.sans_date.add(pid)
        else: self.sans_date.discard(pid); self.hors_horizon.discard(pid)
    def variance_remplissage_medecin(self, m):
        n = self.inst.nb_vacations_medecin.get(m, 0)
        if n == 0: return 0.0
        s, s2 = self.s_taux[m], self.s_taux2[m]; return max(0.0, s2 / n - (s / n) ** 2)
    def variance_remplissage(self):
        n = max(1, len(self.inst.vacations_du_medecin))
        return sum(self.variance_remplissage_medecin(m) for m in self.inst.vacations_du_medecin) / n
    def variance_places(self):
        n = len(self._jours_places)
        return max(0.0, self.s_places2 / n - (self.s_places / n) ** 2) if n else 0.0
    def variance_lits(self):
        n = self._n_jours_lits
        return max(0.0, self.s_lits2 / n - (self.s_lits / n) ** 2) if n else 0.0

PENALITE = 1e6
def _variance(xs):
    n = len(xs)
    if n == 0: return 0.0
    m = sum(xs) / n; return sum((x - m) ** 2 for x in xs) / n
@dataclass
class PoidsRessources:
    places: float = 1.0
    lits: float = 1.0
def cout_places(sol):
    cap = sol.inst.capacite_places_jour; cap_prog = cap - sol.inst.reserve_places
    a = [sol.places_jour[j] for j in sol._jours_places]
    s = sum(max(0, x - cap) for x in a) + sum(max(0, sol.places_jour[j] - sol.places_urg_jour[j] - cap_prog) for j in sol._jours_places)
    return _variance(a) / cap ** 2 + PENALITE * s
def cout_lits(sol):
    cap = sol.inst.capacite_lits; cap_prog = cap - sol.inst.reserve_lits; l = list(sol.lits_jour)
    s = sum(max(0, x - cap) for x in l) + sum(max(0, x - u - cap_prog) for x, u in zip(l, sol.lits_urg_jour))
    return _variance(l) / cap ** 2 + PENALITE * s
def cout_ressources(sol, w=None):
    w = w or PoidsRessources(); return w.places * cout_places(sol) + w.lits * cout_lits(sol)
@dataclass
class PoidsPlanning:
    remplissage: float = 1.0
    delai: float = 1.0
def cout_remplissage(sol):
    inst = sol.inst
    v = [_variance([sol.taux_remplissage(x) for x in vids]) for vids in inst.vacations_du_medecin.values() if vids]
    dep = sum(sol.depassement(x) + sol.depassement_programme(x) for x in inst.vacations)
    return (sum(v) / len(v) if v else 0.0) + PENALITE * dep
def cout_delai(sol):
    inst = sol.inst; total, n = 0.0, 0
    for pid, vid in sol.affectation.items():
        if vid is None and pid not in sol.hors_horizon: continue
        p = inst.patients[pid]; jour = inst.vacations[vid].jour if vid is not None else inst.nb_jours
        total += p.priorite * max(0, jour - (p.jour_demande + p.fenetre_jours)); n += 1
    return total / (n * inst.nb_jours) if n else 0.0
def cout_planning(sol, w=None):
    w = w or PoidsPlanning(); return w.remplissage * cout_remplissage(sol) + w.delai * cout_delai(sol)
@dataclass
class CoutTotal:
    w_ressources: PoidsRessources | None = None
    w_planning: PoidsPlanning | None = None
    alpha: float = 1.0
    def termes(self, sol):
        return {"places": cout_places(sol), "lits": cout_lits(sol), "remplissage": cout_remplissage(sol), "delai": cout_delai(sol)}
    def __call__(self, sol):
        return self.alpha * cout_ressources(sol, self.w_ressources) + cout_planning(sol, self.w_planning)
