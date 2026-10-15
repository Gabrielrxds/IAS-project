r"""
absences.py — Absences des chirurgiens (congé, maladie) et reprogrammation des patients.

Le chirurgien déclare une période d'absence [debut, fin] le jour `t` (déclaration).
  1. FERMER : chacune de ses vacations de la période devient indisponible (capacité nulle) ;
     les patients fantômes qu'elle portait disparaissent (la règle hybride ne prévoit plus
     d'activité ce jour-là) ; la salle est signalée « libérée » pour le cadre de bloc.
  2. LIBÉRER : les patients déjà programmés dans ces vacations perdent leur date.
  3. REPROGRAMMER, patient par patient, dans l'ordre de leur date initiale (le plus proche
     d'abord : il a le moins de marge) :
       - même chirurgien (règle du modèle : le chirurgien ne change jamais) ;
       - pas avant t + PREAVIS_PATIENT jours (le temps de prévenir le patient) ni avant son
         délai minimum ; pas après son délai maximum (6 mois après la consultation) ;
       - congé déclaré à l'avance (≥ 8 jours) : seulement des journées non figées ;
         maladie (préavis court) : aussi les journées figées de la semaine si la vacation a
         encore de la place, sinon la première journée non figée ;
       - choix du jour : selon la politique —
           « premier »  : premier créneau réalisable ;
           « hybride »  : comme à la consultation : coût = hausse de la variance des lits
                          attendus (réels + fantômes) sur une fenêtre de Δ jours après le premier
                          jour réalisable, deux jours proposés, le plus proche retenu.
       - aucun créneau dans le délai maximum : le patient est « à arbitrer » (sans date).
  4. RATTRAPAGE (option) : avant de reprogrammer, on cherche des demi-journées de salle LIBRES
     (aucune vacation, ni d'urgence) dans les 21 jours qui suivent la fin de l'absence, où le
     chirurgien est présent et n'opère pas déjà ; on lui en ouvre autant qu'il faut pour couvrir
     les minutes des patients déplacés, sans dépasser le temps de bloc perdu. Les patients y
     passent en priorité puisque ce sont les premières dates réalisables.
Les patients du jour même ne sont pas opérés : ils sont reprogrammés au plus tôt dans 2 jours.

PATIENT NON VENU (règle simple, SANS priorité) : le jour de son opération, le patient ne se
présente pas. Sa place est perdue pour la journée (le SMA peut la donner à une urgence). Il est
traité comme une NOUVELLE demande vue en consultation ce jour-là : délai minimum et délai maximum
recomptés depuis ce jour, même règle de date que tous les patients (deux jours proposés), mêlé
aux consultations du jour, sans rattrapage ni passe-droit.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import dataclasses
import datetime as dt
import random
from collections import Counter, defaultdict

import numpy as np

from politiques import simuler as _simuler_base  # noqa: F401  (même boucle, cf. simuler_absences)
from regle_hybride import RegleHybride

PREAVIS_PATIENT = 2           # jours minimum entre l'appel au patient et sa nouvelle date
SEUIL_CONGE = 8               # préavis à partir duquel l'absence est traitée comme un congé


@dataclasses.dataclass
class Absence:
    med_id: int
    debut: int                # premier jour d'absence (indice de jour)
    fin: int                  # dernier jour d'absence (inclus)
    declaration: int          # jour où l'absence est saisie dans l'interface
    motif: str = "conge"      # "conge" | "maladie"


@dataclasses.dataclass
class NonVenue:
    pid: int
    jour_initial: int          # jour de l'opération manquée
    jour_nouveau: int | None   # None : pas de date dans les 6 mois
    rang: int                  # 1re, 2e… non-venue de ce patient


@dataclasses.dataclass
class Reprogrammation:
    pid: int
    absence: int              # indice de l'absence
    jour_initial: int
    jour_nouveau: int | None  # None : à arbitrer
    vid_nouveau: int | None
    declaration: int


# ---------------------------------------------------------------------------
# Absences : réelles (déduites des données) et maladies tirées au hasard
# ---------------------------------------------------------------------------

def absences_reelles(don, preavis: int = 42, grille_ref=None) -> list[Absence]:
    """Semaines de 2022 où un chirurgien avait des vacations dans la grille actuelle mais n'a
    opéré AUCUN patient en réalité : on les considère comme des absences (congés, formation…),
    déclarées `preavis` jours à l'avance. C'est une approximation des congés réels."""
    from protocole import grille_actuelle_neutre, charger
    df, _ = charger()
    d22 = df[df["Date Inter"].dt.year == 2022]
    reel = Counter((c, d.date() - dt.timedelta(days=d.weekday())) for d, c in zip(d22["Date Inter"], d22["Praticien"]))
    offre = set()
    for date, salle, deb, fin, code in (grille_ref or grille_actuelle_neutre(don)):
        x = dt.date.fromisoformat(date)
        if x.year == 2022:
            offre.add((code, x - dt.timedelta(days=x.weekday())))
    code_id = {c: m for m, c in don.med_codes.items()}
    out = []
    for code, lundi in sorted(offre):
        if reel.get((code, lundi), 0) == 0 and code in code_id:
            j0 = (lundi - don.jour_zero).days
            out.append(Absence(code_id[code], j0, j0 + 4, max(0, j0 - preavis), "conge"))
    return out


def absences_maladie(inst, taux: float, graine: int, preavis=(0, 1, 2)) -> list[Absence]:
    """Arrêts maladie : chaque journée opératoire d'un chirurgien est perdue avec la probabilité
    `taux` ; durée 1 à 5 jours ouvrés ; déclarés 0 à 2 jours avant."""
    rng = random.Random(graine)
    jours_med = defaultdict(set)
    for v in inst.vacations.values():
        if not v.urgence:
            jours_med[v.med_id].add(v.jour)
    out = []
    for m in sorted(jours_med):
        for j in sorted(jours_med[m]):
            if rng.random() < taux:
                d = rng.choice((1, 1, 1, 2, 3, 5))
                out.append(Absence(m, j, j + d - 1, max(0, j - rng.choice(preavis)), "maladie"))
    return out


# ---------------------------------------------------------------------------
# Le moteur de reprogrammation
# ---------------------------------------------------------------------------

DEMI_JOURNEES = ((480, 780), (810, 1050))      # 8h-13h, 13h30-17h30 (grille hybride)


class Reprogrammateur:
    def __init__(self, inst, regle, politique: str = "hybride", delta: int = 14, rattrapage: bool = False,
                 fenetre_rattrapage: int = 21, feries=frozenset()):
        assert politique in ("premier", "hybride")
        self.inst, self.regle, self.politique, self.delta = inst, regle, politique, delta
        self.rattrapage, self.fen_ratt, self.feries = rattrapage, fenetre_rattrapage, feries
        self.fermees: set[int] = set()
        self.journal: list[Reprogrammation] = []
        self.salles_liberees: list[int] = []
        self.rattrapages: list[int] = []
        self.absences_connues: list[Absence] = []

    # 4. rattrapage --------------------------------------------------------------
    def _absent(self, m, j):
        return any(a.med_id == m and a.debut <= j <= a.fin for a in self.absences_connues)

    def _ouvrir(self, sol, m, j, salle, debut, fin):
        from modele import Vacation
        inst = self.inst
        vid = max(inst.vacations) + 1
        v = Vacation(vid, bloc_id=salle, med_id=m, jour=j, debut=debut, fin=fin, etiquette="rattrapage")
        inst.vacations[vid] = v
        lst = inst.vacations_du_medecin.setdefault(m, [])
        lst.append(vid); lst.sort(key=lambda x: inst.vacations[x].jour)
        inst.vacations_du_jour.setdefault(j, []).append(vid)
        for d in (sol.nb, sol.somme_durees, sol.somme_marges, sol.somme_marges2,
                  sol.nb_urg, sol.urg_durees, sol.urg_marges, sol.urg_marges2):
            d[vid] = 0
        self.rattrapages.append(vid)
        return vid

    def _rattraper(self, sol, a, t, minutes, perdu):
        """Ouvre des demi-journées libres pour le chirurgien, au plus tôt."""
        inst, m = self.inst, a.med_id
        ouvert = 0
        for j in range(t + PREAVIS_PATIENT, a.fin + self.fen_ratt + 1):
            if ouvert >= minutes or ouvert >= perdu:
                break
            if j >= inst.nb_jours or self._absent(m, j) or inst.date_du_jour(j).weekday() > 4 \
                    or inst.date_du_jour(j) in self.feries:
                continue
            vj = [inst.vacations[x] for x in inst.vacations_du_jour.get(j, ())]
            for debut, fin in DEMI_JOURNEES:
                if ouvert >= minutes or ouvert >= perdu:
                    break
                if any(v.med_id == m and v.id not in self.fermees and v.debut < fin and v.fin > debut for v in vj):
                    continue                                   # il opère déjà à cette heure
                for salle in (2, 3, 4, 5):
                    if any(v.bloc_id == salle and v.id not in self.fermees and v.debut < fin and v.fin > debut for v in vj):
                        continue
                    vid = self._ouvrir(sol, m, j, salle, debut, fin)
                    vj.append(inst.vacations[vid])
                    ouvert += inst.capacite_programme(vid)
                    break

    # 1. fermer --------------------------------------------------------------
    def _fermer(self, sol, vid):
        inst = self.inst
        v = inst.vacations[vid]
        if vid in self.fermees:
            return []
        self.fermees.add(vid)
        patients = [p for p, x in sol.affectation.items() if x == vid]
        for p in patients:                                   # 2. libérer
            self._liberer(sol, p, v.jour)
        inst.vacations[vid] = dataclasses.replace(v, fin=v.debut)   # capacité nulle
        if isinstance(self.regle, RegleHybride) and self.regle.pret and vid in self.regle.dans:
            self.regle._retirer_fantomes(vid)
        self.salles_liberees.append(vid)
        return patients

    def _liberer(self, sol, pid, jour):
        sol.affecter(pid, None)
        R = self.regle
        if isinstance(R, RegleHybride) and R.pret:
            for k in self.inst.nuits_occupees(pid, jour):
                if k < len(R.E):
                    R.E[k] -= 1

    # 3. reprogrammer ----------------------------------------------------------
    def _candidats(self, sol, pid, t, figes, motif):
        inst = self.inst
        jmin = t + PREAVIS_PATIENT
        out = []
        for vid in inst.vacations_possibles(pid):
            j = inst.vacations[vid].jour
            if j < jmin or vid in self.fermees:
                continue
            if motif == "conge" and j in figes:
                continue
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if ok:
                out.append(vid)
        return out

    def _placer(self, sol, pid, t, figes, motif):
        inst, R = self.inst, self.regle
        cands = self._candidats(sol, pid, t, figes, motif)
        if not cands:
            return None
        if self.politique == "premier" or not isinstance(R, RegleHybride):
            vid = cands[0]
            sol.affecter(pid, vid)
            if isinstance(R, RegleHybride) and R.pret:
                for k in inst.nuits_occupees(pid, inst.vacations[vid].jour):
                    if k < len(R.E):
                        R.E[k] += 1
            return vid
        premier = inst.vacations[cands[0]].jour
        fen = R._fenetre(t)
        par_jour = {}
        for vid in cands:
            j = inst.vacations[vid].jour
            if j > premier + self.delta:
                break
            s, retrait = R._score(sol, pid, vid, fen)
            if j not in par_jour or s < par_jour[j][0]:
                par_jour[j] = (s, j, vid, retrait)
        deux = sorted(par_jour.values(), key=lambda c: (c[0], c[1]))[:2]
        s, j, vid, retrait = min(deux, key=lambda c: c[1])
        sol.affecter(pid, vid)
        for k in inst.nuits_occupees(pid, j):
            if k < len(R.E):
                R.E[k] += 1
        if retrait is not None:
            m = inst.vacations[vid].med_id
            for c in range(3):
                if retrait[c]:
                    R._ajouter_noyau(m, c, j, -retrait[c])
            R.r[vid] = np.maximum(0.0, R.r[vid] - retrait)
        return vid

    def declarer(self, sol, a: Absence, ia: int, t: int, figes: set[int]):
        """Applique l'absence `a` déclarée le jour t ; renvoie les reprogrammations."""
        inst = self.inst
        motif = "conge" if a.debut - t >= SEUIL_CONGE else "maladie"
        vids = [vid for vid in inst.vacations_du_medecin.get(a.med_id, ())
                if a.debut <= inst.vacations[vid].jour <= a.fin and inst.vacations[vid].jour >= t]
        self.absences_connues.append(a)
        a_placer, perdu = [], 0
        for vid in vids:
            j = inst.vacations[vid].jour
            if vid not in self.fermees:
                perdu += inst.capacite_programme(vid)
            for p in self._fermer(sol, vid):
                a_placer.append((j, p))
        if self.rattrapage and a_placer:
            minutes = sum(inst.patients[p].duree_op + inst.tis for _, p in a_placer)
            self._rattraper(sol, a, t, minutes, perdu)
        res = []
        for j, p in sorted(a_placer):
            vid = self._placer(sol, p, t, figes, motif)
            if vid is None:
                sol.hors_horizon.add(p)
            r = Reprogrammation(p, ia, j, None if vid is None else inst.vacations[vid].jour, vid, t)
            res.append(r)
        self.journal += res
        return res


def simuler_absences(inst, regle, absences: list[Absence], politique="hybride", graine=0,
                     delai_fige=7, rattrapage=False, feries=frozenset(), taux_non_venue: float = 0.0):
    """La boucle de politiques.simuler, avec, au début de chaque jour t, les absences déclarées
    ce jour-là. Renvoie (solution, reprogrammateur)."""
    from collections import defaultdict as dd
    from modele import Solution
    sol = Solution(inst)
    rng = random.Random(graine)
    RP = Reprogrammateur(inst, regle, politique, rattrapage=rattrapage, feries=feries)
    RP.non_venues = []
    rng_nv = random.Random(10_000 + graine)
    nb_nv = Counter()
    par_jour = dd(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    decl = dd(list)
    for i, a in enumerate(absences):
        decl[a.declaration].append((i, a))
    figes: set[int] = set()
    for t in range(inst.nb_jours):
        sol.jour_courant = t
        if t in decl:
            if isinstance(regle, RegleHybride) and not regle.pret:
                regle._init_instance(sol)
            if isinstance(regle, RegleHybride):
                regle.t = t
            for i, a in decl[t]:
                RP.declarer(sol, a, i, t, figes)
        pids = list(par_jour.get(t, []))
        nv_jour = []
        if taux_non_venue > 0:                           # patients non venus ce jour-là
            if isinstance(regle, RegleHybride) and not regle.pret:
                regle._init_instance(sol)
            du_jour = [p for p, v in sol.affectation.items() if v is not None and inst.vacations[v].jour == t]
            for p in sorted(du_jour):
                if rng_nv.random() < taux_non_venue:
                    RP._liberer(sol, p, t)
                    inst.patients[p] = dataclasses.replace(inst.patients[p], jour_demande=t)
                    nb_nv[p] += 1
                    nv = NonVenue(p, t, None, nb_nv[p])
                    RP.non_venues.append(nv); nv_jour.append(nv)
                    pids.append(p)                       # nouvelle demande du jour, sans priorité
        if pids:
            rng.shuffle(pids)
            regle.preparer(sol, t)
            for pid in pids:
                vid = regle.consulter(sol, pid, t, figes)
                if vid is None:
                    sol.hors_horizon.add(pid)
                else:
                    regle.apres(sol, vid, t, figes)
        for nv in nv_jour:
            v = sol.affectation.get(nv.pid)
            nv.jour_nouveau = None if v is None else inst.vacations[v].jour
        j = t + delai_fige
        if j < inst.nb_jours:
            figes.add(j)
    sol.jour_courant = None
    sol.definir_fenetre(0, inst.nb_jours - 1)
    return sol, RP
