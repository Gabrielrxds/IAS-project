r"""
gestion_aleas.py — Gestion des aléas du planning opératoire (hors jour J, géré par le SMA).

Un aléa est déclaré le jour `t` avec un MOTIF (liste fermée), une RAISON libre et, pour les
indisponibilités, une DURÉE ESTIMÉE saisie par le déclarant (« 90 min », « 3 h », « 2 j ») ou une
date de fin. La durée estimée peut se révéler fausse : on PROLONGE (nouvel aléa qui couvre la
suite) ou on LÈVE l'aléa plus tôt (les vacations encore à venir sont rouvertes et proposées aux
patients en attente).

Aléas gérés (catalogue complet dans GESTION_ALEAS.md) :

  RESSOURCES — période [début, fin[ à la minute près
    IndispoMedecin   congé, maladie, formation, urgence dans un autre service, retard, raison
                     personnelle, autre. Ses vacations touchées sont FERMÉES (période qui les
                     couvre) ou RACCOURCIES (période partielle : on garde le plus grand morceau
                     restant s'il dure au moins MIN_RESTE minutes). Les patients qui ne tiennent
                     plus sont reprogrammés (même chirurgien), avec rattrapage en option.
    IndispoSalle     panne de matériel, stérilisation, maintenance, dégât des locaux, personnel
                     manquant, autre. Chaque vacation touchée est d'abord RELOGÉE dans une autre
                     salle libre sur toute sa plage horaire ; sinon fermée ou raccourcie.
    IndispoMateriel  un matériel (ancillaire, implant, colonne…) manque : seules les
                     interventions des types concernés sont déplacées hors de la période ; les
                     autres patients ne bougent pas, et aucun nouveau patient de ces types n'est
                     programmé pendant la période.
    FermetureLits    n lits fermés sur des nuits : capacité réduite ; si les lits programmés
                     dépassent, on déplace les patients hospitalisés programmés les plus récents
                     (pas les journées figées si possible, jamais un patient déjà admis).

  PATIENTS
    AnnulationPatient  le patient annule à l'avance : définitivement, ou report demandé à partir
                       d'une date (nouvelle demande, SANS priorité). Sa place est COMBLÉE.
    PatientNonApte     contre-indication médicale (infection, bilan incomplet, anticoagulants…)
                       avec une durée estimée avant d'être apte : reprogrammé AVEC priorité,
                       délai maximum prolongé d'autant. Sa place est COMBLÉE.
    NonVenue           le jour même : nouvelle demande vue ce jour-là, SANS priorité (règle du
                       parcours E). La place du jour est perdue (le SMA peut la donner).

  COMBLEMENT : une place libérée au moins PREAVIS_PATIENT jours à l'avance est proposée aux
  patients du MÊME chirurgien opérés plus tard (le plus ancien demandeur d'abord), qui acceptent
  avec une probabilité `p_accepte` en simulation ; dans l'interface, la secrétaire les appelle.

Ne sont PAS traités ici (jour J, temps réel) : urgences, dépassements, retards de la journée,
annulations de dernière minute — c'est le rôle du SMA (hybride_sma/).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import dataclasses
import random
import re
from collections import defaultdict
from typing import ClassVar

from absences import Reprogrammateur, Reprogrammation, PREAVIS_PATIENT, SEUIL_CONGE, DEMI_JOURNEES
from regle_hybride import RegleHybride, RegleHybrideV1

JOUR = 24 * 60                 # minutes
MIN_RESTE = 60                 # en dessous, une vacation raccourcie est fermée
OUVERTURE, FERMETURE = 7 * 60 + 30, 18 * 60 + 30   # bornes pour DÉCALER une vacation
SALLES = (2, 3, 4, 5)          # salles où l'on peut reloger une vacation (la salle 1 n'existe pas)

MOTIFS = {
    "medecin": ("conge", "maladie", "formation", "urgence_autre_service", "retard", "personnel", "autre"),
    "salle": ("panne_materiel", "sterilisation", "maintenance", "degat_locaux", "personnel_manquant", "autre"),
    "materiel": ("ancillaire_indisponible", "rupture_implant", "colonne_endoscopie", "autre"),
    "lits": ("epidemie", "travaux", "manque_personnel", "autre"),
    "annulation": ("choix_patient", "empechement", "deces", "transfert", "autre"),
    "non_apte": ("infection", "bilan_incomplet", "traitement_a_arreter", "decompensation", "autre"),
    "non_venu": ("absent_sans_prevenir", "annulation_jour_meme", "contre_indication_admission", "autre"),
}


# ---------------------------------------------------------------------------
# Durées estimées
# ---------------------------------------------------------------------------

_UNITES = {"min": 1, "mn": 1, "m": 1, "h": 60, "j": JOUR, "jour": JOUR, "jours": JOUR, "d": JOUR,
           "sem": 7 * JOUR, "semaine": 7 * JOUR, "semaines": 7 * JOUR}


def lire_duree(texte: str | int) -> int:
    """« 90 min », « 1 h 30 », « 3h », « 2 j », « 1 sem » -> minutes. Un entier = minutes."""
    if isinstance(texte, int):
        return texte
    total = 0
    for nombre, unite in re.findall(r"(\d+(?:[.,]\d+)?)\s*([a-zA-Zé]*)", texte.lower()):
        u = unite.replace("é", "e") or "min"
        if u not in _UNITES:
            raise ValueError(f"unité inconnue : {unite!r} dans {texte!r}")
        total += round(float(nombre.replace(",", ".")) * _UNITES[u])
    if total <= 0:
        raise ValueError(f"durée vide : {texte!r}")
    return total


def periode(jour: int, heure: int | None = None, duree: str | int | None = None,
            jusqu_au: tuple[int, int] | None = None) -> tuple[int, int]:
    """Période [début, fin[ en minutes depuis le jour 0.
      - `duree` en jours (« 2 j ») : journées entières à partir de `jour` (heure ignorée) ;
      - `duree` en minutes ou heures : à partir de `heure` (8 h par défaut) ;
      - `jusqu_au=(jour_fin, heure_fin)` : date de fin explicite."""
    if jusqu_au is not None:
        a = jour * JOUR + (heure or 0)
        return a, jusqu_au[0] * JOUR + jusqu_au[1]
    m = lire_duree(duree)
    if m >= JOUR and m % JOUR == 0:
        return jour * JOUR, (jour + m // JOUR) * JOUR
    a = jour * JOUR + (8 * 60 if heure is None else heure)
    return a, a + m


# ---------------------------------------------------------------------------
# Les aléas
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Alea:
    declaration: int                       # jour de la saisie
    motif: str = "autre"
    raison: str = ""                       # texte libre (« raison x »)
    duree_estimee: str = ""                # telle que saisie, pour l'historique
    id: int = -1                           # attribué par le gestionnaire
    parent: int | None = None              # prolongation d'un autre aléa
    famille: ClassVar[str] = ""

    def __post_init__(self):
        if self.famille and self.motif not in MOTIFS[self.famille]:
            raise ValueError(f"motif {self.motif!r} inconnu pour {self.famille} : {MOTIFS[self.famille]}")


@dataclasses.dataclass
class _Periode(Alea):
    debut: int = 0                         # minutes depuis le jour 0
    fin: int = 0                           # exclu

    @property
    def jour_debut(self) -> int:
        return self.debut // JOUR

    @property
    def jour_fin(self) -> int:             # dernier jour touché (inclus)
        return (self.fin - 1) // JOUR

    def touche(self, jour: int, debut: int, fin: int) -> tuple[int, int] | None:
        """Intersection avec la plage [debut, fin[ du jour (minutes du jour), ou None."""
        a, b = max(self.debut, jour * JOUR + debut), min(self.fin, jour * JOUR + fin)
        return (a - jour * JOUR, b - jour * JOUR) if a < b else None


@dataclasses.dataclass
class IndispoMedecin(_Periode):
    med_id: int = -1
    famille: ClassVar[str] = "medecin"


@dataclasses.dataclass
class IndispoSalle(_Periode):
    salle: int = -1
    famille: ClassVar[str] = "salle"


@dataclasses.dataclass
class IndispoMateriel(_Periode):
    types: frozenset = frozenset()         # types d'intervention qui ont besoin du matériel
    famille: ClassVar[str] = "materiel"


@dataclasses.dataclass
class FermetureLits(_Periode):
    nb: int = 0                            # lits fermés chaque nuit de la période
    famille: ClassVar[str] = "lits"


@dataclasses.dataclass
class AnnulationPatient(Alea):
    pid: int = -1
    report_a_partir: int | None = None     # None : annulation définitive
    famille: ClassVar[str] = "annulation"


@dataclasses.dataclass
class PatientNonApte(Alea):
    pid: int = -1
    apte_a_partir: int = 0                 # déclaration + durée estimée
    famille: ClassVar[str] = "non_apte"


@dataclasses.dataclass
class NonVenue(Alea):
    pid: int = -1
    famille: ClassVar[str] = "non_venu"


@dataclasses.dataclass
class Levee:
    """Fin anticipée d'un aléa de ressource (la durée estimée était trop longue). On désigne
    l'aléa par son objet (simulation) ou par son id (interface)."""
    declaration: int
    alea_id: int = -1
    alea: object = None


@dataclasses.dataclass
class Action:
    """Une ligne du journal : ce que le gestionnaire a fait, pour l'interface et les mesures."""
    alea: int
    famille: str
    action: str          # fermee, raccourcie, relogee, liberee, reprogramme, a_arbitrer, avance,
                         # annule, nouvelle_demande, deficit_lits, rouverte, rattrapage
    declaration: int
    pid: int | None = None
    vid: int | None = None
    jour_initial: int | None = None
    jour_nouveau: int | None = None
    detail: str = ""


# ---------------------------------------------------------------------------
# Le gestionnaire
# ---------------------------------------------------------------------------

class GestionnaireAleas(Reprogrammateur):
    """Applique les aléas à une Solution en cours de construction (simulation en ligne) ou à
    un planning existant (interface). Hérite du moteur de reprogrammation des absences."""

    def __init__(self, inst, regle, politique: str = "hybride", delta: int = 14, rattrapage: bool = False,
                 comblement: bool = True, p_accepte: float = 0.5, essais_comblement: int = 8,
                 fenetre_rattrapage: int = 21, feries=frozenset(), salles=SALLES, graine: int = 0):
        super().__init__(inst, regle, politique, delta, rattrapage, fenetre_rattrapage, feries)
        self.comblement, self.p_accepte, self.essais = comblement, p_accepte, essais_comblement
        self.salles = tuple(salles)
        self.rng = random.Random(7_000_000 + graine)
        self.aleas: dict[int, Alea] = {}
        self.actifs: set[int] = set()
        self.modifs: dict[int, dict[int, object]] = defaultdict(dict)   # alea -> {vid: vacation d'origine}
        self.lits_origine: dict[int, dict[int, int]] = defaultdict(dict)
        self.actions: list[Action] = []
        self.annules: set[int] = set()
        self.demandes: dict[int, list[int]] = defaultdict(list)       # jour -> nouvelles demandes
        self.non_venues: list = []
        self._jmin: int | None = None
        self._materiels: set[int] = set()
        # Les règles de date (consultation) passent par inst.vacations_possibles : on y retire les
        # vacations fermées et les jours interdits par un matériel manquant.
        self._vp = inst.vacations_possibles
        inst.vacations_possibles = self._vacations_possibles

    # -- filtres --------------------------------------------------------------
    def _autorise(self, pid: int, vid: int) -> bool:
        v = self.inst.vacations[vid]
        if vid in self.fermees or v.tvo <= 0:
            return False
        if not self._materiels:
            return True
        t = self.inst.patients[pid].type_interv
        for i in self._materiels:
            a = self.aleas[i]
            if t in a.types and a.touche(v.jour, v.debut, v.fin):
                return False
        return True

    def terminer(self, t: int):
        """Retire des aléas actifs ceux dont la période est passée (appelé chaque jour)."""
        for i in [i for i in self.actifs if self.aleas[i].fin <= t * JOUR]:
            self.actifs.discard(i)
            self._materiels.discard(i)

    def _vacations_possibles(self, pid, fenetre=None):
        return [v for v in self._vp(pid, fenetre) if self._autorise(pid, v)]

    def _candidats(self, sol, pid, t, figes, motif):
        jmin = max(t + PREAVIS_PATIENT, self._jmin or 0)
        out = []
        for vid in self.inst.vacations_possibles(pid):
            j = self.inst.vacations[vid].jour
            if j < jmin or (motif == "conge" and j in figes):
                continue
            sol.affecter(pid, vid)
            ok = not sol.viole(pid, vid)
            sol.affecter(pid, None)
            if ok:
                out.append(vid)
        return out

    def _absent(self, m, j):
        for i in self.actifs:
            a = self.aleas[i]
            if isinstance(a, IndispoMedecin) and a.med_id == m and a.jour_debut <= j <= a.jour_fin:
                return True
        return super()._absent(m, j)

    def _salle_libre(self, salle: int, jour: int, debut: int, fin: int, sauf: int | None = None) -> bool:
        inst = self.inst
        for x in inst.vacations_du_jour.get(jour, ()):
            v = inst.vacations[x]
            if x != sauf and v.bloc_id == salle and x not in self.fermees and v.tvo > 0 \
                    and v.debut < fin and v.fin > debut:
                return False
        for i in self.actifs:
            a = self.aleas[i]
            if isinstance(a, IndispoSalle) and a.salle == salle and a.touche(jour, debut, fin):
                return False
        return True

    # -- outils sur les patients ------------------------------------------------
    def _poser(self, sol, pid, vid):
        sol.affecter(pid, vid)
        R = self.regle
        if isinstance(R, RegleHybrideV1) and R.pret:
            R.lits_patient(pid, self.inst.vacations[vid].jour, +1)

    def _jour(self, sol, pid):
        v = sol.affectation.get(pid)
        return None if v is None else self.inst.vacations[v].jour

    def _note(self, a: Alea, action: str, t: int, **kw):
        self.actions.append(Action(a.id, a.famille, action, t, **kw))

    # -- outils sur les vacations -------------------------------------------------
    def _memoriser(self, a, vid):
        self.modifs[a.id].setdefault(vid, self.inst.vacations[vid])

    def _reduire(self, sol, a: _Periode, vid: int, t: int) -> list[tuple[int, int]]:
        """Retire de la vacation la partie couverte par l'aléa. Renvoie les (jour, patient) à
        reprogrammer."""
        inst = self.inst
        v = inst.vacations[vid]
        inter = a.touche(v.jour, v.debut, v.fin)
        if inter is None or vid in self.fermees:
            return []
        gauche, droite = (v.debut, inter[0]), (inter[1], v.fin)
        g = max(gauche, droite, key=lambda x: x[1] - x[0])
        self._memoriser(a, vid)
        if g[1] - g[0] < MIN_RESTE:
            ps = self._fermer(sol, vid)
            self._note(a, "fermee", t, vid=vid, jour_initial=v.jour, detail=f"{len(ps)} patients")
            return [(v.jour, p) for p in ps]
        g0 = g
        g = self._decaler(v, g, (v.fin - v.debut) - (g[1] - g[0]))
        inst.vacations[vid] = dataclasses.replace(v, debut=g[0], fin=g[1])
        retires = []
        if not v.urgence:
            while sol.depassement(vid) > 0 or sol.depassement_programme(vid) > 0:
                ps = sol.patients_de(vid)
                if not ps:
                    break
                p = max(ps, key=lambda q: (inst.patients[q].duree_op, q))
                self._liberer(sol, p, v.jour)
                retires.append(p)
        self._note(a, "decalee" if g != g0 else "raccourcie", t, vid=vid, jour_initial=v.jour,
                   detail=f"{v.debut}-{v.fin} -> {g[0]}-{g[1]} ; {len(retires)} patients sortis")
        return [(v.jour, p) for p in retires]

    def _decaler(self, v, g, perdu: int) -> tuple[int, int]:
        """Récupère les minutes perdues en prolongeant le morceau gardé du côté opposé à
        l'indisponibilité (ex. chirurgien en retard de 2 h : 8h-13h devient 10h-15h), si la
        salle et le chirurgien sont libres et sans sortir de [OUVERTURE, FERMETURE]."""
        if perdu <= 0 or v.urgence:
            return g
        inst = self.inst
        for sens in ((1, -1) if g[0] > v.debut else (-1, 1)):
            if sens == 1:
                a, b = g[1], min(FERMETURE, g[1] + perdu)
            else:
                a, b = max(OUVERTURE, g[0] - perdu), g[0]
            if b - a < 15:
                continue
            libre = self._salle_libre(v.bloc_id, v.jour, a, b, sauf=v.id) and not any(
                w.med_id == v.med_id and w.id != v.id and w.id not in self.fermees and w.tvo > 0
                and w.debut < b and w.fin > a
                for w in (inst.vacations[x] for x in inst.vacations_du_jour.get(v.jour, ())))
            if libre and not self._absent_entre(v.med_id, v.jour, a, b):
                return (g[0], b) if sens == 1 else (a, g[1])
        return g

    def _absent_entre(self, m, jour, a, b) -> bool:
        return any(isinstance(x, IndispoMedecin) and x.med_id == m and x.touche(jour, a, b)
                   for x in (self.aleas[i] for i in self.actifs))

    def _reloger(self, sol, a, vid, t) -> bool:
        v = self.inst.vacations[vid]
        for s in self.salles:
            if s != v.bloc_id and self._salle_libre(s, v.jour, v.debut, v.fin, sauf=vid):
                self._memoriser(a, vid)
                self.inst.vacations[vid] = dataclasses.replace(v, bloc_id=s)
                self._note(a, "relogee", t, vid=vid, jour_initial=v.jour, detail=f"salle {v.bloc_id} -> {s}")
                return True
        return False

    def _rattraper(self, sol, a, t, minutes, perdu):
        """Comme Reprogrammateur._rattraper, mais avec toutes les salles et en évitant les salles
        indisponibles. `a` a un med_id et un attribut `fin` en jours (cf. _Rattrapage)."""
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
                if any(v.med_id == m and v.id not in self.fermees and v.tvo > 0 and v.debut < fin and v.fin > debut
                       for v in vj):
                    continue
                for salle in self.salles:
                    if not self._salle_libre(salle, j, debut, fin):
                        continue
                    vid = self._ouvrir(sol, m, j, salle, debut, fin)
                    vj.append(inst.vacations[vid])
                    ouvert += inst.capacite_programme(vid)
                    break

    # -- reprogrammer et combler ------------------------------------------------
    def _reprogrammer(self, sol, a: Alea, a_placer, t, figes, motif, jmin=None):
        inst = self.inst
        self._jmin = jmin
        for j, p in sorted(a_placer):
            vid = self._placer(sol, p, t, figes, motif)
            jn = None if vid is None else inst.vacations[vid].jour
            if vid is None:
                sol.hors_horizon.add(p)
            self.journal.append(Reprogrammation(p, a.id, j, jn, vid, t))
            self._note(a, "a_arbitrer" if vid is None else "reprogramme", t, pid=p, vid=vid,
                       jour_initial=j, jour_nouveau=jn)
        self._jmin = None

    def _combler(self, sol, a: Alea, vid: int, t: int, figes: set[int]):
        """Propose la place libérée aux patients du même chirurgien opérés plus tard."""
        inst = self.inst
        if not self.comblement or vid in self.fermees:
            return
        v = inst.vacations[vid]
        if v.urgence or v.jour < t + PREAVIS_PATIENT:
            return
        cands = []
        for p, x in sol.affectation.items():
            if x is None or x == vid:
                continue
            w = inst.vacations[x]
            q = inst.patients[p]
            if w.med_id == v.med_id and w.jour > v.jour and w.jour not in figes and not q.est_urgent \
                    and q.jour_min <= v.jour <= inst.jour_max(p) and self._autorise(p, vid):
                cands.append((q.jour_demande, p, x, w.jour))
        essais = 0
        for _, p, x, jx in sorted(cands):
            if essais >= self.essais:
                break
            essais += 1
            if self.rng.random() >= self.p_accepte:          # le patient décline
                continue
            self._liberer(sol, p, jx)
            sol.affecter(p, vid)
            if sol.viole(p, vid):
                sol.affecter(p, None)
                self._poser(sol, p, x)
                continue
            sol.affecter(p, None)
            self._poser(sol, p, vid)
            self._note(a, "avance", t, pid=p, vid=vid, jour_initial=jx, jour_nouveau=v.jour)

    # -- point d'entrée ---------------------------------------------------------
    def declarer_alea(self, sol, a, t: int, figes: set[int]):
        """Applique l'aléa (ou la levée) déclaré le jour t."""
        R = self.regle
        if isinstance(R, RegleHybrideV1):
            if not R.pret:
                R._init_instance(sol)
            R.t = t
        if isinstance(a, Levee):
            return self._lever(sol, a, t, figes)
        a.id = len(self.aleas)
        self.aleas[a.id] = a
        if isinstance(a, _Periode):
            self.actifs.add(a.id)
            if isinstance(a, IndispoMateriel):
                self._materiels.add(a.id)
        traitement = {IndispoMedecin: self._medecin, IndispoSalle: self._salle, IndispoMateriel: self._materiel,
                      FermetureLits: self._lits, AnnulationPatient: self._annulation,
                      PatientNonApte: self._non_apte, NonVenue: self._non_venu}[type(a)]
        traitement(sol, a, t, figes)
        return a.id

    def _motif(self, a: _Periode, t: int) -> str:
        """« conge » (prévu : seulement les journées non figées) ou « maladie » (imprévu)."""
        return "conge" if a.jour_debut - t >= SEUIL_CONGE else "maladie"

    def _medecin(self, sol, a: IndispoMedecin, t, figes):
        inst = self.inst
        a_placer, perdu = [], 0
        for vid in list(inst.vacations_du_medecin.get(a.med_id, ())):
            v = inst.vacations[vid]
            if v.jour < t or a.touche(v.jour, v.debut, v.fin) is None:
                continue
            avant = inst.capacite_programme(vid)
            a_placer += self._reduire(sol, a, vid, t)
            perdu += avant - inst.capacite_programme(vid)
        if self.rattrapage and a_placer:
            n = len(self.rattrapages)
            minutes = sum(inst.patients[p].duree_op + inst.tis for _, p in a_placer)
            self._rattraper(sol, _Rattrapage(a.med_id, a.jour_fin), t, minutes, perdu)
            for vid in self.rattrapages[n:]:
                self._note(a, "rattrapage", t, vid=vid, jour_nouveau=inst.vacations[vid].jour)
        self._reprogrammer(sol, a, a_placer, t, figes, self._motif(a, t))

    def _salle(self, sol, a: IndispoSalle, t, figes):
        inst = self.inst
        a_placer = []
        for j in range(max(t, a.jour_debut), a.jour_fin + 1):
            for vid in list(inst.vacations_du_jour.get(j, ())):
                v = inst.vacations[vid]
                if v.bloc_id != a.salle or vid in self.fermees or v.tvo <= 0 \
                        or a.touche(j, v.debut, v.fin) is None:
                    continue
                if not self._reloger(sol, a, vid, t):
                    a_placer += self._reduire(sol, a, vid, t)
        self._reprogrammer(sol, a, a_placer, t, figes, self._motif(a, t))

    def _materiel(self, sol, a: IndispoMateriel, t, figes):
        inst = self.inst
        a_placer = []
        for p, x in list(sol.affectation.items()):
            if x is None or inst.patients[p].type_interv not in a.types:
                continue
            v = inst.vacations[x]
            if v.jour >= t and a.touche(v.jour, v.debut, v.fin):
                self._liberer(sol, p, v.jour)
                self._note(a, "liberee", t, pid=p, vid=x, jour_initial=v.jour)
                a_placer.append((v.jour, p))
        self._reprogrammer(sol, a, a_placer, t, figes, self._motif(a, t))

    def _cap_nuit(self, k):
        inst = self.inst
        return max(0, min(self._cap_calendrier[k], inst.capacite_lits - self._lits_fermes[k]))

    def _lits(self, sol, a: FermetureLits, t, figes):
        inst = self.inst
        if inst.capacite_lits_nuit is None:
            inst.capacite_lits_nuit = [inst.capacite_lits] * inst.nb_jours
        if not hasattr(self, "_cap_calendrier"):
            # capacité « armée » de chaque nuit avant tout aléa (calendrier : moins de lits en
            # vacances) ; les lits fermés se retirent des lits PHYSIQUES, et la plus petite des
            # deux capacités s'applique (sinon une fermeture en vacances compterait deux fois)
            self._cap_calendrier = list(inst.capacite_lits_nuit)
            self._lits_fermes = defaultdict(int)
        nuits = range(max(0, a.jour_debut), min(inst.nb_jours, a.jour_fin + 1))
        for k in nuits:
            self.lits_origine[a.id][k] = inst.capacite_lits_nuit[k]
            self._lits_fermes[k] += a.nb
            inst.capacite_lits_nuit[k] = self._cap_nuit(k)
        a_placer = []
        for k in nuits:
            def exces():
                return max(sol.lits_jour[k] - inst.cap_lits(k),
                           sol.lits_jour[k] - sol.lits_urg_jour[k] - (inst.cap_lits(k) - inst.reserve_lits))
            if exces() <= 0:
                continue
            mobiles = []
            for p, x in sol.affectation.items():
                if x is None:
                    continue
                q, j = inst.patients[p], inst.vacations[x].jour
                if q.est_urgent or q.ambulatoire or j < t + PREAVIS_PATIENT or k not in inst.nuits_occupees(p, j):
                    continue
                mobiles.append((j in figes, -q.jour_demande, p, j))
            for _, _, p, j in sorted(mobiles):
                if exces() <= 0:
                    break
                self._liberer(sol, p, j)
                self._note(a, "liberee", t, pid=p, jour_initial=j, detail=f"nuit {k}")
                a_placer.append((j, p))
            if exces() > 0:
                self._note(a, "deficit_lits", t, jour_initial=k, detail=f"{exces()} lits")
        self._reprogrammer(sol, a, a_placer, t, figes, "maladie")

    def _annulation(self, sol, a: AnnulationPatient, t, figes):
        inst, p = self.inst, a.pid
        x = sol.affectation.get(p)
        j = self._jour(sol, p)
        if x is not None and j >= t:
            self._liberer(sol, p, j)
            self._combler(sol, a, x, t, figes)
        sol.hors_horizon.discard(p)
        if a.report_a_partir is None:
            self.annules.add(p)
            self._note(a, "annule", t, pid=p, vid=x, jour_initial=j)
        else:
            d = max(t, a.report_a_partir)
            inst.patients[p] = dataclasses.replace(inst.patients[p], jour_demande=d)
            self.demandes[d].append(p)
            self._note(a, "nouvelle_demande", t, pid=p, vid=x, jour_initial=j, detail=f"à partir du jour {d}")

    def _non_apte(self, sol, a: PatientNonApte, t, figes):
        inst, p = self.inst, a.pid
        x = sol.affectation.get(p)
        j = self._jour(sol, p)
        if x is not None and j < t:
            return                                            # déjà opéré
        if x is not None:
            self._liberer(sol, p, j)
            self._combler(sol, a, x, t, figes)
        q = inst.patients[p]
        gain = max(0, a.apte_a_partir - (j if j is not None else t))
        inst.patients[p] = dataclasses.replace(q, delai_max=inst.jour_max(p) - q.jour_demande + gain)
        sol.hors_horizon.discard(p)
        self._reprogrammer(sol, a, [(j if j is not None else t, p)], t, figes, "maladie", jmin=a.apte_a_partir)

    def _non_venu(self, sol, a: NonVenue, t, figes):
        inst, p = self.inst, a.pid
        j = self._jour(sol, p)
        if j != t:
            raise ValueError(f"non venu : le patient {p} n'est pas opéré le jour {t}")
        self._liberer(sol, p, t)
        inst.patients[p] = dataclasses.replace(inst.patients[p], jour_demande=t)
        self.demandes[t].append(p)
        self._note(a, "nouvelle_demande", t, pid=p, jour_initial=t, detail="non venu, sans priorité")

    def _lever(self, sol, lv: Levee, t, figes):
        """L'aléa se termine avant la fin estimée : rouvre ce qui est encore à venir."""
        a = lv.alea if lv.alea is not None else self.aleas.get(lv.alea_id)
        if a is None or a.id < 0 or a.id not in self.aleas:
            return None                                       # aléa jamais déclaré
        self.actifs.discard(a.id)
        self._materiels.discard(a.id)
        inst = self.inst
        for k, cap in self.lits_origine.get(a.id, {}).items():
            if k >= t:
                self._lits_fermes[k] -= a.nb
                inst.capacite_lits_nuit[k] = self._cap_nuit(k)
        for vid, orig in self.modifs.get(a.id, {}).items():
            if orig.jour < t + PREAVIS_PATIENT or self._encore_touchee(orig):
                continue
            inst.vacations[vid] = orig
            self.fermees.discard(vid)
            self._note(a, "rouverte", t, vid=vid, jour_initial=orig.jour)
            self._combler(sol, a, vid, t, figes)
        return a.id

    def _encore_touchee(self, v) -> bool:
        """La vacation est-elle encore touchée par un autre aléa actif ?"""
        for i in self.actifs:
            a = self.aleas[i]
            if isinstance(a, IndispoMedecin) and a.med_id == v.med_id and a.touche(v.jour, v.debut, v.fin):
                return True
            if isinstance(a, IndispoSalle) and a.salle == v.bloc_id and a.touche(v.jour, v.debut, v.fin):
                return True
        return False


@dataclasses.dataclass
class _Rattrapage:
    med_id: int
    fin: int                       # dernier jour d'absence


# ---------------------------------------------------------------------------
# Simulation en ligne (même boucle que politiques.simuler)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TauxPatients:
    """Probabilités par patient programmé (hypothèses, cf. GESTION_ALEAS.md §4)."""
    annulation: float = 0.0          # annule à l'avance (3 à 20 jours avant)
    part_definitive: float = 0.3     # parmi les annulations : définitives (sinon report)
    report_jours: tuple = (14, 45)   # report demandé : disponible de t+14 à t+45
    non_apte: float = 0.0            # contre-indication découverte à J-7..J
    apte_jours: tuple = (7, 30)      # durée estimée avant d'être apte
    non_venue: float = 0.0           # absent le jour même


def simuler_aleas(inst, regle, aleas: list, politique="hybride", graine=0, delai_fige=7,
                  rattrapage=False, comblement=True, p_accepte=0.5, feries=frozenset(),
                  taux: TauxPatients | None = None, salles=SALLES):
    """Simulation en ligne sur l'horizon : chaque jour t, (1) les aléas déclarés ce jour-là,
    (2) les aléas patients tirés au hasard, (3) les consultations du jour + nouvelles demandes
    (reports, non venus), (4) on fige le jour t + delai_fige. Renvoie (solution, gestionnaire)."""
    from modele import Solution
    taux = taux or TauxPatients()
    sol = Solution(inst)
    G = GestionnaireAleas(inst, regle, politique, rattrapage=rattrapage, comblement=comblement,
                          p_accepte=p_accepte, feries=feries, salles=salles, graine=graine)
    rng = random.Random(graine)
    par_jour = defaultdict(list)
    for pid in sorted(inst.patients):
        par_jour[inst.patients[pid].jour_demande].append(pid)
    decl = defaultdict(list)
    for a in aleas:
        decl[a.declaration].append(a)
    tires = _Tirages(graine, taux)
    figes: set[int] = set()
    for t in range(inst.nb_jours):
        sol.jour_courant = t
        G.terminer(t)
        for a in sorted(decl.get(t, ()), key=lambda x: isinstance(x, Levee)):
            G.declarer_alea(sol, a, t, figes)
        if taux.annulation or taux.non_apte or taux.non_venue:
            for a in tires.du_jour(sol, inst, t):
                G.declarer_alea(sol, a, t, figes)
        pids = list(par_jour.get(t, [])) + G.demandes.pop(t, [])
        pids = [p for p in pids if p not in G.annules]
        if pids:
            if isinstance(regle, RegleHybrideV1) and regle.pret:
                regle.t = t
            rng.shuffle(pids)
            regle.preparer(sol, t)
            for pid in pids:
                vid = regle.consulter(sol, pid, t, figes)
                if vid is None:
                    sol.hors_horizon.add(pid)
                else:
                    regle.apres(sol, vid, t, figes)
        if t + delai_fige < inst.nb_jours:
            figes.add(t + delai_fige)
    sol.jour_courant = None
    sol.definir_fenetre(0, inst.nb_jours - 1)
    return sol, G


class _Tirages:
    """Aléas patients : chaque patient a ses propres tirages (même graine => mêmes aléas quelle que
    soit la politique), appliqués quand sa date programmée arrive à la bonne distance."""

    def __init__(self, graine, taux: TauxPatients):
        self.g, self.taux, self.faits, self._cache = graine, taux, set(), {}

    def _tirage(self, pid, j):
        """Un tirage par (patient, date programmée) : les aléas suivent la date, pas le jour."""
        c = self._cache.get((pid, j))
        if c is None:
            T = self.taux
            r = random.Random((self.g * 1_000_003 + pid) * 7 + j)
            c = (r.random(), r.randint(3, 20), r.random(), r.randint(*T.report_jours),
                 r.random(), r.randint(0, 7), r.randint(*T.apte_jours), r.random())
            self._cache[(pid, j)] = c
        return c

    def du_jour(self, sol, inst, t):
        T, out = self.taux, []
        for p, x in list(sol.affectation.items()):
            if x is None or inst.patients[p].est_urgent:
                continue
            j = inst.vacations[x].jour
            if j < t:
                continue
            cle = (p, j)
            if cle in self.faits or j - t > 20:
                continue
            u_ann, avance_ann, u_def, rep, u_apte, avance_apte, duree_apte, u_nv = self._tirage(p, j)
            if u_ann < T.annulation and j - t == avance_ann:
                self.faits.add(cle)
                out.append(AnnulationPatient(t, "choix_patient", pid=p,
                                             report_a_partir=None if u_def < T.part_definitive else t + rep))
            elif u_apte < T.non_apte and j - t == avance_apte:
                self.faits.add(cle)
                out.append(PatientNonApte(t, "infection", duree_estimee=f"{duree_apte} j", pid=p,
                                          apte_a_partir=t + duree_apte))
            elif u_nv < T.non_venue and j == t:
                self.faits.add(cle)
                out.append(NonVenue(t, "absent_sans_prevenir", pid=p))
        return out
