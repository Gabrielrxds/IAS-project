r"""
sma_aleas.py — Système multi-agents (SMA) de réaction aux aléas du jour J.

Construit sur le modele.py COMMUN du groupe (V1 corrigée) :
  - pas de patients mobiles : une date annoncée ne change jamais ; seules
    l'HEURE et, au besoin, la vacation (même chirurgien, même jour) bougent ;
  - nuits = Date Sortie − Date Entrée, lit occupé de J − nuits_avant pendant
    nb_nuits nuits (`Instance.nuits_occupees`) ;
  - aucun dépassement accepté : TIS et marges P90 comptés EN ENTIER, en
    somme, dans tout test de faisabilité ;
  - urgences et réserves du modèle : tampon de TVO, lits et places réservés,
    créneaux URGENCES (`Vacation.urgence`). Le programmé n'y touche jamais ;
  - une urgence du jour se place le jour même, sinon le jour OUVRÉ suivant
    (`Instance.jour_ouvre_suivant`), sinon c'est un ÉCHEC compté
    (`hors_horizon`, comme `Solution.placer_urgence_jour`) ;
  - les semi-urgences restent au niveau global (`placer_semi_urgence`).

RÔLE
----
Les méthodes du groupe (AG, tabou, recuit, glouton...) construisent le
planning HORS LIGNE. Ce module le fait VIVRE minute par minute face aux
aléas : urgences du jour, annulations, lits fermés, retards au bloc.
Il ne dépend d'AUCUNE métaheuristique :
  1. une `Solution`, construite par n'importe quelle méthode ;
  2. facultatif : un OPTIMISEUR JOURNALIER, n'importe quelle fonction
            f(inst, jour, sequences) -> sequences       ({vid: [pid, ...]})
     C'est la signature de l'AG journalier ; pour un tabou ou un recuit, un
     adaptateur d'une ligne suffit (cf. `adapter`). Il peut renvoyer un dict,
     un `PlanningJour` ou un objet à `.planning` (comme `ResultatAG`). Sa
     sortie est VÉRIFIÉE (mêmes patients, bon chirurgien) ; une sortie
     invalide est rejetée et comptée, la simulation continue ;
  3. facultatif : des PLANS B, les plannings de la journée produits par les
     AUTRES méthodes du groupe (cf. `construire_plans_b`).

LA SOLUTION EST TENUE À JOUR (sur une copie de travail)
  `simuler_periode` travaille sur `copie_de_travail(sol)` : la Solution
  reçue n'est jamais modifiée. Dans la copie, chaque décision passe par les
  méthodes du modèle (`ajouter_urgent`, `affecter`) : une urgence placée
  occupe ses lits les nuits suivantes, une annulation les libère, et
  `verifier()` / `controle_coherence()` restent utilisables. Seuls deux
  événements n'ont pas d'équivalent dans le modèle et vivent dans
  `RegistreLits` : les ambulatoires convertis en nuit et les lits fermés.

AGENTS (ils ne communiquent que par messages, tous journalisés)
  AgentSalle (une par vacation du jour) : file de patients, acte en cours,
      fin prévue (médiane et P90). Répond aux appels d'offres, détecte ses
      retards, cède ou reçoit des patients.
  AgentLits : lits libres par nuit ; accepte ou refuse une réservation.
  AgentAmbulatoire : admissions du jour, sorties tardives de l'UCA,
      conversions en nuit.
  AgentRegulateur (le cadre de bloc) : reçoit les aléas, lance les
      négociations (Contract Net Protocol), décide l'escalade.
Performatifs FIPA : CFP, PROPOSE, REFUSE, ACCEPT_PROPOSAL, REJECT_PROPOSAL,
REQUEST, AGREE, INFORM.

CE QU'UN AGENT SAIT. Jamais la durée réelle d'un acte : il voit l'heure
d'entrée en salle, la durée estimée et la marge P90, et constate qu'un acte
n'est pas fini quand l'heure estimée est passée. Seul l'environnement (le
simulateur) lit `duree_reelle`.

TROIS POLITIQUES, MÊMES ALÉAS (comparaison de la partie 6)
  "statique"       : la règle du modèle commun, sans information temps réel.
                     Urgence : `placer_urgence_jour` (charge P90 statique),
                     ajoutée en fin de programme ; aucun rééquilibrage.
  "reoptimisation" : même placement, puis l'optimiseur journalier refait
                     TOUTE la fin de journée à chaque aléa (centralisé).
  "sma"            : négociation locale sur l'état RÉEL des salles ;
                     l'optimiseur n'est appelé qu'en escalade, sur les salles
                     concernées.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import random
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field

from modele import URGENCE_JOUR, Patient, Solution

POLITIQUES = ("statique", "reoptimisation", "sma")


def hhmm(t: float) -> str:
    t = int(round(t))
    return f"{t // 60:02d}h{t % 60:02d}"


# ---------------------------------------------------------------------------
# 0. Outils sur la Solution
# ---------------------------------------------------------------------------


def copie_de_travail(sol: Solution) -> Solution:
    """Copie indépendante : ses propres agrégats ET son propre dictionnaire de
    patients (les urgences ajoutées n'apparaissent pas dans l'original)."""
    inst2 = copy(sol.inst)
    inst2.patients = dict(sol.inst.patients)
    s = sol.copie()
    s.inst = inst2
    return s


def sequences_du_jour(sol: Solution, jour: int) -> dict[int, list[int]]:
    """Vacations du jour et leurs patients, d'après la Solution."""
    inst = sol.inst
    seq = {vid: [] for vid in inst.vacations_du_jour.get(jour, ())}
    for pid in sol.patients_du_jour(jour):
        seq[sol.affectation[pid]].append(pid)
    return {vid: s for vid, s in seq.items() if s}


def plan_du_jour(sol: Solution, jour: int, plan: dict | None) -> dict[int, list[int]]:
    """Le plan du matin (ordre choisi par un optimiseur) restreint aux
    patients que la Solution opère vraiment ce jour-là, complété par ceux
    qu'il ne connaît pas (urgences réservées la veille), en fin de file."""
    actuels = sequences_du_jour(sol, jour)
    presents = {p for s in actuels.values() for p in s}
    out, vus = {}, set()
    for vid, s in (plan or {}).items():
        garde = [p for p in s if p in presents and p not in vus]
        if garde:
            out[vid] = garde
            vus.update(garde)
    for vid, s in actuels.items():
        for p in s:
            if p not in vus:
                out.setdefault(vid, []).append(p)
    return out


# ---------------------------------------------------------------------------
# 1. Optimiseurs journaliers interchangeables
# ---------------------------------------------------------------------------


def vers_sequences(res) -> dict[int, list[int]] | None:
    """Normalise la sortie d'un optimiseur en {vid: [pid]}."""
    if res is None:
        return None
    if isinstance(res, dict):
        return {int(k): [int(p) for p in v] for k, v in res.items()}
    if isinstance(getattr(res, "sequences", None), dict):       # PlanningJour
        return vers_sequences(res.sequences)
    if hasattr(res, "planning"):                                 # ResultatAG
        return vers_sequences(res.planning)
    if isinstance(res, (tuple, list)) and res:
        return vers_sequences(res[0])
    raise TypeError(f"sortie d'optimiseur non reconnue : {type(res).__name__}")


def adapter(fonction, nom: str | None = None):
    """Rend utilisable par le SMA toute fonction (inst, jour, sequences) -> X,
    X = dict {vid: [pid]}, PlanningJour ou objet à `.planning`."""
    def opt(inst, jour, sequences):
        return vers_sequences(fonction(inst, jour, {v: list(s) for v, s in sequences.items()}))
    opt.nom = nom or getattr(fonction, "__name__", "optimiseur")
    return opt


def optimiseur_glouton(inst, jour, sequences):
    """Référence sans métaheuristique : dans chaque vacation, ambulatoires
    d'abord (courts d'abord), puis hospitalisés (longs d'abord)."""
    P = inst.patients
    return {vid: sorted(s, key=lambda pid: (not P[pid].ambulatoire,
                                            P[pid].duree_op if P[pid].ambulatoire
                                            else -P[pid].duree_op))
            for vid, s in sequences.items()}


optimiseur_glouton.nom = "glouton (ambulatoires d'abord)"


def optimiseur_ag_journalier(temps_max: float = 0.5, graine: int = 0,
                             reaffecter: bool = True, **kw_ag):
    """L'AG journalier du groupe (genetique_jour.py), borné en temps."""
    from genetique_jour import AGJournee, ParametresAG

    def opt(inst, jour, sequences):
        params = ParametresAG(temps_max=temps_max, graine=graine)
        return AGJournee(inst, jour, sequences, params=params,
                         reaffecter=reaffecter, **kw_ag).resoudre()
    return adapter(opt, f"AG journalier ({temps_max} s)")


def construire_plans_b(inst, jour: int, sequences: dict[int, list[int]],
                       optimiseurs: dict) -> dict[str, dict[int, list[int]]]:
    """Plannings alternatifs de la journée, un par méthode du groupe.
    `optimiseurs` = {nom: fonction (inst, jour, sequences) -> ...}."""
    plans = {}
    attendus = sorted(p for s in sequences.values() for p in s)
    for nom, f in optimiseurs.items():
        try:
            s = vers_sequences(f(inst, jour, {v: list(x) for v, x in sequences.items()}))
        except Exception as e:                        # une méthode qui échoue
            print(f"[plans B J{jour}] {nom} : échec ({type(e).__name__}: {e})")
            continue
        if s and sorted(p for x in s.values() for p in x) == attendus:
            plans[nom] = s
        else:
            print(f"[plans B J{jour}] {nom} : sortie rejetée (patients différents)")
    return plans


# ---------------------------------------------------------------------------
# 2. Paramètres, aléas, bilans
# ---------------------------------------------------------------------------


@dataclass
class ParametresSMA:
    politique: str = "sma"                # "statique" | "reoptimisation" | "sma"
    seuil_retard: int = 10                # min de dépassement prévu qui déclenche
    max_transferts: int = 3               # par traitement de retard
    controle_toutes: int = 15             # min entre deux constats d'un acte qui dure
    arrivee_reprises: int = 7 * 60 + 30   # heure de réception des urgences de la veille
    decalage_significatif: int = 30       # min : un patient « décalé » (stabilité)
    # coûts utilisés par les agents pour comparer deux plans
    poids_attente: float = 1.0            # par minute d'attente d'une urgence
    poids_depassement: float = 20.0       # par minute (scénario médian)
    poids_depassement_p90: float = 3.0    # par minute (scénario P90)
    poids_hors_uca: float = 300.0         # par ambulatoire sorti après la fermeture
    poids_decalage: float = 2.0           # par patient décalé par une insertion
    strict: bool = False                  # True : une erreur d'optimiseur est levée


@dataclass
class Alea:
    t: int                       # minute de la journée
    type: str                    # "urgence" | "annulation" | "lits"
    patient: Patient | None = None   # urgence : le Patient (urgence=URGENCE_JOUR)
    pid: int | None = None       # annulation : le patient programmé
    nb_lits: int = 0             # lits : nombre de lits fermés
    nuits: int = 1               # lits : nombre de nuits concernées


@dataclass
class RegistreLits:
    """Ce que le modèle ne sait pas représenter, conservé d'un jour à
    l'autre : ambulatoires convertis en nuit (+1) et lits fermés."""
    conversions: defaultdict = field(default_factory=lambda: defaultdict(int))
    indisponibles: defaultdict = field(default_factory=lambda: defaultdict(int))


@dataclass
class BilanJour:
    jour: int
    date: str
    politique: str
    programmes: int = 0
    annulations: int = 0
    urgences_arrivees: int = 0           # nouvelles urgences du jour
    urgences_reprises: int = 0           # urgences de la veille à placer aujourd'hui
    urgences_operees_jour_meme: int = 0
    urgences_operees_lendemain: int = 0
    urgences_reportees: int = 0          # renvoyées au jour ouvré suivant
    urgences_echec: int = 0              # ni J ni jour ouvré suivant : échec compté
    attente_urgence_moy: float = 0.0     # min, urgences opérées le jour même
    depassement_min: int = 0
    salles_en_depassement: int = 0
    conversions_nuit: int = 0
    hebergements: int = 0
    deficit_lits: int = 0
    pic_places: int = 0
    surcapacite_places: int = 0
    transferts: int = 0
    reordonnancements: int = 0
    patients_decales: int = 0
    changements_salle: int = 0
    messages: int = 0
    echecs_optimiseur: int = 0
    temps_optimiseur_s: float = 0.0

    def en_dict(self) -> dict:
        return dataclasses.asdict(self)


def resumer(bilans: list[BilanJour]) -> dict:
    """Totaux sur une période (moyenne pondérée pour l'attente)."""
    if not bilans:
        return {}
    tot = defaultdict(float)
    for b in bilans:
        for k, v in b.en_dict().items():
            if isinstance(v, (int, float)) and k not in ("jour", "attente_urgence_moy"):
                tot[k] += v
    n = sum(b.urgences_operees_jour_meme for b in bilans)
    tot["attente_urgence_moy"] = (sum(b.attente_urgence_moy * b.urgences_operees_jour_meme
                                      for b in bilans) / n if n else 0.0)
    tot["jours"] = len(bilans)
    return dict(tot)


# ---------------------------------------------------------------------------
# 3. Messages et plateforme
# ---------------------------------------------------------------------------


@dataclass
class Message:
    t: int
    emetteur: str
    destinataire: str
    performatif: str
    contenu: dict

    def __str__(self) -> str:
        c = self.contenu.get("resume")
        if c is None:
            c = ", ".join(f"{k}={v}" for k, v in self.contenu.items()
                          if not (isinstance(v, (list, dict)) and len(v) > 6))
        return (f"{hhmm(self.t)}  {self.emetteur:>13} -> {self.destinataire:<13} "
                f"{self.performatif:<15} {c}")


class Plateforme:
    """Acheminement SYNCHRONE des messages (requête -> réponse) et journal."""

    def __init__(self, env):
        self.env = env
        self.agents: dict[str, Agent] = {}
        self.journal: list[Message] = []

    def inscrire(self, agent: "Agent") -> None:
        self.agents[agent.nom] = agent

    def envoyer(self, emetteur: str, destinataire: str, performatif: str,
                **contenu) -> Message | None:
        m = Message(self.env.t, emetteur, destinataire, performatif, contenu)
        self.journal.append(m)
        rep = self.agents[destinataire].recevoir(m)
        if rep is not None:
            self.journal.append(rep)
        return rep

    def texte(self, debut: int = 0, fin: int = 24 * 60, agents=None,
              max_lignes: int | None = None) -> str:
        lignes = [str(m) for m in self.journal
                  if debut <= m.t <= fin and (agents is None or m.emetteur in agents
                                               or m.destinataire in agents)]
        if max_lignes is not None and len(lignes) > max_lignes:
            lignes = lignes[:max_lignes] + [f"... ({len(lignes) - max_lignes} messages de plus)"]
        return "\n".join(lignes)


class Agent:
    def __init__(self, env, nom: str):
        self.env, self.nom = env, nom

    def envoyer(self, dest, performatif: str, **contenu) -> Message | None:
        nom = dest if isinstance(dest, str) else dest.nom
        return self.env.pf.envoyer(self.nom, nom, performatif, **contenu)

    def recevoir(self, m: Message) -> Message | None:
        h = getattr(self, "sur_" + m.performatif.lower(), None)
        return h(m) if h else None

    def repondre(self, m: Message, performatif: str, **contenu) -> Message:
        return Message(self.env.t, self.nom, m.emetteur, performatif, contenu)


# ---------------------------------------------------------------------------
# 4. Les agents
# ---------------------------------------------------------------------------


@dataclass
class Projection:
    fin: int                 # fin de programme, scénario médian
    fin90: int               # fin de programme, scénario P90 (TIS plein, marges sommées)
    dep: int                 # dépassement prévu (médian)
    dep90: int               # dépassement prévu (P90)
    hors: int                # ambulatoires qui sortiraient après la fermeture UCA
    attente: int             # somme des attentes des urgences de la file
    debuts: dict             # pid -> heure d'entrée prévue (médian)


class AgentSalle(Agent):
    """Une vacation du jour. Ne connaît que ce qu'une salle voit."""

    def __init__(self, env, vid: int, file: list[int]):
        v = env.inst.vacations[vid]
        super().__init__(env, f"{'URG' if v.urgence else 'S'}{v.bloc_id}/V{vid}")
        self.vid, self.v = vid, v
        self.file: list[int] = list(file)
        self.en_cours: tuple[int, int] | None = None     # (pid, heure d'entrée)
        self.realises: list[tuple[int, int, int]] = []   # (pid, entrée, sortie)
        self.libre_a = v.debut
        self.dernier_type = None
        self.jeton = 0           # réveil en attente
        self.version = 0         # change à chaque évolution de la salle

    # -- règles du modèle ---------------------------------------------------

    def peut_accueillir(self, p: Patient) -> bool:
        """Créneau URGENCES : tout urgent, jamais un programmé. Sinon : le
        chirurgien de la vacation seulement."""
        if self.v.urgence:
            return p.est_urgent
        return p.med_id == self.v.med_id

    def limite(self, p: Patient) -> int:
        """Fin P90 admissible : le TVO entier pour un urgent, TVO − tampon
        pour un programmé (le tampon est réservé aux urgences)."""
        if p.est_urgent:
            return self.v.fin
        return self.v.fin - round(self.env.inst.tampon_urgence * self.v.tvo)

    # -- prévision (sans connaître les durées réelles) ----------------------

    def fin_acte_en_cours(self, t: int, p90: bool = False) -> int:
        pid, deb = self.en_cours
        p = self.env.patient(pid)
        e1, e2 = deb + p.duree_op, deb + p.duree_op + p.marge_perso
        if not p90 and t < e1:
            return e1
        if t < e2:
            return e2
        return t + self.env.P.controle_toutes        # « encore un quart d'heure »

    def _depart(self, t: int, p90: bool):
        if self.en_cours:
            return (self.fin_acte_en_cours(t, p90),
                    self.env.patient(self.en_cours[0]).type_interv, True)
        if self.realises:
            return self.libre_a, self.dernier_type, True
        return self.v.debut, None, False

    def projeter(self, t: int, file=None, p90: bool = False):
        """Enchaînement au plus tôt de la file. Médian : TIS réduit entre
        actes identiques (comme `deriver_creneaux`). P90 : TIS PLEIN et marge
        de chaque acte, comme `Solution.charge` (règle « aucun dépassement »)."""
        inst = self.env.inst
        file = self.file if file is None else file
        pos, prec, a_prec = self._depart(t, p90)
        debuts = {}
        for pid in file:
            p = self.env.patient(pid)
            if a_prec:
                pos += (inst.tis if p90 or not (p.type_interv and p.type_interv == prec)
                        else inst.tis_meme_acte)
            pos = max(pos, t)
            debuts[pid] = pos
            pos += p.duree_op + (p.marge_perso if p90 else 0)
            prec, a_prec = p.type_interv, True
        return pos, debuts

    def debut_disponible(self, t: int) -> int:
        pos, _, a_prec = self._depart(t, True)
        return max(pos + (self.env.inst.tis if a_prec else 0), t)

    def evaluer(self, t: int, file=None) -> Projection:
        env, inst = self.env, self.env.inst
        file = self.file if file is None else file
        fin, debuts = self.projeter(t, file)
        fin90, _ = self.projeter(t, file, p90=True)
        hors = sum(1 for pid, d in debuts.items()
                   if env.patient(pid).ambulatoire
                   and d + env.patient(pid).duree_op + inst.surveillance_ambu > inst.fermeture_uca)
        if self.en_cours:
            p = env.patient(self.en_cours[0])
            if p.ambulatoire and self.fin_acte_en_cours(t) + inst.surveillance_ambu > inst.fermeture_uca:
                hors += 1
        attente = sum(d - env.arrivee[pid] for pid, d in debuts.items() if pid in env.urgents)
        actif = bool(file) or self.en_cours is not None
        return Projection(fin, fin90,
                          max(0, fin - self.v.fin) if actif else 0,
                          max(0, fin90 - self.v.fin) if actif else 0,
                          hors, attente, debuts)

    def cout(self, pr: Projection) -> float:
        P = self.env.P
        return (P.poids_depassement * pr.dep + P.poids_depassement_p90 * pr.dep90
                + P.poids_hors_uca * pr.hors + P.poids_attente * pr.attente)

    # -- détection ---------------------------------------------------------

    def controler(self, t: int) -> None:
        pr = self.evaluer(t)
        if pr.dep > self.env.P.seuil_retard:
            self.envoyer(self.env.reg, "INFORM", sujet="retard", minutes=pr.dep,
                         resume=f"retard : fin prévue {hhmm(pr.fin)} "
                                f"(+{pr.dep} min sur {hhmm(self.v.fin)})")

    # -- protocoles --------------------------------------------------------

    def sur_cfp(self, m: Message) -> Message:
        """Appel d'offres : où insérer ce patient, à quel coût ? Seules les
        positions qui tiennent en P90 (TIS plein, marges sommées) dans la
        limite de la vacation sont proposées."""
        env, P, t = self.env, self.env.P, self.env.t
        pid = m.contenu["patient"]
        p = env.patient(pid)
        if not self.peut_accueillir(p):
            return self.repondre(m, "REFUSE", patient=pid, version=self.version,
                                 resume="hors compétence")
        if self.en_cours is None and not self.file and t >= self.v.fin:
            return self.repondre(m, "REFUSE", patient=pid, version=self.version,
                                 resume="vacation terminée")
        limite = self.limite(p)
        c0 = self.cout(self.evaluer(t))
        meilleur, fin90_min = None, None
        for k in range(len(self.file) + 1):
            f = self.file[:k] + [pid] + self.file[k:]
            pr = self.evaluer(t, f)
            fin90_min = pr.fin90 if fin90_min is None else min(fin90_min, pr.fin90)
            if pr.fin90 > limite:
                continue
            c = self.cout(pr) - c0 + P.poids_decalage * (len(self.file) - k)
            if meilleur is None or c < meilleur[0]:
                meilleur = (c, k, pr.debuts[pid])
        if meilleur is None:
            return self.repondre(m, "REFUSE", patient=pid, version=self.version,
                                 resume=f"P{pid} : pas de place (avec lui, fin P90 "
                                        f"{hhmm(fin90_min)} > {hhmm(limite)})")
        c, k, d = meilleur
        return self.repondre(m, "PROPOSE", patient=pid, position=k, cout=round(c, 1),
                             debut=d, resume=f"P{pid} en position {k + 1}, entrée "
                                             f"{hhmm(d)}, coût {c:.0f}")

    def _modifie(self) -> None:
        self.version += 1
        self.env.reveiller(self)

    def sur_accept_proposal(self, m: Message) -> Message:
        pid, k = m.contenu["patient"], m.contenu["position"]
        self.file.insert(min(k, len(self.file)), pid)
        self.env.inscrire(pid, self.vid)
        self._modifie()
        return self.repondre(m, "INFORM", sujet="fait",
                             resume=f"P{pid} inscrit en position {k + 1}")

    def sur_request(self, m: Message) -> Message:
        env = self.env
        a = m.contenu["action"]
        pid = m.contenu.get("patient")
        if a == "ceder":
            cand = [x for x in self.file if x not in env.urgents]
            return self.repondre(m, "INFORM", candidats=cand,
                                 resume=f"{len(cand)} patient(s) transférable(s)")
        if a in ("retirer", "annuler"):
            if pid in self.file:
                self.file.remove(pid)
                self._modifie()
                return self.repondre(m, "AGREE", patient=pid, resume=f"P{pid} retiré")
            return self.repondre(m, "REFUSE", patient=pid,
                                 resume=f"P{pid} déjà en salle ou absent")
        if a == "inserer":
            self.file.insert(min(m.contenu["position"], len(self.file)), pid)
            self._modifie()
            return self.repondre(m, "AGREE", patient=pid, resume=f"P{pid} ajouté")
        if a == "nouvelle_file":
            self.file = list(m.contenu["file"])
            for x in self.file:
                env.inscrire(x, self.vid)
            self._modifie()
            return self.repondre(m, "AGREE", resume="nouvel ordre appliqué")
        if a == "avancer":
            return self._avancer(m, pid, m.contenu.get("p90", False))
        return self.repondre(m, "REFUSE", resume=f"action inconnue {a}")

    def _avancer(self, m: Message, pid: int, p90: bool) -> Message:
        """Avancer un ambulatoire pour qu'il sorte avant la fermeture de
        l'UCA, sans aggraver le dépassement P90, les autres sorties tardives
        ni l'attente des urgences. On l'avance le MOINS possible."""
        env, inst, t = self.env, self.env.inst, self.env.t
        if pid not in self.file:
            return self.repondre(m, "REFUSE", patient=pid, resume="patient absent")
        i = self.file.index(pid)
        if i == 0:
            return self.repondre(m, "REFUSE", patient=pid,
                                 resume=f"P{pid} est déjà le prochain à passer")
        p, ref = env.patient(pid), self.evaluer(t)
        for k in range(i - 1, -1, -1):
            f = self.file[:]
            f.insert(k, f.pop(i))
            pr = self.evaluer(t, f)
            _, deb = self.projeter(t, f, p90=p90)
            sortie = (deb[pid] + p.duree_op + (p.marge_perso if p90 else 0)
                      + inst.surveillance_ambu)
            if (sortie <= inst.fermeture_uca and pr.dep90 <= ref.dep90
                    and pr.hors <= ref.hors and pr.attente <= ref.attente):
                self.file = f
                self._modifie()
                return self.repondre(m, "AGREE", patient=pid,
                                     resume=f"P{pid} avancé en position {k + 1}, "
                                            f"sortie prévue {hhmm(sortie)}")
        return self.repondre(m, "REFUSE", patient=pid,
                             resume=f"P{pid} : impossible sans pénaliser la salle")


class AgentLits(Agent):
    """Lits par nuit : capacité (réserve comprise, accessible aux urgences)
    − lits fermés − (lits de la Solution + conversions en nuit)."""

    def __init__(self, env):
        super().__init__(env, "Lits")

    def occupes(self, j: int) -> int:
        lits = self.env.sol.lits_jour
        return (lits[j] if 0 <= j < len(lits) else 0) + self.env.registre.conversions[j]

    def libres(self, j: int) -> int:
        return (self.env.inst.cap_lits(j) - self.env.registre.indisponibles[j]
                - self.occupes(j))

    def sur_request(self, m: Message) -> Message:
        a, nuits = m.contenu["action"], list(m.contenu.get("nuits", []))
        if a == "verifier":             # urgence : un lit chaque nuit de son séjour ?
            manque = [j for j in nuits if self.libres(j) < 1]
            if manque:
                return self.repondre(m, "REFUSE", nuits=manque,
                                     resume=f"pas de lit la nuit J{manque[0]} "
                                            f"({self.libres(manque[0])} libre)")
            return self.repondre(m, "AGREE", resume=f"lits disponibles ({len(nuits)} nuit(s))")
        if a == "convertir":            # ambulatoire sorti trop tard : il DOIT dormir
            manque = [j for j in nuits if self.libres(j) < 1]
            for j in nuits:
                self.env.registre.conversions[j] += 1
            if manque:
                return self.repondre(m, "INFORM", sujet="hebergement", nuits=manque,
                                     resume=f"hébergement hors service (nuit J{manque[0]})")
            return self.repondre(m, "AGREE", resume="lit attribué pour la nuit")
        return self.repondre(m, "REFUSE", resume=f"action inconnue {a}")

    def sur_inform(self, m: Message) -> None:
        if m.contenu.get("sujet") != "indisponibilite":
            return None
        for j in m.contenu["nuits"]:
            self.env.registre.indisponibles[j] += m.contenu["nb"]
        for j in m.contenu["nuits"]:
            if self.libres(j) < 0:
                self.envoyer(self.env.reg, "INFORM", sujet="saturation_lits", nuit=j,
                             deficit=-self.libres(j),
                             resume=f"saturation nuit J{j} : {-self.libres(j)} lit(s) manquant(s)")
        return None


class AgentAmbulatoire(Agent):
    """Admissions du jour, sorties tardives de l'UCA, conversions en nuit."""

    def __init__(self, env):
        super().__init__(env, "Ambulatoire")
        self.places: list[tuple[int, int]] = []    # (arrivée, sortie UCA)
        self.deja_demande: set[int] = set()
        self.strict = False        # True : on raisonne en P90 (lits saturés)

    def sur_request(self, m: Message) -> Message:
        """Une admission ambulatoire de plus aujourd'hui ? (capacité du jour,
        réserve comprise : seule une urgence le demande)."""
        env = self.env
        n, cap = env.sol.places_jour[env.jour], env.inst.capacite_places_jour
        if n + 1 > cap:
            return self.repondre(m, "REFUSE", resume=f"UCA pleine ({n}/{cap} admissions)")
        return self.repondre(m, "AGREE", resume=f"place disponible ({n + 1}/{cap})")

    def surveiller(self, t: int) -> None:
        env, inst = self.env, self.env.inst
        if env.P.politique != "sma":
            return
        for s in env.salles.values():
            if not s.file:
                continue
            _, debuts = s.projeter(t, p90=self.strict)
            for pid, d in debuts.items():
                p = env.patient(pid)
                if not p.ambulatoire or pid in self.deja_demande:
                    continue
                duree = p.duree_op + (p.marge_perso if self.strict else 0)
                if d + duree + inst.surveillance_ambu > inst.fermeture_uca:
                    self.deja_demande.add(pid)
                    self.envoyer(s, "REQUEST", action="avancer", patient=pid, p90=self.strict,
                                 resume=f"avancer P{pid} (sortie UCA prévue "
                                        f"{hhmm(d + duree + inst.surveillance_ambu)})")

    def sortie_de_salle(self, pid: int, debut: int, fin: int) -> None:
        env, inst = self.env, self.env.inst
        sortie = fin + inst.surveillance_ambu
        self.places.append((max(inst.ouverture_uca, debut - inst.avance_ambu), sortie))
        if sortie <= inst.fermeture_uca:
            return
        env.bilan.conversions_nuit += 1
        nuits = [env.jour] if env.jour < inst.nb_jours else []
        r = self.envoyer(env.lits, "REQUEST", action="convertir", patient=pid, nuits=nuits,
                         resume=f"P{pid} sort à {hhmm(sortie)} : conversion en nuit")
        if r is not None and r.contenu.get("sujet") == "hebergement":
            env.bilan.hebergements += 1

    def pic_places(self) -> int:
        ev = sorted([(a, 1) for a, _ in self.places] + [(b, -1) for _, b in self.places])
        n = m = 0
        for _, d in ev:
            n += d
            m = max(m, n)
        return m


class AgentRegulateur(Agent):
    """Le cadre de bloc : reçoit les aléas, négocie, décide l'escalade."""

    def __init__(self, env):
        super().__init__(env, "Régulateur")
        self.attente: list[int] = []                     # urgences sans salle
        self.refus: dict[tuple[int, int], int] = {}      # (pid, vid) -> version refusante
        self.dep_traite: dict[int, int] = {}             # vid -> dépassement déjà traité

    @property
    def pol(self) -> str:
        return self.env.P.politique

    def sur_inform(self, m: Message) -> None:
        sujet = m.contenu.get("sujet")
        if sujet == "urgence":
            self.nouvelle_urgence(m.contenu["patient"])
        elif sujet == "annulation":
            self.annulation(m.contenu["patient"])
        elif sujet == "retard":
            self.retard(self.env.par_nom[m.emetteur])
        elif sujet == "creneau_libre" and self.pol == "sma":
            self.liberation(self.env.par_nom[m.emetteur])
        elif sujet == "saturation_lits" and self.pol == "sma":
            # un ambulatoire converti en nuit n'aurait pas de lit :
            # l'UCA raisonne désormais en P90 pour faire sortir tout le monde
            self.env.ambu.strict = True
            self.env.ambu.deja_demande.clear()
            self.env.ambu.surveiller(self.env.t)
        return None

    # -- urgences ------------------------------------------------------------

    def nouvelle_urgence(self, pid: int) -> None:
        env = self.env
        if self.pol == "sma":
            if not self.contract_net(pid):
                self.attente.append(pid)
            return
        # statique / réoptimisation : la règle du modèle, au moment de l'arrivée
        if env.reprise(pid):
            vid = env.placer_statique_reprise(pid)
        else:
            vid = env.sol.placer_urgence_jour(pid)
        if vid is None:
            env.echecs.append(pid)
            self.envoyer(self, "INFORM", sujet="echec", patient=pid,
                         resume=f"P{pid} : aucune place, ni aujourd'hui ni le jour ouvré "
                                f"suivant (règle du modèle) -> échec")
            return
        v = env.inst.vacations[vid]
        if v.jour != env.jour:
            env.reportees_statique.append(pid)
            self.envoyer(self, "INFORM", sujet="report", patient=pid,
                         resume=f"P{pid} réservé J{v.jour} (V{vid}) par la règle du modèle")
            return
        s = env.salles[vid]
        self.envoyer(s, "REQUEST", action="inserer", patient=pid, position=len(s.file),
                     resume=f"urgence P{pid} en fin de programme (règle du modèle)")
        if self.pol == "reoptimisation":
            self.reoptimiser(list(env.salles.values()), "urgence")

    def _ressources_ok(self, pid: int) -> bool:
        env = self.env
        p = env.patient(pid)
        if p.ambulatoire:
            r = self.envoyer(env.ambu, "REQUEST", action="admettre", patient=pid,
                             resume=f"une admission ambulatoire pour P{pid} ?")
        else:
            r = self.envoyer(env.lits, "REQUEST", action="verifier", patient=pid,
                             nuits=list(env.inst.nuits_occupees(pid, env.jour)),
                             resume=f"lit pour l'urgence P{pid} ({p.nb_nuits} nuit(s))")
        return r is not None and r.performatif == "AGREE"

    def contract_net(self, pid: int) -> bool:
        """Contract Net Protocol : CFP aux salles compétentes, PROPOSE/REFUSE,
        ressources vérifiées (lit ou place), ACCEPT à la meilleure offre,
        REJECT aux autres."""
        env = self.env
        p = env.patient(pid)
        competentes = [s for s in env.salles.values() if s.peut_accueillir(p)]
        if not competentes:
            if pid not in self.attente:
                self.envoyer(self, "INFORM", sujet="sans_salle", patient=pid,
                             resume=f"P{pid} : ni vacation du chirurgien {p.med_id} ni "
                                    f"créneau URGENCES aujourd'hui")
            return False
        offres = []
        for s in competentes:
            if self.refus.get((pid, s.vid)) == s.version:
                continue        # a déjà refusé et n'a pas changé depuis
            r = self.envoyer(s, "CFP", tache="urgence", patient=pid,
                             resume=f"urgence P{pid} ({p.duree_op} min, "
                                    f"{'ambu' if p.ambulatoire else f'{p.nb_nuits} nuit(s)'})")
            if r.performatif == "PROPOSE":
                offres.append((r.contenu["cout"], s, r.contenu))
            else:
                self.refus[(pid, s.vid)] = r.contenu.get("version")
        if not offres:
            return False
        offres.sort(key=lambda o: o[0])
        if not self._ressources_ok(pid):
            for _, s, _ in offres:
                self.envoyer(s, "REJECT_PROPOSAL", patient=pid, resume="pas de lit / place")
            return False
        _, gagnante, c = offres[0]
        self.envoyer(gagnante, "ACCEPT_PROPOSAL", patient=pid, position=c["position"],
                     resume=f"P{pid} chez vous, position {c['position'] + 1}")
        for _, s, _ in offres[1:]:
            self.envoyer(s, "REJECT_PROPOSAL", patient=pid, resume="offre non retenue")
        env.ambu.surveiller(env.t)
        return True

    def retenter(self) -> None:
        for pid in list(self.attente):
            if self.contract_net(pid):
                self.attente.remove(pid)

    # -- annulations ---------------------------------------------------------

    def annulation(self, pid: int) -> None:
        env = self.env
        s = env.salle_de(pid)
        if s is None:
            return
        r = self.envoyer(s, "REQUEST", action="annuler", patient=pid, resume=f"annuler P{pid}")
        if r.performatif != "AGREE":
            return
        p = env.patient(pid)
        nuits = list(env.inst.nuits_occupees(pid, env.jour))
        env.desinscrire(pid)                 # libère sa place ou ses lits dans la Solution
        env.annules.append(pid)
        env.bilan.annulations += 1
        if not p.ambulatoire:
            self.envoyer(env.lits, "INFORM", sujet="liberation", nuits=nuits,
                         resume=f"{len(nuits)} nuit(s) libérée(s) par l'annulation de P{pid}")
        if self.pol == "sma":
            self.liberation(s)
        elif self.pol == "reoptimisation":
            self.reoptimiser(list(env.salles.values()), "annulation")

    def liberation(self, s: AgentSalle) -> None:
        """Du temps se libère dans s : d'abord les urgences en attente, puis
        les salles en retard du même chirurgien."""
        env, t = self.env, self.env.t
        if self.attente:
            self.retenter()
        for autre in env.salles.values():
            if (autre is not s and autre.v.med_id == s.v.med_id and not autre.v.urgence
                    and autre.evaluer(t).dep > env.P.seuil_retard):
                self.transferer(autre)

    # -- retards -------------------------------------------------------------

    def retard(self, s: AgentSalle) -> None:
        env, P, t = self.env, self.env.P, self.env.t
        if self.pol == "statique":
            return
        dep = s.evaluer(t).dep
        if dep <= P.seuil_retard or dep < self.dep_traite.get(s.vid, 0) + P.seuil_retard:
            return                                                    # rien de neuf
        if self.pol == "reoptimisation":
            self.reoptimiser(list(env.salles.values()), "retard")
        else:
            self.transferer(s)                                        # 1. local
            if s.evaluer(t).dep > P.seuil_retard:                     # 2. escalade
                meme = [x for x in env.salles.values() if x.v.med_id == s.v.med_id]
                self.reoptimiser(meme, f"retard {s.nom}")
            if s.evaluer(t).dep > P.seuil_retard:                     # 3. on assume
                # Aucun patient daté n'est déplacé à un autre jour : le
                # dépassement réel est constaté et mesuré.
                self.envoyer(s, "INFORM", sujet="depassement_accepte",
                             resume=f"dépassement accepté ({s.evaluer(t).dep} min prévues)")
            env.ambu.surveiller(t)
        self.dep_traite[s.vid] = s.evaluer(t).dep

    def transferer(self, s: AgentSalle) -> None:
        """Transfert de patients de s vers une autre vacation du MÊME
        chirurgien aujourd'hui (Contract Net, un patient à la fois). Le
        receveur doit tenir en P90 dans TVO − tampon."""
        env, P, t = self.env, self.env.P, self.env.t
        autres = [x for x in env.salles.values()
                  if x is not s and x.v.med_id == s.v.med_id and not x.v.urgence]
        if not autres or s.v.urgence:
            return
        for _ in range(P.max_transferts):
            ref = s.evaluer(t)
            if ref.dep <= P.seuil_retard:
                return
            r = self.envoyer(s, "REQUEST", action="ceder")
            meilleur = None
            for pid in r.contenu["candidats"]:
                gain = s.cout(s.evaluer(t, [x for x in s.file if x != pid])) - s.cout(ref)
                for x in autres:
                    o = self.envoyer(x, "CFP", tache="transfert", patient=pid, depuis=s.nom,
                                     resume=f"reprendre P{pid} de {s.nom} ?")
                    if o.performatif == "PROPOSE":
                        total = gain + o.contenu["cout"]
                        if meilleur is None or total < meilleur[0]:
                            meilleur = (total, pid, x, o.contenu["position"])
            if meilleur is None or meilleur[0] >= 0:
                return
            _, pid, x, k = meilleur
            self.envoyer(s, "REQUEST", action="retirer", patient=pid)
            self.envoyer(x, "ACCEPT_PROPOSAL", patient=pid, position=k,
                         resume=f"P{pid} transféré depuis {s.nom}")
            env.bilan.transferts += 1

    # -- escalade : plans B et optimiseur --------------------------------------

    def _score(self, files: dict[int, list[int]]) -> float:
        t = self.env.t
        return sum(self.env.salles[v].cout(self.env.salles[v].evaluer(t, f))
                   for v, f in files.items())

    def _tient(self, files: dict[int, list[int]]) -> bool:
        """Un plan proposé ne doit faire déborder aucune vacation au-delà de
        ce qu'elle déborde déjà (P90, limite de chaque patient)."""
        t = self.env.t
        for v, f in files.items():
            s = self.env.salles[v]
            if f == s.file:
                continue
            fin90 = s.evaluer(t, f).fin90
            lim = min((s.limite(self.env.patient(p)) for p in f), default=s.v.fin)
            if fin90 > max(lim, s.evaluer(t).fin90):
                return False
        return True

    def reoptimiser(self, declencheurs: list[AgentSalle], motif: str) -> bool:
        """Compare le plan courant aux plans B et à la proposition de
        l'optimiseur ; applique le meilleur s'il améliore le coût."""
        env, t = self.env, self.env.t
        meds = {s.v.med_id for s in declencheurs if not s.v.urgence}
        scope = {s.vid: s for s in env.salles.values()
                 if (s.v.med_id in meds and not s.v.urgence) or s in declencheurs}
        files = {v: list(s.file) for v, s in scope.items()}
        if sum(len(f) for f in files.values()) < 2:
            return False
        # Les créneaux URGENCES restent tels quels : les optimiseurs du groupe
        # raisonnent par vacation de chirurgien.
        fixes = {v: f for v, f in files.items() if scope[v].v.urgence}
        a_optimiser = {v: list(f) for v, f in files.items() if v not in fixes}
        candidats = [(f"plan B « {nom} »", env.deriver_plan_b(plan, files))
                     for nom, plan in env.plans_b.items()]
        if env.optimiseur is not None and sum(map(len, a_optimiser.values())) >= 2:
            t0 = time.perf_counter()
            try:
                res = vers_sequences(env.optimiseur(env.sous_instance(t, files), env.jour,
                                                    a_optimiser))
                candidats.append((getattr(env.optimiseur, "nom", "optimiseur"),
                                  {**res, **fixes}))
            except Exception as e:
                if env.P.strict:
                    raise
                env.bilan.echecs_optimiseur += 1
                env.erreurs.append(f"J{env.jour} {hhmm(t)} {type(e).__name__}: {e}")
                self.envoyer(self, "INFORM", sujet="echec_optimiseur",
                             resume=f"optimiseur en échec : {type(e).__name__}")
            env.bilan.temps_optimiseur_s += time.perf_counter() - t0
        actuel = self._score(files)
        meilleur = None
        for nom, c in candidats:
            c = env.valider(files, c)
            if c is None:
                env.bilan.echecs_optimiseur += 1
                env.erreurs.append(f"J{env.jour} {hhmm(t)} sortie invalide de {nom}")
                continue
            c = env.urgences_en_tete(c)
            if not self._tient(c):
                continue
            sc = self._score(c)
            if sc < actuel - 1e-6 and (meilleur is None or sc < meilleur[0]):
                meilleur = (sc, nom, c)
        if meilleur is None:
            return False
        sc, nom, c = meilleur
        for v, f in c.items():
            if f != scope[v].file:
                self.envoyer(scope[v], "REQUEST", action="nouvelle_file", file=f,
                             resume=f"nouvel ordre ({nom}, {motif}) : coût "
                                    f"{actuel:.0f} -> {sc:.0f}")
        env.bilan.reordonnancements += 1
        return True


# ---------------------------------------------------------------------------
# 5. L'environnement : simulation à événements discrets d'une journée
# ---------------------------------------------------------------------------


class Journee:
    """Une journée de bloc, simulée minute par minute avec ses aléas.

    `sol` est la Solution DE TRAVAIL : elle est modifiée (urgences inscrites,
    annulations, transferts). Passez `copie_de_travail(sol)` pour garder
    l'original ; `simuler_periode` le fait pour vous.

        work = copie_de_travail(sol)
        j = Journee(work, jour, plan_du_jour(work, jour, plan), aleas, ParametresSMA())
        bilan = j.executer()
        print(j.pf.texte())          # journal des messages
        print(j.gantt())             # ce qui s'est réellement passé
    """

    _PRIO = {"fin": 0, "alea": 1, "controle": 2, "debut": 3}

    def __init__(self, sol: Solution, jour: int, sequences: dict[int, list[int]],
                 aleas=(), P: ParametresSMA | None = None, optimiseur="glouton",
                 plans_b: dict | None = None, registre: RegistreLits | None = None,
                 durees: str = "reelles", graine: int = 0):
        self.sol, self.inst, self.jour = sol, sol.inst, jour
        inst = self.inst
        self.P = P or ParametresSMA()
        if self.P.politique not in POLITIQUES:
            raise ValueError(f"politique inconnue : {self.P.politique} ({POLITIQUES})")
        self.optimiseur = optimiseur_glouton if optimiseur == "glouton" else optimiseur
        self.plans_b = dict(plans_b or {})
        self.registre = registre if registre is not None else RegistreLits()
        self.durees = durees                     # "reelles" | "estimees" | "tirage"
        self.rng = random.Random(graine)
        self.t = 0
        self._duree: dict[int, int] = {}

        vids = sorted(inst.vacations_du_jour.get(jour, ()),
                      key=lambda v: (inst.vacations[v].bloc_id, inst.vacations[v].debut))
        inconnues = set(sequences) - set(vids)
        if inconnues:
            raise ValueError(f"J{jour} : vacations hors de la journée {sorted(inconnues)}")
        # La Solution suit le plan : un optimiseur du matin a pu changer la
        # vacation d'un patient (même chirurgien, même jour).
        for vid, seq in sequences.items():
            for pid in seq:
                p, v = inst.patients[pid], inst.vacations[vid]
                if sol.affectation.get(pid) is None or \
                        inst.vacations[sol.affectation[pid]].jour != jour:
                    raise ValueError(f"P{pid} n'est pas opéré J{jour} dans la Solution")
                if not (v.med_id == p.med_id or (v.urgence and p.est_urgent)):
                    raise ValueError(f"P{pid} dans une vacation d'un autre chirurgien")
                if sol.affectation[pid] != vid:
                    sol.affecter(pid, vid)

        # urgences déjà inscrites pour aujourd'hui (réservées la veille)
        self.urgents: dict[int, Patient] = {}
        self.arrivee: dict[int, int] = {}
        for seq in sequences.values():
            for pid in seq:
                if inst.patients[pid].est_urgent:
                    self.urgents[pid] = inst.patients[pid]
                    self.arrivee[pid] = self.P.arrivee_reprises

        self.pf = Plateforme(self)
        self.salles = {v: AgentSalle(self, v, sequences.get(v, [])) for v in vids}
        self.par_nom = {s.nom: s for s in self.salles.values()}
        self.lits = AgentLits(self)
        self.ambu = AgentAmbulatoire(self)
        self.reg = AgentRegulateur(self)
        for a in (*self.salles.values(), self.lits, self.ambu, self.reg):
            self.pf.inscrire(a)

        self.aleas = sorted(aleas, key=lambda a: a.t)
        self.prevu, self.vac_initiale = {}, {}
        for s in self.salles.values():
            for pid, d in s.projeter(0)[1].items():
                self.prevu[pid], self.vac_initiale[pid] = d, s.vid
        self.debut_reel, self.fin_reelle, self.salle_reelle = {}, {}, {}
        self.annules: list[int] = []
        self.erreurs: list[str] = []
        self.echecs: list[int] = []               # urgences définitivement sans place
        self.reportees_statique: list[int] = []   # réservées au jour suivant (statique)
        self.a_reprendre: list[Patient] = []      # urgences SMA pour le jour ouvré suivant
        self.bilan = BilanJour(jour, str(inst.date_du_jour(jour)), self.P.politique,
                               programmes=sum(1 for p in self.prevu if p not in self.urgents))
        self._tas: list = []
        self._n = 0

    # -- outils pour les agents ----------------------------------------------

    def patient(self, pid: int) -> Patient:
        return self.inst.patients[pid]

    def reprise(self, pid: int) -> bool:
        """Urgence arrivée un jour précédent (dernière chance aujourd'hui)."""
        return self.inst.patients[pid].jour_demande < self.jour

    def salle_de(self, pid: int) -> AgentSalle | None:
        return next((s for s in self.salles.values() if pid in s.file), None)

    def inscrire(self, pid: int, vid: int) -> None:
        if self.sol.affectation.get(pid) != vid:
            self.sol.affecter(pid, vid)

    def desinscrire(self, pid: int) -> None:
        self.sol.affecter(pid, None)

    def echouer(self, pid: int) -> None:
        """Urgence sans place ni J ni le jour ouvré suivant (règle du modèle)."""
        self.sol.sans_date.discard(pid)
        self.sol.hors_horizon.add(pid)
        self.echecs.append(pid)

    def placer_statique_reprise(self, pid: int) -> int | None:
        """Reprise d'une urgence de la veille, règle du modèle : aujourd'hui
        seulement (créneau URGENCES, puis vacation du chirurgien)."""
        inst, p = self.inst, self.patient(pid)
        chir = [v for v in inst.vacations_du_jour.get(self.jour, ())
                if not inst.vacations[v].urgence and inst.vacations[v].med_id == p.med_id]
        for vid in list(inst.vacations_urgence_du_jour.get(self.jour, ())) + chir:
            if self.sol._essayer(pid, vid):
                return vid
        self.echouer(pid)
        return None

    def duree_reelle(self, pid: int) -> int:
        """Connue de l'ENVIRONNEMENT seulement."""
        if pid not in self._duree:
            p = self.patient(pid)
            if self.durees == "estimees":
                d = p.duree_op
            elif self.durees == "reelles" and p.duree_reelle > 0:
                d = p.duree_reelle
            else:     # tirage : médiane = duree_op, P90 = duree_op + marge
                d = round(self.rng.gauss(p.duree_op, max(1.0, p.marge_perso / 1.2816)))
            self._duree[pid] = max(5, int(d))
        return self._duree[pid]

    def sous_instance(self, t: int, files: dict[int, list[int]]):
        """Copie de l'instance pour un optimiseur journalier QUELCONQUE : les
        vacations du jour commencent quand leur salle se libère. Si une file
        dépasse déjà, sa fin est ÉLARGIE pour que l'optimiseur voie une journée
        faisable : c'est le SMA (fins réelles) qui accepte ou rejette ce qu'il
        propose (cf. `AgentRegulateur._tient`)."""
        inst = self.inst
        i2 = copy(inst)
        i2.vacations = dict(inst.vacations)
        for vid, s in self.salles.items():
            debut = max(s.v.debut, s.debut_disponible(t))
            f = files.get(vid, s.file)
            charge = (sum(self.patient(p).duree_op + self.patient(p).marge_perso for p in f)
                      + inst.tis * max(0, len(f) - 1))
            fin = max(s.v.fin, debut + charge)
            i2.vacations[vid] = dataclasses.replace(s.v, debut=debut, fin=fin)
        return i2

    def valider(self, files, c) -> dict[int, list[int]] | None:
        """Mêmes patients, aucun chez un autre chirurgien, aucun programmé
        dans un créneau URGENCES."""
        if c is None:
            return None
        if sorted(p for f in files.values() for p in f) != sorted(p for f in c.values() for p in f):
            return None
        for vid, f in c.items():
            if vid not in files:
                if f:
                    return None
                continue
            if not all(self.salles[vid].peut_accueillir(self.patient(p)) for p in f):
                return None
        return {vid: list(c.get(vid, [])) for vid in files}

    def urgences_en_tete(self, c):
        return {v: sorted([p for p in f if p in self.urgents], key=lambda p: self.arrivee[p])
                + [p for p in f if p not in self.urgents] for v, f in c.items()}

    def deriver_plan_b(self, plan, files):
        """Applique un plan B (planning complet du matin) aux patients
        restants : chacun reprend la vacation et le rang qu'il y avait, si
        c'est possible."""
        rang, vac = {}, {}
        for vid, seq in plan.items():
            for k, pid in enumerate(seq):
                rang[pid], vac[pid] = k, vid
        out = {v: [] for v in files}
        for v, f in files.items():
            for k, pid in enumerate(f):
                cible = vac.get(pid, v)
                if cible not in out or not self.salles[cible].peut_accueillir(self.patient(pid)):
                    cible = v
                out[cible].append((rang.get(pid, 1000 + k), pid))
        return {v: [pid for _, pid in sorted(l)] for v, l in out.items()}

    # -- moteur d'événements ---------------------------------------------------

    def _pousser(self, t: int, type_: str, donnee) -> None:
        self._n += 1
        heapq.heappush(self._tas, (t, self._PRIO[type_], self._n, type_, donnee))

    def reveiller(self, s: AgentSalle) -> None:
        """La salle libre démarre son prochain patient dès que possible."""
        if s.en_cours is not None or not s.file:
            return
        s.jeton += 1
        d = s.projeter(self.t)[1][s.file[0]]
        self._pousser(max(d, self.t), "debut", (s.vid, s.jeton))

    def _demarrer(self, s: AgentSalle, t: int) -> None:
        pid = s.file.pop(0)
        p = self.patient(pid)
        s.en_cours = (pid, t)
        s.version += 1
        self.debut_reel[pid], self.salle_reelle[pid] = t, s.vid
        self._pousser(t + self.duree_reelle(pid), "fin", (s.vid, pid))
        self._pousser(t + p.duree_op, "controle", (s.vid, pid))

    def _sur_debut(self, t, donnee):
        vid, jeton = donnee
        s = self.salles[vid]
        if jeton != s.jeton or s.en_cours is not None or not s.file:
            return
        if s.projeter(t)[1][s.file[0]] > t:
            self.reveiller(s)
            return
        self._demarrer(s, t)

    def _sur_fin(self, t, donnee):
        vid, pid = donnee
        s = self.salles[vid]
        p = self.patient(pid)
        debut = s.en_cours[1]
        s.realises.append((pid, debut, t))
        s.en_cours, s.libre_a, s.dernier_type = None, t, p.type_interv
        s.version += 1
        self.fin_reelle[pid] = t
        if p.ambulatoire:
            self.ambu.sortie_de_salle(pid, debut, t)
        self.reveiller(s)
        if self.P.politique == "sma":
            if self.reg.attente:
                self.reg.retenter()
            if not s.file and t < s.v.fin - 30:
                s.envoyer(self.reg, "INFORM", sujet="creneau_libre",
                          resume=f"salle libre à {hhmm(t)} (fin {hhmm(s.v.fin)})")

    def _sur_controle(self, t, donnee):
        vid, pid = donnee
        s = self.salles[vid]
        if s.en_cours is None or s.en_cours[0] != pid:
            return
        s.controler(t)                  # l'acte dure plus que prévu : constat
        p = self.patient(pid)
        p90 = s.en_cours[1] + p.duree_op + p.marge_perso
        self._pousser(p90 if t < p90 else t + self.P.controle_toutes, "controle", donnee)

    def _sur_alea(self, t, a: Alea):
        if a.type == "urgence":
            p = a.patient
            if p.id not in self.sol.affectation:          # première arrivée
                self.sol.ajouter_urgent(p)
            self.urgents[p.id], self.arrivee[p.id] = self.patient(p.id), t
            if self.reprise(p.id):
                self.bilan.urgences_reprises += 1
            else:
                self.bilan.urgences_arrivees += 1
            self.pf.envoyer("SAU", self.reg.nom, "INFORM", sujet="urgence", patient=p.id,
                            resume=(f"{'reprise de la veille : ' if self.reprise(p.id) else ''}"
                                    f"urgence P{p.id} (chir {p.med_id}, {p.duree_op} min, "
                                    f"{'ambu' if p.ambulatoire else f'{p.nb_nuits} nuit(s)'})"))
        elif a.type == "annulation":
            if self.salle_de(a.pid) is not None:
                self.pf.envoyer("Secrétariat", self.reg.nom, "INFORM", sujet="annulation",
                                patient=a.pid, resume=f"P{a.pid} ne viendra pas")
        elif a.type == "lits":
            nuits = [j for j in range(self.jour, self.jour + a.nuits) if j < self.inst.nb_jours]
            self.pf.envoyer("Gestion lits", self.lits.nom, "INFORM", sujet="indisponibilite",
                            nb=a.nb_lits, nuits=nuits,
                            resume=f"{a.nb_lits} lit(s) fermé(s) pour {a.nuits} nuit(s)")
        else:
            raise ValueError(f"aléa inconnu : {a.type}")

    def executer(self) -> BilanJour:
        for s in self.salles.values():
            self.reveiller(s)
        for a in self.aleas:
            self._pousser(a.t, "alea", a)
        actions = {"fin": self._sur_fin, "alea": self._sur_alea,
                   "controle": self._sur_controle, "debut": self._sur_debut}
        while self._tas:
            t, _, _, type_, donnee = heapq.heappop(self._tas)
            self.t = t
            actions[type_](t, donnee)
        self._cloturer()
        return self.bilan

    # -- fin de journée ------------------------------------------------------

    def _cloturer(self) -> None:
        b, inst = self.bilan, self.inst
        for pid in self.reg.attente:        # SMA : pas de salle aujourd'hui
            if self.reprise(pid):
                self.echouer(pid)           # c'était déjà le jour ouvré suivant
            else:
                self.a_reprendre.append(self.patient(pid))
        b.urgences_reportees = len(self.a_reprendre) + len(self.reportees_statique)
        b.urgences_echec = len(self.echecs)
        operees = [pid for pid in self.urgents if pid in self.debut_reel]
        jm = [pid for pid in operees if not self.reprise(pid)]
        b.urgences_operees_jour_meme = len(jm)
        b.urgences_operees_lendemain = len(operees) - len(jm)
        b.attente_urgence_moy = (sum(self.debut_reel[p] - self.arrivee[p] for p in jm) / len(jm)
                                 if jm else 0.0)
        for s in self.salles.values():
            if s.realises:
                d = max(0, max(f for _, _, f in s.realises) - s.v.fin)
                b.depassement_min += d
                b.salles_en_depassement += d > 0
        b.deficit_lits = max(0, -self.lits.libres(self.jour))
        b.pic_places = self.ambu.pic_places()
        b.surcapacite_places = max(0, b.pic_places - inst.capacite_places)
        for pid, d in self.prevu.items():
            if pid in self.debut_reel and pid not in self.urgents:
                b.patients_decales += abs(self.debut_reel[pid] - d) > self.P.decalage_significatif
                b.changements_salle += self.salle_reelle[pid] != self.vac_initiale[pid]
        b.messages = len(self.pf.journal)

    # -- contrôles et affichage ---------------------------------------------

    def controle_coherence(self) -> None:
        """Invariants de la journée (AssertionError sinon)."""
        inst, sol = self.inst, self.sol
        for pid in self.prevu:
            assert (pid in self.debut_reel) + (pid in self.annules) == 1, \
                f"P{pid} perdu ou compté deux fois"
        for pid in self.urgents:
            fait = pid in self.debut_reel
            sans_place = pid in self.reg.attente or pid in self.echecs
            assert fait + sans_place + (pid in self.reportees_statique) == 1, \
                f"urgence P{pid}"
        for s in self.salles.values():
            assert s.en_cours is None and not s.file, f"{s.nom} n'a pas fini"
            fin = -1
            for pid, d, f in sorted(s.realises, key=lambda x: x[1]):
                assert s.peut_accueillir(self.patient(pid)), f"P{pid} au mauvais endroit"
                assert d >= s.v.debut and d >= fin, f"{s.nom} : P{pid} chevauche"
                assert d >= self.arrivee.get(pid, 0), f"P{pid} opéré avant son arrivée"
                assert sol.affectation[pid] == s.vid, f"P{pid} : Solution non à jour"
                fin = f
        for pid in self.debut_reel:
            assert inst.vacations[self.salle_reelle[pid]].jour == self.jour, \
                f"P{pid} a changé de jour"
        for pid in self.annules:
            assert sol.affectation[pid] is None, f"P{pid} annulé mais toujours inscrit"

    def gantt(self) -> str:
        lignes = [f"=== J{self.jour} {self.bilan.date} — politique « {self.P.politique} » ==="]
        for s in self.salles.values():
            if not s.realises:
                continue
            fin = max(f for _, _, f in s.realises)
            lignes.append(f"{s.nom:<12} vacation {hhmm(s.v.debut)}-{hhmm(s.v.fin)}  "
                          f"fin réelle {hhmm(fin)}"
                          + (f"  (+{fin - s.v.fin} min)" if fin > s.v.fin else ""))
            for pid, d, f in sorted(s.realises, key=lambda x: x[1]):
                p = self.patient(pid)
                tag = "URG " if pid in self.urgents else ""
                ec = self.prevu.get(pid)
                lignes.append(f"    {hhmm(d)}-{hhmm(f)}  {tag}P{pid:<9} "
                              f"{'ambu' if p.ambulatoire else str(p.nb_nuits) + ' n.':<6}"
                              f"estimé {p.duree_op:>3} min, réel {f - d:>3} min"
                              + (f", prévu {hhmm(ec)}" if ec is not None and ec != d else ""))
        return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 6. Générateur d'aléas (reproductible) et simulation d'une période
# ---------------------------------------------------------------------------


def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


@dataclass
class GenerateurAleas:
    """Aléas tirés à partir des données. MÊME graine => MÊMES aléas pour
    toutes les politiques (comparaison appariée). Les RETARDS ne sont pas
    tirés : ils viennent des durées réelles de l'historique.

    Une urgence copie le profil (acte, durées, nuits APRÈS l'opération) d'un
    patient programmé du même chirurgien, choisi au hasard : elle arrive le
    jour même, donc `nuits_avant = 0`."""
    urgences_par_jour: float = 1.5
    p_annulation: float = 0.04
    p_lits: float = 0.15
    lits_max: int = 3
    graine: int = 0
    id_base: int = 10_000_000
    _profils: dict = field(default_factory=dict, repr=False)

    def _profils_du(self, inst, med: int) -> list[Patient]:
        if not self._profils:
            par_med = defaultdict(list)
            for p in inst.patients.values():
                if not p.est_urgent:
                    par_med[p.med_id].append(p)
            self._profils.update({m: sorted(l, key=lambda p: p.id) for m, l in par_med.items()})
        return self._profils.get(med, [])

    def aleas_du_jour(self, inst, jour: int, sequences) -> list[Alea]:
        rng = random.Random(self.graine * 1_000_003 + jour)
        vids = inst.vacations_du_jour.get(jour, ())
        if not vids:
            return []
        aleas = []
        meds = sorted({inst.vacations[v].med_id for v in vids if not inst.vacations[v].urgence})
        for k in range(_poisson(rng, self.urgences_par_jour)):
            if not meds:
                break
            profils = self._profils_du(inst, rng.choice(meds))
            if not profils:
                continue
            m = rng.choice(profils)
            nuits = max(0, m.nb_nuits - m.nuits_avant)
            p = Patient(id=self.id_base + 100 * jour + k, med_id=m.med_id,
                        duree_op=m.duree_op, marge_perso=m.marge_perso,
                        duree_sejour=nuits + 1, nuits_avant=0, jour_demande=jour,
                        fenetre_jours=0, type_interv=m.type_interv,
                        duree_reelle=m.duree_reelle, urgence=URGENCE_JOUR, delai_max=None)
            aleas.append(Alea(rng.randint(8 * 60, 16 * 60), "urgence", patient=p))
        for vid, seq in sorted(sequences.items()):
            for pid in seq:
                if not inst.patients[pid].est_urgent and rng.random() < self.p_annulation:
                    v = inst.vacations[vid]
                    aleas.append(Alea(rng.randint(7 * 60, max(7 * 60, v.debut + 90)),
                                      "annulation", pid=pid))
        if rng.random() < self.p_lits:
            aleas.append(Alea(rng.randint(10 * 60, 16 * 60), "lits",
                              nb_lits=rng.randint(1, self.lits_max), nuits=1))
        return sorted(aleas, key=lambda a: a.t)


def preparer_plans(sol: Solution, jours, planificateur=None) -> dict[int, dict]:
    """Plans du matin (une fois pour toutes, pour comparer les politiques
    sur les MÊMES plans). `planificateur` : n'importe quel optimiseur
    journalier (AG, tabou, recuit...) ; None = ordre de la Solution."""
    inst, plans = sol.inst, {}
    for j in jours:
        seq = sequences_du_jour(sol, j)
        if planificateur is not None and seq:
            try:
                s2 = vers_sequences(planificateur(inst, j, seq))
                if sorted(p for f in s2.values() for p in f) == \
                        sorted(p for f in seq.values() for p in f):
                    seq = {v: f for v, f in s2.items() if f}
            except Exception as e:
                print(f"[plan J{j}] planificateur en échec ({type(e).__name__}: {e}) : "
                      f"ordre de la Solution conservé")
        plans[j] = seq
    return plans


def simuler_periode(sol: Solution, jours, generateur: GenerateurAleas | None = None,
                    P: ParametresSMA | None = None, optimiseur="glouton",
                    plans: dict | None = None, plans_b: dict | None = None,
                    durees: str = "reelles", garder_journees: bool = False):
    """Simule des journées consécutives sur une COPIE de la Solution.

    Une urgence que le SMA ne place pas aujourd'hui revient le jour ouvré
    suivant à `P.arrivee_reprises` ; si elle n'y trouve pas de place, c'est un
    échec. (En politique statique, `placer_urgence_jour` réserve directement
    le jour suivant.) Les jours doivent donc être consécutifs dans
    `jours_ouvres` ; une urgence à reprendre un jour non simulé compte comme
    échec.

    plans   : {jour: sequences} du matin (cf. preparer_plans)
    plans_b : {jour: {nom: sequences}} (cf. construire_plans_b)
    Renvoie (bilans, journees, solution_de_travail)."""
    P = P or ParametresSMA()
    work = copie_de_travail(sol)
    inst = work.inst
    registre = RegistreLits()
    reprises: list[Patient] = []
    bilans, journees = [], []
    for j in jours:
        seq = plan_du_jour(work, j, (plans or {}).get(j))
        aleas = generateur.aleas_du_jour(inst, j, seq) if generateur else []
        aleas += [Alea(P.arrivee_reprises, "urgence", patient=p) for p in reprises]
        J = Journee(work, j, seq, aleas, P, optimiseur, (plans_b or {}).get(j),
                    registre, durees, graine=j)
        bilans.append(J.executer())
        suivant = inst.jour_ouvre_suivant(j)
        reprises = J.a_reprendre
        if reprises and suivant not in jours:
            for p in reprises:
                J.echouer(p.id)
            bilans[-1].urgences_echec += len(reprises)
            bilans[-1].urgences_reportees -= len(reprises)
            reprises = []
        if garder_journees:
            journees.append(J)
    return bilans, journees, work
