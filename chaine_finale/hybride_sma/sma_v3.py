r"""
sma_v3.py — Le jour de l'opération (rapport V3, après discussion avec le groupe).

Deux façons de vivre la journée, avec EXACTEMENT les mêmes aléas :
  « sans agents » : le plan du matin est suivi tel quel. Une urgence est placée par une règle de
      capacité (créneau d'urgence, puis vacation d'un chirurgien compétent qui a de la place au
      planning), le jour même ou dans son délai ; retards et pannes sont subis.
  « avec agents » : système multi-agents (une salle = un agent, une urgence = un agent, lits,
      ambulatoire) qui négocient sur l'état réel des salles (Contract Net).

Les aléas et les règles :
  urgences      vraies urgences de 2022 avec leur priorité (diagnostic, classification FSSA) :
                U1 à opérer dans les 24 h, U2 dans les 72 h. Une urgence n'a pas de chirurgien
                attitré : elle va au premier créneau possible, avec le chirurgien de la salle
                (un chirurgien qui opère des urgences dans l'historique). Les urgences en attente
                sont servies par priorité (U1 d'abord), puis par ordre d'arrivée.
  retard        un acte dure plus que prévu, la fin du programme de la salle dépasse l'horaire.
                Agents : (1) l'ambulatoire fait avancer les patients qui sortiraient après 20 h ;
                (2) le dépassement est accepté jusqu'à LIMITE_RETARD minutes ; (3) au-delà, le
                dernier patient programmé non commencé est reporté au plus tôt.
  annulation    la place est rendue ; agents : la salle annonce un créneau libre.
  salle         une panne bloque une salle quelques heures. Agents : (1) la vacation (son
  indisponible  programme restant) est déplacée entière dans une salle libre ; (2) sinon elle
                reprend après la panne, dépassement accepté jusqu'à LIMITE_RETARD ; (3) sinon le
                dernier patient est reporté au plus tôt. Sans agents : la panne est subie.
  chirurgien    absent sans prévenir le matin : ses patients du jour sont reportés au plus tôt
  absent        dans ses vacations suivantes, en acceptant qu'il déborde de LIMITE_REPORT minutes
                au plus. Agents : ses salles libérées sont proposées aux urgences.
  lits fermés   l'agent lits prévient l'ambulatoire, qui fait sortir les ambulatoires à temps.
« Reporter au plus tôt » : première vacation du même chirurgien, à partir du lendemain, où le
patient tient avec un débordement d'au plus LIMITE_REPORT minutes.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import dataclasses
import math
import random
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

import sma_pont as SP
import sma_aleas as S
from modele import URGENCE_JOUR

hhmm = S.hhmm
LIMITE_RETARD = 60          # min de dépassement acceptées avant de reporter un patient
LIMITE_REPORT = 60          # min de débordement acceptées dans une vacation de report
SEUIL_RETARD = 10
DELAI_URGENCE = {"U1": 1, "U2": 3}     # jours calendaires
HORIZON_REPORT = 90


@dataclasses.dataclass
class BilanV3(S.BilanJour):
    reportes: int = 0
    reportes_jours: int = 0
    a_arbitrer: int = 0
    absences: int = 0
    indisponibilites: int = 0
    relogements: int = 0
    urgences_dans_delai: int = 0
    urgences_u1: int = 0
    urgences_u1_jour_meme: int = 0
    urgences_par_creneau_libre: int = 0


def _viol(pr):
    return (pr.hors, pr.dep)


def _plus(a, b):
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Salle
# ---------------------------------------------------------------------------

class SalleV3(SP.SalleQ):
    def __init__(self, env, vid, file):
        super().__init__(env, vid, file)
        self.bloques: list[tuple[int, int]] = []
        self.invites: set[int] = set()
        self.absente = False

    # -- projection avec plages bloquées
    def _decaler(self, debut, fin):
        for a, b in self.bloques:
            if debut < b and fin > a:
                return b
        return debut

    def projeter(self, t, file=None, p90=False):
        if not self.bloques:
            return super().projeter(t, file, p90)
        inst = self.env.inst
        quad = p90 and getattr(inst, "cumul_marges", "somme") == "quadratique"
        file = self.file if file is None else file
        pos, prec, a_prec = self._depart(t, p90)
        m2, debuts = 0.0, {}
        for pid in file:
            p = self.env.patient(pid)
            if a_prec:
                pos += inst.tis
            pos = max(pos, t)
            marge = (math.sqrt(m2 + p.marge_perso ** 2) if quad else p.marge_perso) if p90 else 0
            d2 = self._decaler(pos, pos + p.duree_op + marge)
            if d2 != pos:
                pos, m2 = d2, 0.0
            debuts[pid] = int(round(pos))
            pos += p.duree_op
            if p90:
                if quad:
                    m2 += float(p.marge_perso) ** 2
                else:
                    pos += p.marge_perso
            prec, a_prec = p.type_interv, True
        return int(round(pos + (math.sqrt(m2) if quad else 0))), debuts

    def libre(self):
        """Programme terminé et temps restant dans la vacation."""
        return (not self.v.urgence and not self.file and self.en_cours is None
                and self.env.t < self.v.fin - 30 and not self.bloque_maintenant())

    def bloque_maintenant(self):
        t = self.env.t
        return any(a <= t < b for a, b in self.bloques)

    # -- compétences
    def peut_accueillir(self, p):
        if p.id in self.invites:
            return True
        if p.est_urgent:
            if self.v.urgence:
                return True
            return self.v.med_id in self.env.competents and not self.absente
        return p.med_id == self.v.med_id and not self.absente

    def sur_accept_proposal(self, m):
        pid = m.contenu["patient"]
        p = self.env.patient(pid)
        if p.est_urgent and not self.v.urgence and p.med_id != self.v.med_id:
            self.env.inst.patients[pid] = dataclasses.replace(p, med_id=self.v.med_id)
        self.invites.add(pid)
        return super().sur_accept_proposal(m)

    # -- offres (agents)
    def sur_cfp(self, m):
        env, t = self.env, self.env.t
        pid = m.contenu["patient"]
        p = env.patient(pid)
        if not self.peut_accueillir(p):
            return self.repondre(m, "REFUSE", patient=pid, resume="hors compétence")
        if self.en_cours is None and not self.file and t >= self.v.fin:
            return self.repondre(m, "REFUSE", patient=pid, resume="vacation terminée")
        ref = self.evaluer(t)
        limite = self.v.fin if p.est_urgent else self.limite(p)
        meilleur = None
        k0 = len(self.file) if not p.est_urgent else 0
        for k in range(k0, len(self.file) + 1):
            f = self.file[:k] + [pid] + self.file[k:]
            pr = self.evaluer(t, f)
            if pr.fin90 > limite or pr.hors > ref.hors:
                continue
            viol = (pr.hors - ref.hors, max(0, pr.dep - ref.dep))
            decales = len(self.file) - k
            cle = (viol, pr.debuts[pid], decales)
            if meilleur is None or cle < meilleur[0]:
                meilleur = (cle, k, pr.debuts[pid], viol, decales)
        if meilleur is None:
            return self.repondre(m, "REFUSE", patient=pid, resume="pas de place")
        _, k, d, viol, decales = meilleur
        return self.repondre(m, "PROPOSE", patient=pid, position=k, debut=d, viol=viol,
                             decales=decales, resume=f"P{pid} position {k + 1}, entrée {hhmm(d)}")

    # -- retards
    def controler(self, t):
        if self.env.P.politique != "sma":
            return None
        pr = self.evaluer(t)
        if pr.dep <= SEUIL_RETARD:
            return None
        env = self.env
        self.envoyer(env.ambu, "INFORM", sujet="retard", resume=f"retard prévu {pr.dep} min")
        self.reduire(t)
        return None

    def reduire(self, t):
        """Au-delà de LIMITE_RETARD min de dépassement prévu, reporter le dernier programmé."""
        env = self.env
        while self.evaluer(t).dep > LIMITE_RETARD:
            pid = next((p for p in reversed(self.file) if p not in env.urgents), None)
            if pid is None:
                break
            self.file.remove(pid)
            self._modifie()
            env.reporter(pid, "retard")

    # -- annulation, panne
    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet == "annulation":
            return self.annuler(m.contenu["patient"])
        if sujet == "indisponibilite_salle":
            return self.indisponible(m.contenu["debut"], m.contenu["fin"])
        return None

    def annuler(self, pid):
        env = self.env
        if pid not in self.file:
            return None
        self.file.remove(pid)
        self._modifie()
        env.desinscrire(pid)
        env.annules.append(pid)
        env.bilan.annulations += 1
        if env.P.politique == "sma" and self.libre():
            env.diffuser_creneau(self)
        return None

    def indisponible(self, a, b):
        env, t = self.env, self.env.t
        if self.en_cours is not None:
            fin = self.fin_acte_en_cours(t, True)
            if fin > a:
                b, a = b + (fin - a), fin
        self.bloques.append((a, b))
        self._modifie()
        env.bilan.indisponibilites += 1
        if env.P.politique != "sma" or not self.file:
            return None
        # 1. déplacer la vacation entière dans une salle libre
        besoin = sum(env.patient(p).duree_op for p in self.file) + env.inst.tis * len(self.file)
        cands = [x for x in env.salles.values() if x is not self and x.libre()
                 and max(t, x.libre_a) + besoin <= x.v.fin + LIMITE_RETARD]
        if cands:
            x = min(cands, key=lambda x: max(t, x.libre_a))
            for pid in self.file:
                x.invites.add(pid)
                x.file.append(pid)
                env.inscrire(pid, x.vid)
            self.file = []
            self._modifie()
            x._modifie()
            env.bilan.relogements += 1
            return None
        # 2-3. reprendre après la panne, dépassement accepté jusqu'à la limite, sinon reporter
        self.reduire(t)
        return None


# ---------------------------------------------------------------------------
# Lits, ambulatoire, urgence
# ---------------------------------------------------------------------------

class LitsV3(SP.LitsC):
    def sur_inform(self, m):
        if m.contenu.get("sujet") != "indisponibilite":
            return None
        env = self.env
        reg = env.reg
        env.reg = SimpleNamespace(nom=env.ambu.nom)
        try:
            return super().sur_inform(m)
        finally:
            env.reg = reg


class AmbuV3(S.AgentAmbulatoire):
    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet == "saturation_lits":
            self.strict = True
            self.deja_demande.clear()
        if sujet in ("saturation_lits", "retard"):
            self.surveiller(self.env.t)
        return None


class AgentUrgence(S.Agent):
    def __init__(self, env, pid):
        super().__init__(env, f"U{pid}")
        self.pid = pid

    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet in ("arrivee", "creneau_libre"):
            ok = self.negocier()
            if ok:
                if sujet == "creneau_libre":
                    self.env.bilan.urgences_par_creneau_libre += 1
                self.env.desabonner(self.pid)
            elif sujet == "arrivee":
                self.env.abonner(self.pid)
        return None

    def negocier(self) -> bool:
        env = self.env
        p = env.patient(self.pid)
        offres = []
        for s in env.salles.values():
            if not s.peut_accueillir(p):
                continue
            r = self.envoyer(s, "CFP", tache="urgence", patient=self.pid, resume="urgence")
            if r.performatif == "PROPOSE":
                offres.append(((tuple(r.contenu["viol"]), r.contenu["debut"]), s, r.contenu))
        if not offres:
            return False
        if p.ambulatoire:
            r = self.envoyer(env.ambu, "REQUEST", action="admettre", patient=self.pid, resume="place ?")
        else:
            r = self.envoyer(env.lits, "REQUEST", action="verifier", patient=self.pid,
                             nuits=list(env.inst.nuits_occupees(self.pid, env.jour)), resume="lit ?")
        if r is None or r.performatif != "AGREE":
            for _, s, _ in offres:
                self.envoyer(s, "REJECT_PROPOSAL", patient=self.pid, resume="pas de lit / place")
            return False
        offres.sort(key=lambda o: o[0])
        _, s, c = offres[0]
        self.envoyer(s, "ACCEPT_PROPOSAL", patient=self.pid, position=c["position"], resume="chez vous")
        for _, x, _ in offres[1:]:
            self.envoyer(x, "REJECT_PROPOSAL", patient=self.pid, resume="non retenue")
        env.ambu.surveiller(env.t)
        return True


# ---------------------------------------------------------------------------
# La journée
# ---------------------------------------------------------------------------

class JourneeV3(S.Journee):
    def __init__(self, sol, jour, seq, aleas, P, optimiseur, plans_b, registre, durees, graine,
                 contexte):
        super().__init__(sol, jour, seq, aleas, P, optimiseur, plans_b, registre, durees, graine=graine)
        self.ctx = contexte
        self.competents = contexte.competents
        anciens = self.salles
        self.pf = S.Plateforme(self)
        self.salles = {v: SalleV3(self, v, list(s.file)) for v, s in anciens.items()}
        self.par_nom = {s.nom: s for s in self.salles.values()}
        self.lits = LitsV3(self)
        self.ambu = AmbuV3(self)
        self.reg = SimpleNamespace(nom="Cadre", attente=[])
        for x in (*self.salles.values(), self.lits, self.ambu):
            self.pf.inscrire(x)
        self.bilan = BilanV3(**dataclasses.asdict(self.bilan))
        self.reportes: list[tuple[int, int | None]] = []

    # -- outils
    def classe(self, pid):
        return self.ctx.classe.get(pid, "U2")

    def delai_ok(self, pid, jour):
        return jour - self.inst.patients[pid].jour_demande <= DELAI_URGENCE[self.classe(pid)]

    def reprise(self, pid):
        return self.inst.patients[pid].jour_demande < self.jour

    def reporter(self, pid, motif):
        """Reporter au plus tôt : première vacation du même chirurgien, dès le lendemain, où le
        patient tient avec un débordement d'au plus LIMITE_REPORT minutes."""
        sol, inst = self.sol, self.inst
        p = inst.patients[pid]
        self.desinscrire(pid)
        cible = None
        for vid in self.ctx.vac_med.get(p.med_id, ()):
            v = inst.vacations[vid]
            if v.jour <= self.jour:
                continue
            if v.jour > self.jour + HORIZON_REPORT:
                break
            n = sol.nb[vid]
            charge = sol.somme_durees[vid] + inst.tis * n
            if charge + p.duree_op + (inst.tis if n else 0) <= v.tvo + LIMITE_REPORT:
                cible = vid
                break
        if cible is not None:
            sol.affecter(pid, cible)
            self.bilan.reportes += 1
            self.bilan.reportes_jours += inst.vacations[cible].jour - self.jour
        else:
            self.bilan.a_arbitrer += 1
        self.reportes.append((pid, cible))

    def abonner(self, pid):
        if pid not in self.reg.attente:
            self.reg.attente.append(pid)

    def desabonner(self, pid):
        if pid in self.reg.attente:
            self.reg.attente.remove(pid)

    def diffuser_creneau(self, s):
        attente = sorted(self.reg.attente, key=lambda p: (self.classe(p) != "U1", self.arrivee.get(p, 0)))
        for pid in attente:
            if pid in self.reg.attente:
                s.envoyer(f"U{pid}", "INFORM", sujet="creneau_libre", resume=f"créneau libre ({s.nom})")

    # -- urgences sans agents : règle de capacité
    def placer_sans_agents(self, pid):
        inst, sol = self.inst, self.sol
        p = inst.patients[pid]
        for j in [self.jour] + [x for x in inst.jours_ouvres if self.jour < x]:
            if not self.delai_ok(pid, j):
                break
            vids = list(inst.vacations_urgence_du_jour.get(j, ()))
            vids += sorted((v for v in inst.vacations_du_jour.get(j, ())
                            if not inst.vacations[v].urgence and inst.vacations[v].med_id in self.competents),
                           key=lambda v: inst.vacations[v].debut)
            for vid in vids:
                v = inst.vacations[vid]
                if j == self.jour and vid in self.salles:
                    s = self.salles[vid]
                    if s.absente or (s.en_cours is None and not s.file and self.t >= v.fin):
                        continue
                if not v.urgence:
                    inst.patients[pid] = dataclasses.replace(p, med_id=v.med_id)
                if sol._essayer(pid, vid):
                    return vid
                inst.patients[pid] = p
        return None

    # -- aléas
    def _sur_alea(self, t, a):
        pol = self.P.politique
        if a.type == "urgence":
            p = a.patient
            if p.id not in self.sol.affectation:
                self.sol.ajouter_urgent(p)
            self.urgents[p.id], self.arrivee[p.id] = self.patient(p.id), t
            if self.reprise(p.id):
                self.bilan.urgences_reprises += 1
            else:
                self.bilan.urgences_arrivees += 1
                if self.classe(p.id) == "U1":
                    self.bilan.urgences_u1 += 1
            if pol == "sma":
                u = AgentUrgence(self, p.id)
                self.pf.inscrire(u)
                self.pf.envoyer("SAU", u.nom, "INFORM", sujet="arrivee", resume=f"urgence P{p.id}")
            else:
                vid = self.placer_sans_agents(p.id)
                if vid is None:
                    self.echouer(p.id)
                elif self.inst.vacations[vid].jour != self.jour:
                    self.reportees_statique.append(p.id)
                else:
                    s = self.salles[vid]
                    s.invites.add(p.id)
                    s.file.append(p.id)
                    s._modifie()
            return
        if a.type == "annulation":
            s = self.salle_de(a.pid)
            if s is not None:
                self.pf.envoyer("Secrétariat", s.nom, "INFORM", sujet="annulation", patient=a.pid,
                                resume="annulation")
            return
        if a.type == "lits":
            nuits = [j for j in range(self.jour, self.jour + a.nuits) if j < self.inst.nb_jours]
            self.pf.envoyer("Gestion lits", self.lits.nom, "INFORM", sujet="indisponibilite",
                            nb=a.nb_lits, nuits=nuits, resume="lits fermés")
            return
        if a.type == "indispo":
            s = self.salles.get(a.pid)
            if s is not None:
                self.pf.envoyer("Cadre", s.nom, "INFORM", sujet="indisponibilite_salle",
                                debut=max(t, a.nb_lits), fin=max(t, a.nb_lits) + a.nuits, resume="panne")
            return
        if a.type == "absence":
            self.bilan.absences += 1
            for s in self.salles.values():
                if s.v.med_id == a.pid and not s.v.urgence:
                    s.absente = True
                    for pid in list(s.file):
                        s.file.remove(pid)
                        self.reporter(pid, "absence")
                    s._modifie()
            if pol == "sma":
                for s in self.salles.values():
                    if s.v.med_id == a.pid and not s.v.urgence:
                        s.absente = False          # salle libre : un chirurgien de garde peut y opérer
                        s.v_garde = True
                        self.diffuser_creneau(s)
            return
        raise ValueError(a.type)

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
        if self.P.politique == "sma" and s.libre():
            self.diffuser_creneau(s)

    def _cloturer(self):
        b = self.bilan
        for pid in list(self.reg.attente):
            suivant = self.inst.jour_ouvre_suivant(self.jour)
            if suivant is not None and self.delai_ok(pid, suivant):
                self.a_reprendre.append(self.patient(pid))
            else:
                self.echouer(pid)
        self.reg.attente = []
        for pid in self.reportees_statique:
            pass
        super()._cloturer()
        operees = [p for p in self.urgents if p in self.debut_reel]
        b.urgences_dans_delai = len(operees)
        b.urgences_u1_jour_meme = sum(1 for p in operees if not self.reprise(p) and self.classe(p) == "U1")


# ---------------------------------------------------------------------------
# Aléas
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GenerateurV3:
    reservoir: dict                      # jour réel -> [Patient urgents]
    classes: dict                        # pid réel -> "U1" | "U2"
    intensite: float | None = None
    p_annulation: float = 0.04
    p_lits: float = 0.15
    lits_max: int = 3
    p_indispo: float = 0.05
    indispo_min: tuple = (60, 180)
    p_absence: float = 0.01              # par chirurgien et par jour opératoire
    graine: int = 0
    id_base: int = 20_000_000
    _arrivees: dict = dataclasses.field(default_factory=dict, repr=False)
    classe_de: dict = dataclasses.field(default_factory=dict, repr=False)

    def preparer(self, inst, jours):
        jours = sorted(jours)
        js = set(jours)
        arr = {j: [] for j in jours}
        if self.intensite is None:
            for j, ps in sorted(self.reservoir.items()):
                k = j if j in js else next((x for x in jours if x > j), None)
                if k is not None and (j in js or k - j <= 4):
                    for p in ps:
                        arr[k].append((p, j == k))
        else:
            pool = [p for _, ps in sorted(self.reservoir.items()) for p in ps]
            lam = self.intensite * len(pool) / max(1, len(jours))
            for j in jours:
                rng = random.Random(self.graine * 7_000_003 + j)
                for _ in range(S._poisson(rng, lam)):
                    arr[j].append((rng.choice(pool), True))
        self._arrivees = arr

    def aleas_du_jour(self, inst, jour, sequences):
        rng = random.Random(self.graine * 1_000_003 + jour)
        out = []
        for k, (m, meme_jour) in enumerate(self._arrivees.get(jour, [])):
            t = rng.randint(8 * 60, 16 * 60) if meme_jour else 7 * 60 + 30
            post = max(0, m.nb_nuits - m.nuits_avant)
            q = dataclasses.replace(m, id=self.id_base + 100 * jour + k, urgence=URGENCE_JOUR, jour_demande=jour,
                                    fenetre_jours=0, delai_max=None, nuits_avant=0, duree_sejour=post + 1,
                                    ambulatoire=None)
            self.classe_de[q.id] = self.classes.get(m.id, "U2")
            out.append(S.Alea(t, "urgence", patient=q))
        # tirages indépendants par type d'aléa (et par patient / chirurgien) : les deux politiques
        # reçoivent les mêmes aléas même si leurs plannings diffèrent après un report
        meds = sorted({inst.vacations[v].med_id for v, s in sequences.items() if s and not inst.vacations[v].urgence})
        for m in meds:
            if random.Random(self.graine * 9_000_011 + jour * 101 + m).random() < self.p_absence:
                out.append(S.Alea(7 * 60, "absence", pid=m))
        for vid, seq in sorted(sequences.items()):
            for pid in seq:
                ra = random.Random(self.graine * 5_000_011 + pid)
                if not inst.patients[pid].est_urgent and ra.random() < self.p_annulation:
                    v = inst.vacations[vid]
                    out.append(S.Alea(ra.randint(7 * 60, max(7 * 60, v.debut + 90)), "annulation", pid=pid))
        rl = random.Random(self.graine * 3_000_017 + jour)
        if rl.random() < self.p_lits:
            out.append(S.Alea(rl.randint(10 * 60, 16 * 60), "lits", nb_lits=rl.randint(1, self.lits_max), nuits=1))
        rng = random.Random(self.graine * 2_000_003 + jour)
        vids = [v for v, s in sorted(sequences.items()) if s and not inst.vacations[v].urgence]
        if vids and rng.random() < self.p_indispo:
            vid = rng.choice(vids)
            v = inst.vacations[vid]
            debut = rng.randint(v.debut, max(v.debut, v.fin - 60))
            out.append(S.Alea(max(7 * 60, debut - rng.randint(0, 60)), "indispo", pid=vid, nb_lits=debut,
                              nuits=rng.randint(*self.indispo_min)))
        return sorted(out, key=lambda a: (a.t, a.type != "absence"))


def contexte(df, don, inst):
    """Chirurgiens qui opèrent des urgences dans l'historique, classes des urgences réelles,
    vacations de chaque chirurgien par jour."""
    from protocole import est_urgent_diag
    import urgences_fssa as U
    h = df[df["annee"].isin([2019, 2020, 2021])]
    u = h[est_urgent_diag(h["CIM Diag Pr."].astype("string").fillna("").tolist())]
    cnt = u["Praticien"].value_counts()
    codes = {c: m for m, c in don.med_codes.items()}
    competents = {codes[c] for c, n in cnt.items() if n >= 10 and c in codes}
    classes = {p: U.classer(str(df.loc[p, "CIM Diag Pr."]))["classe_urgence"] for p in don.urgents}
    vac_med = defaultdict(list)
    for vid, v in sorted(inst.vacations.items(), key=lambda kv: (kv[1].jour, kv[1].debut)):
        if not v.urgence:
            vac_med[v.med_id].append(vid)
    return SimpleNamespace(competents=competents, classes_reelles=classes, vac_med=dict(vac_med), classe={})


def reservoir_urgences(don, debut, fin):
    res = {}
    for p, j in don.jour_urgent.items():
        if debut <= j <= fin:
            res.setdefault(j, []).append(don.urgents[p])
    return res


def simuler(matin, generateur, politique, jours, ctx, durees="reelles", don_reel=None):
    """Journées consécutives sur une copie de la Solution du matin. Les lits sont mesurés avec les
    séjours réels (don_reel) quand le plan a été fait sur des séjours prévus."""
    P = S.ParametresSMA(politique=politique)
    work = S.copie_de_travail(matin["sol"])
    inst = work.inst
    inst.capacite_lits_nuit = (list(inst.capacite_lits_nuit) if inst.capacite_lits_nuit is not None
                               else [inst.capacite_lits] * inst.nb_jours)
    inst._cap_calendrier = list(inst.capacite_lits_nuit)
    if generateur is not None:
        generateur.preparer(inst, jours)
        ctx.classe = generateur.classe_de
    registre = S.RegistreLits()
    reprises, bilans, salles = [], [], []
    t0 = time.perf_counter()
    for j in jours:
        seq = S.plan_du_jour(work, j, matin["plans"].get(j))
        aleas = generateur.aleas_du_jour(inst, j, seq) if generateur else []
        for p in reprises:
            aleas.append(S.Alea(P.arrivee_reprises, "urgence", patient=inst.patients[p.id]))
        J = JourneeV3(work, j, seq, aleas, P, None, None, registre, durees, j, ctx)
        b = J.executer()
        bilans.append(b)
        for s in J.salles.values():
            if s.realises:
                fin = max(f for _, _, f in s.realises)
                salles.append(dict(jour=j, tvo=s.v.tvo, opere=sum(f - d for _, d, f in s.realises),
                                   depassement=max(0, fin - s.v.fin), urgence=bool(s.v.urgence)))
        suivant = inst.jour_ouvre_suivant(j)
        reprises = J.a_reprendre
        if reprises and suivant not in jours:
            for p in reprises:
                J.echouer(p.id)
            b.urgences_echec += len(reprises)
            reprises = []
    # lits réellement occupés : séjours réels des programmés, urgences à leur date d'opération
    n = len(work.lits_jour)
    lits, places = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for pid, vid in work.affectation.items():
        if vid is None:
            continue
        p = (don_reel.patients.get(pid) if don_reel is not None else None) or inst.patients[pid]
        j = inst.vacations[vid].jour
        if p.nb_nuits == 0:
            places[j] += 1
        else:
            d = j - p.nuits_avant
            lits[max(0, d):min(n, d + p.nb_nuits)] += 1
    lits = lits + np.array([registre.conversions[k] for k in range(n)])
    R = S.resumer(bilans)
    return dict(resume=R, lits=lits.tolist(), places=places.tolist(), salles=salles,
                secondes=time.perf_counter() - t0)
