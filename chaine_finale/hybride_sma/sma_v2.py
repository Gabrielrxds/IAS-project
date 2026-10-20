r"""
sma_v2.py — Système multi-agents, version 2 (compte-rendu SMA du groupe), adapté au protocole.

Différences avec sma_aleas.py (version 1, avec un agent « Régulateur » central) :

  * PLUS DE RÉGULATEUR. Chaque aléa est remis à l'agent qu'il concerne, qui lance lui-même la
    négociation (Contract Net) :
        urgence             -> un AgentUrgence (un par patient urgent) ;
        retard              -> la salle en retard ;
        annulation          -> la salle du patient ;
        lits fermés         -> l'agent Lits, qui prévient directement l'agent Ambulatoire ;
        salle indisponible  -> la salle touchée (panne, matériel, personnel : une plage bloquée).
  * RÈGLE DE DÉCISION LEXICOGRAPHIQUE, sans formule de coût pondérée :
        1. contraintes dures : fin P90 dans le TVO (urgent) ou dans TVO − tampon (programmé),
           aucune sortie ambulatoire de plus après la fermeture de l'UCA, accord des agents
           Lits ou Ambulatoire ;
        2. le moins de violations : d'abord les ambulatoires qui sortiraient après la fermeture,
           puis les minutes de dépassement prévues ;
        3. le critère de date, le même qu'en consultation (la hausse de l'écart des lits à la
           cible) ; toutes les offres d'une même journée y sont égales ;
        4. départage : urgence -> l'entrée au bloc la plus tôt ; transfert -> le moins de
           patients décalés ; nouveau plan -> les urgences le plus tôt, puis le moins de
           patients décalés.
  * Une urgence sans offre s'abonne aux « créneaux libres » : toute salle qui se libère
    (annulation, fin de programme en avance) le diffuse, et l'urgence retente sa chance.
    Le soir, elle passe au jour ouvré suivant ; si elle n'y trouve pas de place : échec.
  * Les urgences sont les VRAIES urgences 2022 (diagnostic FSSA, cf. protocole.separer_urgences),
    arrivées le jour de leur opération réelle à une heure tirée entre 8 h et 16 h (week-end et
    jours sans bloc : le jour ouvré suivant à 7 h 30). Pour faire varier l'intensité, on tire
    un nombre de Poisson par jour et on rééchantillonne ce même réservoir de patients.

Les politiques « statique » et « reoptimisation » (références) restent celles de sma_aleas.py,
avec la gestion d'une salle indisponible ajoutée. Les salles sont les salles 2 à 5.
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
from types import SimpleNamespace

import numpy as np

import sma_pont as SP          # installe SalleQ / LitsC (marges quadratiques, lits fermés)
import sma_aleas as S
from modele import URGENCE_JOUR

hhmm = S.hhmm


@dataclasses.dataclass
class BilanV2(S.BilanJour):
    indisponibilites: int = 0
    transferts_indispo: int = 0
    reprogrammes_indispo: int = 0
    creneaux_libres: int = 0
    urgences_par_creneau_libre: int = 0


def _viol(pr) -> tuple:
    """Violations d'une projection : (ambulatoires hors UCA, minutes de dépassement)."""
    return (pr.hors, pr.dep)


def _plus(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Salle
# ---------------------------------------------------------------------------

class SalleV2(SP.SalleQ):
    """Agent salle V2 : gère lui-même ses retards, ses annulations et ses indisponibilités."""

    def __init__(self, env, vid, file):
        super().__init__(env, vid, file)
        self.bloques: list[tuple[int, int]] = []     # plages indisponibles [a, b)
        self.dep_traite = 0
        self.invites: set[int] = set()               # urgences reçues sur un créneau libre

    def sur_accept_proposal(self, m):
        p = self.env.patient(m.contenu["patient"])
        if not super(SP.SalleQ, self).peut_accueillir(p):
            self.invites.add(p.id)
        return super().sur_accept_proposal(m)

    # -- projection avec plages bloquées --------------------------------------
    def _decaler(self, debut: int, fin: int) -> int:
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
        m2 = 0.0
        debuts = {}
        for pid in file:
            p = self.env.patient(pid)
            if a_prec:
                pos += inst.tis if (p90 or not (p.type_interv and p.type_interv == prec)) else inst.tis_meme_acte
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
        fin = pos + (math.sqrt(m2) if quad else 0)
        return int(round(fin)), debuts

    def peut_accueillir(self, p):
        """SMA : une salle qui a fini son programme avant la fin de sa vacation devient un
        « créneau libre » ouvert à toute urgence (son chirurgien vient y opérer)."""
        if super().peut_accueillir(p) or p.id in self.invites:
            return True
        env = self.env
        return (env.P.politique == "sma" and p.est_urgent and not self.v.urgence
                and not self.file and self.en_cours is None and env.t < self.v.fin - 30
                and not env.chirurgien_occupe(p.med_id))

    # -- offres -------------------------------------------------------------------
    def sur_cfp(self, m):
        env, t = self.env, self.env.t
        if env.P.politique != "sma":
            return super().sur_cfp(m)
        pid, tache = m.contenu["patient"], m.contenu.get("tache", "urgence")
        p = env.patient(pid)
        if not self.peut_accueillir(p):
            return self.repondre(m, "REFUSE", patient=pid, resume="hors compétence")
        if self.en_cours is None and not self.file and t >= self.v.fin:
            return self.repondre(m, "REFUSE", patient=pid, resume="vacation terminée")
        ref = self.evaluer(t)
        limite = self.limite(p)
        meilleur = None
        for k in range(len(self.file) + 1):
            f = self.file[:k] + [pid] + self.file[k:]
            pr = self.evaluer(t, f)
            if pr.fin90 > limite or pr.hors > ref.hors:          # contraintes dures
                continue
            viol = (pr.hors - ref.hors, max(0, pr.dep - ref.dep))
            decales = len(self.file) - k
            cle = (viol, pr.debuts[pid]) if tache == "urgence" else (viol, decales, pr.debuts[pid])
            if meilleur is None or cle < meilleur[0]:
                meilleur = (cle, k, pr.debuts[pid], viol, decales)
        if meilleur is None:
            return self.repondre(m, "REFUSE", patient=pid, resume=f"P{pid} : pas de place")
        _, k, d, viol, decales = meilleur
        return self.repondre(m, "PROPOSE", patient=pid, position=k, debut=d, viol=viol,
                             decales=decales, resume=f"P{pid} position {k + 1}, entrée {hhmm(d)}")

    def sur_request(self, m):
        if m.contenu.get("action") == "examiner_plan":
            return self._examiner_plan(m)
        return super().sur_request(m)

    def _examiner_plan(self, m):
        """Un plan propose une nouvelle file : la salle l'accepte si elle respecte ses contraintes
        dures, ou si elle ne fait pas déborder la salle plus qu'aujourd'hui."""
        t, f = self.env.t, list(m.contenu["file"])
        pr, ref = self.evaluer(t, f), self.evaluer(t)
        lim = min((self.limite(self.env.patient(p)) for p in f), default=self.v.fin)
        ok = (pr.fin90 <= lim or pr.fin90 <= ref.fin90) and pr.hors <= max(ref.hors, 0) + 0
        if not ok:
            return self.repondre(m, "REFUSE", resume="plan refusé (fin P90 ou sortie UCA)")
        return self.repondre(m, "AGREE", viol=_viol(pr), resume="plan acceptable")

    # -- retards ---------------------------------------------------------------
    def controler(self, t):
        if self.env.P.politique != "sma":
            return super().controler(t)
        dep = self.evaluer(t).dep
        if dep > self.env.P.seuil_retard and dep >= self.dep_traite + self.env.P.seuil_retard:
            self.traiter_retard()

    def soeurs(self):
        return [x for x in self.env.salles.values()
                if x is not self and not x.v.urgence and x.v.med_id == self.v.med_id]

    def transferer(self, n_max=None, forcer=False) -> int:
        """Cède des patients aux vacations sœurs (même chirurgien, même jour). Renvoie le nombre
        de transferts."""
        env, P, t = self.env, self.env.P, self.env.t
        if self.v.urgence:
            return 0
        n = 0
        for _ in range(n_max or P.max_transferts):
            ref = self.evaluer(t)
            if not forcer and ref.dep <= P.seuil_retard:
                break
            soeurs = self.soeurs()
            if not soeurs:
                break
            cand = [x for x in self.file if x not in env.urgents]
            meilleur = None
            for pid in cand:
                apres = _viol(self.evaluer(t, [x for x in self.file if x != pid]))
                for x in soeurs:
                    o = self.envoyer(x, "CFP", tache="transfert", patient=pid, depuis=self.nom,
                                     resume=f"reprendre P{pid} ?")
                    if o.performatif != "PROPOSE":
                        continue
                    cle = (_plus(apres, tuple(o.contenu["viol"])), o.contenu["decales"])
                    if meilleur is None or cle < meilleur[0]:
                        meilleur = (cle, pid, x, o.contenu["position"])
            if meilleur is None or (not forcer and meilleur[0][0] >= _viol(ref)):
                break
            _, pid, x, k = meilleur
            self.file.remove(pid)
            self._modifie()
            self.envoyer(x, "ACCEPT_PROPOSAL", patient=pid, position=k,
                         resume=f"P{pid} transféré depuis {self.nom}")
            env.bilan.transferts += 1
            n += 1
        return n

    def traiter_retard(self):
        env, P, t = self.env, self.env.P, self.env.t
        self.transferer()                                           # 1. transferts
        if self.evaluer(t).dep > P.seuil_retard:                    # 2. escalade : plans B / optimiseur
            self.escalade()
        if self.evaluer(t).dep > P.seuil_retard:                    # 3. dépassement accepté
            self.envoyer(env.ambu, "INFORM", sujet="depassement_accepte",
                         resume=f"dépassement accepté ({self.evaluer(t).dep} min prévues)")
        self.dep_traite = self.evaluer(t).dep

    def escalade(self) -> bool:
        env, t = self.env, self.env.t
        scope = {self.vid: self, **{x.vid: x for x in self.soeurs()}}
        files = {v: list(s.file) for v, s in scope.items()}
        if sum(map(len, files.values())) < 2:
            return False
        candidats = [(f"plan B « {nom} »", env.deriver_plan_b(plan, files))
                     for nom, plan in env.plans_b.items()]
        if env.optimiseur is not None:
            t0 = time.perf_counter()
            try:
                res = S.vers_sequences(env.optimiseur(env.sous_instance(t, files), env.jour, files))
                candidats.append((getattr(env.optimiseur, "nom", "optimiseur"), res))
            except Exception as e:                                    # noqa: BLE001
                env.bilan.echecs_optimiseur += 1
                env.erreurs.append(f"J{env.jour} {hhmm(t)} {type(e).__name__}: {e}")
            env.bilan.temps_optimiseur_s += time.perf_counter() - t0
        actuel = (sum((_viol(s.evaluer(t)) for s in scope.values()), (0, 0)), 0, 0)
        meilleur = None
        for nom, c in candidats:
            c = env.valider(files, c)
            if c is None:
                continue
            c = env.urgences_en_tete(c)
            viol, ok = (0, 0), True
            for v, f in c.items():
                if f == scope[v].file:
                    viol = _plus(viol, _viol(scope[v].evaluer(t)))
                    continue
                r = self.envoyer(scope[v], "REQUEST", action="examiner_plan", file=f,
                                 resume=f"examiner {nom}")
                if r.performatif != "AGREE":
                    ok = False
                    break
                viol = _plus(viol, tuple(r.contenu["viol"]))
            if not ok:
                continue
            urg = sum(d for v, f in c.items() for p, d in scope[v].evaluer(t, f).debuts.items()
                      if p in env.urgents)
            dec = sum(1 for v, f in c.items() for k, p in enumerate(f)
                      if p not in scope[v].file or scope[v].file.index(p) != k)
            cle = (viol, urg, dec)
            if cle[0] < actuel[0] and (meilleur is None or cle < meilleur[0]):
                meilleur = (cle, nom, c)
        if meilleur is None:
            return False
        _, nom, c = meilleur
        for v, f in c.items():
            if f != scope[v].file:
                self.envoyer(scope[v], "REQUEST", action="nouvelle_file", file=f,
                             resume=f"nouvel ordre ({nom})")
        env.bilan.reordonnancements += 1
        return True

    # -- annulation, créneau libre ----------------------------------------------
    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet == "annulation":
            return self.annuler(m.contenu["patient"])
        if sujet == "indisponibilite_salle":
            return self.indisponible(m.contenu["debut"], m.contenu["fin"])
        if sujet == "creneau_libre" and self.env.P.politique == "sma":
            if self.evaluer(self.env.t).dep > self.env.P.seuil_retard:
                self.transferer()
        return None

    def annuler(self, pid):
        env = self.env
        if pid not in self.file:
            return None
        self.file.remove(pid)
        self._modifie()
        p = env.patient(pid)
        nuits = list(env.inst.nuits_occupees(pid, env.jour))
        env.desinscrire(pid)
        env.annules.append(pid)
        env.bilan.annulations += 1
        if not p.ambulatoire:
            self.envoyer(env.lits, "INFORM", sujet="liberation", nuits=nuits,
                         resume=f"{len(nuits)} nuit(s) libérée(s)")
        if env.P.politique == "sma":
            env.diffuser_creneau(self)
        elif env.P.politique == "reoptimisation":
            env.reg.reoptimiser(list(env.salles.values()), "annulation")
        return None

    def indisponible(self, a, b):
        """Plage [a, b) bloquée : les patients qui ne tiennent plus (fin P90 au-delà de la limite)
        sont proposés aux vacations sœurs (SMA) puis rendus à la reprogrammation."""
        env, t = self.env, self.env.t
        if self.en_cours is not None:
            fin = self.fin_acte_en_cours(t, True)
            if fin > a:
                b, a = b + (fin - a), fin
        self.bloques.append((a, b))
        self._modifie()
        env.bilan.indisponibilites += 1
        pol = env.P.politique
        if pol == "reoptimisation":
            env.reg.reoptimiser(list(env.salles.values()), "indisponibilité")

        def deborde():
            if not self.file:
                return False
            lim = min(self.limite(env.patient(p)) for p in self.file)
            return self.evaluer(t).fin90 > lim

        while deborde():
            if pol == "sma":
                n = self.transferer(n_max=1, forcer=True)
                if n:
                    env.bilan.transferts_indispo += 1
                    continue
            pid = next((p for p in reversed(self.file) if p not in env.urgents), None)
            if pid is None:
                break
            self.file.remove(pid)
            self._modifie()
            env.reprogrammer(pid)
        return None


# ---------------------------------------------------------------------------
# Lits, Ambulatoire, Urgence
# ---------------------------------------------------------------------------

class LitsV2(SP.LitsC):
    def sur_inform(self, m):
        if m.contenu.get("sujet") != "indisponibilite":
            return None
        env = self.env
        reg = env.reg
        env.reg = SimpleNamespace(nom=env.ambu.nom)       # la saturation va à l'Ambulatoire
        try:
            return super().sur_inform(m)
        finally:
            env.reg = reg


class AmbuV2(S.AgentAmbulatoire):
    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet == "saturation_lits":
            self.strict = True
            self.deja_demande.clear()
            self.surveiller(self.env.t)
        elif sujet == "depassement_accepte":
            self.surveiller(self.env.t)
        return None

    def sur_request(self, m):
        return super().sur_request(m)


class AgentUrgence(S.Agent):
    def __init__(self, env, pid):
        super().__init__(env, f"U{pid}")
        self.pid = pid

    def sur_inform(self, m):
        sujet = m.contenu.get("sujet")
        if sujet in ("arrivee", "creneau_libre"):
            ok = self.negocier()
            if ok and sujet == "creneau_libre":
                self.env.bilan.urgences_par_creneau_libre += 1
            if ok:
                self.env.desabonner(self.pid)
            elif sujet == "arrivee":
                self.env.abonner(self.pid)
        return None

    def negocier(self) -> bool:
        env = self.env
        p = env.patient(self.pid)
        competentes = [s for s in env.salles.values() if s.peut_accueillir(p)]
        offres = []
        for s in competentes:
            r = self.envoyer(s, "CFP", tache="urgence", patient=self.pid,
                             resume=f"urgence P{self.pid} ({p.duree_op} min)")
            if r.performatif == "PROPOSE":
                offres.append(((tuple(r.contenu["viol"]), r.contenu["debut"]), s, r.contenu))
        if not offres:
            return False
        if p.ambulatoire:
            r = self.envoyer(env.ambu, "REQUEST", action="admettre", patient=self.pid,
                             resume="une admission ?")
        else:
            r = self.envoyer(env.lits, "REQUEST", action="verifier", patient=self.pid,
                             nuits=list(env.inst.nuits_occupees(self.pid, env.jour)),
                             resume=f"lit pour {p.nb_nuits} nuit(s) ?")
        if r is None or r.performatif != "AGREE":
            for _, s, _ in offres:
                self.envoyer(s, "REJECT_PROPOSAL", patient=self.pid, resume="pas de lit / place")
            return False
        offres.sort(key=lambda o: o[0])
        _, gagnante, c = offres[0]
        self.envoyer(gagnante, "ACCEPT_PROPOSAL", patient=self.pid, position=c["position"],
                     resume=f"P{self.pid} chez vous")
        for _, s, _ in offres[1:]:
            self.envoyer(s, "REJECT_PROPOSAL", patient=self.pid, resume="offre non retenue")
        env.ambu.surveiller(env.t)
        return True


# ---------------------------------------------------------------------------
# Journées
# ---------------------------------------------------------------------------

class _Base(S.Journee):
    """Agents salle V2 (plages bloquées) pour toutes les politiques ; bilan V2."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        anciens = self.salles
        self.pf = S.Plateforme(self)
        self.salles = {v: SalleV2(self, v, list(s.file)) for v, s in anciens.items()}
        self.par_nom = {s.nom: s for s in self.salles.values()}
        self.lits = self._lits()
        self.ambu = self._ambu()
        self.reg = self._reg()
        for x in (*self.salles.values(), self.lits, self.ambu):
            self.pf.inscrire(x)
        if isinstance(self.reg, S.Agent):
            self.pf.inscrire(self.reg)
        b = self.bilan
        self.bilan = BilanV2(**dataclasses.asdict(b))
        self.reprogrammes: list[int] = []

    def _lits(self):
        return SP.LitsC(self)

    def _ambu(self):
        return S.AgentAmbulatoire(self)

    def _reg(self):
        return S.AgentRegulateur(self)

    def reprogrammer(self, pid):
        """Rendu à la reprogrammation (hors de la journée) : libère sa place et ses lits."""
        self.desinscrire(pid)
        self.reprogrammes.append(pid)
        self.bilan.reprogrammes_indispo += 1

    def _sur_alea(self, t, a):
        if a.type == "indispo":
            s = self.salles.get(a.pid)
            if s is not None:
                self.pf.envoyer("Cadre de bloc", s.nom, "INFORM", sujet="indisponibilite_salle",
                                debut=max(t, a.nb_lits), fin=max(t, a.nb_lits) + a.nuits,
                                resume=f"salle indisponible {a.nuits} min")
            return
        if a.type == "annulation" and self.P.politique != "sma":
            s = self.salle_de(a.pid)
            if s is not None:
                self.pf.envoyer("Secrétariat", s.nom, "INFORM", sujet="annulation", patient=a.pid,
                                resume=f"P{a.pid} ne viendra pas")
            return
        super()._sur_alea(t, a)

    def controle_coherence(self):
        # les patients rendus à la reprogrammation comptent comme annulés pour l'invariant
        self.annules = self.annules + [p for p in self.reprogrammes if p not in self.annules]
        super().controle_coherence()


class JourneeRef(_Base):
    """Politiques de référence (statique, réoptimisation) avec le régulateur de la V1."""


class JourneeV2(_Base):
    """Politique SMA V2 : pas de régulateur."""

    def _lits(self):
        return LitsV2(self)

    def _ambu(self):
        return AmbuV2(self)

    def _reg(self):
        return SimpleNamespace(nom="—", attente=[])

    def chirurgien_occupe(self, med):
        """Le chirurgien opère en ce moment, ou a encore des patients dans une de ses salles."""
        return any(s.v.med_id == med and not s.v.urgence and (s.en_cours is not None or s.file)
                   for s in self.salles.values())

    def abonner(self, pid):
        if pid not in self.reg.attente:
            self.reg.attente.append(pid)

    def desabonner(self, pid):
        if pid in self.reg.attente:
            self.reg.attente.remove(pid)

    def diffuser_creneau(self, s):
        self.bilan.creneaux_libres += 1
        for pid in list(self.reg.attente):
            s.envoyer(f"U{pid}", "INFORM", sujet="creneau_libre", resume=f"créneau libre ({s.nom})")
        for x in s.soeurs():
            if x.evaluer(self.t).dep > self.P.seuil_retard:
                s.envoyer(x, "INFORM", sujet="creneau_libre", resume="créneau libre")

    def _sur_alea(self, t, a):
        if a.type == "urgence":
            p = a.patient
            if p.id not in self.sol.affectation:
                self.sol.ajouter_urgent(p)
            self.urgents[p.id], self.arrivee[p.id] = self.patient(p.id), t
            if self.reprise(p.id):
                self.bilan.urgences_reprises += 1
            else:
                self.bilan.urgences_arrivees += 1
            u = AgentUrgence(self, p.id)
            self.pf.inscrire(u)
            self.pf.envoyer("SAU", u.nom, "INFORM", sujet="arrivee",
                            resume=f"urgence P{p.id} (chir {p.med_id}, {p.duree_op} min)")
            return
        if a.type == "annulation":
            s = self.salle_de(a.pid)
            if s is not None:
                self.pf.envoyer("Secrétariat", s.nom, "INFORM", sujet="annulation", patient=a.pid,
                                resume=f"P{a.pid} ne viendra pas")
            return
        if a.type == "lits":
            nuits = [j for j in range(self.jour, self.jour + a.nuits) if j < self.inst.nb_jours]
            self.pf.envoyer("Gestion lits", self.lits.nom, "INFORM", sujet="indisponibilite",
                            nb=a.nb_lits, nuits=nuits, resume=f"{a.nb_lits} lit(s) fermé(s)")
            return
        super()._sur_alea(t, a)

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
        if not s.file and t < s.v.fin - 30:
            self.diffuser_creneau(s)


# ---------------------------------------------------------------------------
# Aléas : vraies urgences 2022, annulations, lits fermés, salles indisponibles
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GenerateurV2:
    """MÊME graine => MÊMES aléas pour toutes les politiques.
    intensite = None : les urgences réelles, à leur date ; un nombre x : Poisson de moyenne
    x fois le débit réel par jour ouvré, patients rééchantillonnés dans le même réservoir."""
    reservoir: dict                      # jour réel -> [Patient urgents]
    intensite: float | None = None
    p_annulation: float = 0.04
    p_lits: float = 0.15
    lits_max: int = 3
    p_indispo: float = 0.05              # par jour : une salle indisponible
    indispo_min: tuple = (60, 180)
    graine: int = 0
    id_base: int = 20_000_000
    _arrivees: dict = dataclasses.field(default_factory=dict, repr=False)

    def preparer(self, inst, jours):
        """Arrivées par jour ouvré simulé."""
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
            pool = [p for _, ps in sorted(self.reservoir.items()) for p in ps
                    if jours[0] <= _ <= jours[-1]]
            lam = self.intensite * len(pool) / max(1, len(jours))
            for j in jours:
                rng = random.Random(self.graine * 7_000_003 + j)
                for _ in range(S._poisson(rng, lam)):
                    arr[j].append((rng.choice(pool), True))
        self._arrivees = arr

    def _patient(self, m, jour, k):
        post = max(0, m.nb_nuits - m.nuits_avant)
        return dataclasses.replace(m, id=self.id_base + 100 * jour + k, urgence=URGENCE_JOUR,
                                   jour_demande=jour, fenetre_jours=0, delai_max=None,
                                   nuits_avant=0, duree_sejour=post + 1, ambulatoire=None)

    def aleas_du_jour(self, inst, jour, sequences):
        rng = random.Random(self.graine * 1_000_003 + jour)
        out = []
        for k, (m, meme_jour) in enumerate(self._arrivees.get(jour, [])):
            t = rng.randint(8 * 60, 16 * 60) if meme_jour else 7 * 60 + 30
            p = chirurgien_de_garde(inst, jour, self._patient(m, jour, k), rng)
            out.append(S.Alea(t, "urgence", patient=p))
        for vid, seq in sorted(sequences.items()):
            for pid in seq:
                if not inst.patients[pid].est_urgent and rng.random() < self.p_annulation:
                    v = inst.vacations[vid]
                    out.append(S.Alea(rng.randint(7 * 60, max(7 * 60, v.debut + 90)), "annulation", pid=pid))
        if rng.random() < self.p_lits:
            out.append(S.Alea(rng.randint(10 * 60, 16 * 60), "lits",
                              nb_lits=rng.randint(1, self.lits_max), nuits=1))
        vids = [v for v, s in sorted(sequences.items()) if s and not inst.vacations[v].urgence]
        if vids and rng.random() < self.p_indispo:
            vid = rng.choice(vids)
            v = inst.vacations[vid]
            t = rng.randint(max(7 * 60, v.debut - 60), max(7 * 60, v.debut + 60))
            debut = rng.randint(v.debut, max(v.debut, v.fin - 60))
            # Alea(pid = vacation, nb_lits = début de la plage, nuits = durée en minutes)
            out.append(S.Alea(min(t, debut), "indispo", pid=vid, nb_lits=debut,
                              nuits=rng.randint(*self.indispo_min)))
        return sorted(out, key=lambda a: a.t)


def chirurgien_de_garde(inst, jour, p, rng=None):
    """Une urgence est opérée par le chirurgien de garde : le sien s'il opère ce jour-là, sinon un
    chirurgien qui a une vacation ce jour-là (tiré au hasard, de façon reproductible)."""
    presents = sorted({inst.vacations[v].med_id for v in inst.vacations_du_jour.get(jour, ())
                       if not inst.vacations[v].urgence})
    if not presents or p.med_id in presents:
        return p
    rng = rng or random.Random(p.id * 31 + jour)
    return dataclasses.replace(p, med_id=rng.choice(presents))


def reservoir_urgences(don, debut, fin):
    res = {}
    for p, j in don.jour_urgent.items():
        if debut <= j <= fin:
            res.setdefault(j, []).append(don.urgents[p])
    return res


def simuler(matin, generateur, politique, optimiseur, avec_plans_b, jours, durees="reelles",
            garder=None):
    P = S.ParametresSMA(politique=politique)
    work = S.copie_de_travail(matin["sol"])
    inst = work.inst
    inst.capacite_lits_nuit = (list(inst.capacite_lits_nuit) if inst.capacite_lits_nuit is not None
                               else [inst.capacite_lits] * inst.nb_jours)
    inst._cap_calendrier = list(inst.capacite_lits_nuit)
    if generateur is not None:
        generateur.preparer(inst, jours)
    registre = S.RegistreLits()
    reprises, bilans, gardees, salles = [], [], {}, []
    reprog = 0
    t0 = time.perf_counter()
    Classe = JourneeV2 if politique == "sma" else JourneeRef
    for j in jours:
        seq = S.plan_du_jour(work, j, matin["plans"].get(j))
        aleas = generateur.aleas_du_jour(inst, j, seq) if generateur else []
        for p in reprises:                                  # reprise : chirurgien de garde du jour
            q = chirurgien_de_garde(inst, j, p)
            inst.patients[p.id] = q
            aleas.append(S.Alea(P.arrivee_reprises, "urgence", patient=q))
        pb = matin["plans_b"].get(j) if avec_plans_b else None
        J = Classe(work, j, seq, aleas, P, optimiseur, pb, registre, durees, graine=j)
        b = J.executer()
        J.controle_coherence()
        bilans.append(b)
        reprog += len(J.reprogrammes)
        for s in J.salles.values():
            if s.realises:
                fin = max(f for _, _, f in s.realises)
                salles.append(dict(jour=j, tvo=s.v.tvo, opere=sum(f - d for _, d, f in s.realises),
                                   depassement=max(0, fin - s.v.fin), n=len(s.realises),
                                   urgence=bool(s.v.urgence)))
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
           "salles": salles, "secondes": time.perf_counter() - t0, "reprogrammes": reprog,
           "nuits_deficit": int(sum(l > c for l, c in zip(lits, cap))), "premiere_nuit": min(jours)}
    return mes, gardees
