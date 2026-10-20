r"""
genetique_jour.py — Algorithme génétique pour le planning d'UNE journée.

Même rôle que le tabou n°2 : la solution globale a déjà fixé QUI est opéré CE
JOUR-LÀ et par QUEL chirurgien. Il reste à décider :
  - l'ORDRE de passage dans chaque vacation (donc les heures) ;
  - éventuellement la VACATION, parmi celles du même chirurgien ce jour-là
    (option `reaffecter`) — le jour et le chirurgien ne bougent jamais.
Les numéros de place / de lit ne sont PAS des gènes : une fois les heures
connues, `colorier_intervalles` les affecte de façon optimale (cf. modele.py).

CODAGE (étape 0 du cours)
-------------------------
Un chromosome a deux parties, indexées par patient i = 0..n-1 :

    ordre = [3, 0, 5, 1, 4, 2]     permutation de tous les patients du jour
    vac   = [12, 12, 15, 12, 15, 15]   vacation de chaque patient

Décodage : on parcourt `ordre` et on range chaque patient dans la file de sa
vacation. La séquence d'une vacation est donc l'ordre RELATIF de ses patients
dans la permutation. Tout chromosome est décodable et respecte les
contraintes dures par construction (bon chirurgien, bon jour) : pas besoin de
réparation.

OPÉRATEURS
----------
  Croisement  : OX (Davis 1985, celui du cours) sur `ordre` — il préserve la
                validité d'une permutation, contrairement au croisement à un
                point (cf. l'exemple ABCDDE du cours sur le PVC) ;
                uniforme sur `vac` (gène par gène, hérité de l'un des parents).
  Mutation    : échange de deux gènes quelconques, transposition de deux gènes
                consécutifs (les deux du cours), insertion, inversion d'un
                segment, réaffectation d'un patient à une autre vacation.
  Sélection   : tournoi (défaut) ou roulette (celle du cours, sur un fitness
                linéarisé puisqu'on MINIMISE un coût).
  Remplacement: générationnel avec élitisme ; immigration aléatoire en cas de
                stagnation, pour garder de la diversité.

AUCUN DÉPASSEMENT (contrainte ABSOLUE)
--------------------------------------
L'ordre ne change pas la somme des durées : seul le TIS réduit (actes
identiques consécutifs) fait gagner quelques minutes. Si une vacation reçoit
plus qu'elle ne peut contenir, AUCUN ordre ne supprime le dépassement. D'où :
  - `depassement="p90"` (défaut) : interdit dans le scénario pessimiste
    (chaque acte dure médiane + marge_perso) — même convention que
    `Solution.charge` au niveau global. Interdit a fortiori en médian.
    `"median"` : interdit seulement avec les durées estimées.
  - une minute de dépassement coûte plus que TOUT le reste réuni ; l'AG part
    d'individus faisables (graines réparées) et l'élitisme les conserve ;
  - si la journée reçue est impossible, deux comportements :
      `autoriser_report=True`  : l'AG REPORTE le moins de patients possible
                                 (gène vacation = REPORT) — ils repartent au
                                 niveau global pour une nouvelle date ;
      `autoriser_report=False` : (DÉFAUT) on lève `JourneeInfaisable`, avec le détail.
Garantie : le planning renvoyé a TOUJOURS 0 minute de dépassement dans le
scénario choisi. Avec les durées RÉELLES, l'aléa reste possible au-delà du
P90 (1 acte sur 10 dépasse son P90) : c'est à mesurer avec `rejouer_reel`.

POURQUOI LA CONTRAINTE EST NORMALEMENT DÉJÀ SATISFAITE
Au niveau global, on n'accepte un patient dans une vacation que si
    Σ (durée + marge) + TIS × (n − 1)  ≤  TVO          (Solution.charge)
Or, pour N'IMPORTE QUEL ordre, la fin de vacation (scénario P90) vaut
    début + Σ (durée + marge) + Σ TIS_k,   avec chaque TIS_k ≤ TIS
donc elle est ≤ début + TVO = fin. Tout ordre est sans dépassement : la
contrainte ne restreint pas l'AG, elle sert de GARDE-FOU. Elle ne peut être
violée que si :
  - la journée ne vient pas du moteur (saisie manuelle, rejeu d'historique) ;
  - une durée a été ré-estimée à la hausse après l'attribution de la date ;
  - l'option `reaffecter` déplace un patient vers une vacation trop pleine
    (l'AG l'interdit : un tel individu est réparé ou rejeté).
D'où `autoriser_report=False` par défaut : une journée infaisable en entrée
est une ANOMALIE du niveau global, signalée par `JourneeInfaisable`, et non
quelque chose que l'AG doit « arranger » en reportant des patients.

FONCTION DE COÛT (à minimiser)
------------------------------
Somme pondérée, poids dans `Poids`. Ordre de grandeur voulu :
dépassement interdit >> patient reporté >> tout le reste.

  INTERDIT : depassement_interdit (min, dans le scénario choisi)
  REPORT   : report (par patient reporté)
  DUR (scénario médian, durées estimées)
    depassement        min au-delà de la fin de vacation (si non interdit)
    hors_uca           ambulatoires qui sortent après la fermeture de l'UCA
    surcapacite_places places simultanées au-delà de `capacite_places`
  ROBUSTESSE (scénario pessimiste : chaque acte dure médiane + marge_perso ≈ P90)
    depassement_p90, hors_uca_p90
  CONFORT
    pic_places         pic d'occupation des places ambulatoires
    ecart_type_places  écart-type TEMPOREL de cette occupation (lissage)
    creux              min de vacation perdues (la séquence agit via le TIS réduit)
    dispersion         écart-type des taux de remplissage entre les vacations
                       d'un même chirurgien ce jour-là (utile si `reaffecter`)

Le calcul du coût réimplémente `deriver_creneaux` sans créer d'objets (x10
plus rapide) ; `verifier_evaluateur` prouve que les deux donnent la même chose.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from copy import copy

from modele import (MED_URGENCE, Instance, PlanningJour, Solution, deriver_creneaux,
                    moments_temporels, pic, profil_cumulatif)

# ---------------------------------------------------------------------------
# 1. Paramètres
# ---------------------------------------------------------------------------


@dataclass
class Poids:
    depassement_interdit: float = 1e7  # par minute, scénario contraint
    report: float = 1e5                # par patient reporté
    depassement: float = 100.0         # par minute
    hors_uca: float = 2000.0           # par patient
    surcapacite_places: float = 3000.0  # par place au-delà de la capacité
    depassement_p90: float = 3.0       # par minute, scénario pessimiste
    hors_uca_p90: float = 150.0        # par patient, scénario pessimiste
    pic_places: float = 15.0           # par place
    ecart_type_places: float = 30.0    # par unité d'écart-type
    creux: float = 1.0                 # par minute
    dispersion: float = 200.0          # par unité d'écart-type de taux


@dataclass
class ParametresAG:
    taille_population: int = 60
    nb_generations: int = 400
    p_croisement: float = 0.9
    p_mutation: float = 0.4
    selection: str = "tournoi"          # "tournoi" | "roulette"
    taille_tournoi: int = 3
    elitisme: int = 2
    stagnation_max: int = 80            # arrêt si pas d'amélioration
    immigration: float = 0.2            # part renouvelée après stagnation/2
    recherche_locale: bool = True       # AG « mémétique » : descente sur l'élite
    temps_max: float | None = None      # secondes
    graine: int | None = None


# ---------------------------------------------------------------------------
# 2. Évaluation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Evaluation:
    cout: float
    depassement: int
    hors_uca: int
    pic_places: int
    places_variance: float
    creux_total: int
    dispersion: float
    depassement_p90: int
    hors_uca_p90: int
    reportes: int = 0
    violation: int = 0        # minutes de dépassement dans le scénario contraint

    def detail(self) -> dict:
        return {k: getattr(self, k) for k in self.__slots__}


REPORT = -1   # valeur du gène « vacation » pour un patient reporté


class JourneeInfaisable(Exception):
    """La journée ne tient pas sans dépassement et le report est interdit."""


class Individu:
    __slots__ = ("ordre", "vac", "eval")

    def __init__(self, ordre: list[int], vac: list[int]):
        self.ordre, self.vac, self.eval = ordre, vac, None

    def copie(self) -> "Individu":
        c = Individu(list(self.ordre), list(self.vac))
        c.eval = self.eval
        return c

    @property
    def cout(self) -> float:
        return self.eval.cout


# ---------------------------------------------------------------------------
# 3. L'algorithme
# ---------------------------------------------------------------------------


class AGJournee:
    """AG pour une journée. Usage :

        ag = AGJournee(inst, jour, sequences)      # sequences = {vid: [pid...]}
        res = ag.resoudre()
        print(afficher_journee(inst, res.planning))
    """

    def __init__(self, inst: Instance, jour: int,
                 sequences: dict[int, list[int]],
                 poids: Poids | None = None,
                 params: ParametresAG | None = None,
                 reaffecter: bool = True,
                 depassement: str = "p90",
                 autoriser_report: bool = False):
        if depassement not in ("p90", "median"):
            raise ValueError("depassement doit valoir 'p90' ou 'median'")
        self.inst, self.jour = inst, jour
        self.mode_dep = depassement
        self.autoriser_report = autoriser_report
        self.W = poids or Poids()
        self.P = params or ParametresAG()
        self.rng = random.Random(self.P.graine)

        # --- gènes : patients du jour, vacation initiale -------------------
        self.pids: list[int] = [pid for seq in sequences.values() for pid in seq]
        self.n = len(self.pids)
        vac_init = {pid: vid for vid, seq in sequences.items() for pid in seq}
        for pid, vid in vac_init.items():
            if (inst.vacations[vid].med_id != inst.patients[pid].med_id
                    and not (inst.vacations[vid].urgence and inst.patients[pid].est_urgent)):
                raise ValueError(f"patient {pid} dans une vacation d'un autre chirurgien")

        # URGENCES : un urgent posé dans un créneau URGENCES appartient au
        # « groupe » MED_URGENCE (il peut passer d'un créneau URGENCES du jour
        # à un autre) ; tous les autres au groupe de leur chirurgien.
        self._grp = {p: (MED_URGENCE if inst.vacations[vac_init[p]].urgence
                         else inst.patients[p].med_id) for p in self.pids}

        # vacations concernées : celles fournies + (si réaffectation) les
        # autres vacations du jour des mêmes chirurgiens
        meds = set(self._grp.values())
        self.vids = list(sequences)
        if reaffecter:
            for vid in inst.vacations_du_jour.get(jour, ()):
                if vid not in sequences and inst.vacations[vid].med_id in meds:
                    self.vids.append(vid)
        self.vac_par_med: dict[int, list[int]] = {}
        for vid in self.vids:
            self.vac_par_med.setdefault(inst.vacations[vid].med_id, []).append(vid)
        self.choix: list[list[int]] = [
            (self.vac_par_med[self._grp[p]] if reaffecter else [vac_init[p]])
            for p in self.pids]
        if autoriser_report:
            # un urgent n'est JAMAIS reporté : l'AG ne peut que bouger son heure
            self.choix = [c if inst.patients[p].est_urgent else c + [REPORT]
                          for p, c in zip(self.pids, self.choix)]
        self.vac0 = [vac_init[p] for p in self.pids]

        # --- données patient en tableaux (accès rapide) --------------------
        P = inst.patients
        self._duree = [P[p].duree_op for p in self.pids]
        self._duree90 = [P[p].duree_op + P[p].marge_perso for p in self.pids]
        self._type = [P[p].type_interv for p in self.pids]
        self._ambu = [P[p].ambulatoire for p in self.pids]
        self._v = {vid: (inst.vacations[vid].debut, inst.vacations[vid].fin,
                         inst.vacations[vid].med_id) for vid in self.vids}

        self._cache: dict[tuple, Evaluation] = {}
        self.nb_evaluations = 0
        self.convergence: list[tuple[int, float, float]] = []   # (gen, meilleur, moyen)

    # -- décodage ----------------------------------------------------------

    def decoder(self, ind: Individu) -> dict[int, list[int]]:
        """Chromosome -> {vid: [indices patients dans l'ordre]}."""
        files: dict[int, list[int]] = {vid: [] for vid in self.vids}
        for i in ind.ordre:
            if ind.vac[i] != REPORT:
                files[ind.vac[i]].append(i)
        return files

    def reportes(self, ind: Individu) -> list[int]:
        return [self.pids[i] for i in range(self.n) if ind.vac[i] == REPORT]

    def sequences(self, ind: Individu) -> dict[int, list[int]]:
        """Chromosome -> {vid: [pid...]} au format de `deriver_creneaux`."""
        return {vid: [self.pids[i] for i in seq]
                for vid, seq in self.decoder(ind).items()}

    # -- évaluation --------------------------------------------------------

    def _derouler(self, files, durees):
        """Enchaînement au plus tôt, identique à `deriver_creneaux`."""
        inst = self.inst
        tis, tis2, surv = inst.tis, inst.tis_meme_acte, inst.surveillance_ambu
        ferm = inst.fermeture_uca
        ouv_uca, avance = inst.ouverture_uca, inst.avance_ambu
        dep = hors = creux = 0
        ambus: list[tuple[int, int]] = []
        taux: dict[int, float] = {}
        ouverture = None
        for vid, seq in files.items():
            debut, fin_v, _ = self._v[vid]
            if not seq:
                taux[vid] = 0.0
                creux += fin_v - debut
                continue
            ouverture = debut if ouverture is None else min(ouverture, debut)
            t, prec = debut, None
            for i in seq:
                if prec is not None:
                    typ = self._type[i]
                    t += tis2 if (typ and typ == self._type[prec]) else tis
                deb = t
                t += durees[i]
                if self._ambu[i]:
                    # la place est prise à l'ARRIVÉE, pas à la sortie de salle
                    ambus.append((max(ouv_uca, deb - avance), t + surv))
                    if t + surv > ferm:
                        hors += 1
                prec = i
            tvo = fin_v - debut
            taux[vid] = (min(t, fin_v) - debut) / tvo if tvo else 0.0
            creux += max(0, fin_v - t)
            dep += max(0, t - fin_v)
        return dep, hors, creux, ambus, taux, min(8 * 60 if ouverture is None else ouverture,
                                                  inst.ouverture_uca)

    def evaluer(self, ind: Individu) -> Evaluation:
        files = self.decoder(ind)
        cle = tuple(tuple(files[v]) for v in self.vids)
        ev = self._cache.get(cle)
        if ev is None:
            ev = self._evaluer_files(files)
            self._cache[cle] = ev
            self.nb_evaluations += 1
        ind.eval = ev
        return ev

    def _evaluer_files(self, files) -> Evaluation:
        W, inst = self.W, self.inst
        dep, hors, creux, ambus, taux, ouv = self._derouler(files, self._duree)
        profil = profil_cumulatif(ambus)
        pic_p = pic(profil)
        _, var_p = moments_temporels(profil, ouv, inst.fermeture_uca)

        # dispersion du remplissage : par chirurgien, entre SES vacations du jour
        disp, k = 0.0, 0
        for vids in self.vac_par_med.values():
            if len(vids) >= 2:
                ts = [taux[v] for v in vids]
                m = sum(ts) / len(ts)
                disp += math.sqrt(sum((x - m) ** 2 for x in ts) / len(ts))
                k += 1
        disp = disp / k if k else 0.0

        dep90, hors90, *_ = self._derouler(files, self._duree90)
        violation = dep90 if self.mode_dep == "p90" else dep
        nrep = self.n - sum(len(s) for s in files.values())

        cout = (W.depassement_interdit * violation + W.report * nrep
                + W.depassement * dep + W.hors_uca * hors
                + W.surcapacite_places * max(0, pic_p - inst.capacite_places)
                + W.depassement_p90 * dep90 + W.hors_uca_p90 * hors90
                + W.pic_places * pic_p + W.ecart_type_places * math.sqrt(var_p)
                + W.creux * creux + W.dispersion * disp)
        return Evaluation(cout, dep, hors, pic_p, var_p, creux, disp, dep90, hors90,
                          nrep, violation)

    # -- réparation : rendre un individu faisable ---------------------------

    def _durees_contraintes(self):
        return self._duree90 if self.mode_dep == "p90" else self._duree

    def _fin(self, seq, durees, debut) -> int:
        t, prec = debut, None
        for i in seq:
            if prec is not None:
                t += (self.inst.tis_meme_acte
                      if self._type[i] and self._type[i] == self._type[prec]
                      else self.inst.tis)
            t += durees[i]
            prec = i
        return t

    def reparer(self, ind: Individu) -> Individu:
        """Supprime tout dépassement (scénario contraint) :
          1. un patient d'une vacation qui déborde est d'abord DÉPLACÉ vers une
             autre vacation de son chirurgien ce jour-là, s'il y tient ;
          2. sinon il est REPORTÉ : on choisit le plus court qui suffit à
             résorber le débordement, à défaut le plus long (peu de reports).
        Sans report autorisé, la réparation peut échouer : l'individu garde
        alors sa violation (et son coût énorme)."""
        d = self._durees_contraintes()
        vac = list(ind.vac)
        for _ in range(2 * self.n + 1):
            files = self.decoder(Individu(ind.ordre, vac))
            deborde = [(v, self._fin(s, d, self._v[v][0]) - self._v[v][1])
                       for v, s in files.items() if s]
            deborde = [(v, x) for v, x in deborde if x > 0]
            if not deborde:
                break
            v, exces = deborde[0]
            seq = files[v]
            # 1. déplacement vers une vacation sœur qui a la place
            deplace = False
            for i in sorted(seq, key=lambda i: -d[i]):
                for w in self.choix[i]:
                    if w in (v, REPORT):
                        continue
                    essai = files[w] + [i]
                    if self._fin(essai, d, self._v[w][0]) <= self._v[w][1]:
                        vac[i], deplace = w, True
                        break
                if deplace:
                    break
            if deplace:
                continue
            # 2. report
            if not self.autoriser_report:
                break
            suffisants = [i for i in seq if d[i] >= exces]
            i = (min(suffisants, key=lambda i: d[i]) if suffisants
                 else max(seq, key=lambda i: d[i]))
            vac[i] = REPORT
        r = Individu(list(ind.ordre), vac)
        self.evaluer(r)
        return r

    # -- population initiale ----------------------------------------------

    def _ind_depuis_ordre(self, ordre, vac=None) -> Individu:
        ind = Individu(list(ordre), list(vac if vac is not None else self.vac0))
        self.evaluer(ind)
        return ind

    def _vac_equilibree(self) -> list[int]:
        """LPT : chaque patient (du plus long au plus court) va dans la
        vacation de son chirurgien la moins remplie en proportion."""
        charge = {v: 0 for v in self.vids}
        vac = list(self.vac0)
        for i in sorted(range(self.n), key=lambda i: -self._duree[i]):
            if len([v for v in self.choix[i] if v != REPORT]) > 1:
                vac[i] = min([v for v in self.choix[i] if v != REPORT],
                             key=lambda v: charge[v] /
                             max(1, self._v[v][1] - self._v[v][0]))
            charge[vac[i]] += self._duree[i]
        return vac

    def population_initiale(self) -> list[Individu]:
        """Quelques graines heuristiques + le reste au hasard."""
        idx = list(range(self.n))
        amb = self._ambu
        graines = [
            idx,                                                        # ordre fourni
            sorted(idx, key=lambda i: (not amb[i], self._type[i])),     # ambu d'abord, groupés par acte
            sorted(idx, key=lambda i: (not amb[i], self._duree[i])),    # ambu d'abord, courts d'abord
            sorted(idx, key=lambda i: (not amb[i], -self._duree90[i] + self._duree[i])),  # ambu prévisibles d'abord
            sorted(idx, key=lambda i: (amb[i], self._type[i])),         # hospitalisés d'abord
        ]
        pop = [self._ind_depuis_ordre(o) for o in graines]
        if any(len([v for v in c if v != REPORT]) > 1 for c in self.choix):
            vac_eq = self._vac_equilibree()
            pop += [self._ind_depuis_ordre(o, vac_eq) for o in graines[:3]]
        # versions RÉPARÉES des graines : la population contient toujours des
        # individus sans dépassement, que l'élitisme ne perdra plus
        pop += [self.reparer(i) for i in list(pop) if i.eval.violation > 0]
        while len(pop) < self.P.taille_population:
            pop.append(self._aleatoire())
        return pop[:self.P.taille_population]

    def _aleatoire(self) -> Individu:
        ordre = list(range(self.n))
        self.rng.shuffle(ordre)
        vac = [self.rng.choice([v for v in c if v != REPORT] or c) for c in self.choix]
        return self._ind_depuis_ordre(ordre, vac)

    # -- sélection ---------------------------------------------------------

    def selectionner(self, pop: list[Individu]) -> Individu:
        if self.P.selection == "roulette":
            # Roulette du cours. On MINIMISE un coût : fitness linéarisée
            # f_i = c_max - c_i + ε, puis tirage proportionnel à f_i.
            cmax = max(i.cout for i in pop)
            fit = [cmax - i.cout + 1e-9 for i in pop]
            y = self.rng.random() * sum(fit)
            acc = 0.0
            for ind, f in zip(pop, fit):
                acc += f
                if acc >= y:
                    return ind
            return pop[-1]
        cand = self.rng.sample(pop, min(self.P.taille_tournoi, len(pop)))
        return min(cand, key=lambda i: i.cout)

    # -- croisement --------------------------------------------------------

    @staticmethod
    def croisement_ox(p1: list, p2: list, a: int, b: int) -> list:
        """Croisement d'ordre OX (Davis 1985), exactement comme dans le cours :
        l'enfant reçoit le segment [a, b) du parent 2, puis les gènes
        manquants dans l'ordre où ils apparaissent dans le parent 1 à partir
        de la position b (circulairement), placés à partir de b.
        """
        n = len(p1)
        enfant = [None] * n
        enfant[a:b] = p2[a:b]
        pris = set(p2[a:b])
        reste = [p1[(b + k) % n] for k in range(n) if p1[(b + k) % n] not in pris]
        pos = b % n
        for g in reste:
            enfant[pos] = g
            pos = (pos + 1) % n
        return enfant

    def croiser(self, p1: Individu, p2: Individu) -> tuple[Individu, Individu]:
        n = self.n
        a, b = sorted(self.rng.sample(range(n + 1), 2)) if n >= 2 else (0, n)
        o1 = self.croisement_ox(p1.ordre, p2.ordre, a, b)
        o2 = self.croisement_ox(p2.ordre, p1.ordre, a, b)
        v1, v2 = list(p1.vac), list(p2.vac)
        for i in range(n):                       # uniforme sur les vacations
            if self.rng.random() < 0.5:
                v1[i], v2[i] = v2[i], v1[i]
        return Individu(o1, v1), Individu(o2, v2)

    # -- mutation ----------------------------------------------------------

    def muter(self, ind: Individu) -> None:
        n, r, o = self.n, self.rng, ind.ordre
        if n < 2:
            return
        peut_reaffecter = [i for i in range(n) if len(self.choix[i]) > 1]
        ops = ["echange", "transposition", "insertion", "inversion"]
        if peut_reaffecter:
            ops.append("reaffectation")
        op = r.choice(ops)
        if op == "echange":                       # deux gènes quelconques
            i, j = r.sample(range(n), 2)
            o[i], o[j] = o[j], o[i]
        elif op == "transposition":               # deux gènes consécutifs
            i = r.randrange(n - 1)
            o[i], o[i + 1] = o[i + 1], o[i]
        elif op == "insertion":                   # déplacer un patient
            i, j = r.sample(range(n), 2)
            o.insert(j, o.pop(i))
        elif op == "inversion":                   # inverser un segment (2-opt)
            i, j = sorted(r.sample(range(n), 2))
            o[i:j + 1] = reversed(o[i:j + 1])
        else:                                     # changer de vacation
            i = r.choice(peut_reaffecter)
            ind.vac[i] = r.choice([v for v in self.choix[i] if v != ind.vac[i]])
        ind.eval = None

    # -- recherche locale (option mémétique) ---------------------------------

    def descente(self, ind: Individu, max_passes: int = 3) -> Individu:
        """Première amélioration par insertion d'un patient à une autre
        position de la permutation. Appliquée au meilleur individu seulement :
        l'AG explore, la descente finit le travail localement."""
        best = ind.copie()
        if best.eval is None:
            self.evaluer(best)
        for _ in range(max_passes):
            ameliore = False
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    c = Individu(list(best.ordre), best.vac)
                    c.ordre.insert(j, c.ordre.pop(i))
                    if self.evaluer(c).cout < best.cout - 1e-9:
                        best, ameliore = Individu(c.ordre, list(best.vac)), True
                        best.eval = c.eval
            if not ameliore:
                break
        return best

    # -- boucle principale ---------------------------------------------------

    def resoudre(self) -> "ResultatAG":
        t0 = time.perf_counter()
        P = self.P
        if self.n == 0:
            pj = deriver_creneaux(self.inst, self.jour, {v: [] for v in self.vids})
            return ResultatAG(pj, None, None, 0, 0, 0.0, [])

        self.verifier_faisabilite()
        pop = self.population_initiale()
        initial = pop[0].copie()                                  # ordre fourni
        meilleur = min(pop, key=lambda i: i.cout).copie()
        stagnation, gen = 0, 0

        for gen in range(1, P.nb_generations + 1):
            pop.sort(key=lambda i: i.cout)
            nouvelle = [i.copie() for i in pop[:P.elitisme]]

            while len(nouvelle) < P.taille_population:
                p1, p2 = self.selectionner(pop), self.selectionner(pop)
                if self.rng.random() < P.p_croisement:
                    e1, e2 = self.croiser(p1, p2)
                else:
                    e1, e2 = p1.copie(), p2.copie()
                for e in (e1, e2):
                    if self.rng.random() < P.p_mutation:
                        self.muter(e)
                    if e.eval is None:
                        self.evaluer(e)
                    nouvelle.append(e)
            pop = nouvelle[:P.taille_population]

            champion = min(pop, key=lambda i: i.cout)
            if P.recherche_locale and (gen % 20 == 0 or stagnation == P.stagnation_max // 2):
                champion = self.descente(champion)
                pop[pop.index(min(pop, key=lambda i: i.cout))] = champion

            if champion.cout < meilleur.cout - 1e-9:
                meilleur, stagnation = champion.copie(), 0
            else:
                stagnation += 1

            # immigration : on renouvelle le bas de la population
            if stagnation and stagnation % max(1, P.stagnation_max // 2) == 0:
                pop.sort(key=lambda i: i.cout)
                k = int(P.immigration * len(pop))
                if k:
                    pop[-k:] = [self._aleatoire() for _ in range(k)]

            self.convergence.append((gen, meilleur.cout,
                                     sum(i.cout for i in pop) / len(pop)))
            if stagnation >= P.stagnation_max:
                break
            if P.temps_max and time.perf_counter() - t0 > P.temps_max:
                break

        if P.recherche_locale:
            meilleur = self.descente(meilleur)
        if meilleur.eval.violation > 0:
            meilleur = self.reparer(meilleur)
        if meilleur.eval.violation > 0:          # seulement si report interdit
            raise JourneeInfaisable(
                f"J{self.jour} : {meilleur.eval.violation} min de dépassement "
                f"({self.mode_dep}) dans le meilleur ordre trouvé")
        pj = deriver_creneaux(self.inst, self.jour, self.sequences(meilleur))
        return ResultatAG(pj, meilleur.eval, initial.eval, gen,
                          self.nb_evaluations, time.perf_counter() - t0,
                          self.convergence, self.reportes(meilleur))

    def verifier_faisabilite(self) -> None:
        """Borne inférieure, sans lancer l'AG : même avec le meilleur TIS
        possible (tous les actes identiques) et la meilleure répartition entre
        ses vacations du jour, chaque chirurgien doit tenir dans son TVO.
        Si ce n'est pas le cas et que le report est interdit : exception."""
        if self.autoriser_report:
            return
        d = self._durees_contraintes()
        besoin: dict[int, list[int]] = {}
        for i in range(self.n):
            besoin.setdefault(self._grp[self.pids[i]], []).append(d[i])
        messages = []
        for med, durees in besoin.items():
            vids = sorted({v for i in range(self.n) for v in self.choix[i]
                           if v != REPORT and self._v[v][2] == med})
            tvo = sum(self._v[v][1] - self._v[v][0] for v in vids)
            mini = sum(durees) + max(0, len(durees) - len(vids)) * self.inst.tis_meme_acte
            if mini > tvo:
                messages.append(f"chirurgien {med} : au moins {mini - tvo} min de trop "
                                f"({len(durees)} patients, TVO {tvo} min)")
        if messages:
            raise JourneeInfaisable(f"J{self.jour} ({self.mode_dep}) : " + " ; ".join(messages))


@dataclass
class ResultatAG:
    planning: PlanningJour
    evaluation: Evaluation | None          # meilleure solution trouvée
    evaluation_initiale: Evaluation | None  # ordre fourni en entrée
    generations: int
    nb_evaluations: int
    duree_s: float
    convergence: list = field(default_factory=list)
    reportes: list = field(default_factory=list)   # pid à reprogrammer au niveau global

    def resume(self) -> str:
        if self.evaluation is None:
            return "journée vide"
        a, b = self.evaluation_initiale, self.evaluation
        gain = (a.cout - b.cout) / a.cout if a.cout else 0.0
        rep = f" | {len(self.reportes)} patient(s) reporté(s)" if self.reportes else ""
        return (f"coût {a.cout:.0f} -> {b.cout:.0f} ({gain:+.0%} de mieux){rep} | "
                f"{self.generations} générations, {self.nb_evaluations} "
                f"évaluations, {self.duree_s:.2f} s")


# ---------------------------------------------------------------------------
# 4. Raccourcis
# ---------------------------------------------------------------------------


def sequences_du_jour(sol: Solution, jour: int) -> dict[int, list[int]]:
    """Extrait d'une Solution globale les vacations du jour et leurs patients
    (ordre arbitraire : c'est l'AG qui le fixe)."""
    inst = sol.inst
    seq = {vid: [] for vid in inst.vacations_du_jour.get(jour, ())}
    for pid in sol.patients_du_jour(jour):
        seq[sol.affectation[pid]].append(pid)
    return {vid: s for vid, s in seq.items() if s}


def optimiser_journee(sol: Solution, jour: int, **kw) -> ResultatAG:
    """Point d'entrée depuis la solution globale."""
    return AGJournee(sol.inst, jour, sequences_du_jour(sol, jour), **kw).resoudre()


def rejouer_reel(inst: Instance, pj: PlanningJour) -> PlanningJour:
    """Rejoue la séquence avec les durées RÉELLES (banc d'essai uniquement).
    La séquence est figée ; seules les durées changent."""
    inst2 = copy(inst)
    inst2.patients = {pid: (replace(p, duree_op=p.duree_reelle) if p.duree_reelle else p)
                      for pid, p in inst.patients.items()}
    return deriver_creneaux(inst2, pj.jour, pj.sequences)


def verifier_evaluateur(ag: AGJournee, nb: int = 200) -> None:
    """Contrôle que l'évaluateur rapide == `deriver_creneaux` (champs communs)."""
    for _ in range(nb):
        ind = ag._aleatoire()
        ev = ind.eval
        pj = deriver_creneaux(ag.inst, ag.jour, ag.sequences(ind))
        assert ev.depassement == pj.depassement, (ev.depassement, pj.depassement)
        assert ev.hors_uca == pj.hors_uca
        assert ev.pic_places == pj.pic_places
        assert abs(ev.places_variance - pj.places_variance) < 1e-9
        assert ev.creux_total == pj.creux_total
