r"""
tabou.py — Les deux recherches tabou.

OÙ LE TABOU SERT VRAIMENT, ET OÙ IL NE SERT PAS
-----------------------------------------------
Le processus réel est le suivant : le patient est vu en consultation, on lui
propose DEUX dates, le chirurgien en choisit une, et cette date ne bouge plus.
Il n'y a pas de liste d'attente, et aucun patient déjà programmé n'est
déplacé.

Conséquence qu'il faut assumer : **à l'étape de proposition, il n'y a pas de
recherche combinatoire.** Le chirurgien a une vingtaine de vacations sur six
mois ; on les évalue toutes, on trie, on rend les deux meilleurs jours. C'est
une énumération exhaustive — exacte, instantanée, et donc strictement
meilleure qu'une métaheuristique. Elle est dans `propositions.py`.

Le tabou garde deux emplois, tous deux réels :

  1. TABOU LOCAL (`TabouLocal`, partie II de ce fichier) — la veille de
     l'opération, sur une journée : séquence des patients dans chaque
     vacation, donc horaires, donc pic de places et de lits. Là, le voisinage
     est combinatoire et le problème est NP-difficile. C'est un vrai tabou.

  2. TABOU HORS-LIGNE (`TabouGlobal`, partie I) — il calcule le meilleur
     planning atteignable AVEC LE RECUL : tous les patients connus d'avance,
     tout déplaçable. Ce planning-là n'est pas réalisable en pratique, et ce
     n'est pas le but. Il sert d'ÉTALON : l'écart entre lui et ce que produit
     le processus réel (insertion au fil de l'eau, dates figées) mesure le
     PRIX DE LA CONTRAINTE « une date annoncée ne bouge plus ». C'est un
     chiffre, il se mesure, et c'est le résultat le plus parlant qu'on puisse
     tirer de ce travail.

     Le même moteur sert à la REPLANIFICATION EXCEPTIONNELLE : congé imprévu
     d'un praticien, fermeture d'une salle. Là on a le droit de déplacer des
     patients, et `TabouGlobal.figer()` permet de sanctuariser ceux dont la
     convocation est déjà partie.

RAPPEL : le découpage en deux niveaux (quel jour / quel ordre) suit la réalité
de l'hôpital — on annonce une DATE des semaines à l'avance, on fixe l'HEURE la
veille. C'est la décomposition hiérarchique classique en ordonnancement de
bloc (*advance scheduling* / *allocation scheduling*).

RAPPEL SUR LA RECHERCHE TABOU (Glover, 1986)
--------------------------------------------
À chaque itération, on prend le MEILLEUR voisin, même s'il dégrade la
solution — c'est ce qui permet de sortir d'un optimum local, contrairement à
une descente. Pour ne pas y revenir immédiatement, l'attribut du mouvement
inverse est déclaré « tabou » pendant quelques itérations. Trois ingrédients
indispensables :
  - la LISTE TABOU : on interdit un attribut (ici « remettre le patient p
    dans la vacation v »), pas une solution entière — sinon la mémoire coûte
    trop cher et n'interdit presque rien ;
  - l'ASPIRATION : un mouvement tabou est quand même accepté s'il produit la
    meilleure solution jamais vue. Sans elle, on s'interdit des optima ;
  - la DIVERSIFICATION : après N itérations sans amélioration, on repart du
    meilleur connu avec une perturbation aléatoire.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

from modele import Instance, PlanningJour, Solution, deriver_creneaux

# ===========================================================================
#                          PARTIE I — TABOU GLOBAL
# ===========================================================================


@dataclass
class Poids:
    """DEUX critères pour choisir le jour d'une opération. C'est tout.

        f = w_remplissage · variance des taux de remplissage, par praticien
          + w_lits        · variance de l'occupation des lits, jour par jour

    REMPLISSAGE. On ne cherche pas à remplir au maximum, on cherche à ce que
    les journées d'un même praticien se RESSEMBLENT. Un patient ira donc
    spontanément dans la vacation la moins remplie de son chirurgien, à
    l'intérieur de la fenêtre demandée. On évite ainsi les journées à 95 %,
    qui débordent au moindre aléa, et les journées à 30 %, qui mobilisent une
    équipe pour rien. Les vacations vides comptent dans la variance, avec un
    taux de zéro : laisser une journée à l'abandon coûte.

    LITS. Variance de l'occupation jour par jour, sur TOUS les jours de
    l'horizon, week-ends compris. C'est la courbe à aplatir.

    IL N'Y A PLUS DE TERME DE DÉLAI, et c'est un progrès. Le délai est
    désormais porté par la FENÊTRE que le chirurgien choisit en consultation
    (une semaine, trois semaines, six semaines). Un poids de délai est une
    abstraction qu'il faut calibrer et défendre ; une fenêtre est une
    décision que le praticien prend en connaissance de cause, patient par
    patient. « Je veux une date dans les trois semaines » est une phrase
    qu'il prononce vraiment ; « je donne un poids de 5 au délai », non.

    Les capacités — durée de vacation, lits, places — ne sont pas dans
    l'objectif : ce sont des contraintes DURES. Un mouvement qui en viole une
    est écarté, une date qui en viole une n'est pas proposée.

    LE NOMBRE DE PATIENTS PROGRAMMÉS N'EST PAS DANS L'OBJECTIF non plus, et
    ce point mérite une explication parce que la première version le mettait.

    Avec un poids, il écrasait tout : sur cette instance, le terme de report
    valait 5,5 pour 0,08 de remplissage et 0,11 de lits, soit 96 % de
    l'objectif. La recherche ne faisait plus qu'une chose — caser des
    patients — et les deux critères qu'on lui demandait d'optimiser étaient
    du bruit. Baisser le poids aurait eu l'effet inverse : elle aurait
    abandonné des patients pour gagner trois décimales de lissage.

    C'est le signe qu'il ne s'agit pas d'un compromis mais d'un ORDRE DE
    PRIORITÉ. On compare donc d'abord le nombre de patients sans date, puis
    le coût, lexicographiquement — exactement comme le tabou local traite ses
    violations. Entre deux plannings, celui qui opère plus de monde gagne
    toujours ; à nombre égal, le mieux lissé gagne. Aucun taux de change à
    inventer, et un poids de moins.
    """

    w_remplissage: float = 1.0
    w_lits: float = 1.0


@dataclass
class ParametresTabou:
    """Réglages de la recherche. Les valeurs par défaut sont des points de
    départ raisonnables, pas des vérités."""

    iterations: int = 3000
    duree_tabou_min: int = 8
    duree_tabou_max: int = 25
    taille_voisinage: int = 120     # nombre de mouvements évalués par itération
    candidats_par_patient: int = 12  # vacations testées pour un patient donné
    stagnation_max: int = 300       # avant diversification
    force_perturbation: float = 0.08
    fenetre: int | None = None      # jours ; None = tout l'horizon
    graine: int | None = 0
    temps_max: float | None = None  # secondes, None = pas de limite
    verbeux: bool = True


@dataclass(slots=True)
class Mouvement:
    """Un déplacement candidat, déjà évalué mais PAS appliqué."""

    patient: int
    vers: int | None
    depuis: int | None
    delta: float            # variation du coût (remplissage + lits)
    delta_sans_date: int    # -1 on programme, +1 on déprogramme, 0 on déplace
    viole_vacation: bool
    viole_ressource: bool


class Evaluateur:
    """Fonction objectif + calcul incrémental des variations.

    Les deux termes sont des variances, donc déjà sans dimension. Celui des
    lits est divisé par le carré de l'occupation moyenne cible, ce qui en fait
    un carré de coefficient de variation ; celui du remplissage porte sur des
    taux, qui sont déjà dans [0, 1]. Les deux poids sont ainsi comparables
    entre eux.

    La constante de normalisation des lits est calculée UNE FOIS sur
    l'instance, jamais sur la solution courante — sinon l'objectif se déforme
    pendant la recherche et comparer deux itérations n'a plus de sens.
    """

    def __init__(self, inst: Instance, poids: Poids):
        self.inst = inst
        self.poids = poids

        self.tvo_total = max(1, sum(v.tvo for v in inst.vacations.values()))
        nuits = sum(p.nb_nuits for p in inst.patients.values())
        # occupation moyenne des lits si la charge était parfaitement étalée :
        # le repère naturel pour juger une variance.
        self.ref_lits = max(1.0, nuits / max(1, inst.nb_jours))
        self.nb_medecins = max(1, len(inst.vacations_du_medecin))

    # -- évaluation complète (rare : contrôle, démarrage, journalisation) ---

    def cout(self, sol: Solution) -> float:
        """Les deux critères, et rien d'autre."""
        p = self.poids
        return (p.w_remplissage * sol.variance_remplissage()
                + p.w_lits * sol.variance_lits() / self.ref_lits ** 2)

    def score(self, sol: Solution) -> tuple[int, float]:
        """(patients sans date, coût) — comparé dans cet ordre.

        Un tuple Python se compare terme à terme : `(3, 0.01) < (4, 0.001)`.
        Programmer un patient de plus l'emporte donc toujours sur n'importe
        quel gain de lissage, sans qu'on ait eu à inventer un taux de change
        entre « un patient opéré » et « un écart-type de lits ».
        """
        return len(sol.sans_date), self.cout(sol)

    def detail(self, sol: Solution) -> dict:
        """Les termes séparés : indispensable pour régler les poids."""
        p = self.poids
        return {
            "sans_date": len(sol.sans_date),
            "remplissage": p.w_remplissage * sol.variance_remplissage(),
            "lits": p.w_lits * sol.variance_lits() / self.ref_lits ** 2,
            "total": self.cout(sol),
        }

    # -- jours de ressource consommés par un patient posé dans une vacation -

    def _jours_ressource(self, pid: int, vid: int | None) -> list[int]:
        if vid is None:
            return []
        inst = self.inst
        p = inst.patients[pid]
        j0 = inst.vacations[vid].jour
        if p.ambulatoire:
            return [j0]
        return list(range(j0, min(j0 + p.nb_nuits, inst.nb_jours)))

    # -- LE calcul central : variation de l'objectif sans rien modifier ------

    def evaluer_mouvement(self, sol: Solution, pid: int,
                          vers: int | None) -> Mouvement | None:
        """Variation exacte de l'objectif si on déplaçait `pid` vers `vers`.

        Coût : O(nombre de nuits du patient), soit 0 à 12 opérations. C'est ce
        qui permet d'évaluer une centaine de voisins par itération, et les
        1 998 consultations d'un semestre en une fraction de seconde.
        """
        depuis = sol.affectation[pid]
        if depuis == vers:
            return None
        inst, poids = self.inst, self.poids
        p = inst.patients[pid]
        delta = 0.0
        viole_vac = False

        # --- 1. remplissage des vacations du praticien ----------------------
        # Le patient ne change jamais de chirurgien : les deux vacations
        # concernées appartiennent au même praticien, un seul terme bouge.
        med = p.med_id
        n_vac = inst.nb_vacations_medecin.get(med, 0)
        if n_vac:
            s_t, s_t2 = sol.s_taux[med], sol.s_taux2[med]
            var_avant = max(0.0, s_t2 / n_vac - (s_t / n_vac) ** 2)
            ds = ds2 = 0.0
            for vid, signe in ((depuis, -1), (vers, +1)):
                if vid is None:
                    continue
                tvo = inst.vacations[vid].tvo
                n = sol.nb[vid] + signe
                charge = (0 if n == 0 else
                          sol.somme_durees[vid] + signe * p.duree_op
                          + sol.somme_marges[vid] + signe * p.marge_perso
                          + inst.tis * (n - 1))
                t0 = sol.taux_remplissage(vid)
                t1 = charge / tvo if tvo else 0.0
                ds += t1 - t0
                ds2 += t1 * t1 - t0 * t0
                if signe > 0 and charge > inst.capacite_utile(vid):
                    viole_vac = True
            var_apres = max(0.0, (s_t2 + ds2) / n_vac - ((s_t + ds) / n_vac) ** 2)
            delta += poids.w_remplissage * (var_apres - var_avant) / self.nb_medecins
        elif vers is not None:
            n = sol.nb[vers] + 1
            charge = (sol.somme_durees[vers] + p.duree_op
                      + sol.somme_marges[vers] + p.marge_perso
                      + inst.tis * (n - 1))
            viole_vac = charge > inst.capacite_utile(vers)

        # --- 2. ressources cumulatives (places OU lits) ---------------------
        variations: dict[int, int] = defaultdict(int)
        for j in self._jours_ressource(pid, depuis):
            variations[j] -= 1
        for j in self._jours_ressource(pid, vers):
            variations[j] += 1

        viole_res = False
        if variations:
            if p.ambulatoire:
                # Les admissions ambulatoires ne sont pas dans l'objectif : le
                # lissage de l'ambulatoire se joue à l'échelle de la journée,
                # dans le tabou local, où les heures sont connues. Ici on ne
                # vérifie que la capacité.
                cap = inst.capacite_places_jour
                for j, dd in variations.items():
                    if dd > 0 and sol.places_jour[j] + dd > cap:
                        viole_res = True
            else:
                courant = sol.lits_jour
                n_jours = sol._n_jours_lits
                var_avant = sol.variance_lits()
                cap = inst.capacite_lits
                ds = ds2 = 0
                for j, dd in variations.items():
                    if dd == 0:
                        continue
                    a = courant[j]
                    b = a + dd
                    ds += b - a
                    ds2 += b * b - a * a
                    if dd > 0 and b > cap:
                        viole_res = True
                if ds or ds2:
                    s_ap = sol.s_lits + ds
                    s2_ap = sol.s_lits2 + ds2
                    var_apres = max(0.0, s2_ap / n_jours - (s_ap / n_jours) ** 2)
                    delta += poids.w_lits * (var_apres - var_avant) / self.ref_lits ** 2

        # --- 3. le patient gagne-t-il ou perd-il sa date ? ------------------
        delta_sans = (-1 if depuis is None else (1 if vers is None else 0))

        return Mouvement(pid, vers, depuis, delta, delta_sans,
                         viole_vac, viole_res)


# ---------------------------------------------------------------------------
# Solution initiale
# ---------------------------------------------------------------------------


def solution_initiale(inst: Instance, ev: Evaluateur,
                      alea: random.Random | None = None,
                      fenetre: int | None = None) -> Solution:
    """Construction gloutonne « meilleur ajustement » (*best fit*).

    On traite les patients par ancienneté de demande, puis par durée
    décroissante. Les gros d'abord : c'est la règle LPT du *bin packing*, et
    elle est nettement meilleure que l'ordre inverse — placer les gros quand
    il ne reste que des miettes est la meilleure façon de gaspiller du temps
    de salle.

    Pour chaque patient on prend la vacation qui dégrade le moins l'objectif.
    C'est donc déjà la fonction objectif à trois plans qui pilote la
    construction : on ne démarre pas d'une solution absurde.
    """
    alea = alea or random.Random(0)
    sol = Solution(inst)
    ordre = sorted(inst.patients.values(),
                   key=lambda p: (p.jour_demande, -p.duree_op))

    for p in ordre:
        candidats = inst.vacations_possibles(p.id, fenetre)
        if not candidats:
            continue
        meilleur, meilleur_delta = None, math.inf
        for vid in candidats:
            m = ev.evaluer_mouvement(sol, p.id, vid)
            if m is None or m.viole_vacation or m.viole_ressource:
                continue
            if m.delta < meilleur_delta:
                meilleur, meilleur_delta = vid, m.delta
        if meilleur is not None:
            sol.affecter(p.id, meilleur)
        else:
            # aucune vacation ne peut l'accueillir sans violer une capacité :
            # il est HORS HORIZON, à reconvoquer quand l'horizon aura glissé.
            sol.hors_horizon.add(p.id)
    return sol


# ---------------------------------------------------------------------------
# La recherche
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """Historique de la recherche, pour tracer les courbes de convergence."""

    couts: list[float] = field(default_factory=list)
    meilleurs: list[float] = field(default_factory=list)
    iterations: int = 0
    temps: float = 0.0
    diversifications: int = 0
    mouvements_tabou_acceptes: int = 0


class TabouGlobal:
    """Tabou hors-ligne : affectation patient -> vacation (donc jour + salle).

    Toutes les contraintes sont DURES : chirurgien, capacité de vacation,
    lits, places. Un mouvement qui en viole une est écarté du voisinage, et
    la recherche ne quitte donc jamais le domaine faisable.

    C'est une simplification assumée par rapport à la version précédente, qui
    proposait trois modes de gestion des contraintes (tout pénalisé, hybride,
    tout dur) avec des coefficients adaptatifs. Mesuré sur cette instance, les
    pénalités n'apportaient rien — le détail est dans la docstring de `Poids`.
    Moins de code, moins de paramètres, mêmes résultats.

    Ce que le tabou conserve, en revanche, ce sont ses trois mécanismes
    essentiels :
      - la LISTE TABOU sur l'attribut (patient, vacation quittée) ;
      - l'ASPIRATION, qui lève l'interdit quand le mouvement produit la
        meilleure solution jamais vue ;
      - la DIVERSIFICATION, qui repart du meilleur connu avec une
        perturbation après N itérations sans amélioration.
    """

    def __init__(self, inst: Instance, poids: Poids | None = None,
                 params: ParametresTabou | None = None):
        self.inst = inst
        self.poids = poids or Poids()
        self.params = params or ParametresTabou()
        self.ev = Evaluateur(inst, self.poids)
        self.alea = random.Random(self.params.graine)
        self.tabou: dict[tuple[int, int | None], int] = {}
        self.trace = Trace()
        # Patients sanctuarisés : leur convocation est partie, on n'y touche
        # plus. Vide pour le calcul de l'étalon hors-ligne ; rempli pour une
        # replanification exceptionnelle.
        self.figes: set[int] = set()

    def figer(self, patients) -> "TabouGlobal":
        """Interdit tout déplacement de ces patients. Renvoie self."""
        self.figes = set(patients)
        return self

    def figer_avant(self, sol: Solution, jour_limite: int) -> "TabouGlobal":
        """Fige tous les patients programmés avant un jour donné.

        Usage typique en replanification : `figer_avant(sol, aujourdhui + 30)`
        laisse le tabou réorganiser l'arrière-plan sans annuler une seule
        convocation déjà envoyée.
        """
        inst = self.inst
        self.figes = {pid for pid, vid in sol.affectation.items()
                      if vid is not None and inst.vacations[vid].jour < jour_limite}
        return self

    # -- filtre de faisabilité ---------------------------------------------

    def _admissible(self, m: Mouvement) -> bool:
        """Les capacités sont dures : un mouvement qui en viole une n'existe pas."""
        return not (m.viole_vacation or m.viole_ressource)

    # -- construction du voisinage ----------------------------------------

    def _patients_candidats(self, sol: Solution, k: int) -> list[int]:
        """Choix des patients à bouger : on ne tire pas au hasard uniforme.

        Une itération de tabou coûte cher ; il faut la dépenser là où il y a
        quelque chose à gagner. On vise donc en priorité :
          - les patients encore sans date (ils coûtent le plus) ;
          - ceux posés dans une vacation qui déborde ;
          - ceux posés un jour où les lits ou les places saturent ;
        et on complète au hasard pour ne pas s'enfermer. C'est une *candidate
        list strategy*, l'un des leviers les plus rentables d'un tabou.
        """
        inst = self.inst
        chauds: list[int] = []

        libres = sol.sans_date - self.figes
        if libres:
            chauds.extend(self.alea.sample(sorted(libres),
                                           min(len(libres), k // 2)))

        vac_chaudes = [vid for vid in inst.vacations if sol.depassement(vid) > 0]
        jours_chauds = {j for j, n in enumerate(sol.lits_jour) if n > inst.capacite_lits}
        jours_chauds |= {j for j in sol._jours_places
                         if sol.places_jour[j] > inst.capacite_places_jour}

        if vac_chaudes or jours_chauds:
            vids_chauds = set(vac_chaudes)
            for j in jours_chauds:
                vids_chauds.update(inst.vacations_du_jour.get(j, ()))
            pool = [pid for pid, vid in sol.affectation.items()
                    if vid in vids_chauds and pid not in self.figes]
            if pool:
                chauds.extend(self.alea.sample(pool, min(len(pool), k // 2)))

        tous = [pid for pid in inst.patients if pid not in self.figes]
        while len(chauds) < k and tous:
            chauds.append(self.alea.choice(tous))
        return chauds[:k]

    def _voisinage(self, sol: Solution) -> list[Mouvement]:
        """Mouvements évalués à cette itération.

        Deux types :
          - DÉPLACEMENT : le patient change de vacation, ou sort
            temporairement du planning. C'est le mouvement structurant.
          - ÉCHANGE implicite : inutile de le coder séparément ici. Un échange
            est une suite de deux déplacements, et comme on accepte les
            mouvements dégradants, le tabou les enchaîne naturellement. On
            garde ainsi un voisinage O(n) au lieu de O(n²).
        """
        p = self.params
        voisins: list[Mouvement] = []
        par_patient = max(2, p.taille_voisinage // max(1, p.candidats_par_patient))
        patients = self._patients_candidats(sol, par_patient)

        for pid in patients:
            possibles = self.inst.vacations_possibles(pid, p.fenetre)
            if not possibles:
                continue
            if len(possibles) > p.candidats_par_patient:
                possibles = self.alea.sample(possibles, p.candidats_par_patient)
            for vid in possibles:
                m = self.ev.evaluer_mouvement(sol, pid, vid)
                if m is not None and self._admissible(m):
                    voisins.append(m)
            # Sortir un patient du planning est un état TRANSITOIRE : il faut
            # parfois en sortir un pour en faire rentrer deux. Comme le nombre
            # de patients sans date est comparé AVANT le coût, un mouvement qui
            # en laisse un de plus sur le carreau ne sera retenu que s'il n'y a
            # rien d'autre — il n'y a pas de liste d'attente dans ce modèle.
            if sol.affectation[pid] is not None:
                m = self.ev.evaluer_mouvement(sol, pid, None)
                if m is not None and self._admissible(m):
                    voisins.append(m)
        return voisins

    # -- boucle principale -------------------------------------------------

    def resoudre(self, depart: Solution | None = None) -> Solution:
        p = self.params
        sol = (depart.copie() if depart is not None
               else solution_initiale(self.inst, self.ev, self.alea, p.fenetre))

        # `cout` est un TUPLE (patients sans date, coût) : on compare d'abord
        # le nombre de patients programmés, puis la qualité du lissage.
        cout = self.ev.score(sol)
        meilleure, meilleur_cout = sol.copie(), cout
        stagnation = 0
        t0 = time.perf_counter()

        for it in range(1, p.iterations + 1):
            voisins = self._voisinage(sol)
            if not voisins:
                stagnation += 1
                continue

            # sélection : meilleur voisin non tabou, ou tabou par aspiration
            choisi, meilleur_score = None, (math.inf, math.inf)
            par_aspiration = False
            for m in voisins:
                score = (cout[0] + m.delta_sans_date, cout[1] + m.delta)
                interdit = self.tabou.get((m.patient, m.vers), 0) > it
                if interdit:
                    # ASPIRATION : on lève l'interdit si le résultat est le
                    # meilleur jamais atteint. Sans cela, la liste tabou finit
                    # par interdire l'optimum lui-même.
                    if score < meilleur_cout and score < meilleur_score:
                        choisi, meilleur_score, par_aspiration = m, score, True
                    continue
                if score < meilleur_score:
                    choisi, meilleur_score, par_aspiration = m, score, False

            if choisi is None:
                stagnation += 1
                continue

            # application
            sol.affecter(choisi.patient, choisi.vers)
            cout = meilleur_score if False else (cout[0] + choisi.delta_sans_date,
                                                 cout[1] + choisi.delta)
            if par_aspiration:
                self.trace.mouvements_tabou_acceptes += 1

            # l'attribut interdit est le RETOUR à la position quittée
            duree = self.alea.randint(p.duree_tabou_min, p.duree_tabou_max)
            self.tabou[(choisi.patient, choisi.depuis)] = it + duree

            if cout < meilleur_cout:
                meilleure, meilleur_cout = sol.copie(), cout
                stagnation = 0
            else:
                stagnation += 1

            self.trace.couts.append(cout)
            self.trace.meilleurs.append(meilleur_cout)

            # DIVERSIFICATION
            if stagnation >= p.stagnation_max:
                sol = self._perturber(meilleure)
                cout = self.ev.score(sol)
                self.tabou.clear()
                stagnation = 0
                self.trace.diversifications += 1
                if p.verbeux:
                    print(f"    it {it:5d} : diversification "
                          f"(meilleur = {meilleur_cout[0]} sans date, "
                          f"{meilleur_cout[1]:.5f})")

            if p.verbeux and it % 500 == 0:
                print(f"    it {it:5d} : courant {cout[1]:.5f} | "
                      f"meilleur {meilleur_cout[1]:.5f} | "
                      f"sans date {len(sol.sans_date)}")

            if p.temps_max and time.perf_counter() - t0 > p.temps_max:
                break

        self.trace.iterations = it
        self.trace.temps = time.perf_counter() - t0
        return meilleure

    def _perturber(self, base: Solution) -> Solution:
        """Redémarrage depuis le meilleur connu, avec un bruit contrôlé.

        On déplace aléatoirement une fraction des patients. Trop peu : on
        retombe dans le même optimum local. Trop : on perd tout le travail et
        on refait une recherche aléatoire.
        """
        sol = base.copie()
        n = max(1, int(self.params.force_perturbation * len(self.inst.patients)))
        for pid in self.alea.sample(list(self.inst.patients), n):
            if pid in self.figes:
                continue
            possibles = self.inst.vacations_possibles(pid, self.params.fenetre)
            if not possibles:
                continue
            # La perturbation doit rester DANS le domaine faisable, sinon
            # l'invariant « aucune solution visitée ne viole une capacité »
            # saute et la recherche peut rendre un planning inexploitable.
            cible = self.alea.choice(possibles)
            m = self.ev.evaluer_mouvement(sol, pid, cible)
            if m is not None and self._admissible(m):
                sol.affecter(pid, cible)
        return sol


# ===========================================================================
#                          PARTIE II — TABOU LOCAL
# ===========================================================================


@dataclass
class PoidsLocaux:
    """DEUX familles de critères pour organiser une journée : creux et places.

        g = b_creux       · part du TVO de la journée laissée inoccupée
          + b_creux_var   · dispersion du remplissage entre les vacations du jour
          + b_pic_places  · pic de places ambulatoires simultanées
          + b_var_places  · variance TEMPORELLE de l'occupation des places

    CREUX. Le temps de vacation payé et non utilisé. Deux poids parce qu'il y
    a deux questions distinctes : combien de temps est perdu (`b_creux`), et
    est-il réparti également entre les salles ouvertes ce jour-là
    (`b_creux_var`) ? Une journée où une salle finit à 12h et l'autre à 18h
    n'est pas équivalente à deux salles qui finissent à 15h, même à creux
    total identique.

    Avec l'enchaînement au plus tôt, la séquence agit sur le creux par un seul
    canal : le TIS est réduit entre deux actes identiques, donc regrouper les
    actes similaires libère du temps en fin de vacation. Le second levier est
    le transfert d'un patient vers une autre salle du même praticien.

    PLACES. Deux poids également, parce que les deux chiffres ne servent pas à
    la même décision. Le PIC dimensionne : c'est le nombre de places à
    équiper. La VARIANCE TEMPORELLE lisse : à pic égal, une journée où les
    places se remplissent et se vident régulièrement demande moins de
    personnel de surveillance qu'une journée avec un bouchon à 11h et un
    désert à 16h.

    CE QUI N'EST PAS DANS LA FONCTION DE COÛT. Le dépassement de vacation, la
    sortie d'un ambulatoire après la fermeture de l'UCA et le dépassement de
    la capacité en places sont des VIOLATIONS, pas des coûts. Elles sont
    comparées avant le coût, lexicographiquement : entre deux séquences, celle
    qui viole le moins gagne toujours, quel que soit son coût. Ça évite
    d'avoir à inventer un poids pour convertir « un patient qui rentre chez
    lui à 20h30 » en minutes de creux — conversion qui n'a aucun sens.

    Les lits ne sont plus dans l'objectif local non plus. `contexte_lits` et
    `pic_lits` restent calculés et affichés comme INDICATEURS : si vous
    voulez remettre le levier « opérer les hospitalisés après 10h », il suffit
    d'ajouter un poids sur `pj.pic_lits`.
    """

    b_creux: float = 1.0            # part du TVO inoccupée
    b_creux_var: float = 2.0        # dispersion du remplissage entre salles
    b_pic_places: float = 0.10      # par place simultanée
    b_var_places: float = 0.05      # par unité de variance temporelle


def contexte_lits(inst: Instance, sol: Solution, jour: int) -> tuple[int, int]:
    """Lits déjà occupés le matin du jour J, et sorties prévues ce matin-là.

    C'EST CE QUI DONNE UN SENS AU TABOU LOCAL CÔTÉ LITS. Sans ce contexte, le
    nombre de lits d'une journée ne dépendrait pas de l'ordre des opérations
    et il n'y aurait rien à optimiser.

    Un patient opéré le jour d avec n nuits est présent du jour d au jour d+n,
    et libère son lit le matin du jour d+n (à `heure_sortie_hospit`). Donc au
    matin du jour J :
      report  = patients opérés AVANT J et encore présents
      sortants = ceux qui partent ce matin
    Un patient hospitalisé qui sort de salle AVANT l'heure des sorties
    s'ajoute au report complet : il faut un lit de plus. Après, il récupère un
    lit libéré. Décaler les hospitalisations après 10h est donc un vrai levier,
    et c'est exactement ce que fait l'hôpital en pratique.
    """
    report = sortants = 0
    for pid, vid in sol.affectation.items():
        if vid is None:
            continue
        p = inst.patients[pid]
        if p.ambulatoire:
            continue
        d = inst.vacations[vid].jour
        if d < jour <= d + p.nb_nuits:
            report += 1
            if d + p.nb_nuits == jour:
                sortants += 1
    return report, sortants


def violations_journee(inst: Instance, pj: PlanningJour) -> int:
    """Ce qui rend une journée inacceptable, quel qu'en soit le coût.

    Trois choses : une vacation qui déborde, un ambulatoire libéré après la
    fermeture de l'UCA, et plus de places occupées en même temps qu'il n'en
    existe. On les COMPTE, on ne les convertit pas en coût : convertir « un
    patient qui rentre chez lui à 20h30 » en minutes de creux demanderait un
    taux de change arbitraire, et un poids arbitraire est un poids qu'on ne
    sait pas défendre.
    """
    return (pj.depassement
            + 100 * pj.hors_uca
            + 100 * max(0, pj.pic_places - inst.capacite_places))


def evaluer_journee(inst: Instance, pj: PlanningJour,
                    poids: PoidsLocaux) -> float:
    """Coût d'un planning de journée : creux et places, rien d'autre.

    Recalcul complet : une journée, c'est une trentaine de patients,
    l'évaluation coûte quelques microsecondes. On ne complique pas ce qui n'a
    pas besoin de l'être.
    """
    return (poids.b_creux * pj.part_creux
            + poids.b_creux_var * pj.variance_taux
            + poids.b_pic_places * pj.pic_places
            + poids.b_var_places * pj.places_variance)


def contexte_lits_pic(inst: Instance, pj: PlanningJour,
                      report: int, sortants: int) -> int:
    """Pic de lits de la journée, report de la veille inclus. INDICATEUR.

    Un patient opéré le jour d avec n nuits libère son lit le matin du jour
    d+n. Un hospitalisé qui sort de salle AVANT l'heure des sorties s'ajoute
    donc au report complet de la nuit : il faut un lit de plus. Ce n'est plus
    un critère d'optimisation, mais c'est un chiffre à surveiller.
    """
    evenements = [(0, report), (inst.heure_sortie_hospit, -sortants)]
    evenements += [(c.fin, 1) for c in pj.creneaux.values() if not c.ambulatoire]
    evenements.sort()
    courant = sommet = 0
    for _, d in evenements:
        courant += d
        sommet = max(sommet, courant)
    return sommet


class TabouLocal:
    """Tabou n°2 : la veille de l'opération, sur une seule journée.

    DÉCISIONS
      - la SÉQUENCE des patients dans chaque vacation, d'où les heures ;
      - le transfert éventuel d'un patient vers une autre vacation du MÊME
        chirurgien le même jour (donc changement de salle) ;
      - en dernier recours, le report d'un patient qui ne tient pas.

    L'affectation des places n'est PAS une décision : une fois les horaires
    connus, le nombre minimal de places est le pic du profil, et le coloriage
    glouton l'atteint (cf. `colorier_intervalles` dans modele.py). Confondre
    les deux est l'erreur classique — on cherche à « optimiser l'affectation
    des places » alors que tout se joue sur les horaires.

    VOISINAGE
      - échange de deux patients (même vacation ou vacations différentes) ;
      - déplacement d'un patient à une autre position.
    Sur ~30 patients, le voisinage complet fait quelques centaines de
    mouvements : on l'énumère entièrement, pas besoin d'échantillonner.
    """

    def __init__(self, inst: Instance, poids: PoidsLocaux | None = None,
                 iterations: int = 120, duree_tabou: int = 7,
                 voisinage_max: int = 250, graine: int | None = 0):
        self.inst = inst
        self.poids = poids or PoidsLocaux()
        self.iterations = iterations
        self.duree_tabou = duree_tabou
        self.voisinage_max = voisinage_max
        self.alea = random.Random(graine)

    # -- séquence de départ ------------------------------------------------

    def _sequences_initiales(self, sol: Solution, jour: int) -> dict[int, list[int]]:
        """Heuristique de départ : ambulatoires d'abord, du plus court au plus
        long, puis les hospitalisés.

        Deux justifications :
          - l'ambulatoire doit sortir avant la fermeture de l'UCA, donc plus il
            est tôt, mieux c'est ;
          - l'hospitalisé opéré après 10h récupère un lit libéré le matin même,
            il ne s'ajoute pas au report de la nuit.
        C'est déjà une bonne solution ; le tabou l'affine.
        """
        seqs: dict[int, list[int]] = {}
        for vid in self.inst.vacations_du_jour.get(jour, ()):
            pats = [pid for pid, v in sol.affectation.items() if v == vid]
            pats.sort(key=lambda pid: (not self.inst.patients[pid].ambulatoire,
                                       self.inst.patients[pid].duree_op))
            seqs[vid] = pats
        return seqs

    # -- résolution --------------------------------------------------------

    def resoudre(self, sol: Solution, jour: int) -> tuple[PlanningJour, list[int]]:
        inst = self.inst
        report, sortants = contexte_lits(inst, sol, jour)
        seqs = self._sequences_initiales(sol, jour)
        reportes: list[int] = []

        # les vacations du jour, groupées par chirurgien : un patient ne peut
        # être transféré qu'entre vacations de SON praticien.
        par_med: dict[int, list[int]] = defaultdict(list)
        for vid in seqs:
            par_med[inst.vacations[vid].med_id].append(vid)

        def evaluer(s, rep):
            """(violations, coût) — comparé dans cet ordre, lexicographiquement.

            Un tuple Python se compare terme à terme : `(0, 3.2) < (1, 0.1)`.
            Une journée qui viole moins gagne donc toujours, quel que soit son
            coût, sans qu'on ait eu à inventer un taux de change entre les
            deux. C'est plus honnête qu'une pénalité géante, et ça marche
            exactement pareil.
            """
            pj = deriver_creneaux(inst, jour, s)
            pj.pic_lits = contexte_lits_pic(inst, pj, report, sortants)
            score = (violations_journee(inst, pj) + 1000 * len(rep),
                     evaluer_journee(inst, pj, self.poids))
            return score, pj

        cout, pj = evaluer(seqs, reportes)
        meilleur_rep = list(reportes)
        meilleur_cout, meilleur_pj = cout, pj
        tabou: dict[tuple[str, int, int], int] = {}

        for it in range(1, self.iterations + 1):
            mouvements = self._voisinage(seqs, par_med, reportes, pj.depassement > 0)
            if not mouvements:
                break
            choisi, choisi_cout = None, (math.inf, math.inf)
            choisi_seqs = choisi_rep = choisi_pj = None

            for cle, nouvelles, nouveau_rep in mouvements:
                c, ppj = evaluer(nouvelles, nouveau_rep)
                interdit = tabou.get(cle, 0) > it
                if interdit and not c < meilleur_cout:
                    continue
                if c < choisi_cout:
                    choisi, choisi_cout = cle, c
                    choisi_seqs, choisi_rep, choisi_pj = nouvelles, nouveau_rep, ppj

            if choisi is None:
                break
            seqs, reportes, cout, pj = choisi_seqs, choisi_rep, choisi_cout, choisi_pj
            tabou[choisi] = it + self.duree_tabou
            if cout < meilleur_cout:
                meilleur_cout = cout
                meilleur_rep = list(reportes)
                meilleur_pj = pj

        return meilleur_pj, meilleur_rep

    def _voisinage(self, seqs, par_med, reportes, deborde: bool):
        """Énumération des mouvements. Trois familles.

        `deborde` dit si la journée ne tient pas dans ses vacations. Les
        mouvements de REPORT (sortir un patient de la journée) ne sont
        engendrés que dans ce cas : sinon ils encombrent le voisinage sans
        jamais être choisis, et chaque candidat coûte une évaluation.
        """
        inst = self.inst
        sortie = []
        vids = list(seqs)

        # 1. ÉCHANGE de deux patients — même vacation, ou deux vacations du
        #    même chirurgien (sinon on violerait la règle « le patient ne
        #    change pas de docteur »).
        for i, va in enumerate(vids):
            for vb in vids[i:]:
                if va != vb and inst.vacations[va].med_id != inst.vacations[vb].med_id:
                    continue
                for pa in seqs[va]:
                    for pb in seqs[vb]:
                        if pa >= pb:
                            continue
                        nouvelles = {k: list(v) for k, v in seqs.items()}
                        ia, ib = nouvelles[va].index(pa), nouvelles[vb].index(pb)
                        nouvelles[va][ia], nouvelles[vb][ib] = pb, pa
                        sortie.append((("echange", pa, pb), nouvelles, list(reportes)))

        # 2. INSERTION à une autre position, éventuellement dans une autre
        #    salle du même chirurgien.
        for va in vids:
            for pa in list(seqs[va]):
                position_actuelle = seqs[va].index(pa)
                for vb in par_med[inst.vacations[va].med_id]:
                    for pos in range(len(seqs[vb]) + 1):
                        if vb == va and pos in (position_actuelle, position_actuelle + 1):
                            continue
                        nouvelles = {k: list(v) for k, v in seqs.items()}
                        nouvelles[va].remove(pa)
                        cible = nouvelles[vb]
                        cible.insert(min(pos, len(cible)), pa)
                        sortie.append((("insertion", pa, vb), nouvelles, list(reportes)))

        # 3. REPORT et REPRISE — la soupape. Un jour surchargé doit pouvoir
        #    sortir un patient du programme du jour, et un jour allégé le
        #    reprendre. Un patient sorti ici doit être REPROPOSÉ en
        #    consultation : c'est une déprogrammation, pas une mise en
        #    attente.
        if deborde:
            for va in vids:
                for pa in seqs[va]:
                    nouvelles = {k: list(v) for k, v in seqs.items()}
                    nouvelles[va].remove(pa)
                    sortie.append((("report", pa, -1), nouvelles, reportes + [pa]))
        for pa in reportes:
            med = inst.patients[pa].med_id
            for vb in par_med.get(med, ()):
                nouvelles = {k: list(v) for k, v in seqs.items()}
                nouvelles[vb].append(pa)
                sortie.append((("reprise", pa, vb), nouvelles,
                               [x for x in reportes if x != pa]))

        # garde-fou : sur une journée très chargée le voisinage complet peut
        # exploser. On échantillonne plutôt que de ralentir toute la chaîne.
        if len(sortie) > self.voisinage_max:
            sortie = self.alea.sample(sortie, self.voisinage_max)
        return sortie


# ---------------------------------------------------------------------------
# Déclenchement du tabou local
# ---------------------------------------------------------------------------


def doit_lancer_local(inst: Instance, sol: Solution, jour: int, aujourdhui: int,
                      seuil_remplissage: float = 0.85) -> bool:
    """Deux déclencheurs, comme convenu.

      1. la VEILLE : on fige toujours le planning du lendemain ;
      2. le SEUIL : dès qu'une journée dépasse un taux de remplissage donné,
         on l'optimise sans attendre. Une journée à 90 % ne pardonne aucun
         mauvais ordre — c'est là que les places et les lits saturent.
    """
    if jour == aujourdhui + 1:
        return True
    vids = inst.vacations_du_jour.get(jour, ())
    if not vids:
        return False
    tvo = sum(inst.vacations[v].tvo for v in vids)
    charge = sum(sol.charge(v) for v in vids)
    return tvo > 0 and charge / tvo >= seuil_remplissage


def planifier_journees(inst: Instance, sol: Solution, jours: list[int],
                       poids: PoidsLocaux | None = None,
                       iterations: int = 300) -> dict[int, PlanningJour]:
    """Applique le tabou local à une liste de journées."""
    local = TabouLocal(inst, poids, iterations=iterations)
    resultats = {}
    for j in jours:
        if not inst.vacations_du_jour.get(j):
            continue
        pj, reportes = local.resoudre(sol, j)
        for pid in reportes:
            # déprogrammé : il faudra lui reproposer deux dates.
            sol.affecter(pid, None)
        resultats[j] = pj
    return resultats


def planifier_horizon(inst: Instance, sol: Solution,
                      poids: PoidsLocaux | None = None,
                      iterations: int = 150,
                      verbeux: bool = False) -> tuple[dict[int, PlanningJour], dict]:
    """Déroule le tabou local sur TOUTES les journées ouvertes de l'horizon.

    En exploitation on ne fait jamais ça : le tabou local tourne la veille,
    une journée à la fois, quand le programme opératoire est arrêté. Le
    dérouler d'un coup sur tout l'horizon sert à autre chose — vérifier que
    les dates distribuées en consultation donnent des journées réellement
    exécutables, et en sortir le planning au créneau près.

    Renvoie les plannings journaliers et un bilan agrégé.
    """
    local = TabouLocal(inst, poids, iterations=iterations)
    plannings: dict[int, PlanningJour] = {}
    avant = {"pic": [], "sd": [], "creux": []}
    apres = {"pic": [], "sd": [], "creux": []}
    deborde = hors_uca = depasse_places = 0
    reportes_total: list[int] = []

    jours = [j for j in inst.jours_ouvres if any(sol.nb[v] for v in
                                                 inst.vacations_du_jour.get(j, ()))]
    for n, j in enumerate(jours, 1):
        depart = deriver_creneaux(inst, j, local._sequences_initiales(sol, j))
        avant["pic"].append(depart.pic_places)
        avant["sd"].append(depart.places_variance ** 0.5)
        avant["creux"].append(depart.creux_total)

        pj, reportes = local.resoudre(sol, j)
        for pid in reportes:
            sol.affecter(pid, None)
        reportes_total.extend(reportes)
        plannings[j] = pj
        apres["pic"].append(pj.pic_places)
        apres["sd"].append(pj.places_variance ** 0.5)
        apres["creux"].append(pj.creux_total)
        deborde += 1 if pj.depassement > 0 else 0
        hors_uca += pj.hors_uca
        depasse_places += 1 if pj.pic_places > inst.capacite_places else 0
        if verbeux and n % 25 == 0:
            print(f"    {n}/{len(jours)} journées planifiées")

    moy = lambda x: sum(x) / len(x) if x else 0.0
    bilan = {
        "journees": len(jours),
        "creneaux": sum(len(p.creneaux) for p in plannings.values()),
        "pic_places_avant": moy(avant["pic"]), "pic_places_apres": moy(apres["pic"]),
        "sd_places_avant": moy(avant["sd"]), "sd_places_apres": moy(apres["sd"]),
        "creux_avant_min": moy(avant["creux"]), "creux_apres_min": moy(apres["creux"]),
        "journees_debordantes": deborde,
        "sorties_hors_uca": hors_uca,
        "journees_places_saturees": depasse_places,
        "patients_deprogrammes": len(reportes_total),
    }
    return plannings, bilan


def creneaux_en_lignes(inst: Instance, plannings: dict[int, PlanningJour]) -> list[dict]:
    """Le planning au créneau près : une ligne par patient.

    C'est la sortie finale du système — ce qu'on imprimerait pour le bloc :
    date, salle, heure d'entrée, heure de sortie, chirurgien, acte, et le
    numéro de place ou de lit attribué.
    """
    lignes = []
    for jour in sorted(plannings):
        pj = plannings[jour]
        for pid, c in sorted(pj.creneaux.items(), key=lambda kv: (kv[1].bloc_id, kv[1].debut)):
            p = inst.patients[pid]
            lignes.append({
                "date": inst.date_du_jour(jour).isoformat(),
                "jour": jour,
                "salle": c.bloc_id,
                "debut": f"{c.debut // 60:02d}:{c.debut % 60:02d}",
                "fin": f"{c.fin // 60:02d}:{c.fin % 60:02d}",
                "minutes": c.fin - c.debut,
                "patient": pid,
                "chirurgien": inst.vacations[c.vacation_id].med_id,
                "acte": p.type_interv,
                "type": "ambulatoire" if p.ambulatoire else f"{p.nb_nuits} nuit(s)",
                "place": ("place " if p.ambulatoire else "lit ") + str(c.place),
                "liberation": f"{min(c.lib_place, 1439) // 60:02d}:{min(c.lib_place, 1439) % 60:02d}",
            })
    return lignes
