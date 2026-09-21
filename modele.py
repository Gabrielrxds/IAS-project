r"""
modele.py — Structures de données pour l'ordonnancement du bloc opératoire.

PRINCIPE DE STRUCTURE (le changement majeur par rapport à votre version)
------------------------------------------------------------------------
On sépare strictement trois choses :

  1. L'INSTANCE  (`Instance`)  — ce qui ne bouge jamais pendant la recherche :
     les patients et leurs caractéristiques, les médecins, les vacations
     offertes, les capacités, l'horizon. C'est le *problème*.

  2. LA SOLUTION GLOBALE (`Solution`) — qui va dans quelle vacation (donc
     quel JOUR). Pas d'heures. C'est ce que construit le moteur de
     consultation (propositions.py), et ce que manipule le tabou hors-ligne
     qui sert d'étalon (tabou.py).

  3. LE PLANNING D'UNE JOURNÉE (`PlanningJour`) — ce que produit le tabou n°2 :
     l'ordre des patients dans chaque vacation, donc les heures, et le numéro
     de place / de lit de chacun.

Pourquoi c'est important : dans votre version, l'état de décision (`debut_op`,
`vacation_id`) était stocké *dans* le patient. Une métaheuristique teste des
millions d'états ; si l'état vit dans les objets métier, il faut les copier en
profondeur à chaque essai, et la moindre incohérence se propage partout. Ici,
copier une solution = copier un dictionnaire d'entiers.

UNITÉS
------
À l'intérieur de la recherche, toutes les durées sont des ENTIERS EN MINUTES
et tous les jours sont des ENTIERS (index du jour dans l'horizon). Les
`timedelta` et les `datetime` sont jolis mais 30 à 50 fois plus lents en
arithmétique, et le tabou fait des dizaines de millions d'opérations. On les
garde uniquement aux frontières (lecture des données, affichage).

VOCABULAIRE (indicateurs ANAP, ceux de votre fichier de vacations)
------------------------------------------------------------------
  TVO   Temps de Vacation Offert          : durée de la vacation mise à dispo.
  TROS  Temps Réel d'Occupation de Salle  : entrée en salle -> sortie de salle.
  TIS   Temps Inter-Salle                 : nettoyage + installation.
  Creux = TVO - (TROS cumulé + TIS cumulé) : le temps de salle payé et perdu.

MODÈLE DE RESSOURCES (après suppression de la SSPI)
---------------------------------------------------
  Dès la sortie de salle, le patient occupe :
    - soit une PLACE si ambulatoire (pool unique : chaise, fauteuil ou lit,
      on ne distingue pas le type) — libérée le soir même ;
    - soit un LIT s'il doit dormir au moins une nuit — libéré le matin de la
      sortie.
  Un patient dont la durée de séjour vaut n jours passe n-1 NUITS. C'est la
  nuit qui compte un lit : `duree_sejour = 1` (convention de votre fichier)
  signifie ambulatoire, donc zéro nuit.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# 0. Outils de base sur les intervalles
# ---------------------------------------------------------------------------


def chevauchent(d1: int, f1: int, d2: int, f2: int) -> bool:
    """Deux intervalles SEMI-OUVERTS [d, f) se recouvrent-ils ?

    Le semi-ouvert est important : un patient qui libère sa place à 10h00 et un
    autre qui l'occupe à 10h00 ne se chevauchent pas. Avec des intervalles
    fermés, on compterait une place de trop à chaque transition.
    """
    return d1 < f2 and d2 < f1


def profil_cumulatif(intervalles) -> list[tuple[int, int]]:
    """Ligne de balayage (*sweep line*) : profil du nombre d'intervalles actifs.

    C'est LA technique à connaître pour compter une ressource cumulative
    (lits, places, brancardiers...). On ne parcourt pas le temps minute par
    minute : on ne regarde que les instants où quelque chose change.
      +1 à chaque arrivée, -1 à chaque départ, somme cumulée sur les instants
      triés. Complexité O(n log n) au lieu de O(horizon).
    """
    evenements: dict[int, int] = defaultdict(int)
    for debut, fin in intervalles:
        if fin <= debut:
            continue
        evenements[debut] += 1
        evenements[fin] -= 1

    courant, profil = 0, []
    for instant in sorted(evenements):
        courant += evenements[instant]
        profil.append((instant, courant))
    return profil


def pic(profil) -> int:
    """Maximum d'un profil cumulatif (0 si vide)."""
    return max((n for _, n in profil), default=0)


def moments_temporels(profil, t0: int, t1: int) -> tuple[float, float]:
    """Moyenne et variance TEMPORELLES d'un profil en escalier sur [t0, t1].

    Un profil cumulatif n'est pas une série de points, c'est une fonction en
    escalier : le niveau n_i vaut de l'instant t_i jusqu'à t_{i+1}. Faire la
    moyenne des points serait une erreur classique — elle donnerait le même
    poids à un palier de 5 minutes et à un palier de 4 heures. On pondère donc
    chaque palier par sa DURÉE :

        moyenne  = (1/T) ∫ n(t) dt
        variance = (1/T) ∫ (n(t) − moyenne)² dt

    C'est ce qui permet de dire « lisser l'occupation des places dans le
    temps » et pas seulement « écrêter le pic ». Deux journées peuvent avoir
    le même pic et des courbes très différentes : l'une en cloche, l'autre en
    dents de scie. Seule la variance temporelle les distingue.
    """
    duree = t1 - t0
    if duree <= 0:
        return 0.0, 0.0

    # on reconstruit les paliers (début, fin, niveau) qui couvrent [t0, t1],
    # niveau 0 avant le premier événement et après le dernier
    paliers: list[tuple[int, int, int]] = []
    precedent, niveau = t0, 0
    for instant, n in profil:
        borne = min(max(instant, t0), t1)
        if borne > precedent:
            paliers.append((precedent, borne, niveau))
        precedent, niveau = max(borne, precedent), n
    if precedent < t1:
        paliers.append((precedent, t1, niveau))

    somme = sum(n * (f - d) for d, f, n in paliers)
    moyenne = somme / duree
    variance = sum((n - moyenne) ** 2 * (f - d) for d, f, n in paliers) / duree
    return moyenne, variance


def colorier_intervalles(intervalles: list[tuple[int, int, int]]) -> dict[int, int]:
    """Affecte un numéro de ressource à chaque intervalle (id, début, fin).

    Résultat garanti OPTIMAL : sur des intervalles, le nombre minimum de
    ressources nécessaires est exactement le pic du profil cumulatif (c'est
    un graphe d'intervalles, donc parfait — le nombre chromatique égale la
    taille de la plus grande clique). L'algorithme glouton par date de début
    croissante l'atteint toujours.

    Conséquence importante pour le tabou n°2 : l'AFFECTATION des places n'est
    pas un problème d'optimisation. Une fois les horaires fixés, elle est
    résolue en O(n log n). La seule vraie décision, c'est la SÉQUENCE.
    """
    libres: list[int] = []          # numéros de places rendues, à recycler
    fins: list[tuple[int, int]] = []  # (fin, numero) des places occupées
    affectation: dict[int, int] = {}
    prochaine = 0

    for ident, debut, fin in sorted(intervalles, key=lambda x: x[1]):
        # on récupère toutes les places libérées avant ce début
        restantes = []
        for f, num in fins:
            if f <= debut:
                libres.append(num)
            else:
                restantes.append((f, num))
        fins = restantes

        if libres:
            num = libres.pop()
        else:
            num, prochaine = prochaine, prochaine + 1
        affectation[ident] = num
        fins.append((fin, num))
    return affectation


# ---------------------------------------------------------------------------
# 1. Données statiques : patients, médecins, vacations
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Patient:
    """Données d'un patient. AUCUNE décision d'ordonnancement ici.

    `duree_op`         TROS estimé en minutes, validé par le chirurgien
                       (cf. estimation.py). N'inclut PAS le TIS.
    `marge_perso`      minutes de risque propres à cet acte : P90 - médiane de
                       l'historique. C'est ce qui remplace la marge forfaitaire
                       de 30 min : un canal carpien est prévisible, une
                       ostéosynthèse ne l'est pas, elles ne méritent pas la
                       même réserve.
    `duree_sejour`     en jours, convention du fichier source : 1 = ambulatoire.
                       Le nombre de NUITS vaut donc duree_sejour - 1.
    `med_id`           CONTRAINTE DURE : un patient ne change jamais de
                       chirurgien. C'est pour cela que le voisinage du tabou
                       n'explore que les vacations de son propre praticien.
    `jour_demande`     index du jour de la CONSULTATION où l'on décide
                       d'opérer. On ne peut pas l'opérer avant, et c'est ce
                       jour-là qu'on lui propose deux dates.
    `fenetre_jours`    délai MINIMUM avant l'opération : « pas avant une
                       semaine », « pas avant trois semaines », « pas avant
                       six semaines ». C'est une borne INFÉRIEURE, pas un
                       intervalle : au-delà, on cherche jusqu'au bout de
                       l'horizon. Le patient ressort donc toujours de
                       consultation avec une date, sauf si l'horizon entier
                       est saturé.
                       C'est une DONNÉE D'ENTRÉE du patient, au même titre que
                       la durée estimée — une décision clinique (bilan,
                       consultation d'anesthésie, délai de réflexion), jamais
                       quelque chose qu'on déduit de l'historique.
    `priorite`         réservé (pondération d'un patient prioritaire).
    """

    id: int
    med_id: int
    duree_op: int = 60
    duree_sejour: int = 1
    ambulatoire: bool = True
    marge_perso: int = 10
    jour_demande: int = 0
    fenetre_jours: int = 7     # délai MINIMUM ; 7 jours = le plus court des trois
    priorite: float = 1.0
    type_interv: str = ""
    diag: str = ""
    sexe: str = "F"
    age: int = 50
    gestes: tuple[str, ...] = ()

    # TROS RÉELLEMENT OBSERVÉ dans l'historique, en minutes. Présent pour une
    # seule raison : permettre au banc d'essai de rejouer un planning avec les
    # vraies durées après l'avoir construit avec les durées estimées. AUCUN
    # algorithme ne doit le lire — ce serait tricher, puisqu'en consultation on
    # ne le connaît évidemment pas. 0 = inconnu.
    duree_reelle: int = 0

    @property
    def nb_nuits(self) -> int:
        """Nombre de nuits d'hospitalisation. 0 pour un ambulatoire."""
        return max(0, self.duree_sejour - 1)

    @property
    def cout_salle(self) -> int:
        """Ce que le patient consomme réellement dans la vacation, hors TIS."""
        return self.duree_op

    def __repr__(self) -> str:
        t = "AMBU" if self.ambulatoire else f"{self.nb_nuits}n"
        return f"<P{self.id} chir{self.med_id} {self.duree_op}min {t}>"


@dataclass(slots=True)
class Medecin:
    """Praticien. Le motif de vacations n'est plus ici : il vit dans la grille.

    Raison : un motif (jour, heure, modulo) est compact mais on ne raisonne
    jamais dessus. On le déroule une fois pour toutes en vacations datées
    (cf. donnees.py). Sinon chaque contrainte devient de l'arithmétique
    modulaire, et la moindre exception — férié, congé, remplacement — casse
    tout.
    """

    id: int
    nom: str = ""
    specialite: str = ""

    def __repr__(self) -> str:
        return f"<Medecin {self.id} {self.nom} ({self.specialite})>"


@dataclass(slots=True)
class Vacation:
    """Une plage de salle attribuée à un praticien : une ligne de votre grille.

    `jour`          index du jour dans l'horizon (0 = premier jour).
    `debut`, `fin`  minutes depuis minuit ce jour-là.
    `bloc_id`       la salle. Elle est portée par la VACATION, pas par le
                    patient : c'est ainsi que « la salle peut changer » — un
                    patient déplacé vers une autre vacation de son chirurgien
                    change de salle mécaniquement.
    """

    id: int
    bloc_id: int
    med_id: int
    jour: int
    debut: int
    fin: int
    etiquette: str = ""

    @property
    def tvo(self) -> int:
        """Temps de Vacation Offert, en minutes."""
        return self.fin - self.debut

    def __repr__(self) -> str:
        return (f"<V{self.id} J{self.jour} S{self.bloc_id} chir{self.med_id} "
                f"{self.debut // 60:02d}h{self.debut % 60:02d}"
                f"-{self.fin // 60:02d}h{self.fin % 60:02d}>")


@dataclass
class Conflit:
    """Une violation de règle. On ne lève pas d'exception : on COLLECTE.

    Un solveur a besoin de savoir *combien* et *lesquelles*, pas de s'arrêter
    à la première. C'est ce qui permet de transformer une contrainte dure en
    pénalité dans la fonction objectif.
    """

    regle: str
    message: str
    patient_id: int | None = None
    vacation_id: int | None = None

    def __str__(self) -> str:
        return f"[{self.regle}] {self.message}"


# ---------------------------------------------------------------------------
# 2. L'instance : le problème, figé
# ---------------------------------------------------------------------------


@dataclass
class Instance:
    """Tout ce qui ne change pas pendant la recherche, + les index utiles.

    Les index (`vacations_du_medecin`, `vacations_du_jour`) sont calculés une
    fois. Dans une boucle qui tourne des millions de fois, une recherche
    linéaire « pour toutes les vacations, si v.med_id == p.med_id » coûte
    l'essentiel du temps de calcul.
    """

    patients: dict[int, Patient] = field(default_factory=dict)
    medecins: dict[int, Medecin] = field(default_factory=dict)
    vacations: dict[int, Vacation] = field(default_factory=dict)

    # --- ressources -------------------------------------------------------
    capacite_lits: int = 42          # lits d'hospitalisation (>= 1 nuit)
    capacite_places: int = 12        # places ambulatoires SIMULTANÉES
    capacite_places_jour: int = 18   # admissions ambulatoires par JOUR
    # Deux capacités pour la même ressource, et c'est volontaire.
    #   Le tabou n°1 ne connaît pas les heures : il ne peut compter que le
    #   NOMBRE d'ambulatoires programmés dans la journée. Or une place tourne :
    #   un patient du matin la libère pour un patient de l'après-midi. Compter
    #   les admissions du jour contre les 12 places physiques interdirait des
    #   journées parfaitement réalisables.
    #   On borne donc le flux journalier (`capacite_places_jour`, de l'ordre de
    #   12 places × rotation observée) au niveau global, et on vérifie le vrai
    #   pic simultané (`capacite_places`) au niveau local, où les heures sont
    #   connues. Sur votre historique 2022 : médiane 10 ambulatoires/jour,
    #   P90 = 15, maximum 20.
    tis: int = 15                    # temps inter-salle, minutes
    tis_meme_acte: int = 10          # TIS réduit si acte identique consécutif
    fermeture_uca: int = 20 * 60     # sortie ambulatoire au plus tard (minutes)
    surveillance_ambu: int = 180     # temps de surveillance d'un ambulatoire
    heure_sortie_hospit: int = 10 * 60   # heure de libération d'un lit

    # --- horizon ----------------------------------------------------------
    jour_zero: date = date(2026, 1, 5)
    nb_jours: int = 182              # 6 mois

    # --- index (remplis par `indexer`) ------------------------------------
    vacations_du_medecin: dict[int, list[int]] = field(default_factory=dict)
    vacations_du_jour: dict[int, list[int]] = field(default_factory=dict)
    jours_ouvres: list[int] = field(default_factory=list)
    nb_vacations_medecin: dict[int, int] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def ajouter_patient(self, p: Patient) -> None:
        self.patients[p.id] = p

    def ajouter_medecin(self, m: Medecin) -> None:
        self.medecins[m.id] = m

    def ajouter_vacation(self, v: Vacation) -> None:
        self.vacations[v.id] = v

    def indexer(self) -> "Instance":
        """À appeler une fois toutes les vacations et patients enregistrés."""
        self.vacations_du_medecin = defaultdict(list)
        self.vacations_du_jour = defaultdict(list)
        for v in self.vacations.values():
            self.vacations_du_medecin[v.med_id].append(v.id)
            self.vacations_du_jour[v.jour].append(v.id)
        for liste in self.vacations_du_medecin.values():
            liste.sort(key=lambda vid: self.vacations[vid].jour)
        self.jours_ouvres = sorted(self.vacations_du_jour)
        # Dénominateur de la variance des taux de remplissage : FIXE, calculé
        # une fois. Les vacations vides comptent, avec un taux de 0.
        self.nb_vacations_medecin = {m: len(v)
                                     for m, v in self.vacations_du_medecin.items()}
        return self

    # -- capacités ---------------------------------------------------------

    def capacite_utile(self, vacation_id: int) -> int:
        """TVO disponible pour du TROS.

        Il n'y a PAS de marge forfaitaire retranchée ici : la marge est
        portée par les patients (`marge_perso`, issue de la dispersion
        historique de leur acte) et retranchée dans `Solution.charge`. Une
        vacation remplie d'actes très prévisibles peut donc légitimement
        être remplie davantage qu'une vacation d'actes variables.
        """
        return self.vacations[vacation_id].tvo

    def date_du_jour(self, j: int) -> date:
        return self.jour_zero + timedelta(days=j)

    def est_weekend(self, j: int) -> bool:
        return self.date_du_jour(j).weekday() >= 5

    # -- fenêtres de recherche offertes au praticien ------------------------
    #
    # Trois échelles, pas plus : un menu de trois choix se lit d'un coup d'œil
    # en consultation, un curseur continu demanderait de réfléchir.
    FENETRE_SEMAINE = 7
    FENETRE_TROIS_SEMAINES = 21
    FENETRE_MOIS_ET_DEMI = 45
    FENETRES = {"semaine": 7, "3 semaines": 21, "mois et demi": 45}

    def horodater(self, j: int, minutes: int) -> datetime:
        """(index de jour, minutes depuis minuit) -> datetime, pour l'affichage."""
        return datetime.combine(self.date_du_jour(j), datetime.min.time()) + timedelta(minutes=minutes)

    # -- vacations éligibles pour un patient -------------------------------

    def vacations_possibles(self, patient_id: int,
                            fenetre: int | None = None) -> list[int]:
        """Vacations où le patient PEUT aller.

        Trois filtres, tous DURS :
          - le chirurgien ne change jamais (votre règle) ;
          - on n'opère pas quelqu'un avant sa consultation ;
          - on n'opère pas avant le DÉLAI MINIMUM fixé par le praticien.

        LA FENÊTRE EST UNE BORNE INFÉRIEURE — « pas avant trois semaines » —
        et non un intervalle. Au-delà, la recherche va jusqu'au bout de
        l'horizon. C'est ce qui garantit qu'un patient ressort toujours de
        consultation avec une date, sauf si l'horizon entier est saturé.

        `fenetre` permet de forcer un autre délai minimum, ce qui ne sert
        qu'aux analyses de sensibilité.
        """
        p = self.patients[patient_id]
        jmin = p.jour_demande + (p.fenetre_jours if fenetre is None else fenetre)
        return [vid for vid in self.vacations_du_medecin.get(p.med_id, ())
                if self.vacations[vid].jour >= jmin]


# ---------------------------------------------------------------------------
# 3. La solution globale : ce que manipule le tabou n°1
# ---------------------------------------------------------------------------


class Solution:
    """Affectation patient -> vacation, avec ÉVALUATION INCRÉMENTALE.

    Le cœur de la performance d'un tabou est là. À chaque itération on évalue
    des centaines de mouvements ; recalculer la fonction objectif à chaque fois
    coûterait O(nb_patients). On maintient donc en permanence :

      nb[v], somme_durees[v], somme_marges[v]  -> la charge de chaque vacation
      places_jour[j], lits_jour[j]             -> les profils journaliers
      s_places, s_places2, s_lits, s_lits2     -> sommes et sommes de carrés,
                                                  d'où la variance en O(1)

    Un déplacement ne touche qu'une poignée de ces compteurs, et
    `delta_deplacement` renvoie la variation exacte de l'objectif SANS rien
    modifier. On applique seulement le mouvement retenu.

    Rappel de la formule de variance utilisée :
        Var(X) = E[X²] - E[X]²  =  s2/n - (s/n)²
    Numériquement imprudente en général, parfaitement sûre ici : les valeurs
    sont de petits entiers positifs.
    """

    __slots__ = ("inst", "affectation", "nb", "somme_durees", "somme_marges",
                 "places_jour", "lits_jour", "s_places", "s_places2",
                 "s_lits", "s_lits2", "sans_date", "hors_horizon",
                 "s_taux", "s_taux2", "_jours_places", "_n_jours_lits")

    def __init__(self, inst: Instance):
        self.inst = inst
        self.affectation: dict[int, int | None] = {pid: None for pid in inst.patients}

        self.nb: dict[int, int] = {vid: 0 for vid in inst.vacations}
        self.somme_durees: dict[int, int] = {vid: 0 for vid in inst.vacations}
        self.somme_marges: dict[int, int] = {vid: 0 for vid in inst.vacations}

        self.places_jour: list[int] = [0] * inst.nb_jours
        self.lits_jour: list[int] = [0] * inst.nb_jours

        self.s_places = self.s_places2 = 0
        self.s_lits = self.s_lits2 = 0

        # Agrégats du REMPLISSAGE par médecin : somme et somme des carrés des
        # taux de remplissage de SES vacations, d'où leur variance en O(1).
        # Les vacations vides comptent, avec un taux de 0 — c'est ce qui fait
        # que laisser une journée à l'abandon pendant qu'une autre sature est
        # pénalisé. C'est le sens de « lisser entre les vacations du médecin ».
        self.s_taux: dict[int, float] = {m: 0.0 for m in inst.vacations_du_medecin}
        self.s_taux2: dict[int, float] = {m: 0.0 for m in inst.vacations_du_medecin}

        # IL N'Y A PAS DE LISTE D'ATTENTE dans ce modèle. Deux états
        # seulement pour un patient sans vacation :
        #   `sans_date`    : il n'est pas encore passé en consultation, il
        #                    n'existe donc pas encore pour le planning ;
        #   `hors_horizon` : il est passé en consultation mais aucune date ne
        #                    tenait dans les 6 mois — il sera reconvoqué quand
        #                    l'horizon aura glissé.
        # Le tabou hors-ligne utilise aussi `sans_date` comme état TRANSITOIRE
        # pendant sa recherche ; comme le nombre de patients sans date est
        # comparé AVANT le coût, il n'y reste que ceux qu'aucune vacation de
        # leur fenêtre ne peut accueillir.
        self.sans_date: set[int] = set(inst.patients)
        self.hors_horizon: set[int] = set()

        # dénominateurs des variances, fixés une fois pour toutes
        self._jours_places = inst.jours_ouvres or list(range(inst.nb_jours))
        self._n_jours_lits = inst.nb_jours

    # -- copie -------------------------------------------------------------

    def copie(self) -> "Solution":
        """Copie de travail. Rapide : que des dict/list de petits entiers."""
        s = Solution.__new__(Solution)
        s.inst = self.inst
        s.affectation = dict(self.affectation)
        s.nb = dict(self.nb)
        s.somme_durees = dict(self.somme_durees)
        s.somme_marges = dict(self.somme_marges)
        s.places_jour = list(self.places_jour)
        s.lits_jour = list(self.lits_jour)
        s.s_places, s.s_places2 = self.s_places, self.s_places2
        s.s_lits, s.s_lits2 = self.s_lits, self.s_lits2
        s.sans_date = set(self.sans_date)
        s.hors_horizon = set(self.hors_horizon)
        s.s_taux = dict(self.s_taux)
        s.s_taux2 = dict(self.s_taux2)
        s._jours_places = self._jours_places
        s._n_jours_lits = self._n_jours_lits
        return s

    # -- lecture -----------------------------------------------------------

    def charge(self, vid: int) -> int:
        """Minutes engagées dans la vacation : TROS + TIS + marges de risque.

        C'est ici qu'on paie la marge : la vacation doit tenir même si tous
        les actes dérapent jusqu'à leur P90. Un `tis * (n-1)` classique, plus
        la somme des marges individuelles.
        """
        n = self.nb[vid]
        if n == 0:
            return 0
        return self.somme_durees[vid] + self.somme_marges[vid] + self.inst.tis * (n - 1)

    def creux(self, vid: int) -> int:
        return max(0, self.inst.capacite_utile(vid) - self.charge(vid))

    def depassement(self, vid: int) -> int:
        return max(0, self.charge(vid) - self.inst.capacite_utile(vid))

    def taux_remplissage(self, vid: int) -> float:
        tvo = self.inst.vacations[vid].tvo
        return self.charge(vid) / tvo if tvo else 0.0

    def patients_de(self, vid: int) -> list[int]:
        return [pid for pid, v in self.affectation.items() if v == vid]

    def patients_du_jour(self, j: int) -> list[int]:
        vids = set(self.inst.vacations_du_jour.get(j, ()))
        return [pid for pid, v in self.affectation.items() if v in vids]

    # -- modification ------------------------------------------------------

    def _appliquer_patient(self, pid: int, vid: int | None, signe: int) -> None:
        """Ajoute (+1) ou retire (-1) la contribution d'un patient posé en vid."""
        if vid is None:
            return
        inst = self.inst
        p = inst.patients[pid]
        v = inst.vacations[vid]

        self.nb[vid] += signe
        self.somme_durees[vid] += signe * p.duree_op
        self.somme_marges[vid] += signe * p.marge_perso

        if p.ambulatoire:
            j = v.jour
            a = self.places_jour[j]
            b = a + signe
            self.places_jour[j] = b
            self.s_places += b - a
            self.s_places2 += b * b - a * a
        else:
            # les nuits occupées : J, J+1, ... J+nb_nuits-1
            for j in range(v.jour, min(v.jour + p.nb_nuits, inst.nb_jours)):
                a = self.lits_jour[j]
                b = a + signe
                self.lits_jour[j] = b
                self.s_lits += b - a
                self.s_lits2 += b * b - a * a



    def affecter(self, pid: int, vid: int | None) -> None:
        """Pose un patient dans une vacation. `None` = le retire du planning.

        Retirer un patient n'est PAS une opération du processus réel : une
        date annoncée ne bouge plus. C'est un mouvement interne au tabou
        hors-ligne, et l'outil de replanification exceptionnelle (congé d'un
        praticien, fermeture de salle) s'en sert aussi.
        """
        ancien = self.affectation[pid]
        if ancien == vid:
            return
        med = self.inst.patients[pid].med_id

        # Les agrégats de remplissage se mettent à jour par différence : on
        # note le taux des deux vacations concernées avant, puis après. Comme
        # le patient ne change jamais de chirurgien, les deux vacations
        # appartiennent au même praticien et un seul médecin est touché.
        avant = [(v, self.taux_remplissage(v)) for v in (ancien, vid) if v is not None]

        self._appliquer_patient(pid, ancien, -1)
        self._appliquer_patient(pid, vid, +1)
        self.affectation[pid] = vid

        for v, t0 in avant:
            t1 = self.taux_remplissage(v)
            self.s_taux[med] += t1 - t0
            self.s_taux2[med] += t1 * t1 - t0 * t0

        if vid is None:
            self.sans_date.add(pid)
        else:
            self.sans_date.discard(pid)
            self.hors_horizon.discard(pid)

    def variance_remplissage_medecin(self, med_id: int) -> float:
        """Variance des taux de remplissage des vacations de ce praticien.

        0 = toutes ses journées sont remplies pareil. C'est le critère
        « % de remplissage associé au médecin », écrit comme un coût à
        minimiser : on ne cherche pas à remplir au maximum, on cherche à ce
        que ses journées se ressemblent.

        Conséquence pratique, et c'est elle qu'il faut retenir : un patient
        ira spontanément dans la vacation la MOINS remplie du praticien, dans
        la fenêtre choisie. On évite ainsi les journées à 95 % — celles qui
        débordent au moindre aléa — et les journées à 30 % — celles qui
        mobilisent une équipe pour rien.
        """
        n = self.inst.nb_vacations_medecin.get(med_id, 0)
        if n == 0:
            return 0.0
        s, s2 = self.s_taux[med_id], self.s_taux2[med_id]
        return max(0.0, s2 / n - (s / n) ** 2)

    def variance_remplissage(self) -> float:
        """Moyenne du précédent sur tous les praticiens de la grille.

        On divise par le nombre TOTAL de praticiens, pas par le nombre de
        praticiens actifs : sinon le dénominateur bougerait pendant la
        recherche et deux itérations ne seraient plus comparables.
        """
        n = max(1, len(self.inst.vacations_du_medecin))
        return sum(self.variance_remplissage_medecin(m)
                   for m in self.inst.vacations_du_medecin) / n

    # -- indicateurs agrégés ----------------------------------------------

    def variance_places(self) -> float:
        # Seuls les jours ouvrés comptent au dénominateur : une place
        # ambulatoire ne peut pas être utilisée un jour sans vacation, ce
        # n'est pas un « creux » qu'on pourrait combler. Les jours non ouvrés
        # valent 0 et ne contribuent donc ni à s_places ni à s_places2 : les
        # sommes maintenues incrémentalement sont directement utilisables.
        n = len(self._jours_places)
        if n == 0:
            return 0.0
        return max(0.0, self.s_places2 / n - (self.s_places / n) ** 2)

    def variance_lits(self) -> float:
        # tous les jours comptent, week-ends compris : un lit occupé le samedi
        # est un lit occupé. C'est précisément ce qu'on veut lisser.
        n = self._n_jours_lits
        if n == 0:
            return 0.0
        return max(0.0, self.s_lits2 / n - (self.s_lits / n) ** 2)

    def moyenne_places(self) -> float:
        n = len(self._jours_places)
        return self.s_places / n if n else 0.0

    def moyenne_lits(self) -> float:
        return self.s_lits / self._n_jours_lits if self._n_jours_lits else 0.0

    def creux_total(self) -> int:
        return sum(self.creux(vid) for vid in self.inst.vacations)

    def depassement_total(self) -> int:
        return sum(self.depassement(vid) for vid in self.inst.vacations)

    def tvo_total(self) -> int:
        return sum(v.tvo for v in self.inst.vacations.values())

    def surcharge_lits(self) -> int:
        cap = self.inst.capacite_lits
        return sum(max(0, n - cap) for n in self.lits_jour)

    def surcharge_places(self) -> int:
        cap = self.inst.capacite_places_jour
        return sum(max(0, self.places_jour[j] - cap) for j in self._jours_places)

    def delai_total(self) -> float:
        """Somme des délais consultation -> opération, pondérés par la priorité.

        Un patient sans date compte comme s'il était opéré au dernier jour de
        l'horizon : c'est une borne basse de ce qu'il coûtera réellement, et
        surtout c'est monotone — sans cela, ne pas placer un patient
        deviendrait gratuit et l'algorithme préférerait ne rien faire.
        """
        total = 0.0
        inst = self.inst
        for pid, vid in self.affectation.items():
            p = inst.patients[pid]
            jour = inst.vacations[vid].jour if vid is not None else inst.nb_jours
            total += p.priorite * (jour - p.jour_demande)
        return total

    # -- audit de faisabilité ----------------------------------------------

    def verifier(self) -> list[Conflit]:
        """Les règles DURES du niveau global. Indépendant de la fonction
        objectif : c'est la preuve que la solution est exploitable."""
        c: list[Conflit] = []
        inst = self.inst
        for pid, vid in self.affectation.items():
            if vid is None:
                continue
            p, v = inst.patients[pid], inst.vacations[vid]
            if v.med_id != p.med_id:
                c.append(Conflit("G1", f"patient {pid} chez le chirurgien {v.med_id} "
                                       f"au lieu de {p.med_id}", pid, vid))
            if v.jour < p.jour_demande:
                c.append(Conflit("G5", f"patient {pid} opéré J{v.jour} alors qu'il "
                                       f"n'est inscrit que J{p.jour_demande}", pid, vid))
        for vid in inst.vacations:
            d = self.depassement(vid)
            if d > 0:
                c.append(Conflit("G2", f"vacation {vid} surchargée de {d} min "
                                       f"(marges de risque incluses)", None, vid))
        for j, n in enumerate(self.lits_jour):
            if n > inst.capacite_lits:
                c.append(Conflit("G3", f"J{j} ({inst.date_du_jour(j)}) : "
                                       f"{n} lits pour {inst.capacite_lits}"))
        for j in self._jours_places:
            n = self.places_jour[j]
            if n > inst.capacite_places_jour:
                c.append(Conflit("G4", f"J{j} ({inst.date_du_jour(j)}) : "
                                       f"{n} admissions ambulatoires pour "
                                       f"{inst.capacite_places_jour} possibles"))
        return c

    # -- tableau de bord ---------------------------------------------------

    def indicateurs(self) -> dict:
        tvo = self.tvo_total()
        ouvertes = [vid for vid in self.inst.vacations if self.nb[vid] > 0]
        taux = [self.taux_remplissage(vid) for vid in self.inst.vacations]
        lits_ouvres = [self.lits_jour[j] for j in range(self.inst.nb_jours)]
        places = [self.places_jour[j] for j in self._jours_places]
        return {
            "patients_programmes": len(self.inst.patients) - len(self.sans_date),
            "patients_sans_date": len(self.sans_date),
            "patients_hors_horizon": len(self.hors_horizon),
            "vacations_ouvertes": f"{len(ouvertes)}/{len(self.inst.vacations)}",
            "taux_remplissage_moyen": sum(taux) / len(taux) if taux else 0.0,
            "remplissage_ecart_type": self.variance_remplissage() ** 0.5,
            "creux_total_h": self.creux_total() / 60,
            "part_creux": self.creux_total() / tvo if tvo else 0.0,
            "lits_moyen": self.moyenne_lits(),
            "lits_ecart_type": self.variance_lits() ** 0.5,
            "lits_pic": max(lits_ouvres, default=0),
            "lits_creux": min(lits_ouvres, default=0),
            "places_moyen": self.moyenne_places(),
            "places_ecart_type": self.variance_places() ** 0.5,
            "places_pic": max(places, default=0),
            "surcharge_lits_j": sum(1 for n in lits_ouvres if n > self.inst.capacite_lits),
            "surcharge_places_j": sum(1 for n in places if n > self.inst.capacite_places_jour),
            "depassement_vacations_h": self.depassement_total() / 60,
            "delai_moyen_j": self.delai_total() / max(1, len(self.inst.patients)),
            "conflits": len(self.verifier()),
        }


# ---------------------------------------------------------------------------
# 4. Le planning d'une journée : ce que produit le tabou n°2
# ---------------------------------------------------------------------------


@dataclass
class Creneau:
    """Un patient posé à une heure précise, dans une salle, sur une place."""

    patient_id: int
    vacation_id: int
    bloc_id: int
    debut: int          # minutes depuis minuit
    fin: int            # = debut + duree_op (sortie de salle)
    lib_place: int      # instant de libération de la place / du lit
    place: int = -1     # numéro de place ambulatoire, ou de lit
    ambulatoire: bool = True


@dataclass
class PlanningJour:
    """Résultat du tabou local pour une journée.

    `sequences[vid]` = liste ordonnée de patient_id.
    `creneaux[pid]`  = le créneau calculé.
    """

    jour: int
    sequences: dict[int, list[int]] = field(default_factory=dict)
    creneaux: dict[int, Creneau] = field(default_factory=dict)
    pic_places: int = 0
    pic_lits: int = 0
    depassement: int = 0
    hors_uca: int = 0
    creux_total: int = 0          # minutes de vacation non utilisées ce jour-là
    tvo_jour: int = 0             # temps de vacation offert ce jour-là
    taux: dict[int, float] = field(default_factory=dict)  # remplissage par vacation
    places_moyenne: float = 0.0   # occupation moyenne des places sur la journée
    places_variance: float = 0.0  # variance temporelle de cette occupation

    @property
    def part_creux(self) -> float:
        return self.creux_total / self.tvo_jour if self.tvo_jour else 0.0

    @property
    def variance_taux(self) -> float:
        """Dispersion du remplissage entre les vacations de la journée."""
        if len(self.taux) < 2:
            return 0.0
        vals = list(self.taux.values())
        moy = sum(vals) / len(vals)
        return sum((t - moy) ** 2 for t in vals) / len(vals)

    def patients(self) -> list[int]:
        return [pid for seq in self.sequences.values() for pid in seq]


def deriver_creneaux(inst: Instance, jour: int,
                     sequences: dict[int, list[int]]) -> PlanningJour:
    """Séquence -> horaires -> places. Le calcul déterministe du tabou n°2.

    ENCHAÎNEMENT AU PLUS TÔT. Chaque patient entre en salle dès que le
    précédent en sort, plus le TIS. L'algorithme n'a pas le droit d'insérer du
    temps mort pour étaler la charge de l'ambulatoire : une salle qui attend
    coûte cher, et le personnel ne l'accepterait pas. La seule décision est
    donc la SÉQUENCE, plus la répartition des patients entre les vacations du
    même praticien ce jour-là.

    LE TIS N'EST PAS UN CREUX. Entre deux interventions il faut nettoyer la
    salle et installer le patient suivant : c'est incompressible, et pendant
    ce temps l'équipe travaille. Le creux, c'est ce qui reste en FIN de
    vacation. Le TIS est réduit entre deux actes identiques (même boîte
    d'instruments, même installation), ce qui récompense le regroupement — et
    c'est par là que la séquence agit sur le creux.

    Trois étapes :
      1. on déroule chaque vacation au plus tôt ;
      2. chaque patient occupe une place (ambulatoire, jusqu'à
         `fin_op + surveillance`) ou un lit (jusqu'à la fermeture du jour) ;
      3. on colorie les intervalles pour attribuer les numéros de place —
         étape OPTIMALE (cf. `colorier_intervalles`), donc tout le travail
         d'optimisation porte sur l'étape 1.
    """
    pj = PlanningJour(jour=jour,
                      sequences={k: list(v) for k, v in sequences.items()})
    fin_journee = 24 * 60

    for vid, seq in pj.sequences.items():
        v = inst.vacations[vid]
        pj.tvo_jour += v.tvo
        t = v.debut
        precedent: Patient | None = None
        for pid in seq:
            p = inst.patients[pid]
            if precedent is not None:
                t += (inst.tis_meme_acte
                      if (p.type_interv and p.type_interv == precedent.type_interv)
                      else inst.tis)
            debut, fin = t, t + p.duree_op
            lib = fin + inst.surveillance_ambu if p.ambulatoire else fin_journee
            pj.creneaux[pid] = Creneau(pid, vid, v.bloc_id, debut, fin, lib,
                                       ambulatoire=p.ambulatoire)
            t = fin
            precedent = p

        utilise = min(t, v.fin) - v.debut if seq else 0
        pj.taux[vid] = utilise / v.tvo if v.tvo else 0.0
        pj.creux_total += max(0, v.fin - t) if seq else v.tvo
        if seq:
            pj.depassement += max(0, t - v.fin)

    # -- places et lits : profil, pic, et FORME de la courbe ----------------
    ouverture = min((inst.vacations[vid].debut for vid in pj.sequences
                     if pj.sequences[vid]), default=8 * 60)

    ambulatoires = [(c.fin, c.lib_place) for c in pj.creneaux.values() if c.ambulatoire]
    profil_places = profil_cumulatif(ambulatoires)
    pj.pic_places = pic(profil_places)
    pj.places_moyenne, pj.places_variance = moments_temporels(
        profil_places, ouverture, inst.fermeture_uca)

    hospitalises = [(c.fin, c.lib_place) for c in pj.creneaux.values()
                    if not c.ambulatoire]
    pj.pic_lits = pic(profil_cumulatif(hospitalises))

    for ambu in (True, False):
        items = [(c.patient_id, c.fin, c.lib_place)
                 for c in pj.creneaux.values() if c.ambulatoire == ambu]
        for pid, num in colorier_intervalles(items).items():
            pj.creneaux[pid].place = num

    pj.hors_uca = sum(1 for c in pj.creneaux.values()
                      if c.ambulatoire and c.lib_place > inst.fermeture_uca)
    return pj


# ---------------------------------------------------------------------------
# 5. Affichage
# ---------------------------------------------------------------------------


def afficher_journee(inst: Instance, pj: PlanningJour) -> str:
    """Gantt textuel, pour déboguer sans dépendance graphique."""
    d = inst.date_du_jour(pj.jour)
    lignes = [f"=== {d:%A %d/%m/%Y} (J{pj.jour}) ==="]
    for vid in sorted(pj.sequences, key=lambda x: (inst.vacations[x].bloc_id,
                                                   inst.vacations[x].debut)):
        v = inst.vacations[vid]
        seq = pj.sequences[vid]
        utilise = sum(inst.patients[pid].duree_op for pid in seq)
        lignes.append(
            f"  Salle {v.bloc_id} {v.debut//60:02d}h{v.debut%60:02d}"
            f"-{v.fin//60:02d}h{v.fin%60:02d}  chir {v.med_id:<3} "
            f"TVO {v.tvo/60:.2f}h | TROS {utilise/60:.2f}h | "
            f"remplissage {utilise/v.tvo:.0%}")
        for pid in seq:
            c = pj.creneaux[pid]
            p = inst.patients[pid]
            tag = f"place {c.place}" if p.ambulatoire else f"lit {c.place} ({p.nb_nuits}n)"
            marque = " !" if c.fin > v.fin else ""
            lignes.append(
                f"      {c.debut//60:02d}h{c.debut%60:02d}-{c.fin//60:02d}h{c.fin%60:02d}"
                f"  P{pid:<5} {p.type_interv[:32]:<32} {tag}{marque}")
    lignes.append(f"  places : pic {pj.pic_places}/{inst.capacite_places}, "
                  f"moyenne {pj.places_moyenne:.1f}, "
                  f"écart-type temporel {pj.places_variance ** 0.5:.2f}")
    lignes.append(f"  creux {pj.creux_total} min ({pj.part_creux:.0%} du TVO), "
                  f"dispersion du remplissage {pj.variance_taux ** 0.5:.3f} | "
                  f"lits ouverts {pj.pic_lits} | "
                  f"dépassement {pj.depassement} min | hors UCA {pj.hors_uca}")
    return "\n".join(lignes)


def profil_hebdo(sol: Solution) -> str:
    """Courbe d'occupation des lits, une ligne par semaine. C'est la courbe
    qu'on cherche à aplatir (votre slide 13, objectif ±2 lits)."""
    inst = sol.inst
    lignes = ["Occupation des lits (une colonne par jour, L M M J V S D) :"]
    for debut in range(0, inst.nb_jours, 7):
        sem = sol.lits_jour[debut:debut + 7]
        pl = [sol.places_jour[j] for j in range(debut, min(debut + 7, inst.nb_jours))]
        lignes.append(f"  S{debut//7:02d} lits " + " ".join(f"{n:3d}" for n in sem)
                      + "   | places " + " ".join(f"{n:2d}" for n in pl))
    m, e = sol.moyenne_lits(), sol.variance_lits() ** 0.5
    lignes.append(f"  moyenne {m:.1f} lits, écart-type {e:.2f} "
                  f"(objectif : écart-type le plus faible possible)")
    return "\n".join(lignes)
