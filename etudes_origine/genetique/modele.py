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
from bisect import bisect_left, bisect_right, insort
from math import sqrt
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

# URGENCES — niveaux d'un patient (`Patient.urgence`)
PROGRAMME = ""          # patient programmé classique
SEMI_URGENCE = "semi"   # à opérer sous `delai_max` jours, connu à l'avance
URGENCE_JOUR = "jour"   # à opérer le jour même (sinon le jour ouvré suivant)

# med_id des créneaux URGENCES de la grille : ils n'appartiennent à aucun
# chirurgien, tout patient urgent peut y être opéré (par son chirurgien).
MED_URGENCE = -1


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
    `duree_sejour`     nombre de NUITS + 1 (1 = ambulatoire). Les nuits se
                       calculent avec les DATES : Date Sortie − Date Entrée.
                       Ne JAMAIS utiliser la colonne « durée de séjour » du
                       fichier : 1 884 séjours y sont codés 1 alors que le
                       patient a passé une ou plusieurs nuits.
    `nuits_avant`      nuits passées AVANT le jour opératoire (entrée la
                       veille ou plus tôt) = Date Inter − Date Entrée. Elles
                       font partie de `nb_nuits` : le lit est occupé de
                       J − nuits_avant à J − nuits_avant + nb_nuits − 1.
    `ambulatoire`      DÉDUIT de `duree_sejour` (ambulatoire ⇔ 0 nuit). On
                       peut le passer, mais s'il contredit `duree_sejour` la
                       création du patient échoue (ValueError) : sinon un
                       patient pouvait n'occuper ni lit ni place.
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
                       Borne SUPÉRIEURE : `Instance.delai_max_programme`
                       (6 mois après la consultation, règle du prof), ou
                       `delai_max` si le patient en a un.
    `priorite`         réservé (pondération d'un patient prioritaire).
    """

    id: int
    med_id: int
    duree_op: int = 60
    duree_sejour: int = 1
    ambulatoire: bool | None = None    # déduit de duree_sejour (cf. __post_init__)
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

    # URGENCES. `urgence` : PROGRAMME, SEMI_URGENCE ou URGENCE_JOUR.
    # `delai_max` : BORNE SUPÉRIEURE du délai, comptée depuis `jour_demande`
    # (jour où l'urgence est connue). None = pas de borne.
    # Comme tout patient de ce modèle, un urgent reçoit sa date dès qu'il est
    # inscrit et ne la change plus. Urgence du jour même : `jour_demande` =
    # jour d'arrivée, `fenetre_jours` = 0.
    urgence: str = PROGRAMME
    delai_max: int | None = None

    nuits_avant: int = 0

    def __post_init__(self) -> None:
        if self.duree_sejour < 1:
            raise ValueError(f"patient {self.id} : duree_sejour = nuits + 1 >= 1")
        ambu = self.duree_sejour == 1
        if self.ambulatoire is None:
            self.ambulatoire = ambu
        elif bool(self.ambulatoire) != ambu:
            raise ValueError(f"patient {self.id} : ambulatoire={self.ambulatoire} "
                             f"mais {self.duree_sejour - 1} nuit(s) — ambulatoire ⇔ 0 nuit")
        if not 0 <= self.nuits_avant <= self.duree_sejour - 1:
            raise ValueError(f"patient {self.id} : nuits_avant={self.nuits_avant} "
                             f"hors de [0, {self.duree_sejour - 1}]")

    @property
    def est_urgent(self) -> bool:
        return self.urgence != PROGRAMME

    @property
    def jour_min(self) -> int:
        """Premier jour où il peut être opéré (délai minimum)."""
        return self.jour_demande + self.fenetre_jours

    @property
    def jour_max(self) -> int | None:
        """Dernier jour autorisé si le patient a son propre `delai_max`,
        None sinon. La borne effective, valable pour TOUS les patients, est
        `Instance.jour_max(pid)`."""
        return None if self.delai_max is None else self.jour_demande + self.delai_max

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
    # URGENCES : créneau « URGENCES » de la grille. Interdit au programmé,
    # ouvert à tout patient urgent quel que soit son chirurgien.
    urgence: bool = False

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
    heure_sortie_hospit: int = 10 * 60   # (inutilisé pour l'instant)
    ouverture_uca: int = 7 * 60 + 30     # arrivée ambulatoire au plus tôt
    avance_ambu: int = 120               # un ambulatoire arrive 2 h avant d'entrer en salle

    # --- délais -----------------------------------------------------------
    # Règle : opérer entre `fenetre_jours` et 6 mois APRÈS LA CONSULTATION.
    # La borne haute est propre à chaque patient (jour_demande + 182), pas
    # la fin de l'horizon.
    delai_max_programme: int = 182

    # --- marges de risque ----------------------------------------------------
    # Comment les marges individuelles (P90 − médiane) s'additionnent :
    #   "somme"       : Σ marges — tous les actes dérapent en même temps.
    #                   Prudent, garantit zéro dépassement (défaut).
    #   "quadratique" : √(Σ marges²) — P90 de la somme si les dérapages sont
    #                   indépendants et à peu près gaussiens. Remplit plus.
    cumul_marges: str = "somme"

    # --- cibles et capacités par nuit / par jour (optionnelles) ----------------
    # Sans cible, les coûts mesurent la VARIANCE, sur les seuls jours ouvrés
    # (nuit qui suit un jour de bloc) : on ne pousse plus à remplir les
    # week-ends et les jours fériés. Avec une cible (liste de longueur
    # nb_jours), on mesure l'écart quadratique à la cible, sur tous les jours
    # de la fenêtre — y compris un week-end, avec la cible qu'on lui donne.
    # En simulation EN LIGNE, donner cible − charge future attendue : les
    # jours lointains, pas encore remplis par les patients à venir, cessent
    # alors d'attirer les patients.
    cible_lits: list[float] | None = None
    cible_places: list[float] | None = None
    # Capacité de lits par nuit (ex. moins de lits le week-end), sinon
    # `capacite_lits` tous les jours. Contrainte DURE.
    capacite_lits_nuit: list[int] | None = None

    # --- réserves pour les urgences ---------------------------------------
    # Le PROGRAMMÉ n'a droit qu'à :
    #   TVO · (1 − tampon_urgence)            dans chaque vacation ;
    #   capacite_lits − reserve_lits          lits chaque nuit ;
    #   capacite_places_jour − reserve_places admissions ambulatoires / jour.
    # Les URGENTS peuvent aller jusqu'aux capacités réelles. Réserves
    # identiques pour toutes les spécialités. Valeurs par défaut nulles :
    # sans urgences, le modèle se comporte exactement comme avant.
    tampon_urgence: float = 0.0
    reserve_lits: int = 0
    reserve_places: int = 0

    # --- horizon ----------------------------------------------------------
    jour_zero: date = date(2026, 1, 5)
    nb_jours: int = 182              # 6 mois

    # --- index (remplis par `indexer`) ------------------------------------
    vacations_du_medecin: dict[int, list[int]] = field(default_factory=dict)
    vacations_du_jour: dict[int, list[int]] = field(default_factory=dict)
    jours_ouvres: list[int] = field(default_factory=list)
    nb_vacations_medecin: dict[int, int] = field(default_factory=dict)
    vacations_urgence_du_jour: dict[int, list[int]] = field(default_factory=dict)

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
        self.vacations_urgence_du_jour = defaultdict(list)
        for v in self.vacations.values():
            self.vacations_du_jour[v.jour].append(v.id)
            if v.urgence:        # hors index chirurgien : ni candidates, ni remplissage
                self.vacations_urgence_du_jour[v.jour].append(v.id)
            else:
                self.vacations_du_medecin[v.med_id].append(v.id)
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

    def capacite_programme(self, vacation_id: int) -> int:
        """Minutes de la vacation accessibles au PROGRAMMÉ : TVO moins le
        tampon réservé aux urgences. 0 pour un créneau URGENCES."""
        v = self.vacations[vacation_id]
        if v.urgence:
            return 0
        return v.tvo - round(self.tampon_urgence * v.tvo)

    def cap_lits(self, j: int) -> int:
        """Capacité de lits de la nuit j."""
        return self.capacite_lits if self.capacite_lits_nuit is None else self.capacite_lits_nuit[j]

    def jour_max(self, patient_id: int) -> int:
        """Dernier jour où le patient peut être opéré : son `delai_max` s'il
        en a un, sinon 6 mois (`delai_max_programme`) après la consultation."""
        p = self.patients[patient_id]
        d = self.delai_max_programme if p.delai_max is None else p.delai_max
        return p.jour_demande + d

    def nuits_occupees(self, patient_id: int, jour: int) -> range:
        """Nuits où le patient opéré le jour `jour` occupe un lit (vide pour
        un ambulatoire), tronquées à l'horizon."""
        p = self.patients[patient_id]
        debut = jour - p.nuits_avant
        return range(max(0, debut), min(debut + p.nb_nuits, self.nb_jours))

    def jour_ouvre_suivant(self, j: int) -> int | None:
        """Premier jour > j où le bloc a des vacations (None si aucun)."""
        k = bisect_left(self.jours_ouvres, j + 1)
        return self.jours_ouvres[k] if k < len(self.jours_ouvres) else None

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

        Filtres, tous DURS :
          - le chirurgien ne change jamais (votre règle) ;
          - on n'opère pas avant le DÉLAI MINIMUM fixé par le praticien ;
          - on n'opère pas plus de 6 mois après la consultation
            (`jour_max`), quelle que soit la fin de l'horizon.

        `fenetre` permet de forcer un autre délai minimum, ce qui ne sert
        qu'aux analyses de sensibilité.
        """
        p = self.patients[patient_id]
        jmin = p.jour_demande + (p.fenetre_jours if fenetre is None else fenetre)
        jmax = self.jour_max(patient_id)
        return [vid for vid in self.vacations_du_medecin.get(p.med_id, ())
                if jmin <= self.vacations[vid].jour <= jmax]


# ---------------------------------------------------------------------------
# 3. La solution globale : ce que manipule le tabou n°1
# ---------------------------------------------------------------------------


class Solution:
    """Affectation patient -> vacation, avec ÉVALUATION INCRÉMENTALE.

    Une métaheuristique évalue des centaines de milliers de mouvements ;
    recalculer le coût à chaque fois coûterait O(patients + jours). On
    maintient donc en permanence, à chaque `affecter` :

      nb, somme_durees, somme_marges(2)[v]   -> charge de chaque vacation
      places_jour, lits_jour[j]              -> profils journaliers
      *_urg*                                 -> part des URGENTS (réserves)
      s_places, s_places2, s_lits, s_lits2   -> Σx, Σx² des écarts à la cible
                                                sur les jours MESURÉS
      s_creux_u, s_tvo_u                     -> temps perdu / TVO des
                                                vacations UTILISÉES (fenêtre)
      s_exces_places, s_exces_lits, s_dep    -> violations des contraintes dures
      s_delai, n_places                      -> délai des patients placés
      s_taux, s_taux2                        -> variance du remplissage par
                                                praticien (indicateur)

    Toutes les fonctions coût de la section 6 sont donc en O(1).
    `controle_coherence()` recalcule tout de zéro et compare (tests).

    PATIENTS SANS VACATION. Deux états :
      `sans_date`    : pas encore vu en consultation (ou retiré le temps d'un
                       mouvement par une métaheuristique) ;
      `hors_horizon` : vu, mais aucune date ne tenait.
    Un patient À PLANIFIER (jour_demande <= jour_courant) sans vacation est
    pénalisé dans le coût, quel que soit son état : déplanifier n'est jamais
    gratuit. `jour_courant=None` (défaut) = tous les patients de l'instance
    sont à planifier (cas hors ligne : un lot de patients connus).
    """

    __slots__ = ("inst", "affectation", "nb", "somme_durees", "somme_marges",
                 "somme_marges2", "places_jour", "lits_jour",
                 "nb_urg", "urg_durees", "urg_marges", "urg_marges2",
                 "places_urg_jour", "lits_urg_jour",
                 "s_places", "s_places2", "s_lits", "s_lits2",
                 "s_creux_u", "s_tvo_u", "s_exces_places", "s_exces_lits", "s_dep",
                 "s_delai", "n_places", "sans_date", "hors_horizon",
                 "s_taux", "s_taux2", "jour_courant", "_jd",
                 "t0", "t1", "_mes_places", "_mes_lits", "_n_mes_places", "_n_mes_lits",
                 "_jours_places", "_n_jours_lits")

    def __init__(self, inst: Instance, jour_courant: int | None = None):
        self.inst = inst
        self.affectation: dict[int, int | None] = {pid: None for pid in inst.patients}
        V = inst.vacations
        self.nb = {vid: 0 for vid in V}
        self.somme_durees = {vid: 0 for vid in V}
        self.somme_marges = {vid: 0 for vid in V}
        self.somme_marges2 = {vid: 0 for vid in V}
        self.nb_urg = {vid: 0 for vid in V}
        self.urg_durees = {vid: 0 for vid in V}
        self.urg_marges = {vid: 0 for vid in V}
        self.urg_marges2 = {vid: 0 for vid in V}
        self.places_jour = [0] * inst.nb_jours
        self.lits_jour = [0] * inst.nb_jours
        self.places_urg_jour = [0] * inst.nb_jours
        self.lits_urg_jour = [0] * inst.nb_jours
        self.s_taux = {m: 0.0 for m in inst.vacations_du_medecin}
        self.s_taux2 = {m: 0.0 for m in inst.vacations_du_medecin}
        self.s_delai = 0.0
        self.n_places = 0
        self.sans_date: set[int] = set(inst.patients)
        self.hors_horizon: set[int] = set()
        self.jour_courant = jour_courant
        self._jd = sorted(p.jour_demande for p in inst.patients.values())
        self._jours_places = inst.jours_ouvres or list(range(inst.nb_jours))
        self._n_jours_lits = inst.nb_jours
        self.definir_fenetre(0, inst.nb_jours - 1)

    # -- fenêtre de mesure ---------------------------------------------------

    def definir_fenetre(self, t0: int, t1: int) -> None:
        """Jours [t0, t1] sur lesquels on MESURE lissage et remplissage (les
        contraintes dures, elles, valent sur tout l'horizon). Recalcule les
        agrégats : O(jours + vacations), à n'appeler qu'au changement de
        fenêtre (ex. chaque jour d'une simulation en ligne)."""
        inst = self.inst
        self.t0, self.t1 = t0, t1
        ouvres = set(inst.jours_ouvres)
        dans = [t0 <= j <= t1 for j in range(inst.nb_jours)]
        self._mes_places = [dans[j] and (inst.cible_places is not None or j in ouvres)
                            for j in range(inst.nb_jours)]
        self._mes_lits = [dans[j] and (inst.cible_lits is not None or j in ouvres)
                          for j in range(inst.nb_jours)]
        self._n_mes_places = sum(self._mes_places)
        self._n_mes_lits = sum(self._mes_lits)
        self._recalculer()

    def _recalculer(self) -> None:
        inst = self.inst
        self.s_places = self.s_places2 = 0.0
        self.s_lits = self.s_lits2 = 0.0
        self.s_exces_places = self.s_exces_lits = 0
        for j in range(inst.nb_jours):
            if self._mes_places[j]:
                x = self._x_places(j)
                self.s_places += x
                self.s_places2 += x * x
            if self._mes_lits[j]:
                x = self._x_lits(j)
                self.s_lits += x
                self.s_lits2 += x * x
            self.s_exces_places += self._exces_places(j)
            self.s_exces_lits += self._exces_lits(j)
        self.s_creux_u = self.s_tvo_u = self.s_dep = 0
        for vid in inst.vacations:
            dep, cr, tv = self._contrib_vac(vid)
            self.s_dep += dep
            self.s_creux_u += cr
            self.s_tvo_u += tv

    # -- contributions élémentaires -------------------------------------------

    def _x_places(self, j: int) -> float:
        c = self.inst.cible_places
        return self.places_jour[j] - (c[j] if c is not None else 0.0)

    def _x_lits(self, j: int) -> float:
        c = self.inst.cible_lits
        return self.lits_jour[j] - (c[j] if c is not None else 0.0)

    def _exces_places(self, j: int) -> int:
        """UNE pénalité par jour : le pire des deux excès (capacité totale,
        ou capacité du programmé hors réserve)."""
        inst = self.inst
        n = self.places_jour[j]
        cap = inst.capacite_places_jour
        return max(0, n - cap, n - self.places_urg_jour[j] - (cap - inst.reserve_places))

    def _exces_lits(self, j: int) -> int:
        inst = self.inst
        n = self.lits_jour[j]
        cap = inst.cap_lits(j)
        return max(0, n - cap, n - self.lits_urg_jour[j] - (cap - inst.reserve_lits))

    def _contrib_vac(self, vid: int) -> tuple[int, int, int]:
        """(violation en minutes, creux si utilisée et dans la fenêtre, TVO
        programmable si utilisée et dans la fenêtre)."""
        v = self.inst.vacations[vid]
        dep = self.depassement(vid)
        if v.urgence:
            return dep, 0, 0
        dep = max(dep, self.depassement_programme(vid))
        if self.nb[vid] == 0 or not (self.t0 <= v.jour <= self.t1):
            return dep, 0, 0
        cap = self.inst.capacite_programme(vid)
        return dep, max(0, cap - self.charge(vid)), cap

    def _bouger_jour(self, j: int, d: int, urgent: bool, lits: bool) -> None:
        """Ajoute d (±1) patient présent le jour/la nuit j, en tenant les
        agrégats à jour par différence avant/après."""
        if lits:
            mes, xf, ef = self._mes_lits[j], self._x_lits, self._exces_lits
        else:
            mes, xf, ef = self._mes_places[j], self._x_places, self._exces_places
        x0, e0 = (xf(j) if mes else 0.0), ef(j)
        if lits:
            self.lits_jour[j] += d
            if urgent:
                self.lits_urg_jour[j] += d
        else:
            self.places_jour[j] += d
            if urgent:
                self.places_urg_jour[j] += d
        x1, e1 = (xf(j) if mes else 0.0), ef(j)
        if lits:
            self.s_lits += x1 - x0
            self.s_lits2 += x1 * x1 - x0 * x0
            self.s_exces_lits += e1 - e0
        else:
            self.s_places += x1 - x0
            self.s_places2 += x1 * x1 - x0 * x0
            self.s_exces_places += e1 - e0

    # -- copie ---------------------------------------------------------------

    def copie(self) -> "Solution":
        """Copie de travail. Rapide : que des dict/list de petits nombres."""
        s = Solution.__new__(Solution)
        for a in Solution.__slots__:
            val = getattr(self, a)
            if isinstance(val, (dict, list, set)):
                val = type(val)(val)
            setattr(s, a, val)
        s.inst = self.inst
        s._mes_places, s._mes_lits = self._mes_places, self._mes_lits   # lecture seule
        s._jours_places = self._jours_places
        return s

    # -- lecture -----------------------------------------------------------

    def _marge_cumulee(self, s1: int, s2: int) -> float:
        return sqrt(s2) if self.inst.cumul_marges == "quadratique" else s1

    def charge(self, vid: int) -> int:
        """Minutes engagées dans la vacation : TROS + TIS + marge de risque
        cumulée (cf. `Instance.cumul_marges`). Le TIS est compté plein : la
        vacation tient donc dans N'IMPORTE QUEL ordre de passage."""
        n = self.nb[vid]
        if n == 0:
            return 0
        m = self._marge_cumulee(self.somme_marges[vid], self.somme_marges2[vid])
        return int(round(self.somme_durees[vid] + m)) + self.inst.tis * (n - 1)

    def charge_programme(self, vid: int) -> int:
        """Charge des seuls patients PROGRAMMÉS de la vacation."""
        n = self.nb[vid] - self.nb_urg[vid]
        if n <= 0:
            return 0
        m = self._marge_cumulee(self.somme_marges[vid] - self.urg_marges[vid],
                                self.somme_marges2[vid] - self.urg_marges2[vid])
        return (int(round(self.somme_durees[vid] - self.urg_durees[vid] + m))
                + self.inst.tis * (n - 1))

    def depassement_programme(self, vid: int) -> int:
        """Débordement du programmé dans le tampon réservé aux urgences."""
        return max(0, self.charge_programme(vid) - self.inst.capacite_programme(vid))

    def creux(self, vid: int) -> int:
        return max(0, self.inst.capacite_utile(vid) - self.charge(vid))

    def depassement(self, vid: int) -> int:
        return max(0, self.charge(vid) - self.inst.capacite_utile(vid))

    def taux_remplissage(self, vid: int) -> float:
        """Charge / capacité du PROGRAMMÉ (TVO − tampon). Sans tampon :
        charge / TVO."""
        cap = self.inst.capacite_programme(vid) or self.inst.vacations[vid].tvo
        return self.charge(vid) / cap if cap else 0.0

    def patients_de(self, vid: int) -> list[int]:
        return [pid for pid, v in self.affectation.items() if v == vid]

    def patients_du_jour(self, j: int) -> list[int]:
        vids = set(self.inst.vacations_du_jour.get(j, ()))
        return [pid for pid, v in self.affectation.items() if v in vids]

    def n_a_planifier(self) -> int:
        """Patients déjà vus en consultation (jour_demande <= jour_courant)."""
        if self.jour_courant is None:
            return len(self._jd)
        return bisect_right(self._jd, self.jour_courant)

    def nb_non_places(self) -> int:
        """Patients à planifier qui n'ont pas de vacation."""
        return max(0, self.n_a_planifier() - self.n_places)

    # -- modification ------------------------------------------------------

    def _appliquer_patient(self, pid: int, vid: int | None, signe: int) -> None:
        """Ajoute (+1) ou retire (-1) la contribution d'un patient posé en vid."""
        if vid is None:
            return
        inst = self.inst
        p = inst.patients[pid]
        v = inst.vacations[vid]
        urgent = p.urgence != PROGRAMME
        m = p.marge_perso

        self.nb[vid] += signe
        self.somme_durees[vid] += signe * p.duree_op
        self.somme_marges[vid] += signe * m
        self.somme_marges2[vid] += signe * m * m
        if urgent:
            self.nb_urg[vid] += signe
            self.urg_durees[vid] += signe * p.duree_op
            self.urg_marges[vid] += signe * m
            self.urg_marges2[vid] += signe * m * m
        self.s_delai += signe * p.priorite * (v.jour - p.jour_demande)
        self.n_places += signe

        if p.ambulatoire:
            self._bouger_jour(v.jour, signe, urgent, lits=False)
        else:
            for j in inst.nuits_occupees(pid, v.jour):
                self._bouger_jour(j, signe, urgent, lits=True)

    def affecter(self, pid: int, vid: int | None) -> None:
        """Pose un patient dans une vacation. `None` = le retire du planning.

        Retirer un patient n'est PAS une opération du processus réel : une
        date annoncée ne bouge plus. C'est un mouvement interne aux
        métaheuristiques hors ligne (et à la replanification exceptionnelle).
        """
        ancien = self.affectation[pid]
        if ancien == vid:
            return
        inst = self.inst
        med = inst.patients[pid].med_id
        touchees = [v for v in (ancien, vid) if v is not None]

        # par différence : contributions des vacations touchées avant/après
        avant_taux = [(v, self.taux_remplissage(v)) for v in touchees
                      if not inst.vacations[v].urgence]
        for v in touchees:
            dep, cr, tv = self._contrib_vac(v)
            self.s_dep -= dep
            self.s_creux_u -= cr
            self.s_tvo_u -= tv

        self._appliquer_patient(pid, ancien, -1)
        self._appliquer_patient(pid, vid, +1)
        self.affectation[pid] = vid

        for v in touchees:
            dep, cr, tv = self._contrib_vac(v)
            self.s_dep += dep
            self.s_creux_u += cr
            self.s_tvo_u += tv
        for v, t0 in avant_taux:
            t1 = self.taux_remplissage(v)
            self.s_taux[med] += t1 - t0
            self.s_taux2[med] += t1 * t1 - t0 * t0

        if vid is None:
            self.sans_date.add(pid)
        else:
            self.sans_date.discard(pid)
            self.hors_horizon.discard(pid)

    def viole(self, pid: int, vid: int) -> bool:
        """Le patient étant POSÉ en vid, une contrainte dure est-elle violée ?

        Pour tous : jour hors de [jour_min, jour_max], dépassement du TVO,
        lits > capacité de la nuit, admissions ambulatoires > capacité.
        En plus pour un PROGRAMMÉ : créneau URGENCES, tampon et réserves de
        lits et de places interdits.
        """
        inst, p = self.inst, self.inst.patients[pid]
        v = inst.vacations[vid]
        if not p.jour_min <= v.jour <= inst.jour_max(pid):
            return True
        if self.depassement(vid) > 0:
            return True
        urgent = p.urgence != PROGRAMME
        if not urgent and (v.urgence or self.depassement_programme(vid) > 0):
            return True
        if p.ambulatoire:
            j = v.jour
            cap = inst.capacite_places_jour
            n = self.places_jour[j]
            return n > cap or (not urgent and n - self.places_urg_jour[j]
                               > cap - inst.reserve_places)
        for k in inst.nuits_occupees(pid, v.jour):
            cap = inst.cap_lits(k)
            n = self.lits_jour[k]
            if n > cap or (not urgent and n - self.lits_urg_jour[k] > cap - inst.reserve_lits):
                return True
        return False

    # -- indicateurs de dispersion (O(1)) -------------------------------------

    def variance_remplissage_medecin(self, med_id: int) -> float:
        """Variance des taux de remplissage des vacations de ce praticien
        (vides comprises, taux 0). INDICATEUR de régularité : elle ne mesure
        PAS le remplissage du bloc (tout à 40 % donne 0). Le coût utilise
        `part_creux_utilisees`."""
        n = self.inst.nb_vacations_medecin.get(med_id, 0)
        if n == 0:
            return 0.0
        s, s2 = self.s_taux[med_id], self.s_taux2[med_id]
        return max(0.0, s2 / n - (s / n) ** 2)

    def variance_remplissage(self) -> float:
        n = max(1, len(self.inst.vacations_du_medecin))
        return sum(self.variance_remplissage_medecin(m)
                   for m in self.inst.vacations_du_medecin) / n

    def _dispersion(self, s: float, s2: float, n: int, cible: bool) -> float:
        if n == 0:
            return 0.0
        if cible:                       # écart quadratique moyen à la cible
            return s2 / n
        return max(0.0, s2 / n - (s / n) ** 2)   # variance

    def variance_places(self) -> float:
        """Admissions ambulatoires, sur les jours MESURÉS (fenêtre ∩ jours
        ouvrés, ou toute la fenêtre si `cible_places`) : variance, ou écart
        quadratique moyen à la cible si `Instance.cible_places`."""
        return self._dispersion(self.s_places, self.s_places2, self._n_mes_places,
                                self.inst.cible_places is not None)

    def variance_lits(self) -> float:
        """Lits occupés, sur les nuits MESURÉES : sans cible, seulement les
        nuits qui suivent un jour de bloc (on ne pousse plus à remplir les
        week-ends) ; avec `Instance.cible_lits`, écart quadratique moyen à la
        cible sur toute la fenêtre."""
        return self._dispersion(self.s_lits, self.s_lits2, self._n_mes_lits,
                                self.inst.cible_lits is not None)

    def part_creux_utilisees(self) -> float:
        """Temps de bloc PERDU dans les vacations utilisées (au moins un
        patient) de la fenêtre, rapporté à leur TVO programmable. Une
        vacation vide n'est pas comptée : elle peut être rendue au bloc."""
        return self.s_creux_u / self.s_tvo_u if self.s_tvo_u else 0.0

    def moyenne_places(self) -> float:
        n = self._n_mes_places
        return sum(self.places_jour[j] for j in range(self.inst.nb_jours)
                   if self._mes_places[j]) / n if n else 0.0

    def moyenne_lits(self) -> float:
        n = self._n_mes_lits
        return sum(self.lits_jour[j] for j in range(self.inst.nb_jours)
                   if self._mes_lits[j]) / n if n else 0.0

    def creux_total(self) -> int:
        """Creux des vacations PROGRAMMÉES, vides comprises (un créneau
        URGENCES vide n'est pas du temps perdu : c'est une réserve)."""
        return sum(self.creux(vid) for vid, v in self.inst.vacations.items()
                   if not v.urgence)

    def depassement_total(self) -> int:
        return sum(self.depassement(vid) for vid in self.inst.vacations)

    def tvo_total(self) -> int:
        return sum(v.tvo for v in self.inst.vacations.values() if not v.urgence)

    def surcharge_lits(self) -> int:
        return sum(max(0, self.lits_jour[j] - self.inst.cap_lits(j))
                   for j in range(self.inst.nb_jours))

    def surcharge_places(self) -> int:
        cap = self.inst.capacite_places_jour
        return sum(max(0, self.places_jour[j] - cap) for j in self._jours_places)

    def delai_total(self) -> float:
        """Σ priorité × (jour opéré − jour de consultation), sur les patients
        PLACÉS. Définition UNIQUE du délai (indicateurs ET coût). Les patients
        non placés ne sont pas « comptés au dernier jour » : ils ont leur
        propre pénalité (`nb_non_places`)."""
        return self.s_delai

    def delai_moyen(self) -> float:
        return self.s_delai / self.n_places if self.n_places else 0.0

    # -- contrôle ------------------------------------------------------------

    def controle_coherence(self) -> None:
        """Recalcule tout de zéro et compare aux agrégats incrémentaux."""
        ref = Solution(self.inst, self.jour_courant)
        ref.definir_fenetre(self.t0, self.t1)
        for pid, vid in self.affectation.items():
            if vid is not None:
                ref.affecter(pid, vid)
        for a in ("nb", "somme_durees", "somme_marges", "somme_marges2", "nb_urg",
                  "urg_durees", "urg_marges", "urg_marges2", "places_jour",
                  "lits_jour", "places_urg_jour", "lits_urg_jour",
                  "s_exces_places", "s_exces_lits", "s_dep", "s_creux_u",
                  "s_tvo_u", "n_places"):
            assert getattr(ref, a) == getattr(self, a), a
        for a in ("s_places", "s_places2", "s_lits", "s_lits2", "s_delai"):
            assert abs(getattr(ref, a) - getattr(self, a)) < 1e-6, a
        for m in self.s_taux:
            assert abs(ref.s_taux[m] - self.s_taux[m]) < 1e-6, ("s_taux", m)

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
            if v.urgence and not p.est_urgent:
                c.append(Conflit("U1", f"patient programmé {pid} dans le créneau "
                                       f"URGENCES {vid}", pid, vid))
            if v.med_id != p.med_id and not (v.urgence and p.est_urgent):
                c.append(Conflit("G1", f"patient {pid} chez le chirurgien {v.med_id} "
                                       f"au lieu de {p.med_id}", pid, vid))
            if v.jour < p.jour_min:
                c.append(Conflit("G5", f"patient {pid} opéré J{v.jour} avant son délai "
                                       f"minimum (J{p.jour_min})", pid, vid))
            if v.jour > inst.jour_max(pid):
                c.append(Conflit("G7", f"patient {pid} opéré J{v.jour} après son délai "
                                       f"maximum (J{inst.jour_max(pid)})", pid, vid))
        for vid, v in inst.vacations.items():
            d = self.depassement(vid)
            if d > 0:
                c.append(Conflit("G2", f"vacation {vid} surchargée de {d} min "
                                       f"(marges de risque incluses)", None, vid))
            d = self.depassement_programme(vid)
            if d > 0 and not v.urgence:
                c.append(Conflit("U3", f"vacation {vid} : le programmé déborde de "
                                       f"{d} min dans le tampon urgences", None, vid))
        for j in range(inst.nb_jours):
            n, cap = self.lits_jour[j], inst.cap_lits(j)
            if n > cap:
                c.append(Conflit("G3", f"J{j} ({inst.date_du_jour(j)}) : {n} lits pour {cap}"))
            n = self.lits_jour[j] - self.lits_urg_jour[j]
            if n > cap - inst.reserve_lits:
                c.append(Conflit("U4", f"J{j} : le programmé occupe {n} lits "
                                       f"pour {cap - inst.reserve_lits} hors réserve"))
        cap = inst.capacite_places_jour
        for j in self._jours_places:
            n = self.places_jour[j]
            if n > cap:
                c.append(Conflit("G4", f"J{j} ({inst.date_du_jour(j)}) : {n} admissions "
                                       f"ambulatoires pour {cap} possibles"))
            n = self.places_jour[j] - self.places_urg_jour[j]
            if n > cap - inst.reserve_places:
                c.append(Conflit("U5", f"J{j} : {n} ambulatoires programmés "
                                       f"pour {cap - inst.reserve_places} hors réserve"))
        return c

    # -- URGENCES --------------------------------------------------------------
    #
    # Règle absolue : un patient qui a une date ne la change JAMAIS (et dans
    # ce modèle, tout patient a sa date dès sa première consultation). Une
    # urgence ne prend donc que de la capacité LIBRE ou RÉSERVÉE : tampon des
    # vacations, réserves de lits et de places, créneaux URGENCES.
    # Les HEURES peuvent bouger : c'est l'AG journalier qui refait l'ordre de
    # la journée. Le niveau global garantit seulement que la charge tient dans
    # la vacation — dans n'importe quel ordre, puisque `charge` compte le TIS
    # plein.

    def ajouter_urgent(self, p: Patient) -> int:
        """Enregistre un patient urgent arrivé en cours de route (dans
        l'instance ET dans la solution). Renvoie son id."""
        if p.urgence == PROGRAMME:
            raise ValueError("ajouter_urgent : patient non urgent")
        if p.id in self.inst.patients:
            raise ValueError(f"id {p.id} déjà utilisé")
        self.inst.ajouter_patient(p)
        self.affectation[p.id] = None
        self.sans_date.add(p.id)
        insort(self._jd, p.jour_demande)
        return p.id

    def _essayer(self, pid: int, vid: int) -> bool:
        """Pose pid en vid ; le garde si aucune contrainte dure n'est violée,
        sinon le retire."""
        self.affecter(pid, vid)
        if self.viole(pid, vid):
            self.affecter(pid, None)
            return False
        return True

    def _echec_urgence(self, pid: int) -> None:
        """Aucune place : le patient est compté `hors_horizon` (vu, sans date)."""
        self.sans_date.discard(pid)
        self.hors_horizon.add(pid)

    def placer_urgence_jour(self, pid: int) -> int | None:
        """URGENCE DU JOUR MÊME, arrivée le jour `jour_demande`.

        Le jour même, puis le jour OUVRÉ suivant ; pour chaque jour :
          1. un créneau URGENCES du jour ;
          2. une vacation de SON chirurgien ce jour-là (capacité libre + tampon).
        Renvoie la vacation, ou None si même le jour suivant est impossible
        (patient compté hors horizon : c'est un échec de dimensionnement des
        réserves). Aucun patient déjà daté n'est déplacé.
        """
        inst, p = self.inst, self.inst.patients[pid]
        if p.urgence != URGENCE_JOUR:
            raise ValueError(f"patient {pid} n'est pas une urgence du jour")
        if self.affectation[pid] is not None:
            raise ValueError(f"patient {pid} déjà placé")
        j = p.jour_demande
        jours = [j] if j in inst.vacations_du_jour else []
        suivant = inst.jour_ouvre_suivant(j)
        if suivant is not None:
            jours.append(suivant)
        for jour in jours:
            chir = [v for v in inst.vacations_du_jour.get(jour, ())
                    if not inst.vacations[v].urgence
                    and inst.vacations[v].med_id == p.med_id]
            for vid in list(inst.vacations_urgence_du_jour.get(jour, ())) + chir:
                if self._essayer(pid, vid):
                    return vid
        self._echec_urgence(pid)
        return None

    def placer_semi_urgence(self, pid: int) -> int | None:
        """SEMI-URGENCE : entre `jour_min` et `jour_max`, au plus tôt :
          1. vacations de SON chirurgien (capacité libre + tampon) ;
          2. créneaux URGENCES, en dernier recours (ils sont faits pour les
             urgences du jour même).
        Renvoie la vacation, ou None si rien ne tient dans le délai.
        """
        inst, p = self.inst, self.inst.patients[pid]
        if p.urgence != SEMI_URGENCE:
            raise ValueError(f"patient {pid} n'est pas une semi-urgence")
        if self.affectation[pid] is not None:
            raise ValueError(f"patient {pid} déjà placé")
        jmax = min(inst.jour_max(pid), inst.nb_jours - 1)
        for vid in inst.vacations_possibles(pid):     # triées par jour, <= jour_max
            if self._essayer(pid, vid):
                return vid
        for jour in range(p.jour_min, jmax + 1):
            for vid in inst.vacations_urgence_du_jour.get(jour, ()):
                if self._essayer(pid, vid):
                    return vid
        self._echec_urgence(pid)
        return None

    # -- tableau de bord ---------------------------------------------------

    def indicateurs(self) -> dict:
        inst = self.inst
        tvo = self.tvo_total()
        prog = [vid for vid, v in inst.vacations.items() if not v.urgence]
        ouvertes = [vid for vid in prog if self.nb[vid] > 0]
        taux = [self.taux_remplissage(vid) for vid in ouvertes]
        lits = [self.lits_jour[j] for j in range(inst.nb_jours) if self._mes_lits[j]]
        places = [self.places_jour[j] for j in range(inst.nb_jours) if self._mes_places[j]]
        return {
            "patients_programmes": self.n_places,          # = placés
            "patients_sans_date": self.nb_non_places(),     # à planifier, sans vacation
            "patients_hors_horizon": len(self.hors_horizon),
            "vacations_ouvertes": f"{len(ouvertes)}/{len(prog)}",   # = utilisées
            "urgents_places": sum(1 for p, v in self.affectation.items()
                                  if v is not None and inst.patients[p].est_urgent),
            "urgents_sans_date": sum(1 for p in self.hors_horizon
                                     if inst.patients[p].est_urgent),
            "taux_remplissage_moyen": sum(taux) / len(taux) if taux else 0.0,
            "part_creux_utilisees": self.part_creux_utilisees(),
            "remplissage_ecart_type": self.variance_remplissage() ** 0.5,
            "creux_total_h": self.creux_total() / 60,
            "part_creux": self.creux_total() / tvo if tvo else 0.0,
            "lits_moyen": self.moyenne_lits(),
            "lits_ecart_type": self.variance_lits() ** 0.5,
            "lits_pic": max(self.lits_jour, default=0),
            "lits_creux": min(lits, default=0),
            "places_moyen": self.moyenne_places(),
            "places_ecart_type": self.variance_places() ** 0.5,
            "places_pic": max(places, default=0),
            "surcharge_lits_j": sum(1 for j in range(inst.nb_jours)
                                    if self.lits_jour[j] > inst.cap_lits(j)),
            "surcharge_places_j": sum(1 for j in self._jours_places
                                      if self.places_jour[j] > inst.capacite_places_jour),
            "depassement_vacations_h": self.depassement_total() / 60,
            "delai_moyen_j": self.delai_moyen(),
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
    arrivee: int = -1   # prise de la place / du lit ce jour-là (cf. deriver_creneaux)


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
    # Lits : l'occupation d'un lit dépend des patients opérés LES AUTRES
    # jours. Si `deriver_creneaux` reçoit la solution globale, pic_lits =
    # lits occupés la nuit du jour (tous patients) ; sinon = nombre
    # d'hospitalisés OPÉRÉS ce jour-là (entrées du bloc), faute de mieux.
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
                     sequences: dict[int, list[int]],
                     sol: "Solution | None" = None) -> PlanningJour:
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
      2. chaque ambulatoire occupe une place de son ARRIVÉE,
         max(ouverture_uca, début_op − avance_ambu), jusqu'à
         `fin_op + surveillance` ; un hospitalisé occupe un lit ;
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
            arr = (max(inst.ouverture_uca, debut - inst.avance_ambu)
                   if p.ambulatoire else fin)
            pj.creneaux[pid] = Creneau(pid, vid, v.bloc_id, debut, fin, lib,
                                       ambulatoire=p.ambulatoire, arrivee=arr)
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
    ouverture = min(ouverture, inst.ouverture_uca)    # les ambulatoires arrivent avant

    ambulatoires = [(c.arrivee, c.lib_place) for c in pj.creneaux.values() if c.ambulatoire]
    profil_places = profil_cumulatif(ambulatoires)
    pj.pic_places = pic(profil_places)
    pj.places_moyenne, pj.places_variance = moments_temporels(
        profil_places, ouverture, inst.fermeture_uca)

    if sol is not None:
        pj.pic_lits = sol.lits_jour[jour]
    else:
        pj.pic_lits = sum(1 for c in pj.creneaux.values() if not c.ambulatoire)

    for ambu in (True, False):
        items = [(c.patient_id, c.arrivee, c.lib_place)
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


# ---------------------------------------------------------------------------
# 6. Fonction coût du niveau global
# ---------------------------------------------------------------------------
#
#   C = α·(w_places·P̃ + w_lits·L̃) + w_rempl·R̃ + w_délai·D̃
#       + PENALITE · violations + PENALITE_NON_PLACE · non_placés
#
# Termes BRUTS (tous en O(1), lus dans les agrégats incrémentaux de Solution) :
#   P = variance_places() / cap_places²   admissions ambulatoires, jours mesurés
#   L = variance_lits()   / cap_lits²     lits, nuits mesurées
#   R = part_creux_utilisees()            temps perdu / TVO, vacations utilisées
#   D = délai moyen des placés / delai_max_programme
# Termes NORMALISÉS : X̃ = X / X_ref, où X_ref est la valeur du terme sur une
# solution de référence (`CoutTotal.calibrer`, ex. la gloutonne). Sans
# calibration, les termes n'ont PAS le même ordre de grandeur (sur 2022 : le
# délai pèse 10 à 100 fois plus que les lits) et les poids n'ont pas de sens.
#
# Contraintes dures : PENALITE par unité de violation (minutes de
# dépassement, lits ou places en trop, UNE fois par jour même avec réserves)
# -> un individu réalisable a un coût < PENALITE.
# Patients à planifier sans vacation : PENALITE_NON_PLACE chacun, qui domine
# les termes normalisés (~1) -> déplanifier ne fait jamais baisser le coût.

PENALITE = 1e6
PENALITE_NON_PLACE = 1e3


def _variance(xs: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    m = sum(xs) / n
    return sum((x - m) ** 2 for x in xs) / n


def terme_places(sol: Solution) -> float:
    return sol.variance_places() / sol.inst.capacite_places_jour ** 2


def terme_lits(sol: Solution) -> float:
    return sol.variance_lits() / sol.inst.capacite_lits ** 2


def terme_remplissage(sol: Solution) -> float:
    return sol.part_creux_utilisees()


def terme_delai(sol: Solution) -> float:
    return sol.delai_moyen() / sol.inst.delai_max_programme


def violations(sol: Solution) -> int:
    """Minutes de dépassement + lits en trop + admissions en trop."""
    return sol.s_dep + sol.s_exces_lits + sol.s_exces_places


# -- fonctions par thème (interface inchangée, désormais O(1)) ----------------

@dataclass
class PoidsRessources:
    places: float = 1.0
    lits: float = 1.0


def cout_places(sol: Solution) -> float:
    return terme_places(sol) + PENALITE * sol.s_exces_places


def cout_lits(sol: Solution) -> float:
    return terme_lits(sol) + PENALITE * sol.s_exces_lits


def cout_ressources(sol: Solution, w: PoidsRessources | None = None) -> float:
    w = w or PoidsRessources()
    return w.places * cout_places(sol) + w.lits * cout_lits(sol)


@dataclass
class PoidsPlanning:
    remplissage: float = 1.0
    delai: float = 1.0


def cout_remplissage(sol: Solution) -> float:
    return terme_remplissage(sol) + PENALITE * sol.s_dep


def cout_delai(sol: Solution) -> float:
    return terme_delai(sol) + PENALITE_NON_PLACE * sol.nb_non_places()


def cout_planning(sol: Solution, w: PoidsPlanning | None = None) -> float:
    w = w or PoidsPlanning()
    return w.remplissage * cout_remplissage(sol) + w.delai * cout_delai(sol)


# -- coût total -------------------------------------------------------------------

@dataclass
class CoutTotal:
    """Appelable : `cout(sol) -> float`, en O(1).

    Usage :
        cout = CoutTotal(PoidsRessources(1, 2), PoidsPlanning(1, 0.5))
        cout.calibrer(sol_gloutonne)     # normalisation (fortement conseillé)
        c = cout(sol)
    `alpha` = poids relatif ressources / planning. Les poids ne sont
    interprétables qu'APRÈS `calibrer`.
    """
    w_ressources: PoidsRessources | None = None
    w_planning: PoidsPlanning | None = None
    alpha: float = 1.0
    ref: dict | None = None      # valeurs de référence des 4 termes

    def calibrer(self, sol: Solution) -> "CoutTotal":
        t = self.termes(sol)
        self.ref = {k: (t[k] if t[k] > 1e-12 else 1.0)
                    for k in ("places", "lits", "remplissage", "delai")}
        return self

    def termes(self, sol: Solution) -> dict:
        """Termes BRUTS (non normalisés) + contraintes."""
        return {"places": terme_places(sol), "lits": terme_lits(sol),
                "remplissage": terme_remplissage(sol), "delai": terme_delai(sol),
                "violations": violations(sol), "non_places": sol.nb_non_places()}

    def termes_normalises(self, sol: Solution) -> dict:
        t = self.termes(sol)
        ref = self.ref or {}
        return {k: t[k] / ref.get(k, 1.0) for k in ("places", "lits", "remplissage", "delai")}

    def __call__(self, sol: Solution) -> float:
        wr = self.w_ressources or PoidsRessources()
        wp = self.w_planning or PoidsPlanning()
        n = self.termes_normalises(sol)
        return (self.alpha * (wr.places * n["places"] + wr.lits * n["lits"])
                + wp.remplissage * n["remplissage"] + wp.delai * n["delai"]
                + PENALITE * violations(sol)
                + PENALITE_NON_PLACE * sol.nb_non_places())


# ---------------------------------------------------------------------------
# 7. Créneaux URGENCES de la grille
# ---------------------------------------------------------------------------

def ajouter_vacations_urgence(inst: Instance, feries=frozenset()) -> int:
    """Ajoute à l'instance les créneaux de type « urgence » de la grille
    (grille.py : les 6 « URGENCES » 13h30-15h30 et « TDO/BS (urg) » 8h-10h),
    comme vacations `urgence=True`, `med_id=MED_URGENCE`, puis réindexe.
    À appeler AVANT de créer la `Solution`. Renvoie le nombre ajouté.
    Import local : grille.py importe modele.py, l'inverse doit rester paresseux.
    """
    from grille import creneaux_du

    vid = max(inst.vacations, default=-1) + 1
    n = 0
    for j in range(inst.nb_jours):
        d = inst.date_du_jour(j)
        if d in feries:
            continue
        for c in creneaux_du(d, types=("urgence",)):
            inst.ajouter_vacation(Vacation(vid, bloc_id=c.salle, med_id=MED_URGENCE,
                                           jour=j, debut=c.debut, fin=c.fin,
                                           etiquette=f"{c.code} S{c.salle} {d:%d/%m}",
                                           urgence=True))
            vid += 1
            n += 1
    inst.indexer()
    return n
