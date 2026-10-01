r"""
propositions.py — Le moteur de consultation : proposer deux dates au chirurgien.

LE PROCESSUS RÉEL
-----------------
    Le patient est vu en consultation
        -> on estime la durée de son acte et le chirurgien la valide
           (estimation.py)
        -> on lui propose DEUX dates
        -> le chirurgien en choisit une, devant le patient
        -> cette date ne bouge plus jamais.

Il n'y a pas de liste d'attente. Un patient sort de consultation avec une
date, ou avec un « pas de date avant tel mois, on vous reconvoque » — jamais
avec un « on vous rappellera ».

POURQUOI CE FICHIER NE CONTIENT PAS DE MÉTAHEURISTIQUE
-------------------------------------------------------
C'est le point le plus important à comprendre, et c'est un bon réflexe à
garder pour la suite : **avant de sortir une métaheuristique, compter la
taille de l'espace de recherche.**

Ici, les décisions déjà prises sont figées et on insère UN patient. L'espace
des solutions, c'est donc l'ensemble des vacations où ce patient peut aller :
celles de SON chirurgien (il n'en change jamais), situées après sa
consultation, et dans la FENÊTRE qu'il a demandée. Une poignée de vacations.
On les évalue toutes en quelques microsecondes et on trie.

Une énumération exhaustive est **exacte** — elle trouve l'optimum, pas une
approximation — et **instantanée**. Un tabou ferait moins bien et plus
lentement. Le tabou reste indispensable là où l'espace explose : l'ordre des
patients dans une journée (tabou local), et le calcul de l'étalon hors-ligne
où tout redevient déplaçable (tabou global). Voir l'en-tête de tabou.py.

CE QUI EST QUAND MÊME NON TRIVIAL ICI
--------------------------------------
Deux choses, et ce sont elles qui font la qualité du résultat :

  1. **Le critère de classement.** On ne classe pas les dates par « la salle
     est libre », on les classe par la DÉGRADATION qu'elles infligent au
     planning entier : la fonction objectif à deux termes — dispersion du
     remplissage des vacations du praticien, et variance de l'occupation des
     lits — évaluée en delta. Une date qui remplit bien une vacation mais crée
     un pic de lits trois jours plus tard sera mal classée, et c'est le
     comportement voulu.

  2. **Ce qu'on montre au chirurgien.** Un classement sans justification n'est
     pas utilisable en consultation : le praticien doit voir POURQUOI une date
     est meilleure, et ce qu'elle coûte. D'où les alertes et l'aperçu
     d'impact attachés à chaque proposition.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date

from modele import Instance, Patient, Solution
from tabou import Evaluateur, Poids

# ---------------------------------------------------------------------------
# 0. Les fenêtres de recherche
# ---------------------------------------------------------------------------

FENETRES: dict[str, int] = {
    "semaine": 7,
    "trois_semaines": 21,
    "mois_et_demi": 45,
}
"""Les trois horizons que le chirurgien peut demander, en jours.

C'est le paramètre qui a remplacé le terme de délai dans la fonction
objectif, et le remplacement est un gain net. Un poids de délai est une
abstraction : il faut le calibrer, le justifier, et il applique le même
arbitrage à tous les patients. Une fenêtre est une décision clinique, prise
patient par patient, par la personne qui sait si le cas peut attendre.

Le chirurgien annonce la fenêtre AVANT la recherche. On ne cherche que
dedans, et on lui rend les deux meilleures dates de cette fenêtre. S'il n'y
a rien, le report lui dit quelle fenêtre plus large aurait une place — on ne
l'élargit jamais à sa place.
"""

ORDRE_FENETRES = ["semaine", "trois_semaines", "mois_et_demi"]

JOURS_PROCHE = 7
"""Ce qu'on appelle « la semaine à venir », en jours après la consultation."""

SEUIL_DENSITE_PROCHE = 0.50
"""En dessous de ce taux de remplissage, la semaine à venir est dite CREUSE.

POURQUOI CETTE RÈGLE EXISTE
---------------------------
La fenêtre est un délai MINIMUM : « pas avant trois semaines » veut dire que
le patient n'est pas prêt, ou que le praticien veut du temps. Mais quand la
semaine qui vient est à moitié vide, refuser d'y placer qui que ce soit coûte
deux fois : on perd du temps de salle qui ne se rattrape jamais (une vacation
non utilisée est perdue, pas reportée), et on allonge les délais de tout le
monde en poussant la demande vers des semaines déjà plus chargées.

On lève donc le délai minimum — et seulement lui — lorsque les vacations du
praticien dans les sept jours à venir sont remplies à moins de
`SEUIL_DENSITE_PROCHE`. Le reste de la fenêtre est inchangé : les dates
au-delà du délai minimum restent candidates, on ne fait qu'ajouter la semaine
creuse à l'ensemble de recherche. Le chirurgien voit alors une date « au plus
tôt » très proche, avec une alerte qui dit pourquoi elle apparaît, et il
garde le dernier mot — s'il a levé la fenêtre pour une raison clinique, il
prend la seconde date.

C'est un seuil, donc un paramètre : il se balaye (`balayer_densite`), il ne
se choisit pas au jugé."""


def fenetre_uniforme(graine: int = 0):
    """Politique de SIMULATION : une chance sur trois pour chaque fenêtre.

    En exploitation, la fenêtre vient du chirurgien. En simulation il faut
    bien la fabriquer, et on la tire uniformément — pas parce que c'est la
    répartition réelle (on ne la connaît pas), mais parce que c'est
    l'hypothèse la plus neutre : elle n'est calée sur aucun fichier et le code
    tourne donc à l'identique sur n'importe quelle base de données.

    Le tirage dépend de la graine ET de l'identifiant du patient, pas de
    l'ordre dans lequel on les parcourt. Deux conséquences utiles : le
    résultat est reproductible, et la fenêtre d'un patient donné ne change pas
    si on ajoute ou retire d'autres patients de la base. Sans cela, comparer
    deux jeux de données ferait varier deux choses à la fois.
    """
    def _politique(patient) -> str:
        # La graine combine le numéro d'expérience et l'identifiant du patient
        # dans un seul entier : `random.Random` n'accepte pas de tuple.
        return ORDRE_FENETRES[random.Random(graine * 1_000_003
                                            + patient.id).randrange(3)]
    return _politique

# ---------------------------------------------------------------------------
# 1. Une proposition de date
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Proposition:
    """Une date proposée au chirurgien, avec de quoi la juger."""

    patient_id: int
    vacation_id: int
    jour: int
    date: date
    cout: float                       # dégradation de l'objectif global
    delai: int                        # jours entre la consultation et l'opération
    bloc_id: int
    remplissage_apres: float          # taux de la vacation après insertion
    marge_restante: int               # minutes encore disponibles, marges déduites
    lits_apres: int                   # lits occupés le jour J après insertion
    places_apres: int                 # admissions ambulatoires du jour après
    role: str = ""                    # "au plus tôt" ou "recommandée"
    alertes: list[str] = field(default_factory=list)

    def ligne(self) -> str:
        return (f"{self.date:%a %d/%m/%Y}  salle {self.bloc_id}  "
                f"dans {self.delai} jours  |  vacation remplie à "
                f"{self.remplissage_apres:.0%} "
                f"(reste {self.marge_restante} min)")


@dataclass
class Report:
    """Réponse quand aucune date ne tient dans l'horizon.

    Ce n'est pas un échec de l'algorithme, c'est une information de gestion :
    le bloc est saturé pour ce praticien sur les six mois à venir. Le patient
    est reconvoqué quand l'horizon aura glissé.
    """

    patient_id: int
    raison: str
    vacations_examinees: int
    fenetre_demandee: str = ""
    fenetre_suffisante: str = ""      # la première fenêtre plus large qui marche
    premiere_date_theorique: date | None = None

    def __str__(self) -> str:
        s = (f"Aucune date à partir de « {self.fenetre_demandee} » "
             f"({self.raison}).")
        if self.fenetre_suffisante:
            s += (f" En raccourcissant le délai minimum à "
                  f"« {self.fenetre_suffisante} », une date se libère")
            if self.premiere_date_theorique:
                s += f" ({self.premiere_date_theorique:%d/%m/%Y})"
            s += "."
        else:
            s += (" Même sans délai minimum il n'y a rien : patient à "
                  "reconvoquer quand l'horizon aura glissé.")
        return s


# ---------------------------------------------------------------------------
# 2. Le moteur
# ---------------------------------------------------------------------------


class MoteurPropositions:
    """Propose des dates, et applique celle que le chirurgien retient.

    Le moteur détient la solution courante : le planning tel qu'il est
    aujourd'hui, avec toutes les dates déjà annoncées. Il ne le réorganise
    jamais — il ne fait qu'y ajouter.
    """

    def __init__(self, inst: Instance, solution: Solution | None = None,
                 poids: Poids | None = None,
                 seuil_densite: float = SEUIL_DENSITE_PROCHE):
        self.inst = inst
        self.solution = solution if solution is not None else Solution(inst)
        self.poids = poids or Poids()
        self.evaluateur = Evaluateur(inst, self.poids)
        self.seuil_densite = seuil_densite
        self.relachements = 0        # combien de fois le délai minimum a sauté
        self.relaches_utilises = 0   # ... et combien de fois la date proche a été prise

    # -- la semaine à venir est-elle creuse ? ------------------------------

    def densite_proche(self, patient: Patient,
                       jours_exclus: set[int] | None = None) -> float | None:
        """Taux de remplissage des vacations DU PRATICIEN dans les 7 jours.

        On ne regarde que ses vacations à lui : c'est le seul temps de salle
        où ce patient peut aller, et c'est donc le seul qui puisse être perdu
        pour lui. Un bloc globalement plein alors que ce praticien a une
        vacation vide mercredi n'est pas une raison de ne pas y aller.

        Les journées déjà figées (programme opératoire arrêté) sont exclues du
        calcul : leur temps libre n'est plus utilisable, le compter ferait
        croire à un creux qui n'existe plus.

        Rend `None` s'il n'a aucune vacation ouverte dans la semaine — la
        question ne se pose alors pas, et la fenêtre reste ce qu'elle est.
        """
        inst, sol = self.inst, self.solution
        exclus = jours_exclus or ()
        j0 = patient.jour_demande
        charge = tvo = 0
        for vid in inst.vacations_du_medecin.get(patient.med_id, ()):
            v = inst.vacations[vid]
            if j0 <= v.jour < j0 + JOURS_PROCHE and v.jour not in exclus:
                charge += sol.charge(vid)
                tvo += v.tvo
        return charge / tvo if tvo else None

    # -- le cœur -----------------------------------------------------------

    def proposer(self, patient_id: int, fenetre: str | None = None,
                 k: int = 2,
                 jours_exclus: set[int] | None = None) -> list[Proposition] | Report:
        """Les `k` meilleures dates de la FENÊTRE demandée, sur des jours distincts.

        Le chirurgien annonce d'abord son horizon — « je veux une date dans
        les trois semaines » — et on ne cherche que dedans. La fenêtre n'est
        jamais élargie automatiquement : si rien ne tient, on le dit, on
        indique quelle fenêtre plus large aurait une place, et c'est lui qui
        décide d'attendre ou non.

        Contrainte de distinction : proposer deux créneaux du même jour n'est
        pas un choix pour le patient. On ne garde donc que la meilleure
        vacation de chaque jour avant de classer — s'il y a deux salles
        possibles le même jour, le moteur prend la meilleure et n'en parle
        pas, c'est un détail d'organisation interne.

        Une date n'est proposable que si elle ne viole RIEN : ni la capacité
        de la vacation (marges de risque incluses), ni les lits, ni les
        places. On ne propose pas une date que le bloc ne tiendra pas.
        """
        inst = self.inst
        patient = inst.patients[patient_id]
        if fenetre is None:
            duree = patient.fenetre_jours
            fenetre = nom_fenetre(duree)
        else:
            duree = FENETRES.get(fenetre)
            if duree is None:
                raise ValueError(f"fenêtre inconnue : {fenetre!r} "
                                 f"(attendu : {', '.join(FENETRES)})")

        # --- le délai minimum saute si la semaine à venir est creuse -------
        # On élargit l'ensemble de recherche vers l'avant, jamais vers
        # l'arrière : les dates au-delà du délai demandé restent toutes
        # candidates, on ne fait qu'y ajouter les sept jours qui viennent.
        densite = (self.densite_proche(patient, jours_exclus)
                   if duree > 0 else None)
        relache = densite is not None and densite < self.seuil_densite
        limite = patient.jour_demande + JOURS_PROCHE

        par_jour = self._meilleures_par_jour(
            patient_id, 0 if relache else duree, jours_exclus)
        if relache:
            jmin = patient.jour_demande + duree
            par_jour = {j: v for j, v in par_jour.items()
                        if j < limite or j >= jmin}
            if any(j < limite for j in par_jour):
                self.relachements += 1
            else:
                relache = False

        if par_jour:
            props = self._contraster(patient, par_jour, k)
            if relache and props[0].jour < limite:
                props[0].alertes.insert(
                    0, f"semaine à venir remplie à {densite:.0%} seulement : "
                       f"délai minimum levé, date avancée")
            return props

        # rien du tout : le seul cas où un patient repart sans date
        examinees = len(inst.vacations_possibles(patient_id, duree))
        suffisante, premiere = "", None
        rang = ORDRE_FENETRES.index(fenetre) if fenetre in ORDRE_FENETRES else 0
        for nom in reversed(ORDRE_FENETRES[:rang]):
            plus_court = self._meilleures_par_jour(patient_id, FENETRES[nom])
            if plus_court:
                suffisante = nom
                premiere = inst.date_du_jour(min(plus_court))
                break
        raison = ("plus aucune vacation de ce praticien avant la fin de l'horizon"
                  if examinees == 0
                  else "toutes ses vacations restantes sont pleines ou saturent "
                       "les lits ou les places")
        return Report(patient_id, raison, examinees, fenetre, suffisante, premiere)

    def _contraster(self, patient: Patient,
                    par_jour: dict[int, tuple[float, int]],
                    k: int) -> list[Proposition]:
        """Deux dates VOLONTAIREMENT différentes : au plus tôt, et recommandée.

        C'est le point de conception le plus important du moteur, et il vient
        d'une mesure. Tant que la fenêtre bornait la recherche des deux côtés,
        proposer les deux meilleures dates suffisait. Depuis qu'elle n'est
        qu'un délai MINIMUM, plus rien ne tire les dates vers l'avant : le
        lissage des taux de remplissage pousse à répartir sur tout l'horizon,
        et les deux « meilleures » dates se retrouvent toutes deux à quatre
        mois. Le chirurgien n'a alors aucun levier — choisir la plus proche
        des deux faisait gagner UN jour sur soixante-quatorze.

        On construit donc deux propositions de nature différente :
          - AU PLUS TÔT   : la première date possible après le délai minimum ;
          - RECOMMANDÉE   : celle qui dégrade le moins le planning.
        Mesuré sur vos données, le choix devient un vrai arbitrage : 50 jours
        de délai médian contre 74, et 4,77 d'écart-type sur les lits contre
        3,63. Le praticien décide, en voyant ce que chaque option coûte.

        Si la date au plus tôt est aussi la mieux placée, la seconde
        proposition est la meilleure des autres journées — on rend toujours
        deux jours distincts, sinon ce n'est pas un choix.
        """
        inst = self.inst
        plus_proche = min(par_jour)
        retenus = [plus_proche]
        for jour, _ in sorted(par_jour.items(), key=lambda kv: kv[1][0]):
            if len(retenus) >= k:
                break
            if jour not in retenus:
                retenus.append(jour)

        propositions = []
        for rang, jour in enumerate(retenus):
            delta, vid = par_jour[jour]
            role = "au plus tôt" if rang == 0 else "recommandée"
            propositions.append(self._detailler(patient, delta, vid, role))
        return propositions

    def _meilleures_par_jour(self, patient_id: int, duree: int,
                             jours_exclus: set[int] | None = None
                             ) -> dict[int, tuple[float, int]]:
        """Pour chaque jour de la fenêtre, la meilleure vacation et son coût.

        C'est l'énumération exhaustive : une vingtaine de vacations, évaluées
        toutes, en quelques microsecondes. Pas de métaheuristique ici — avant
        d'en sortir une, on compte la taille de l'espace de recherche.
        """
        inst, sol, ev = self.inst, self.solution, self.evaluateur
        exclus = jours_exclus or ()
        par_jour: dict[int, tuple[float, int]] = {}
        for vid in inst.vacations_possibles(patient_id, duree):
            if inst.vacations[vid].jour in exclus:
                # journée dont le programme opératoire est déjà arrêté : les
                # heures sont annoncées aux patients, on n'y ajoute personne.
                continue
            m = ev.evaluer_mouvement(sol, patient_id, vid)
            if m is None or m.viole_vacation or m.viole_ressource:
                continue
            j = inst.vacations[vid].jour
            if j not in par_jour or m.delta < par_jour[j][0]:
                par_jour[j] = (m.delta, vid)
        return par_jour

    def _detailler(self, patient: Patient, delta: float, vid: int,
                   role: str = "") -> Proposition:
        """Fabrique l'aperçu d'impact montré au chirurgien."""
        inst, sol = self.inst, self.solution
        v = inst.vacations[vid]

        charge_apres = (sol.somme_durees[vid] + patient.duree_op
                        + sol.somme_marges[vid] + patient.marge_perso
                        + inst.tis * sol.nb[vid])
        remplissage = charge_apres / v.tvo if v.tvo else 0.0
        restant = v.tvo - charge_apres

        if patient.ambulatoire:
            lits_apres = sol.lits_jour[v.jour]
            places_apres = sol.places_jour[v.jour] + 1
        else:
            lits_apres = sol.lits_jour[v.jour] + 1
            places_apres = sol.places_jour[v.jour]

        alertes = []
        if remplissage > 0.90:
            alertes.append(f"vacation remplie à {remplissage:.0%} : plus aucune "
                           f"souplesse en cas de dépassement")
        if restant < 30:
            alertes.append(f"il ne resterait que {restant} min dans la vacation")
        if not patient.ambulatoire:
            pic = max(sol.lits_jour[j] + 1
                      for j in range(v.jour, min(v.jour + patient.nb_nuits,
                                                 inst.nb_jours))) \
                if patient.nb_nuits else lits_apres
            if pic > 0.85 * inst.capacite_lits:
                alertes.append(f"porterait l'occupation des lits à {pic}/"
                               f"{inst.capacite_lits} pendant son séjour")
        if places_apres > 0.85 * inst.capacite_places_jour:
            alertes.append(f"{places_apres}e ambulatoire de la journée "
                           f"(maximum {inst.capacite_places_jour})")
        if inst.est_weekend(v.jour):
            alertes.append("jour de week-end")

        return Proposition(
            patient_id=patient.id, vacation_id=vid, jour=v.jour,
            date=inst.date_du_jour(v.jour), cout=delta,
            delai=v.jour - patient.jour_demande, bloc_id=v.bloc_id,
            remplissage_apres=remplissage, marge_restante=restant,
            lits_apres=lits_apres, places_apres=places_apres, role=role,
            alertes=alertes)

    # -- validation --------------------------------------------------------

    def appliquer(self, proposition: Proposition) -> None:
        """Le chirurgien a tranché : la date est posée, et elle ne bougera plus."""
        self.solution.affecter(proposition.patient_id, proposition.vacation_id)

    def enregistrer_report(self, report: Report) -> None:
        self.solution.hors_horizon.add(report.patient_id)


# ---------------------------------------------------------------------------
# 3. Ce que voit le chirurgien
# ---------------------------------------------------------------------------


def nom_fenetre(jours: int) -> str:
    """Nom de la fenêtre correspondant à une durée en jours."""
    for nom in ORDRE_FENETRES:
        if jours <= FENETRES[nom]:
            return nom
    return ORDRE_FENETRES[-1]


def afficher_propositions(inst: Instance, patient: Patient,
                          resultat: list[Proposition] | Report) -> str:
    """Affichage de consultation.

    Même principe que pour l'estimation de durée : on montre les FAITS avant
    de montrer le classement. Un praticien qui voit « date recommandée » en
    premier ne regarde pas le reste.
    """
    lignes = ["", "─" * 74,
              f"Patient {patient.id} — {patient.type_interv or 'acte non précisé'}",
              f"  praticien {patient.med_id} | {patient.duree_op} min estimées "
              f"(+ {patient.marge_perso} min de marge) | "
              + ("ambulatoire" if patient.ambulatoire
                 else f"{patient.nb_nuits} nuit(s) d'hospitalisation"),
              f"  consultation le {inst.date_du_jour(patient.jour_demande):%d/%m/%Y}",
              ""]
    if isinstance(resultat, Report):
        lignes.append("  " + str(resultat))
        return "\n".join(lignes)

    lignes.insert(-1, f"  fenêtre demandée par le chirurgien : "
                      f"{nom_fenetre(patient.fenetre_jours).replace('_', ' ')} "
                      f"({patient.fenetre_jours} jours)")

    for i, p in enumerate(resultat, start=1):
        etiquette = f" [{p.role}]" if p.role else ""
        lignes.append(f"  Date {i}{etiquette} — {p.ligne()}")
        lignes.append(f"           lits ce jour-là {p.lits_apres}/{inst.capacite_lits}"
                      f" | ambulatoires {p.places_apres}/{inst.capacite_places_jour}")
        for a in p.alertes:
            lignes.append(f"           ⚠ {a}")
    lignes.append("")
    return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 4. La décision du chirurgien
# ---------------------------------------------------------------------------
#
# Le choix final appartient au praticien, pas à l'algorithme. On le modélise
# par une fonction interchangeable : en production c'est un humain devant un
# écran, en simulation c'est une POLITIQUE qu'on peut faire varier pour
# mesurer son effet. C'est une expérience à part entière — « que coûte un
# chirurgien qui prend systématiquement la date la plus proche ? » est une
# vraie question de gestion, et elle se chiffre.


def choix_meilleur(props: list[Proposition]) -> Proposition:
    """Le praticien suit la recommandation : celle qui ménage le planning."""
    return min(props, key=lambda p: p.cout)


def choix_plus_proche(props: list[Proposition]) -> Proposition:
    """Le praticien prend toujours la date la plus proche.

    C'est le comportement humain par défaut, et c'est la politique à comparer
    aux autres : elle montre ce que coûte le fait d'ignorer l'organisation.
    """
    return min(props, key=lambda p: p.delai)


def choix_aleatoire(graine: int = 0):
    """Le praticien tranche selon la convenance du patient, hors du modèle."""
    alea = random.Random(graine)

    def _c(props: list[Proposition]) -> Proposition:
        return alea.choice(props)
    return _c


def choix_console(props: list[Proposition]) -> Proposition:
    """Vrai usage : on demande."""
    reponse = input(f"  Quelle date retenez-vous ? [1-{len(props)}, "
                    f"Entrée = 1] ").strip()
    try:
        i = int(reponse) - 1
    except ValueError:
        i = 0
    return props[i] if 0 <= i < len(props) else props[0]


# ---------------------------------------------------------------------------
# 5. Rejouer un flux de consultations
# ---------------------------------------------------------------------------


@dataclass
class Journal:
    """Trace de la simulation, pour mesurer ce que produit le processus réel."""

    poses: int = 0
    reports: int = 0
    une_seule_date: int = 0
    delais: list[int] = field(default_factory=list)
    couts: list[float] = field(default_factory=list)
    reportes: list[int] = field(default_factory=list)

    def resume(self) -> dict:
        import statistics
        return {
            "patients_dates": self.poses,
            "patients_reportes_hors_horizon": self.reports,
            "consultations_avec_une_seule_date": self.une_seule_date,
            "delai_moyen_j": statistics.fmean(self.delais) if self.delais else 0,
            "delai_median_j": statistics.median(self.delais) if self.delais else 0,
            "delai_max_j": max(self.delais, default=0),
        }


def simuler_consultations(inst: Instance, poids: Poids | None = None,
                          choix=choix_meilleur, k: int = 2,
                          fenetre=None,
                          verbeux: bool = False) -> tuple[Solution, Journal]:
    """Rejoue tout le flux de consultations dans l'ordre chronologique.

    C'est LA simulation qui compte : à l'instant où le patient i est vu, les
    patients vus plus tard n'existent pas encore, et les dates déjà données ne
    peuvent plus bouger. Le planning se construit exactement comme dans la
    vraie vie, et il porte donc la même myopie.

    `fenetre` vaut `None` par défaut : chaque patient garde la fenêtre que le
    chirurgien lui a fixée en consultation (`Patient.fenetre_jours`). On peut
    la forcer — une chaîne pour tout le monde, ou une fonction patient ->
    fenêtre — mais uniquement pour répondre à des questions de sensibilité du
    type « que donnerait cette population si tout le monde demandait six
    semaines ? ». Ce n'est jamais le mode normal.

    Comparer le résultat au planning du tabou hors-ligne — qui, lui, connaît
    tout le monde d'avance — donne le prix de cette myopie.
    """
    moteur = MoteurPropositions(inst, Solution(inst), poids)
    journal = Journal()

    # `fenetre=None` : chaque patient garde la sienne, celle que le
    # chirurgien a fixée en consultation. Une chaîne la force pour tout le
    # monde, ce qui ne sert qu'aux analyses de sensibilité.
    choisir_fenetre = (fenetre if callable(fenetre)
                       else (lambda p: fenetre))

    ordre = sorted(inst.patients.values(), key=lambda p: (p.jour_demande, p.id))
    for patient in ordre:
        resultat = moteur.proposer(patient.id, choisir_fenetre(patient), k=k)
        if isinstance(resultat, Report):
            moteur.enregistrer_report(resultat)
            journal.reports += 1
            journal.reportes.append(patient.id)
            if verbeux:
                print(afficher_propositions(inst, patient, resultat))
            continue
        if len(resultat) == 1:
            journal.une_seule_date += 1
        if verbeux:
            print(afficher_propositions(inst, patient, resultat))
        retenue = choix(resultat)
        moteur.appliquer(retenue)
        journal.poses += 1
        journal.delais.append(retenue.delai)
        journal.couts.append(retenue.cout)

    return moteur.solution, journal
