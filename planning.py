r"""
planning.py — Ordonnancement de bloc opératoire : structure de données et
moteur de vérification des contraintes.

Vocabulaire métier (indicateurs ANAP, ceux de votre fichier de vacations) :
  TVO  Temps de Vacation Offert   : durée de la vacation mise à disposition.
  TROS Temps Réel d'Occupation de Salle : entrée en salle -> sortie de salle.
       C'est CE temps qui remplit la vacation, pas la durée d'incision.
  TIS  Temps Inter-Salle : nettoyage + installation entre deux patients.
  Taux d'occupation = TROS cumulé / TVO.

Découpage du temps d'un patient (ce que montre votre historique) :
  [SSPI pré-op] -> [entrée salle .... sortie salle] -> [SSPI] -> [lit d'hospit.]
                    \_________ TROS __________/                  \___ séjour ___/
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# 1. Outils de base sur les intervalles
# ---------------------------------------------------------------------------

def chevauchent(d1: datetime, f1: datetime, d2: datetime, f2: datetime) -> bool:
    """Deux intervalles SEMI-OUVERTS [d, f) se recouvrent-ils ?

    Le semi-ouvert est important : un patient qui sort à 10h00 et un autre qui
    entre à 10h00 ne se chevauchent pas. Avec des intervalles fermés, on
    compterait un lit en trop à chaque transition.
    """
    return d1 < f2 and d2 < f1


def inclus(d1: datetime, f1: datetime, d2: datetime, f2: datetime) -> bool:
    """[d1, f1) est-il entièrement contenu dans [d2, f2) ?"""
    return d2 <= d1 and f1 <= f2


def profil_cumulatif(intervalles) -> list[tuple[datetime, int]]:
    """Ligne de balayage (sweep line) : profil du nombre d'intervalles actifs.

    C'est LA technique à connaître pour compter une ressource cumulative
    (lits, places de SSPI, brancardiers...). On ne parcourt pas le temps
    minute par minute : on ne regarde que les instants où quelque chose change.
      +1 à chaque arrivée, -1 à chaque départ, somme cumulée sur les dates triées.
    Complexité O(n log n) au lieu de O(horizon).
    """
    evenements: dict[datetime, int] = defaultdict(int)
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


# ---------------------------------------------------------------------------
# 2. Vos classes, corrigées
# ---------------------------------------------------------------------------

class Patient:
    """Données du patient + son positionnement dans le planning.

    Changements par rapport à votre version :
      - plus d'arguments par défaut mutables (list) : en Python, la liste par
        défaut est CRÉÉE UNE SEULE FOIS et partagée par toutes les instances.
        Ajouter un geste à un patient l'ajoutait à tous les autres.
      - `fin_op` et `depart_postop` sont des propriétés CALCULÉES, pas des
        attributs stockés. Un état dérivé stocké finit toujours par devenir
        incohérent quand on déplace une intervention.
      - `vacation_id` : le patient sait dans quelle vacation il est posé.
    """

    def __init__(
        self,
        id: int = 0,
        sexe: str = "F",
        age: int = 26,
        med_id: int = 0,
        bloc_id: int = 0,
        diag: str = "S62.30",
        clim_assoc: list[str] | None = None,
        gestes: list[str] | None = None,
        debut_op: datetime | None = None,
        duree_op: timedelta = timedelta(minutes=45),
        duree_sspi: timedelta = timedelta(minutes=90),
        duree_sejour: timedelta = timedelta(days=1),
        ambulatoire: bool = False,
        vacation_id: int | None = None,
    ):
        self.id = id
        self.sexe = sexe
        self.age = age
        self.diag = diag
        self.clim_assoc = list(clim_assoc) if clim_assoc else []
        self.gestes = list(gestes) if gestes else []
        self.med_id = med_id
        self.bloc_id = bloc_id
        self.vacation_id = vacation_id
        self.ambulatoire = ambulatoire

        self.debut_op = debut_op                 # entrée en salle
        self.duree_op = duree_op                 # TROS (entrée -> sortie salle)
        self.duree_sspi = duree_sspi             # salle de réveil
        self.duree_sejour = duree_sejour         # occupation du lit

    # --- état dérivé : jamais stocké, toujours recalculé -------------------

    @property
    def fin_op(self) -> datetime | None:
        return None if self.debut_op is None else self.debut_op + self.duree_op

    @property
    def fin_sspi(self) -> datetime | None:
        return None if self.debut_op is None else self.fin_op + self.duree_sspi

    @property
    def arrivee_postop(self) -> datetime | None:
        """Montée dans le lit d'hospitalisation = sortie de SSPI."""
        return self.fin_sspi

    @property
    def depart_postop(self) -> datetime | None:
        return None if self.debut_op is None else self.arrivee_postop + self.duree_sejour

    @property
    def programme(self) -> bool:
        return self.debut_op is not None

    def temps_post_op(self) -> timedelta:
        return self.duree_sejour

    def deplacer_op(self, delta: timedelta) -> None:
        """Décale tout le patient. Tout le reste suit automatiquement.

        NB : la validation n'est PAS ici. Un objet ne peut pas savoir s'il
        gêne quelqu'un d'autre : seul le Planning a cette vue. C'est le
        principe de la ressource centralisée.
        """
        if self.debut_op is None:
            raise ValueError(f"Patient {self.id} non programmé.")
        self.debut_op += delta

    def __repr__(self) -> str:
        if not self.programme:
            return f"<Patient {self.id} non programmé>"
        return (f"<Patient {self.id} | salle {self.bloc_id} | chir {self.med_id} | "
                f"{self.debut_op:%d/%m %H:%M}-{self.fin_op:%H:%M}>")


class Medecin:
    """Praticien et son motif de vacations.

    `creneau[s]` = liste des créneaux de la s-ième semaine du cycle.
    Un créneau = (jour, heure_debut, heure_fin), jour 0 = lundi.
    `modulo` = longueur du cycle. Votre grille réelle utilise modulo = 2
    ("semaine paire" / "semaine impaire"), votre intuition est la bonne.
    """

    def __init__(self, id: int = 0, nom: str = "", specialite: str = "",
                 creneau=None, modulo: int = 1, salle_preferee: int | None = None):
        self.id = id
        self.nom = nom
        self.specialite = specialite
        self.creneau = creneau if creneau is not None else [[(3, 14, 18)]]
        self.modulo = modulo
        self.salle_preferee = salle_preferee

    def __repr__(self) -> str:
        return f"<Medecin {self.id} {self.nom} ({self.specialite})>"


@dataclass
class Vacation:
    """Une plage de salle attribuée à un praticien (une ligne de votre grille)."""
    id: int
    bloc_id: int          # salle 1..5
    debut: datetime
    fin: datetime
    med_id: int | None = None
    specialite: str | None = None
    marge: timedelta = timedelta(minutes=30)   # tampon pour l'aléa / dépassements

    @property
    def tvo(self) -> timedelta:
        """Temps de Vacation Offert."""
        return self.fin - self.debut

    @property
    def capacite_utile(self) -> timedelta:
        return self.tvo - self.marge

    def __repr__(self) -> str:
        return (f"<Vacation {self.id} S{self.bloc_id} chir {self.med_id} "
                f"{self.debut:%a %d/%m %H:%M}-{self.fin:%H:%M}>")


@dataclass
class Conflit:
    """Une violation de règle. On ne lève pas d'exception : on COLLECTE.

    Un solveur a besoin de savoir *combien* et *lesquelles*, pas de s'arrêter
    à la première. C'est ce qui permet plus tard de transformer une contrainte
    dure en pénalité dans une fonction objectif.
    """
    regle: str
    message: str
    patient_id: int | None = None
    autre_id: int | None = None

    def __str__(self) -> str:
        return f"[{self.regle}] {self.message}"


# ---------------------------------------------------------------------------
# 3. Le planning
# ---------------------------------------------------------------------------

class Planning:
    """Registre central des ressources + vérificateur de contraintes.

    RÈGLES DURES implémentées
      R1 COHERENCE  chronologie interne du patient valide
      R2 SALLE      pas deux interventions dans la même salle en même temps (+ TIS)
      R3 CHIRURGIEN pas deux interventions du même praticien en même temps (+ trajet)
      R4 VACATION   l'intervention tient dans une vacation ouverte, bonne salle,
                    bon praticien (ou bonne spécialité)
      R5 MARGE      TROS cumulé + TIS <= TVO - marge
      R6 LITS       occupation des lits <= capacité (contrainte CUMULATIVE)
      R7 SSPI       occupation de la salle de réveil <= capacité
      R8 AMBU       un ambulatoire entre et sort le même jour, avant fermeture UCA

    RÈGLES SOUPLES (à mettre en pénalité, pas en interdiction — voir indicateurs())
      S1  lisser l'occupation des lits (objectif ±2 lits de votre slide 13)
      S2  éviter les séjours qui traversent le week-end sans raison
      S3  ambulatoires en début de vacation (sortie le soir même)
      S4  patients septiques / infectés en fin de vacation
      S5  enfants et diabétiques tôt (jeûne)
      S6  regrouper les actes similaires (effet d'apprentissage, même matériel)
    """

    def __init__(
        self,
        capacite_lits: int = 42,
        capacite_ambu: int = 12,
        capacite_sspi: int = 6,
        tis: timedelta = timedelta(minutes=15),
        trajet_chirurgien: timedelta = timedelta(minutes=10),
        fermeture_uca: int = 20,          # heure de fermeture de l'ambulatoire
    ):
        self.patients: dict[int, Patient] = {}
        self.medecins: dict[int, Medecin] = {}
        self.vacations: dict[int, Vacation] = {}

        self.capacite_lits = capacite_lits
        self.capacite_ambu = capacite_ambu
        self.capacite_sspi = capacite_sspi
        self.tis = tis
        self.trajet_chirurgien = trajet_chirurgien
        self.fermeture_uca = fermeture_uca

    # -- enregistrement ----------------------------------------------------

    def ajouter_medecin(self, m: Medecin) -> None:
        self.medecins[m.id] = m

    def ajouter_vacation(self, v: Vacation) -> None:
        self.vacations[v.id] = v

    def ajouter_patient(self, p: Patient) -> None:
        self.patients[p.id] = p

    def generer_vacations(self, med: Medecin, lundi_ref: datetime,
                          nb_semaines: int, bloc_id: int, debut_id: int = 1000):
        """Déroule le motif (creneau, modulo) en vacations concrètes.

        Le motif est compact, mais on ne raisonne JAMAIS sur le motif : on le
        déroule en objets datés. Sinon chaque contrainte devient de
        l'arithmétique modulaire, et la moindre exception (jour férié, congé,
        remplacement) casse tout.
        """
        creees = []
        vid = debut_id
        for semaine in range(nb_semaines):
            for (jour, h_deb, h_fin) in med.creneau[semaine % med.modulo]:
                jour0 = lundi_ref + timedelta(weeks=semaine, days=jour)
                v = Vacation(
                    id=vid,
                    bloc_id=bloc_id,
                    med_id=med.id,
                    specialite=med.specialite,
                    debut=jour0.replace(hour=int(h_deb),
                                        minute=int(round((h_deb % 1) * 60)),
                                        second=0, microsecond=0),
                    fin=jour0.replace(hour=int(h_fin),
                                      minute=int(round((h_fin % 1) * 60)),
                                      second=0, microsecond=0),
                )
                self.ajouter_vacation(v)
                creees.append(v)
                vid += 1
        return creees

    # -- lecture du planning -----------------------------------------------

    def programmes(self) -> list[Patient]:
        return [p for p in self.patients.values() if p.programme]

    def patients_de(self, vacation_id: int) -> list[Patient]:
        return sorted(
            (p for p in self.programmes() if p.vacation_id == vacation_id),
            key=lambda p: p.debut_op,
        )

    # -- R6/R7 : ressources cumulatives (lits, SSPI) ------------------------

    def occupation_lits(self, instant: datetime, ambulatoire: bool = False) -> int:
        """Nombre de lits occupés à un instant donné."""
        return sum(
            1 for p in self.programmes()
            if p.ambulatoire == ambulatoire
            and p.arrivee_postop <= instant < p.depart_postop
        )

    def profil_lits(self, ambulatoire: bool = False):
        """Courbe complète d'occupation : c'est la courbe de votre slide 3."""
        return profil_cumulatif(
            (p.arrivee_postop, p.depart_postop)
            for p in self.programmes() if p.ambulatoire == ambulatoire
        )

    def profil_sspi(self):
        return profil_cumulatif((p.fin_op, p.fin_sspi) for p in self.programmes())

    def pic_lits(self, ambulatoire: bool = False):
        profil = self.profil_lits(ambulatoire)
        return max(profil, key=lambda x: x[1]) if profil else (None, 0)

    def depassements(self, profil, capacite: int):
        """Périodes où la capacité est dépassée : [(début, fin, niveau), ...]."""
        out, ouvert = [], None
        for instant, n in profil:
            if n > capacite and ouvert is None:
                ouvert = (instant, n)
            elif n <= capacite and ouvert is not None:
                out.append((ouvert[0], instant, ouvert[1]))
                ouvert = None
            elif ouvert is not None:
                ouvert = (ouvert[0], max(ouvert[1], n))
        return out

    # -- R5 : charge d'une vacation ----------------------------------------

    def charge_vacation(self, vacation_id: int) -> dict:
        """Bilan d'une vacation : TROS, TIS, temps restant, taux d'occupation."""
        v = self.vacations[vacation_id]
        pats = self.patients_de(vacation_id)
        tros = sum((p.duree_op for p in pats), timedelta())
        tis_total = self.tis * max(0, len(pats) - 1)
        utilise = tros + tis_total
        return {
            "vacation": v,
            "nb_patients": len(pats),
            "tros": tros,
            "tis": tis_total,
            "utilise": utilise,
            "restant": v.capacite_utile - utilise,
            "taux_occupation": tros / v.tvo if v.tvo else 0.0,
            "patients": pats,
        }

    def creneaux_libres(self, vacation_id: int, duree: timedelta) -> list[datetime]:
        """Heures de début possibles pour une intervention de cette durée.

        Technique : les seuls départs intéressants sont le début de vacation et
        les fins d'interventions déjà posées (+ TIS). Inutile de tester toutes
        les minutes — on appelle ça les *points d'ancrage*. C'est la base des
        heuristiques d'insertion (first-fit / best-fit, cf. bin packing).
        """
        v = self.vacations[vacation_id]
        pats = self.patients_de(vacation_id)
        occupes = [(p.debut_op, p.fin_op + self.tis) for p in pats]

        candidats = [v.debut] + [fin for _, fin in occupes]
        libres = []
        for c in sorted(set(candidats)):
            if c + duree > v.fin - v.marge:
                continue
            if any(chevauchent(c, c + duree + self.tis, d, f) for d, f in occupes):
                continue
            libres.append(c)
        return libres

    # -- vérification ------------------------------------------------------

    def conflits_patient(self, p: Patient) -> list[Conflit]:
        """Toutes les règles dures, pour un patient. Le cœur du système."""
        c: list[Conflit] = []
        if not p.programme:
            return c

        # R1 — cohérence interne
        if p.duree_op <= timedelta(0):
            c.append(Conflit("R1", f"durée opératoire nulle ou négative", p.id))
        if p.duree_sejour < timedelta(0):
            c.append(Conflit("R1", f"durée de séjour négative", p.id))

        # R2 / R3 — concurrence (ressources DISJONCTIVES)
        for q in self.programmes():
            if q.id == p.id:
                continue
            if q.bloc_id == p.bloc_id and chevauchent(
                p.debut_op, p.fin_op + self.tis, q.debut_op, q.fin_op + self.tis
            ):
                c.append(Conflit("R2", f"salle {p.bloc_id} : conflit avec le patient {q.id} "
                                       f"({q.debut_op:%H:%M}-{q.fin_op:%H:%M})", p.id, q.id))
            if q.med_id == p.med_id and chevauchent(
                p.debut_op, p.fin_op + self.trajet_chirurgien,
                q.debut_op, q.fin_op + self.trajet_chirurgien
            ):
                c.append(Conflit("R3", f"chirurgien {p.med_id} : conflit avec le patient {q.id}",
                                 p.id, q.id))

        # R4 — appartenance à une vacation ouverte
        v = self.vacations.get(p.vacation_id) if p.vacation_id is not None else None
        if v is None:
            c.append(Conflit("R4", "intervention hors de toute vacation", p.id))
        else:
            if not inclus(p.debut_op, p.fin_op, v.debut, v.fin):
                c.append(Conflit("R4", f"déborde de la vacation {v.id} "
                                       f"({v.debut:%H:%M}-{v.fin:%H:%M})", p.id))
            if v.bloc_id != p.bloc_id:
                c.append(Conflit("R4", f"salle {p.bloc_id} ≠ salle {v.bloc_id} de la vacation", p.id))
            if v.med_id is not None and v.med_id != p.med_id:
                c.append(Conflit("R4", f"vacation attribuée au chirurgien {v.med_id}, "
                                       f"pas au {p.med_id}", p.id))

            # R5 — marge
            bilan = self.charge_vacation(v.id)
            if bilan["restant"] < timedelta(0):
                c.append(Conflit("R5", f"vacation {v.id} surchargée de "
                                       f"{-bilan['restant']} (marge {v.marge} incluse)", p.id))

        # R6 / R7 — ressources CUMULATIVES
        cap = self.capacite_ambu if p.ambulatoire else self.capacite_lits
        profil = self.profil_lits(p.ambulatoire)
        for instant, n in profil:
            if n > cap and p.arrivee_postop <= instant < p.depart_postop:
                c.append(Conflit("R6", f"capacité lits dépassée le {instant:%d/%m %H:%M} "
                                       f"({n}/{cap})", p.id))
                break
        for instant, n in self.profil_sspi():
            if n > self.capacite_sspi and p.fin_op <= instant < p.fin_sspi:
                c.append(Conflit("R7", f"SSPI saturée le {instant:%d/%m %H:%M} "
                                       f"({n}/{self.capacite_sspi})", p.id))
                break

        # R8 — ambulatoire
        if p.ambulatoire:
            if p.depart_postop.date() != p.debut_op.date():
                c.append(Conflit("R8", "ambulatoire : sortie un autre jour que l'entrée", p.id))
            elif p.depart_postop.hour >= self.fermeture_uca:
                c.append(Conflit("R8", f"ambulatoire : sortie à {p.depart_postop:%H:%M}, "
                                       f"après fermeture de l'UCA ({self.fermeture_uca}h)", p.id))

        return c

    def verifier(self) -> list[Conflit]:
        """Tous les conflits du planning (dédoublonnés sur les paires)."""
        vus, out = set(), []
        for p in self.programmes():
            for c in self.conflits_patient(p):
                cle = (c.regle, *sorted(filter(None, (c.patient_id, c.autre_id))), c.message[:30])
                if cle not in vus:
                    vus.add(cle)
                    out.append(c)
        return out

    # -- modification sûre --------------------------------------------------

    def programmer(self, patient_id: int, vacation_id: int,
                   debut: datetime | None = None, forcer: bool = False):
        """Place un patient. Renvoie (succès, conflits).

        Motif *tenter / valider / annuler* : on applique, on vérifie, et on
        remet l'état précédent si ça casse. Simple et sûr.
        Dans un vrai solveur on fait l'inverse (évaluation incrémentale AVANT
        d'appliquer) parce qu'on teste des millions de mouvements.
        """
        p = self.patients[patient_id]
        v = self.vacations[vacation_id]
        sauvegarde = (p.debut_op, p.vacation_id, p.bloc_id, p.med_id)

        if debut is None:
            libres = self.creneaux_libres(vacation_id, p.duree_op)
            if not libres:
                return False, [Conflit("R5", f"aucun créneau libre dans la vacation {vacation_id}",
                                       patient_id)]
            debut = libres[0]

        p.debut_op, p.vacation_id, p.bloc_id = debut, vacation_id, v.bloc_id
        if v.med_id is not None:
            p.med_id = v.med_id

        conflits = self.conflits_patient(p)
        if conflits and not forcer:
            p.debut_op, p.vacation_id, p.bloc_id, p.med_id = sauvegarde
            return False, conflits
        return True, conflits

    def deplacer(self, patient_id: int, delta: timedelta, forcer: bool = False):
        """Décale une intervention en vérifiant qu'elle ne gêne personne."""
        p = self.patients[patient_id]
        sauvegarde = p.debut_op
        p.deplacer_op(delta)
        conflits = self.conflits_patient(p)
        if conflits and not forcer:
            p.debut_op = sauvegarde
            return False, conflits
        return True, conflits

    def deprogrammer(self, patient_id: int) -> None:
        p = self.patients[patient_id]
        p.debut_op, p.vacation_id = None, None

    # -- pilotage -----------------------------------------------------------

    def indicateurs(self, heure_mesure: int = 12) -> dict:
        """Les chiffres qui disent si le planning est bon.

        L'écart-type de l'occupation des lits est votre vraie fonction
        objectif : c'est lui qui traduit « lisser la courbe » (slide 13).
        """
        occ = []
        pats = [p for p in self.programmes() if not p.ambulatoire]
        if pats:
            d = min(p.arrivee_postop for p in pats).date()
            f = max(p.depart_postop for p in pats).date()
            while d <= f:
                occ.append(self.occupation_lits(datetime.combine(d, datetime.min.time())
                                                .replace(hour=heure_mesure)))
                d += timedelta(days=1)

        taux = [self.charge_vacation(v)["taux_occupation"] for v in self.vacations]
        profil = self.profil_lits()
        return {
            "patients_programmes": len(self.programmes()),
            "patients_en_attente": len(self.patients) - len(self.programmes()),
            "taux_occupation_moyen": sum(taux) / len(taux) if taux else 0.0,
            "lits_moyen": statistics.mean(occ) if occ else 0,
            "lits_ecart_type": statistics.pstdev(occ) if len(occ) > 1 else 0.0,
            "lits_pic": self.pic_lits()[1],
            "lits_creux": min(occ) if occ else 0,
            "depassements_lits": len(self.depassements(profil, self.capacite_lits)),
            "conflits": len(self.verifier()),
        }

    def afficher_journee(self, jour) -> str:
        """Gantt textuel, pour déboguer sans dépendance graphique."""
        if isinstance(jour, datetime):
            jour = jour.date()
        lignes = [f"=== {jour:%A %d/%m/%Y} ==="]
        vacs = sorted((v for v in self.vacations.values() if v.debut.date() == jour),
                      key=lambda v: (v.bloc_id, v.debut))
        for v in vacs:
            b = self.charge_vacation(v.id)
            lignes.append(f"  Salle {v.bloc_id} {v.debut:%H:%M}-{v.fin:%H:%M} "
                          f"chir {v.med_id} | TVO {v.tvo} | occup. "
                          f"{b['taux_occupation']:.0%} | reste {b['restant']}")
            for p in b["patients"]:
                tag = "AMBU" if p.ambulatoire else f"{p.duree_sejour.days}j"
                lignes.append(f"      {p.debut_op:%H:%M}-{p.fin_op:%H:%M}  "
                              f"patient {p.id:<4} {p.diag:<8} {tag}")
        lignes.append(f"  Lits occupés à 12h : "
                      f"{self.occupation_lits(datetime.combine(jour, datetime.min.time()).replace(hour=12))}"
                      f"/{self.capacite_lits}")
        return "\n".join(lignes)


# ---------------------------------------------------------------------------
# 4. Démonstration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pl = Planning(capacite_lits=6, capacite_sspi=2, tis=timedelta(minutes=15))

    # Deux chirurgiens, cycle de 2 semaines (paire / impaire) comme la vraie grille
    ortho = Medecin(id=1, nom="JT", specialite="Orthopédie",
                    creneau=[[(0, 8, 13), (2, 8, 13)],   # semaine paire
                             [(0, 8, 17.5)]],            # semaine impaire
                    modulo=2)
    digest = Medecin(id=2, nom="SR", specialite="Digestif",
                     creneau=[[(0, 8, 13)]], modulo=1)
    pl.ajouter_medecin(ortho)
    pl.ajouter_medecin(digest)

    lundi = datetime(2026, 1, 5, 0, 0)          # un lundi
    v_ortho = pl.generer_vacations(ortho, lundi, 2, bloc_id=1, debut_id=100)
    v_dig = pl.generer_vacations(digest, lundi, 2, bloc_id=2, debut_id=200)
    print("Vacations générées :")
    for v in v_ortho + v_dig:
        print("   ", v)

    # Patients
    for i, (duree, sejour, ambu) in enumerate([
        (90, 4, False), (75, 3, False), (60, 0, True), (120, 5, False), (45, 0, True),
    ], start=1):
        pl.ajouter_patient(Patient(
            id=i, med_id=1, diag="M17.1",
            duree_op=timedelta(minutes=duree),
            duree_sspi=timedelta(minutes=60),
            duree_sejour=timedelta(days=sejour) if not ambu else timedelta(hours=5),
            ambulatoire=ambu,
        ))

    print("\nRemplissage de la vacation 100 (first-fit) :")
    for i in range(1, 6):
        ok, conf = pl.programmer(i, 100)
        print(f"   patient {i} -> {'placé' if ok else 'REFUSÉ'}"
              + ("" if ok else f" : {conf[0]}"))

    print()
    print(pl.afficher_journee(datetime(2026, 1, 5)))

    # Conflit de concurrence : le même chirurgien se voit attribuer une
    # deuxième salle au même moment (cas réel des "vacations en miroir").
    print("\nTest de concurrence (même chirurgien, deux salles en parallèle) :")
    pl.ajouter_vacation(Vacation(id=300, bloc_id=3, med_id=1,
                                 debut=datetime(2026, 1, 5, 8, 0),
                                 fin=datetime(2026, 1, 5, 13, 0)))
    pl.ajouter_patient(Patient(id=99, med_id=1, duree_op=timedelta(minutes=60),
                               duree_sejour=timedelta(days=2)))
    ok, conf = pl.programmer(99, 300, debut=datetime(2026, 1, 5, 8, 30))
    print(f"   résultat : {'placé' if ok else 'REFUSÉ'}")
    for c in conf:
        print("   ", c)

    print("\nTest de déplacement :")
    ok, conf = pl.deplacer(2, timedelta(minutes=-30))
    print(f"   -30 min sur le patient 2 : {'OK' if ok else 'REFUSÉ'}")
    for c in conf:
        print("   ", c)

    print("\nIndicateurs :")
    for k, val in pl.indicateurs().items():
        print(f"   {k:<24} {val}")
