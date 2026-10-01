r"""
simulation.py — Le temps qui avance : une année de bloc, jour après jour.

POURQUOI CE FICHIER EXISTE
--------------------------
Jusqu'ici les deux moteurs tournaient l'un après l'autre : toutes les
consultations d'abord, puis le tabou local sur toutes les journées. C'est
commode pour mesurer, mais ce n'est pas le fonctionnement réel, et la
différence n'est pas cosmétique.

Dans la vraie vie le temps avance. Chaque jour :

    1. les patients vus en consultation ce jour-là reçoivent leurs deux dates ;
    2. une journée future qui vient de franchir le SEUIL de remplissage voit
       son programme opératoire arrêté tout de suite — on ne laisse pas une
       journée à 90 % dans le flou ;
    3. la journée de DEMAIN voit son programme arrêté, si ce n'est déjà fait ;
    4. on passe au lendemain.

CE QUE ÇA CHANGE, ET QUI EST LE POINT IMPORTANT
------------------------------------------------
Arrêter le programme d'une journée, c'est annoncer une heure de convocation à
chacun de ses patients. On ne peut donc plus rien y ajouter : insérer un
patient décalerait tous ceux qui suivent, et on leur a déjà dit à quelle heure
venir. **Une journée planifiée est une journée fermée.**

Conséquence directe : une journée qui atteint le seuil à 85 % se ferme avec
15 % de son temps inutilisé, et les patients qui seraient allés là vont
ailleurs. Le seuil n'est donc pas un réglage neutre — il arbitre entre
« arrêter tôt le programme, au prix de temps de salle perdu » et « laisser
ouvert jusqu'à la veille, au prix d'une visibilité tardive pour l'équipe ».

C'est exactement le genre de paramètre qu'il faut balayer plutôt que choisir :
`balayer_seuil()` le fait.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from modele import Instance, PlanningJour, Solution, deriver_creneaux
from propositions import (MoteurPropositions, Report, choix_plus_proche)
from tabou import PoidsLocaux, Poids, TabouLocal, contexte_lits_pic


@dataclass
class JournalAnnee:
    """Tout ce qu'on veut pouvoir relire après la simulation."""

    dates_donnees: int = 0
    reports: int = 0
    delais: list[int] = field(default_factory=list)
    figes_par_seuil: list[int] = field(default_factory=list)
    figes_par_veille: list[int] = field(default_factory=list)
    taux_a_la_fermeture: list[float] = field(default_factory=list)
    creux_total_min: int = 0             # temps de salle inoccupé, toutes journées
    deprogrammes: int = 0
    violations: int = 0
    relachements: int = 0        # délais minimum levés pour cause de semaine creuse
    relaches_pris: int = 0       # ... et suivis d'une date effectivement avancée

    def resume(self) -> dict:
        d = sorted(self.delais)
        return {
            "dates_donnees": self.dates_donnees,
            "reports": self.reports,
            "delai_median_j": d[len(d) // 2] if d else 0,
            "delai_p90_j": d[9 * len(d) // 10] if d else 0,
            "journees_figees_par_seuil": len(self.figes_par_seuil),
            "journees_figees_la_veille": len(self.figes_par_veille),
            "taux_moyen_a_la_fermeture": (statistics.fmean(self.taux_a_la_fermeture)
                                          if self.taux_a_la_fermeture else 0.0),
            "creux_total_h": self.creux_total_min / 60,
            "patients_deprogrammes": self.deprogrammes,
            "violations_restantes": self.violations,
            "relachements": self.relachements,
            "relaches_pris": self.relaches_pris,
        }


def simuler_annee(inst: Instance, poids: Poids | None = None,
                  poids_locaux: PoidsLocaux | None = None,
                  choix=choix_plus_proche,
                  seuil: float = 0.85,
                  seuil_densite: float | None = None,
                  iterations_locales: int = 100,
                  fermer_apres_planification: bool = True,
                  verbeux: bool = False
                  ) -> tuple[Solution, dict[int, PlanningJour], JournalAnnee]:
    """Déroule l'horizon jour par jour, les deux moteurs entrelacés.

    `seuil` : taux de remplissage à partir duquel on arrête le programme d'une
    journée sans attendre la veille.

    `fermer_apres_planification` : mettre à False donne la version « on peut
    encore ajouter quelqu'un après coup », qui produit un meilleur remplissage
    mais oblige à re-convoquer. C'est le point de comparaison, pas le mode
    normal.
    """
    moteur = (MoteurPropositions(inst, Solution(inst), poids)
              if seuil_densite is None else
              MoteurPropositions(inst, Solution(inst), poids,
                                 seuil_densite=seuil_densite))
    local = TabouLocal(inst, poids_locaux, iterations=iterations_locales)
    sol = moteur.solution
    journal = JournalAnnee()
    plannings: dict[int, PlanningJour] = {}
    figes: set[int] = set()

    consultations: dict[int, list[int]] = defaultdict(list)
    for p in inst.patients.values():
        consultations[p.jour_demande].append(p.id)

    def taux_du_jour(j: int) -> float:
        vids = inst.vacations_du_jour.get(j, ())
        tvo = sum(inst.vacations[v].tvo for v in vids)
        return sum(sol.charge(v) for v in vids) / tvo if tvo else 0.0

    def arreter_programme(j: int, cause: str) -> None:
        """Le tabou local tourne, les heures sont fixées, la journée se ferme."""
        if j in figes or not inst.vacations_du_jour.get(j):
            return
        if not any(sol.nb[v] for v in inst.vacations_du_jour[j]):
            return
        pj, reportes = local.resoudre(sol, j)
        for pid in reportes:
            sol.affecter(pid, None)
        journal.deprogrammes += len(reportes)
        pj.pic_lits = contexte_lits_pic(inst, pj, *_contexte(inst, sol, j))
        plannings[j] = pj
        journal.violations += (pj.depassement > 0) + pj.hors_uca
        journal.taux_a_la_fermeture.append(taux_du_jour(j))
        journal.creux_total_min += pj.creux_total
        if fermer_apres_planification:
            figes.add(j)
        (journal.figes_par_seuil if cause == "seuil"
         else journal.figes_par_veille).append(j)

    for aujourdhui in range(inst.nb_jours):
        # ---- 1. les consultations du jour ---------------------------------
        for pid in sorted(consultations.get(aujourdhui, ())):
            resultat = moteur.proposer(pid, jours_exclus=figes)
            if isinstance(resultat, Report):
                moteur.enregistrer_report(resultat)
                journal.reports += 1
                continue
            retenue = choix(resultat)
            moteur.appliquer(retenue)
            journal.dates_donnees += 1
            journal.delais.append(retenue.delai)
            if retenue.delai < inst.patients[pid].fenetre_jours:
                journal.relaches_pris += 1

            # ---- 2. cette journée vient-elle de franchir le seuil ? --------
            j = inst.vacations[retenue.vacation_id].jour
            if j > aujourdhui + 1 and j not in figes and taux_du_jour(j) >= seuil:
                arreter_programme(j, "seuil")

        # ---- 3. et de toute façon, le programme de demain ------------------
        arreter_programme(aujourdhui + 1, "veille")

        if verbeux and aujourdhui % 70 == 0:
            print(f"    J{aujourdhui:4d} : {journal.dates_donnees} dates, "
                  f"{len(figes)} journées arrêtées")

    journal.relachements = moteur.relachements
    return sol, plannings, journal


def _contexte(inst: Instance, sol: Solution, jour: int) -> tuple[int, int]:
    """Lits reportés de la veille et sorties du matin (indicateur)."""
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


def balayer_seuil(inst_fabrique, seuils=(0.60, 0.75, 0.85, 0.95, 1.01),
                  **kw) -> list[dict]:
    """Que coûte le seuil de déclenchement ? Un balayage, pas un choix au jugé.

    `inst_fabrique` est une fonction sans argument qui rend une instance
    NEUVE : chaque seuil doit partir du même planning vide, sinon on mesure
    l'ordre des essais et non le seuil.

    Un seuil au-dessus de 1 revient à ne jamais déclencher par remplissage :
    toutes les journées attendent la veille. C'est le témoin.
    """
    lignes = []
    for s in seuils:
        inst = inst_fabrique()
        sol, plannings, journal = simuler_annee(inst, seuil=s, **kw)
        r = journal.resume()
        r["seuil"] = s
        r["remplissage_moyen"] = sol.indicateurs()["taux_remplissage_moyen"]
        r["lits_ecart_type"] = sol.indicateurs()["lits_ecart_type"]
        r["pic_places_moyen"] = (statistics.fmean([p.pic_places for p in plannings.values()])
                                 if plannings else 0)
        lignes.append(r)
    return lignes
