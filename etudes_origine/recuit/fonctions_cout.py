r"""
fonctions_cout.py — Fonctions coût candidates pour le recuit simulé, et
indicateurs « physiques » communs pour les comparer.

Toutes les fonctions s'appliquent à la Solution du modele.py du groupe.

POURQUOI UN « ÉVALUATEUR »
--------------------------
Les fonctions coût du modele.py recalculent tout à chaque appel (tous les
jours, toutes les vacations, tous les patients). Le recuit évalue 50 000 à
200 000 mouvements : ce serait des minutes par exécution. L'Evaluateur tient
à jour, à chaque déplacement, les sommes dont les fonctions coût ont besoin :
un mouvement ne coûte que O(nombre de nuits du patient). Les résultats sont
IDENTIQUES à un recalcul complet (vérifié par `verifier_coherence`).

LES CANDIDATES
--------------
C0  groupe         : CoutTotal du modele.py, tel quel (variance des lits,
                     variance des places, variance des remplissages, délai).
C1  groupe normalisé : les mêmes 4 termes, chacun divisé par sa valeur sur le
                     planning réel, puis pondérés (lits 40 %, places 20 %,
                     bloc 20 %, délai 20 %). Les poids ont enfin un sens.
C2  cible lits     : on dit EXPLICITEMENT le profil voulu. Chaque nuit a une
                     cible = charge moyenne répartie selon le personnel
                     (moins le week-end et en vacances) ; on pénalise l'écart
                     à la cible au carré + le dépassement de la tolérance
                     ±2 lits. Ambulatoire : écart à la moyenne par jour
                     ouvré. Bloc : temps perdu dans les vacations ouvertes.
                     Délai : comme le groupe. Termes normalisés, pondérés.
C3  cible lits seule : uniquement le terme lits de C2 (lissage maximal, pour
                     voir ce que coûte le lissage au bloc et aux délais).

Contraintes dures (toutes les candidates) : lits > capacité, ambulatoires par
jour > capacité, dépassement de vacation, avec la pénalité 1e6 du modèle.
"""

from __future__ import annotations

from dataclasses import dataclass

from modele import PENALITE


class Evaluateur:

    def __init__(self, rejeu, sol, facteur_weekend: float = 0.85, facteur_reduit: float = 0.70,
                 tolerance: float = 2.0, nuits_weekend: tuple[int, ...] = (4, 5, 6)):
        """facteur_weekend : la cible de lits d'une nuit de week-end (vendredi,
        samedi et dimanche soir : personnel de week-end) vaut 85 % d'une nuit
        de semaine. facteur_reduit : 70 % pendant les vacances scolaires, ponts
        et fériés (moins de lits en vacances qu'en week-end, choix du groupe).
        Ce sont des HYPOTHÈSES à ajuster avec l'hôpital."""
        self.rj, self.sol, self.inst = rejeu, sol, sol.inst
        inst = self.inst
        H = inst.nb_jours
        self.H = H
        self.tol = tolerance
        self.facteur_weekend, self.facteur_reduit = facteur_weekend, facteur_reduit
        self.E = list(range(rejeu.eval_debut, rejeu.eval_fin))
        self.dans_E = [rejeu.eval_debut <= j < rejeu.eval_fin for j in range(H)]
        self.we = [inst.date_du_jour(j).weekday() in nuits_weekend for j in range(H)]
        self.reduit = [j in rejeu.periode_reduite for j in range(H)]
        poids_nuit = [(facteur_weekend if self.we[j] else 1.0) * (facteur_reduit if self.reduit[j] else 1.0)
                      for j in range(H)]
        total_lits = sum(sol.lits_jour[j] for j in self.E)
        s = sum(poids_nuit[j] for j in self.E)
        self.cible = [total_lits * poids_nuit[j] / s if self.dans_E[j] else 0.0 for j in range(H)]
        self.JA = [j for j in self.E if j in inst.vacations_du_jour and inst.date_du_jour(j).weekday() < 5]
        self.dans_JA = [False] * H
        for j in self.JA:
            self.dans_JA[j] = True
        self.cibleA = sum(sol.places_jour[j] for j in self.JA) / max(1, len(self.JA))
        self.ouvres = set(inst.jours_ouvres)
        self._recalculer()
        self.ref = {}          # valeurs de référence (planning réel), fixées par figer_reference()

    # -- contributions élémentaires ------------------------------------------

    def _c_lit(self, j, L):
        cap = self.inst.capacite_lits
        s = max(0, L - cap)
        if not self.dans_E[j]:
            return 0.0, 0.0, 0, 0, s
        e = L - self.cible[j]
        t = max(0.0, abs(e) - self.tol)
        return e * e, t * t, L, L * L, s

    def _c_place(self, j, P):
        s = max(0, P - self.inst.capacite_places_jour) if j in self.ouvres else 0
        e = (P - self.cibleA) ** 2 if self.dans_JA[j] else 0.0
        return e, s

    def _c_vac(self, v):
        sol = self.sol
        n = sol.nb[v]
        tvo = self.inst.vacations[v].tvo
        return sol.depassement(v), (sol.creux(v) if n else 0), (tvo if n else 0), (1 if n else 0)

    def _recalculer(self):
        sol, inst = self.sol, self.inst
        self.SE = self.ST = 0.0
        self.SL = self.SL2 = 0
        self.surch_l = 0
        for j in range(self.H):
            a, b, c, d, s = self._c_lit(j, sol.lits_jour[j])
            self.SE += a; self.ST += b; self.SL += c; self.SL2 += d; self.surch_l += s
        self.SEA, self.surch_p = 0.0, 0
        for j in range(self.H):
            a, s = self._c_place(j, sol.places_jour[j])
            self.SEA += a; self.surch_p += s
        self.dep = self.perdu = self.tvo_ouv = self.n_ouv = 0
        for v in inst.vacations:
            a, b, c, d = self._c_vac(v)
            self.dep += a; self.perdu += b; self.tvo_ouv += c; self.n_ouv += d
        self.SD, self.ND, self.SDec, self.nb_deplaces = 0, 0, 0, 0
        self.sans_creneau = sum(1 for pid in self.rj.mobiles if sol.affectation[pid] is None)
        for pid, v in sol.affectation.items():
            if v is not None:
                self.SD += self._delai(pid, v); self.ND += 1
                dec = inst.vacations[v].jour - self.rj.jour_reel[pid]
                self.SDec += dec; self.nb_deplaces += (dec != 0)

    def _delai(self, pid, v):
        p = self.inst.patients[pid]
        return max(0, self.inst.vacations[v].jour - (p.jour_demande + p.fenetre_jours))

    # -- mouvement -----------------------------------------------------------

    def deplacer(self, pid: int, nouveau: int) -> None:
        sol, inst = self.sol, self.inst
        ancien = sol.affectation[pid]
        if ancien == nouveau:
            return
        p = inst.patients[pid]
        H = self.H
        if p.ambulatoire:
            nuits = ()
            jours = {inst.vacations[v].jour for v in (ancien, nouveau) if v is not None}
        else:
            jours = ()
            nuits = set()
            for v in (ancien, nouveau):
                if v is not None:
                    j0 = inst.vacations[v].jour
                    nuits.update(range(j0, min(j0 + p.nb_nuits, H)))
        vacs = [v for v in (ancien, nouveau) if v is not None]
        L, P = sol.lits_jour, sol.places_jour
        for j in nuits:
            a, b, c, d, s = self._c_lit(j, L[j])
            self.SE -= a; self.ST -= b; self.SL -= c; self.SL2 -= d; self.surch_l -= s
        for j in jours:
            a, s = self._c_place(j, P[j]); self.SEA -= a; self.surch_p -= s
        for v in vacs:
            a, b, c, d = self._c_vac(v)
            self.dep -= a; self.perdu -= b; self.tvo_ouv -= c; self.n_ouv -= d
        jr = self.rj.jour_reel[pid]
        self.sans_creneau += (nouveau is None) - (ancien is None)
        if ancien is not None:
            self.SD -= self._delai(pid, ancien); self.ND -= 1
            dec = inst.vacations[ancien].jour - jr; self.SDec -= dec; self.nb_deplaces -= (dec != 0)

        sol.affecter(pid, nouveau)

        for j in nuits:
            a, b, c, d, s = self._c_lit(j, L[j])
            self.SE += a; self.ST += b; self.SL += c; self.SL2 += d; self.surch_l += s
        for j in jours:
            a, s = self._c_place(j, P[j]); self.SEA += a; self.surch_p += s
        for v in vacs:
            a, b, c, d = self._c_vac(v)
            self.dep += a; self.perdu += b; self.tvo_ouv += c; self.n_ouv += d
        if nouveau is not None:
            self.SD += self._delai(pid, nouveau); self.ND += 1
            dec = inst.vacations[nouveau].jour - jr; self.SDec += dec; self.nb_deplaces += (dec != 0)

    # -- termes bruts ----------------------------------------------------------

    def termes_groupe(self) -> dict:
        s, inst = self.sol, self.inst
        return {"lits": s.variance_lits() / inst.capacite_lits ** 2,
                "places": s.variance_places() / inst.capacite_places_jour ** 2,
                "remplissage": s.variance_remplissage(),
                "delai": self.SD / (self.ND * inst.nb_jours) if self.ND else 0.0}

    def termes_cible(self) -> dict:
        nE = len(self.E)
        return {"lits": (self.SE + self.ST) / nE,
                "places": self.SEA / max(1, len(self.JA)),
                "bloc": self.perdu / self.tvo_ouv if self.tvo_ouv else 0.0,
                "delai": self.SD / self.ND if self.ND else 0.0}

    def violations(self) -> float:
        """Lits et ambulatoires en surnombre, minutes de dépassement, et un
        patient sans créneau compte comme 60 minutes de dépassement."""
        return self.surch_l + self.surch_p + self.dep + 60 * self.sans_creneau

    def recalibrer_cible(self) -> None:
        """Recalcule la cible à partir des patients ACTUELLEMENT placés (utile
        pour comparer deux plannings sur exactement les mêmes patients)."""
        sol, H = self.sol, self.H
        poids_nuit = [(self.facteur_weekend if self.we[j] else 1.0) * (self.facteur_reduit if self.reduit[j] else 1.0)
                      for j in range(H)]
        total = sum(sol.lits_jour[j] for j in self.E)
        s = sum(poids_nuit[j] for j in self.E)
        self.cible = [total * poids_nuit[j] / s if self.dans_E[j] else 0.0 for j in range(H)]
        self.cibleA = sum(sol.places_jour[j] for j in self.JA) / max(1, len(self.JA))
        self._recalculer()

    def figer_reference(self) -> None:
        """À appeler sur le PLANNING RÉEL : sert à normaliser C1, C2, C3."""
        self.ref = {"groupe": self.termes_groupe(), "cible": self.termes_cible()}

    # -- indicateurs physiques (identiques pour toutes les fonctions coût) ----------

    def indicateurs(self) -> dict:
        sol, inst = self.sol, self.inst
        L = [sol.lits_jour[j] for j in self.E]
        n = len(L)
        moy = sum(L) / n
        we = [sol.lits_jour[j] for j in self.E if self.we[j]]
        sem = [sol.lits_jour[j] for j in self.E if not self.we[j]]
        red = [sol.lits_jour[j] for j in self.E if self.reduit[j]]
        nor = [sol.lits_jour[j] for j in self.E if not self.reduit[j]]
        A = [sol.places_jour[j] for j in self.JA]
        mA = sum(A) / max(1, len(A))
        m = lambda x: sum(x) / len(x) if x else 0.0
        return {
            "lits_moyen": moy,
            "lits_ecart_type": (sum((x - moy) ** 2 for x in L) / n) ** 0.5,
            "lits_pic": max(L),
            "lits_min": min(L),
            "variation_nuit_a_nuit": sum(abs(a - b) for a, b in zip(L[1:], L)) / (n - 1),
            "nuits_hors_2lits_pct": 100 * sum(abs(sol.lits_jour[j] - self.cible[j]) > self.tol for j in self.E) / n,
            "ecart_cible_lits": ((self.SE) / n) ** 0.5,
            "lits_weekend": m(we), "lits_semaine": m(sem),
            "ratio_weekend_semaine": m(we) / m(sem) if m(sem) else 0.0,
            "lits_vacances": m(red), "lits_hors_vacances": m(nor),
            "ambu_ecart_type": (sum((a - mA) ** 2 for a in A) / max(1, len(A))) ** 0.5,
            "ambu_pic": max(A, default=0),
            "bloc_remplissage_pct": 100 * (1 - self.perdu / self.tvo_ouv) if self.tvo_ouv else 0.0,
            "bloc_heures_perdues": self.perdu / 60,
            "vacations_ouvertes": self.n_ouv,
            "decalage_moyen_j": self.SDec / self.ND if self.ND else 0.0,
            "patients_deplaces_pct": 100 * self.nb_deplaces / max(1, len(self.rj.mobiles)),
            "patients_sans_creneau": self.sans_creneau,
            "depassement_min": self.dep,
            "violations": self.violations(),
        }


# ---------------------------------------------------------------------------
# Fonctions coût candidates
# ---------------------------------------------------------------------------

@dataclass
class Poids:
    lits: float = 0.4
    places: float = 0.2
    bloc: float = 0.2
    delai: float = 0.2


class CoutGroupe:
    """C0 : CoutTotal() du modele.py (poids 1, alpha 1), calculé
    incrémentalement. Dépassements comptés deux fois comme dans le modèle
    (capacité totale + capacité du programmé, identiques sans réserve)."""
    nom = "C0 groupe (actuelle)"

    def __call__(self, ev):
        t = ev.termes_groupe()
        return sum(t.values()) + PENALITE * 2 * ev.violations()

    def termes(self, ev):
        return ev.termes_groupe()


class CoutGroupeNormalise:
    nom = "C1 groupe normalisée"

    def __init__(self, poids: Poids | None = None):
        self.p = poids or Poids()

    def termes(self, ev):
        t, r = ev.termes_groupe(), ev.ref["groupe"]
        return {k: t[k] / r[k] if r[k] else 0.0 for k in t}

    def __call__(self, ev):
        t = self.termes(ev)
        return (self.p.lits * t["lits"] + self.p.places * t["places"] + self.p.bloc * t["remplissage"]
                + self.p.delai * t["delai"] + PENALITE * ev.violations())


class CoutCibleLits:
    nom = "C2 cible lits"

    def __init__(self, poids: Poids | None = None, nom: str | None = None):
        self.p = poids or Poids()
        if nom:
            self.nom = nom

    def termes(self, ev):
        t, r = ev.termes_cible(), ev.ref["cible"]
        return {k: t[k] / r[k] if r[k] else 0.0 for k in t}

    def __call__(self, ev):
        t = self.termes(ev)
        return (self.p.lits * t["lits"] + self.p.places * t["places"] + self.p.bloc * t["bloc"]
                + self.p.delai * t["delai"] + PENALITE * ev.violations())


def candidates() -> list:
    return [CoutGroupe(), CoutGroupeNormalise(), CoutCibleLits(),
            CoutCibleLits(Poids(1, 0, 0, 0), nom="C3 cible lits seule")]


def verifier_coherence(ev) -> float:
    """Écart entre le calcul incrémental et le CoutTotal du modele.py."""
    from modele import CoutTotal
    return abs(CoutGroupe()(ev) - CoutTotal()(ev.sol))
