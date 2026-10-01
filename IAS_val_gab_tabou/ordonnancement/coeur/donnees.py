r"""
donnees.py — Lecture des fichiers de l'hôpital et construction d'une instance.

DEUX SOURCES
------------
1. « donnees bloc anonyme pour centrale 2026.xlsx » : 14 649 interventions
   2019-2022. Sert à deux choses : calibrer l'estimateur de durées, et
   fabriquer une file d'attente réaliste pour tester l'algorithme.

2. « Vacations Opératoires anonymisées.xlsx » : la grille des vacations. Elle
   est en format *visuel* (cellules fusionnées, texte libre, un tableau par
   jour), donc illisible par programme de façon fiable. Elle est ici
   RETRANSCRITE à la main dans `GRILLE`, ce qui a trois avantages : c'est
   relisible par le cadre de bloc, c'est versionnable, et ça résiste à une
   remise en forme du fichier Excel.

NETTOYAGE (à faire, et à documenter — c'est la moitié du travail)
-----------------------------------------------------------------
  - 172 lignes ont un TROS nul ou négatif (heure de sortie <= heure d'entrée) :
    saisies incomplètes, on les écarte du calcul des statistiques.

  - ATTENTION, PIÈGE : la colonne « Durée Séjour en jour (1 pour ambu) » est
    INUTILISABLE. Sa corrélation avec la durée de séjour réellement observée
    (Date Sortie - Date Entrée) est de 0,04, c'est-à-dire nulle. Concrètement,
    la toute première ligne du fichier porte « 1 » pour un patient entré le
    01/01 et sorti le 07/01, et « 1 » aussi pour un vrai ambulatoire entré et
    sorti le 02/01. 32 valeurs de cette colonne sont même des numéros de série
    Excel (jusqu'à 2 458 502) : une date est tombée dans une colonne de durée.

    On reconstruit donc le séjour à partir des dates :
        nuits = Date Sortie - Date Inter
    et on vérifie que le résultat est cliniquement cohérent — c'est le test
    qui compte :
        Varices              0 nuit  (100 % des cas)
        Canal carpien        0 nuit  (97 %)
        Arthroscopie genou   0 nuit  (82 %)
        Prothèse de hanche   4 nuits (médiane)
        Prothèse de genou    5 nuits (médiane)
        Rachis lombaire cx   3 nuits (médiane)
    Ces valeurs sont conformes à la pratique. Celles de la colonne « Durée
    Séjour » ne l'étaient pas. Moralité : ne jamais faire confiance à une
    colonne de durée sans la recalculer depuis les dates.

  - On prend Date Inter et non Date Entrée comme origine : 92 % des patients
    sont admis le jour même de l'intervention, et notre modèle fait commencer
    l'occupation du lit à la sortie de salle.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta

from modele import Instance, Medecin, Patient, Vacation
from estimation import EstimateurDuree, Valideur, estimer, valideur_auto

# ---------------------------------------------------------------------------
# 1. La grille de vacations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MotifVacation:
    """Une ligne de la grille, sous forme exploitable.

    `jour`      0 = lundi ... 4 = vendredi
    `debut/fin` en heures décimales (7.75 = 7h45)
    `semaines`  indices de semaine, modulo 4, où le motif s'applique.

    POURQUOI MODULO 4 ET PAS 2. Votre grille distingue « semaine paire » et
    « semaine impaire », ce qui suggère un cycle de 2. Mais plusieurs cases
    portent « 1 semaine impaire sur 2 » : les semaines impaires elles-mêmes
    alternent entre deux configurations. Le cycle réel est donc de 4 semaines.
        w % 4 == 0 ou 2  ->  semaine paire
        w % 4 == 1       ->  semaine impaire, variante A
        w % 4 == 3       ->  semaine impaire, variante B
    """

    jour: int
    salle: int
    praticien: str
    debut: float
    fin: float
    semaines: tuple[int, ...] = (0, 1, 2, 3)
    etiquette: str = ""


PAIRE = (0, 2)
IMPAIRE_A = (1,)
IMPAIRE_B = (3,)
TOUTES = (0, 1, 2, 3)

# Praticiens hors effectif identifié (vacataires) ou plages réservées : ils
# occupent de la salle mais ne reçoivent pas de patients de la file élective.
RESERVES = {"URGENCES", "LIBRE", "DEVOS", "DS", "TDO/BS", "DN"}

# Alias : dans la grille le même praticien apparaît sous deux écritures.
ALIAS = {"GHREA": "GA"}

GRILLE: list[MotifVacation] = [
    # ---------------- LUNDI ----------------
    MotifVacation(0, 2, "JT",       7.75, 17.5, TOUTES),
    MotifVacation(0, 3, "SR",       8.0,  17.5, PAIRE),
    MotifVacation(0, 3, "GA",       8.0,  13.0, IMPAIRE_A),
    MotifVacation(0, 3, "MT",      13.5,  17.5, IMPAIRE_A),
    MotifVacation(0, 3, "GHREA",    8.0,  10.0, IMPAIRE_B),
    MotifVacation(0, 3, "DEVOS",   10.0,  15.5, IMPAIRE_B),
    MotifVacation(0, 4, "GA",       8.0,  13.0, PAIRE, "2 prothèses, fin 13h"),
    MotifVacation(0, 4, "URGENCES", 13.5, 15.5, PAIRE),
    MotifVacation(0, 4, "TDO/BS",   8.0,  10.0, IMPAIRE_A),
    MotifVacation(0, 4, "DS",      10.0,  15.5, IMPAIRE_A),
    MotifVacation(0, 4, "RL",       8.0,  13.0, IMPAIRE_B),
    MotifVacation(0, 4, "MT",      13.5,  17.5, IMPAIRE_B),
    MotifVacation(0, 5, "FN",       8.0,  15.5, TOUTES),
    # ---------------- MARDI ----------------
    MotifVacation(1, 2, "JT",       7.75, 15.5, PAIRE),
    MotifVacation(1, 2, "JT",       7.75, 17.5, IMPAIRE_A + IMPAIRE_B),
    MotifVacation(1, 3, "TR",       8.0,  13.0, TOUTES),
    MotifVacation(1, 3, "MT",      13.5,  17.5, PAIRE),
    MotifVacation(1, 3, "URGENCES", 13.5, 15.5, IMPAIRE_A + IMPAIRE_B),
    MotifVacation(1, 4, "CT",       8.0,  15.5, TOUTES),
    MotifVacation(1, 5, "MO",       8.0,  17.5, TOUTES),
    # --------------- MERCREDI --------------
    MotifVacation(2, 2, "CL",       7.75, 17.5, TOUTES),
    MotifVacation(2, 3, "DE",       8.0,  17.5, TOUTES),
    MotifVacation(2, 4, "DN",       8.0,  13.0, TOUTES),
    MotifVacation(2, 4, "URGENCES", 13.5, 15.5, TOUTES),
    MotifVacation(2, 5, "MT",       8.0,  13.0, PAIRE),
    MotifVacation(2, 5, "URGENCES", 13.5, 15.5, PAIRE),
    MotifVacation(2, 5, "JE",       8.0,  15.5, IMPAIRE_A + IMPAIRE_B),
    # ---------------- JEUDI ----------------
    MotifVacation(3, 2, "DE",       7.75, 17.5, TOUTES),
    MotifVacation(3, 3, "SR",       8.0,  15.5, TOUTES, "TVO du fichier : 9.5h, "
                                                        "mais 'Stop 15h30' — à trancher"),
    MotifVacation(3, 4, "CT",       8.0,  15.5, TOUTES),
    MotifVacation(3, 5, "LZ",       8.0,  17.5, TOUTES),
    # --------------- VENDREDI --------------
    MotifVacation(4, 2, "CL",       7.75, 13.0, TOUTES),
    MotifVacation(4, 2, "URGENCES", 13.5, 15.5, TOUTES),
    MotifVacation(4, 3, "CU",       8.0,  13.0, TOUTES),
    MotifVacation(4, 3, "MT",      13.5,  15.5, TOUTES),
    MotifVacation(4, 4, "HA",       8.0,  13.0, PAIRE),
    MotifVacation(4, 4, "FP",      13.5,  17.5, PAIRE),
    MotifVacation(4, 4, "SM",       8.0,  17.5, IMPAIRE_A + IMPAIRE_B),
    MotifVacation(4, 5, "LR",       8.0,  13.0, TOUTES),
    MotifVacation(4, 5, "LIBRE",   13.5,  17.5, TOUTES),
]

# La salle 1 n'apparaît dans aucune ligne : elle est fermée dans la version 28
# de la grille. Elle reste disponible comme marge de manœuvre si vous voulez
# tester l'effet d'une ouverture (ajoutez des motifs avec salle=1).


def deployer_grille(inst: Instance, grille=GRILLE, nb_semaines: int = 26,
                    jours_fermes: set[date] | None = None,
                    inclure_reserves: bool = False) -> list[Vacation]:
    """Déroule le motif en vacations DATÉES sur l'horizon.

    On ne raisonne jamais sur le motif lui-même : dès qu'une exception
    apparaît (férié, congé, remplacement), l'arithmétique modulaire devient
    ingérable. On déroule une fois, puis on supprime ou on modifie les objets
    concernés — c'est trivial.
    """
    jours_fermes = jours_fermes or set()
    creees: list[Vacation] = []
    vid = 1
    for semaine in range(nb_semaines):
        for m in grille:
            if semaine % 4 not in m.semaines:
                continue
            praticien = ALIAS.get(m.praticien, m.praticien)
            if praticien in RESERVES and not inclure_reserves:
                continue
            j = semaine * 7 + m.jour
            if j >= inst.nb_jours or inst.date_du_jour(j) in jours_fermes:
                continue
            med_id = code_vers_id(praticien)
            v = Vacation(id=vid, bloc_id=m.salle, med_id=med_id, jour=j,
                         debut=int(round(m.debut * 60)), fin=int(round(m.fin * 60)),
                         etiquette=f"{praticien} {m.etiquette}".strip())
            inst.ajouter_vacation(v)
            creees.append(v)
            vid += 1
    return creees


_CODES: dict[str, int] = {}


def code_vers_id(code: str) -> int:
    """Praticien (code texte) -> identifiant entier stable."""
    if code not in _CODES:
        _CODES[code] = len(_CODES) + 1
    return _CODES[code]


def id_vers_code(ident: int) -> str:
    for code, i in _CODES.items():
        if i == ident:
            return code
    return "?"


# ---------------------------------------------------------------------------
# 2. Lecture de l'historique
# ---------------------------------------------------------------------------


COL_ENTREE = "Heure d'entrée en salle d'opération (calimed)"
COL_SORTIE = "Heure de sortie de salle d'opération (calimed)"
COL_SEJOUR = "DurÈe Sèjour en jour (1 pour ambu)"


def _normaliser(nom: str) -> str:
    """Les en-têtes du fichier sont encodés en latin-1 mal relu (« DurÈe »).
    On compare donc sur une version normalisée sans accents."""
    s = unicodedata.normalize("NFKD", str(nom))
    return "".join(c for c in s if not unicodedata.combining(c)).strip().lower()


def charger_historique(chemin: str):
    """Lit le fichier, calcule le TROS, écarte les lignes inexploitables."""
    import pandas as pd

    df = pd.read_excel(chemin)
    df.columns = [str(c).strip() for c in df.columns]
    corresp = {_normaliser(c): c for c in df.columns}

    c_ent = corresp[_normaliser(COL_ENTREE)]
    c_sor = corresp[_normaliser(COL_SORTIE)]
    c_sej = corresp[_normaliser(COL_SEJOUR)]

    ent = pd.to_datetime(df[c_ent], format="%H:%M:%S", errors="coerce")
    sor = pd.to_datetime(df[c_sor], format="%H:%M:%S", errors="coerce")
    df["tros"] = (sor - ent).dt.total_seconds() / 60

    df["date_inter"] = pd.to_datetime(df["Date Inter"], errors="coerce")
    date_entree = pd.to_datetime(df[corresp[_normaliser("Date EntrÈe")]], errors="coerce")
    date_sortie = pd.to_datetime(df[corresp[_normaliser("Date Sortie")]], errors="coerce")

    # LE point de nettoyage important : on recalcule le séjour depuis les
    # dates au lieu de faire confiance à la colonne dédiée (voir l'en-tête
    # du module).
    df["nuits"] = (date_sortie - df["date_inter"]).dt.days
    df["sejour_declare"] = df[c_sej]
    df["admission_anticipee"] = (df["date_inter"] - date_entree).dt.days

    df["acte"] = df["Interv Type"].fillna("INCONNU").astype(str).str.strip()
    df["praticien"] = df["Praticien"].fillna("?").astype(str).str.strip()

    avant = len(df)
    propre = df[(df["tros"] > 0) & (df["tros"] < 600)
                & df["nuits"].between(0, 60)
                & df["date_inter"].notna()].copy()
    propre.attrs["ecartees"] = avant - len(propre)
    propre.attrs["total_brut"] = avant
    propre.attrs["incoherence_colonne_sejour"] = float(
        propre[["sejour_declare", "nuits"]].corr().iloc[0, 1])
    return propre


def construire_estimateur(df, n_min: int = 5) -> EstimateurDuree:
    historique = zip(df["praticien"], df["acte"], df["tros"])
    return EstimateurDuree(historique, n_min=n_min)


# ---------------------------------------------------------------------------
# 3. Construction d'une instance de test
# ---------------------------------------------------------------------------


def construire_instance(df, estimateur: EstimateurDuree,
                        debut_periode: str = "2022-01-03",
                        nb_semaines: int = 26,
                        capacite_lits: int = 42,
                        capacite_places: int = 12,
                        valideur: Valideur = valideur_auto,
                        facteur_mutualisation: float = 0.45,
                        delai_inscription: tuple[int, int] = (15, 90),
                        politique_fenetre=None,
                        inscriptions_sur: int | None = None,
                        graine: int = 0,
                        grille=None) -> tuple[Instance, dict]:
    """Rejoue une période réelle comme si elle était à planifier.

    On reprend les patients RÉELLEMENT opérés sur la période, on oublie la
    date à laquelle ils l'ont été, et on demande à l'algorithme de la
    retrouver. C'est le protocole de validation le plus honnête dont on
    dispose sans simulateur : la charge de travail est vraie, le case-mix est
    vrai, seule l'affectation est à refaire.

    `delai_inscription` simule la date de consultation : elle n'est pas dans
    le fichier, or sans elle tous les patients seraient disponibles dès J0.

    `inscriptions_sur` change de protocole : au lieu de dériver la date de
    consultation de la date d'opération réelle, on TIRE LES CONSULTATIONS
    UNIFORMÉMENT sur les N premiers jours de l'horizon, et on ne retient que
    les patients réellement opérés pendant cette même période. On obtient un
    flux d'arrivées régulier, sans la saisonnalité ni les à-coups du réel.

    C'est le protocole à utiliser pour observer comment un planning SE
    REMPLIT : avec les vraies dates, la date de consultation est corrélée à la
    date d'opération (on l'a fabriquée à partir d'elle), donc le planning se
    remplit forcément de gauche à droite et on n'apprend rien. Avec des
    arrivées uniformes, la dynamique de remplissage est celle de
    l'algorithme, pas celle des données.

    `politique_fenetre` fabrique la fenêtre de chaque patient. En exploitation
    c'est le chirurgien qui la donne ; en simulation on la tire uniformément
    entre les trois (une chance sur trois chacune), ce qui n'est calé sur
    aucun fichier — le code tourne donc à l'identique sur n'importe quelle
    base de données. Passez votre propre fonction patient -> nom de fenêtre
    pour tester une autre hypothèse de population.
    """
    from propositions import FENETRES, fenetre_uniforme
    politique_fenetre = politique_fenetre or fenetre_uniforme(graine)
    import pandas as pd

    alea = random.Random(graine)
    d0 = pd.Timestamp(debut_periode).date()
    # on cale sur un lundi
    d0 = d0 - timedelta(days=d0.weekday())
    nb_jours = nb_semaines * 7

    inst = Instance(capacite_lits=capacite_lits, capacite_places=capacite_places,
                    jour_zero=d0, nb_jours=nb_jours)

    vacations = deployer_grille(inst, grille=grille if grille is not None else GRILLE,
                                nb_semaines=nb_semaines)
    praticiens_ouverts = {v.med_id for v in vacations}
    for code, ident in _CODES.items():
        inst.ajouter_medecin(Medecin(id=ident, nom=code, specialite="Orthopédie"))

    fenetre_selection = nb_jours if inscriptions_sur is None else inscriptions_sur
    fin = pd.Timestamp(d0 + timedelta(days=fenetre_selection))
    sel = df[(df["date_inter"] >= pd.Timestamp(d0)) & (df["date_inter"] < fin)]

    ignores_praticien = 0
    for i, ligne in enumerate(sel.itertuples(index=False)):
        code = ligne.praticien
        if code not in _CODES or _CODES[code] not in praticiens_ouverts:
            ignores_praticien += 1
            continue
        est = estimer(estimateur, code, ligne.acte, valideur,
                      facteur_mutualisation=facteur_mutualisation)
        jour_reel = (ligne.date_inter.date() - d0).days
        delai = alea.randint(*delai_inscription)
        jour_inscription = (alea.randrange(inscriptions_sur)
                            if inscriptions_sur is not None
                            else max(0, jour_reel - delai))
        nuits = int(ligne.nuits)
        patient = Patient(
            id=i,
            med_id=_CODES[code],
            duree_op=max(5, est.duree),
            marge_perso=est.marge,
            duree_sejour=nuits + 1,          # convention : 1 = ambulatoire
            ambulatoire=(nuits == 0),
            jour_demande=jour_inscription,
            type_interv=ligne.acte,
            priorite=1.0,
            duree_reelle=int(round(ligne.tros)),   # réservé au banc d'essai
        )
        patient.fenetre_jours = FENETRES[politique_fenetre(patient)]
        inst.ajouter_patient(patient)

    inst.indexer()

    diagnostic = {
        "periode": f"{d0} -> {d0 + timedelta(days=nb_jours - 1)}",
        "protocole_inscriptions": ("uniforme sur %d jours" % inscriptions_sur
                                   if inscriptions_sur is not None
                                   else "dérivées des dates réelles"),
        "patients_retenus": len(inst.patients),
        "patients_ignores_praticien_sans_vacation": ignores_praticien,
        "vacations": len(inst.vacations),
        "praticiens_avec_vacation": len(praticiens_ouverts),
        "tvo_total_h": sum(v.tvo for v in inst.vacations.values()) / 60,
        "charge_demandee_h": sum(p.duree_op for p in inst.patients.values()) / 60,
        "ambulatoires": sum(1 for p in inst.patients.values() if p.ambulatoire),
        "fenetres_demandees": {
            nom: sum(1 for p in inst.patients.values()
                     if p.fenetre_jours == jours)
            for nom, jours in FENETRES.items()},
        "part_ambulatoire": (sum(1 for p in inst.patients.values() if p.ambulatoire)
                             / max(1, len(inst.patients))),
        "nuits_totales": sum(p.nb_nuits for p in inst.patients.values()),
    }
    tvo = diagnostic["tvo_total_h"]
    diagnostic["taux_de_charge_theorique"] = (
        diagnostic["charge_demandee_h"] / tvo if tvo else 0.0)
    return inst, diagnostic


def planning_reel(df, inst, debut_periode) -> dict:
    """Indicateurs du planning RÉELLEMENT réalisé, pour servir d'étalon.

    Comparer l'algorithme à rien ne prouve rien. On reconstruit donc
    l'occupation des lits telle qu'elle a eu lieu, et c'est cette courbe
    que le tabou doit aplatir.
    """
    import pandas as pd

    d0 = inst.jour_zero
    fin = pd.Timestamp(d0 + timedelta(days=inst.nb_jours))
    sel = df[(df["date_inter"] >= pd.Timestamp(d0)) & (df["date_inter"] < fin)]

    lits = [0] * inst.nb_jours
    places = [0] * inst.nb_jours
    for ligne in sel.itertuples(index=False):
        j = (ligne.date_inter.date() - d0).days
        nuits = int(ligne.nuits)
        if nuits == 0:
            if 0 <= j < inst.nb_jours:
                places[j] += 1
        else:
            for k in range(j, min(j + nuits, inst.nb_jours)):
                if k >= 0:
                    lits[k] += 1

    import statistics
    ouvres = [j for j in range(inst.nb_jours) if places[j] or lits[j]]
    return {
        "interventions": len(sel),
        "lits_moyen": statistics.fmean(lits),
        "lits_ecart_type": statistics.pstdev(lits),
        "lits_pic": max(lits),
        "lits_creux": min(lits),
        "places_pic": max(places),
        "places_ecart_type": statistics.pstdev([places[j] for j in ouvres]) if ouvres else 0,
        "lits": lits,
        "places": places,
    }


def diagnostic_praticiens(inst) -> list[dict]:
    """Charge demandée vs temps de vacation offert, praticien par praticien.

    À REGARDER AVANT TOUTE OPTIMISATION. Aucun algorithme d'ordonnancement ne
    peut placer un praticien qui a besoin de deux fois son temps de salle : le
    seul résultat possible est une file de patients reportés. Si ce tableau
    montre des taux supérieurs à 100 %, le problème n'est pas l'ordonnancement,
    c'est la RÉPARTITION DES VACATIONS — un autre problème d'optimisation
    (« master surgical schedule »), qu'on résout en déplaçant des plages entre
    praticiens, pas des patients entre jours.

    La charge demandée inclut la marge de risque et le TIS, puisque c'est ce
    qui est réellement consommé dans la vacation.
    """
    from collections import Counter, defaultdict

    demande: dict[int, int] = defaultdict(int)
    nb = Counter()
    for p in inst.patients.values():
        demande[p.med_id] += p.duree_op + p.marge_perso + inst.tis
        nb[p.med_id] += 1

    offert: dict[int, int] = defaultdict(int)
    for v in inst.vacations.values():
        offert[v.med_id] += v.tvo

    lignes = []
    for med_id in offert:
        tvo = offert[med_id]
        lignes.append({
            "praticien": id_vers_code(med_id),
            "med_id": med_id,
            "patients": nb[med_id],
            "demande_h": demande[med_id] / 60,
            "tvo_h": tvo / 60,
            "taux": demande[med_id] / tvo if tvo else float("inf"),
        })
    return sorted(lignes, key=lambda x: -x["taux"])


def afficher_diagnostic_praticiens(inst) -> str:
    lignes = [f"  {'praticien':10} {'patients':>9} {'demandé (h)':>12} "
              f"{'offert (h)':>11} {'taux':>7}"]
    for d in diagnostic_praticiens(inst):
        marque = "  <-- saturé" if d["taux"] > 1.0 else ""
        lignes.append(f"  {d['praticien']:10} {d['patients']:9d} "
                      f"{d['demande_h']:12.0f} {d['tvo_h']:11.0f} "
                      f"{d['taux']:7.0%}{marque}")
    return "\n".join(lignes)
