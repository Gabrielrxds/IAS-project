r"""
urgences.py — Classification de l'urgence chirurgicale à partir du diagnostic
principal (CIM-10), pour décider QUI opérer en premier et dans QUEL DÉLAI.

RÉFÉRENCES
----------
1. FSSA (Federation of Surgical Specialty Associations, Royaume-Uni),
   « Clinical Guide to Surgical Prioritisation », version du 24/08/2020,
   rédigée avec la British Orthopaedic Association. Cinq niveaux :
     P1a  urgence, opérer en moins de 24 h
     P1b  urgence, opérer en moins de 72 h
     P2   moins d'un mois
     P3   moins de trois mois
     P4   plus de trois mois
   et, pour chaque niveau, la liste des situations de traumatologie et
   d'orthopédie (fractures ouvertes, arthrite septique, syndrome des loges,
   fractures instables, ruptures tendineuses, arthroplasties…).
   https://fssa.org.uk/_userfiles/pages/files/covid19/prioritisation_master_240820.pdf
2. NCEPOD (National Confidential Enquiry into Patient Outcome and Death,
   2004) : classification de l'URGENCE D'UNE INTERVENTION en quatre
   catégories — immédiate (minutes), urgente (heures), accélérée
   (« expedited », jours), programmée (« elective »).
   https://www.ncepod.org.uk/classification.html
3. SFAR / SOFCOT / SFGG / SFPC (2017), recommandation R4.1 : opérer une
   fracture de l'extrémité supérieure du fémur (FESF) dans les 48 heures
   suivant l'admission pour réduire la mortalité.

LES CINQ CLASSES DU PROJET
--------------------------
  U1  Urgence             délai cible < 24 h (48 h max pour une FESF)   FSSA P1a  / NCEPOD urgent
  U2  Urgence différée    délai cible < 72 h                           FSSA P1b  / NCEPOD expedited
  P2  Semi-urgent         délai cible < 1 mois                         FSSA P2   / NCEPOD elective
  P3  Programmé prioritaire  < 3 mois                                  FSSA P3   / NCEPOD elective
  P4  Programmé           > 3 mois (dans la fenêtre de 6 mois)         FSSA P4   / NCEPOD elective

LIMITES (À LIRE)
----------------
Le diagnostic seul ne suffit pas toujours : la FSSA demande de RÉÉVALUER
cliniquement (déficit neurologique, douleur, aggravation, ASA…). Exemples :
un canal carpien est P4, mais P2 s'il y a une amyotrophie ; une hernie
discale est P3, mais U1 s'il y a un syndrome de la queue de cheval. Ce module
donne une classe PAR DÉFAUT, que le chirurgien peut toujours surclasser.
La classification doit être validée par le chirurgien référent du projet.
"""

from __future__ import annotations

import pandas as pd

CLASSES = {
    "U1": {"libelle": "Urgence", "delai_cible_jours": 1, "delai_max_jours": 2, "fssa": "P1a",
           "ncepod": "urgent", "rang": 1},
    "U2": {"libelle": "Urgence différée", "delai_cible_jours": 3, "delai_max_jours": 3, "fssa": "P1b",
           "ncepod": "expedited", "rang": 2},
    "P2": {"libelle": "Semi-urgent", "delai_cible_jours": 30, "delai_max_jours": 30, "fssa": "P2",
           "ncepod": "elective", "rang": 3},
    "P3": {"libelle": "Programmé prioritaire", "delai_cible_jours": 90, "delai_max_jours": 90, "fssa": "P3",
           "ncepod": "elective", "rang": 4},
    "P4": {"libelle": "Programmé", "delai_cible_jours": 182, "delai_max_jours": 182, "fssa": "P4",
           "ncepod": "elective", "rang": 5},
}

# (préfixe CIM-10, classe, motif, référence) — la règle au préfixe le plus LONG l'emporte
REGLES = [
    # --- infections et urgences vitales ou fonctionnelles -------------------------------
    ("M00", "U1", "arthrite septique (articulation native)", "FSSA P1a : septic arthritis"),
    ("T84.5", "U1", "infection de prothèse articulaire", "FSSA P1a : septic arthritis (prosthetic joint)"),
    ("T84.6", "U1", "infection de matériel d'ostéosynthèse", "FSSA P1a : infection - other metalwork"),
    ("T84.7", "U1", "infection d'un autre dispositif orthopédique", "FSSA P1a : infection - other metalwork"),
    ("T81.4", "U1", "infection du site opératoire", "FSSA P1a : soft tissue infection not responding"),
    ("T79.6", "U1", "syndrome des loges traumatique", "FSSA P1a : compartment syndrome"),
    ("M72.6", "U1", "fasciite nécrosante", "FSSA P1a : necrotising fasciitis"),
    ("G83.4", "U1", "syndrome de la queue de cheval", "FSSA P1a : cauda equina syndrome"),
    ("S68", "U1", "amputation traumatique (doigt, main)", "FSSA P1a : re-implantation / revascularisation"),
    ("S65", "U1", "plaie vasculaire du poignet ou de la main", "FSSA P1a : vascular injury"),
    # --- fractures du fémur proximal (FESF) et de la diaphyse fémorale ---------------------
    ("S72.0", "U1", "fracture du col du fémur", "FSSA P1a : hip fracture ; SFAR/SOFCOT : < 48 h"),
    ("S72.1", "U1", "fracture pertrochantérienne", "FSSA P1a : hip fracture ; SFAR/SOFCOT : < 48 h"),
    ("S72.2", "U1", "fracture sous-trochantérienne", "FSSA P1a : hip fracture ; SFAR/SOFCOT : < 48 h"),
    ("S72.3", "U1", "fracture de la diaphyse fémorale", "FSSA P1a : femoral shaft"),
    ("S72", "U2", "autre fracture du fémur", "FSSA P1b : unstable articular / lower limb fractures"),
    # --- fractures instables, péri-prothétiques, pathologiques ---------------------------
    ("M96.6", "U2", "fracture sur prothèse ou matériel", "FSSA P1b : peri-prosthetic fractures"),
    ("M84.4", "U2", "fracture pathologique", "FSSA P1b : pathological fractures"),
    ("S82.1", "U2", "fracture du plateau tibial", "FSSA P1b : unstable articular fractures"),
    ("S82.2", "U2", "fracture de la diaphyse tibiale", "FSSA P1b : tibial fracture displaced/unstable"),
    ("S82.3", "U2", "fracture du pilon tibial", "FSSA P1b : unstable articular fractures"),
    ("S82.8", "U2", "fracture bi/trimalléolaire ou complexe de jambe", "FSSA P1b : unstable articular fractures"),
    ("S82.4", "U2", "fracture du péroné isolée", "FSSA P1b : lower limb fractures requiring fixation"),
    ("S82.0", "P2", "fracture de la rotule", "FSSA P2 : knee extensor disruption (fractured patella)"),
    ("S82.5", "P2", "fracture de la malléole interne", "FSSA P2 : displaced intra-articular fracture, ankle"),
    ("S82.6", "P2", "fracture de la malléole externe", "FSSA P2 : displaced intra-articular fracture, ankle"),
    ("S82", "U2", "autre fracture de la jambe", "FSSA P1b"),
    ("S42.2", "U2", "fracture de l'humérus proximal", "FSSA P1b : unstable articular fractures"),
    ("S42.3", "U2", "fracture de la diaphyse humérale", "FSSA P1b : long bone fracture"),
    ("S42.4", "U2", "fracture de l'humérus distal", "FSSA P1b : unstable articular fractures"),
    ("S42.0", "P2", "fracture de la clavicule", "FSSA P2 : fractures NOS"),
    ("S42", "P2", "autre fracture de l'épaule ou du bras", "FSSA P2 : fractures NOS"),
    ("S52.0", "P2", "fracture de l'olécrane", "FSSA P2 : olecranon"),
    ("S52.5", "U2", "fracture du radius distal (poignet)", "FSSA P1b : unstable articular fractures"),
    ("S52.6", "U2", "fracture radius + ulna distaux", "FSSA P1b : unstable articular fractures"),
    ("S52", "U2", "fracture de l'avant-bras", "FSSA P1b : unstable fractures (forearm)"),
    ("S32", "U2", "fracture vertébrale lombaire ou du bassin", "FSSA P1b : spinal trauma requiring stabilisation"),
    ("S22", "U2", "fracture vertébrale thoracique", "FSSA P1b : spinal trauma requiring stabilisation"),
    ("S12", "U2", "fracture vertébrale cervicale", "FSSA P1b : spinal trauma requiring stabilisation"),
    ("S62", "P2", "fracture du poignet ou de la main (carpe, métacarpe, phalanges)", "FSSA P2 : fractures NOS"),
    ("S92", "P2", "fracture du pied", "FSSA P2 : displaced intra-articular fracture, foot"),
    # --- plaies, tendons, nerfs ------------------------------------------------------------
    ("S66", "U2", "plaie d'un tendon de la main ou du poignet", "FSSA P1b : primary tendon/nerve repair"),
    ("S64", "U2", "plaie d'un nerf de la main ou du poignet", "FSSA P1b : primary tendon/nerve repair"),
    ("S54", "U2", "plaie d'un nerf de l'avant-bras", "FSSA P1b : primary nerve repair"),
    ("S94", "U2", "plaie d'un nerf de la cheville ou du pied", "FSSA P1b : primary nerve repair"),
    ("S61", "U2", "plaie du poignet ou de la main", "FSSA P1b : finger tip/nail bed repair, wounds"),
    ("S91", "U2", "plaie de la cheville ou du pied", "FSSA P1b : delayed primary closure of wounds"),
    ("S69", "U2", "autre traumatisme du poignet ou de la main", "FSSA P1b"),
    ("S86", "P2", "rupture du tendon d'Achille ou d'un tendon de jambe", "FSSA P2 : tendon rupture - any site"),
    ("S76", "P2", "rupture du tendon quadricipital", "FSSA P2 : knee extensor disruption"),
    ("S46", "P2", "rupture traumatique de la coiffe ou d'un tendon de l'épaule", "FSSA P2 : tendon rupture"),
    ("S96", "P2", "lésion d'un tendon de la cheville ou du pied", "FSSA P2 : tendon rupture"),
    ("S43", "P2", "luxation ou entorse grave de l'épaule ou de l'acromio-claviculaire", "FSSA P2"),
    ("S63", "P2", "lésion ligamentaire ou luxation du poignet ou des doigts", "FSSA P2"),
    ("S93", "P2", "luxation ou entorse grave de la cheville", "FSSA P2"),
    ("S83", "P3", "lésion du ménisque ou des ligaments du genou", "FSSA P3 : locked knee - ACL / other reconstruction"),
    # --- autres infections et complications --------------------------------------------------
    ("M86", "U2", "ostéomyélite", "FSSA P1b : soft tissue / bone infection not responding"),
    ("M46.3", "U2", "infection discale (spondylodiscite)", "FSSA P1b : infection"),
    ("M65.0", "U2", "abcès de la gaine d'un tendon (phlegmon)", "FSSA P1b : soft tissue infection"),
    ("L02", "U2", "abcès cutané, furoncle", "FSSA P1b : soft tissue infection not responding"),
    ("L03", "U2", "phlegmon, panaris", "FSSA P1b : soft tissue infection not responding"),
    ("T81.0", "U2", "hématome ou hémorragie postopératoire", "FSSA P1b : evacuation of haematoma"),
    ("T81.3", "U2", "désunion de plaie opératoire", "FSSA P1b : repair wound dehiscence"),
    ("T84.1", "P2", "complication mécanique du matériel d'ostéosynthèse", "FSSA P2 / P1b si matériel exposé"),
    ("T84.0", "P3", "complication mécanique d'une prothèse (descellement…)", "FSSA P3 : revision for loosening"),
    ("T84", "P3", "autre complication de prothèse ou de matériel", "FSSA P3 : revision surgery"),
    ("T81", "P3", "autre complication postopératoire", "à réévaluer"),
    ("T85", "P3", "complication d'un autre dispositif", "à réévaluer"),
    ("T87", "P3", "complication d'un moignon d'amputation", "à réévaluer"),
    ("M96", "P3", "complication après chirurgie orthopédique (pseudarthrose…)", "FSSA P3 : revision surgery"),
    # --- tumeurs ---------------------------------------------------------------------------------
    ("C44", "P4", "carcinome cutané (basocellulaire…)", "FSSA P4 : basal cell carcinoma"),
    ("C", "P2", "tumeur maligne (sarcome, mélanome…)", "FSSA P2 : MDT directed sarcoma / skin cancer surgery"),
    ("D48", "P3", "tumeur à évolution imprévisible", "FSSA P3 : benign bone/soft tissue lesion biopsy"),
    ("D", "P4", "tumeur bénigne", "FSSA P4 : excision of benign lesions"),
    # --- rachis -------------------------------------------------------------------------------------
    ("M50.0", "P2", "hernie discale cervicale avec myélopathie", "FSSA P2 : progressive neurological deficit"),
    ("M50", "P3", "hernie discale cervicale", "FSSA P3 : intractable radiculopathy"),
    ("M51", "P3", "hernie discale lombaire (radiculopathie)", "FSSA P3 : intractable radiculopathy"),
    ("M54.1", "P3", "radiculopathie", "FSSA P3 : intractable radiculopathy"),
    ("M54.3", "P3", "sciatique", "FSSA P3 : intractable radiculopathy"),
    ("M54.4", "P3", "lombosciatique", "FSSA P3 : intractable radiculopathy"),
    ("M54", "P4", "lombalgie, dorsalgie", "FSSA P4 : degenerative spinal disease"),
    ("M48", "P4", "canal lombaire étroit", "FSSA P4 : degenerative spinal disease, no deficit"),
    ("M43", "P4", "spondylolisthésis, autres déformations", "FSSA P4 : degenerative / deformity"),
    ("M47", "P4", "arthrose rachidienne", "FSSA P4"),
    ("M53", "P4", "autre dorsopathie", "FSSA P4"),
    ("M99", "P4", "lésion biomécanique du rachis", "FSSA P4"),
    # --- genou, épaule, instabilités ----------------------------------------------------------
    ("M23.4", "P3", "corps étranger articulaire du genou", "FSSA P3 : arthroscopic removal of loose body"),
    ("M23.5", "P3", "instabilité chronique du genou", "FSSA P3 : recurrent joint instability"),
    ("M23", "P4", "lésion méniscale dégénérative ou kyste", "FSSA P4 : NOS"),
    ("M24.4", "P3", "luxation récidivante", "FSSA P3 : recurrent joint instability"),
    ("M24.2", "P3", "laxité ligamentaire (instabilité)", "FSSA P3 : recurrent joint instability"),
    ("M24", "P4", "autre affection articulaire (raideur…)", "FSSA P4"),
    ("M22.0", "P3", "luxation récidivante de la rotule", "FSSA P3 : recurrent joint instability"),
    ("M22", "P4", "autre affection de la rotule", "FSSA P4"),
    ("M75.0", "P3", "capsulite rétractile", "FSSA P3 : frozen shoulder"),
    ("M75", "P4", "lésion de la coiffe des rotateurs (dégénérative)", "FSSA P4 : upper limb NOS"),
    ("M87", "P3", "ostéonécrose", "FSSA P3 : hip avascular necrosis"),
    ("M84.1", "P3", "pseudarthrose", "FSSA P3 : revision / non-union"),
    ("M84", "P4", "cal vicieux, autre trouble de la consolidation", "FSSA P4"),
    ("M93", "P3", "ostéochondrite", "FSSA P2-P3 : osteochondral defect"),
    # --- programmé (arthrose, main, pied, varices…) --------------------------------------
    ("M16", "P4", "coxarthrose", "FSSA P4 : arthroplasty NOS"),
    ("M17", "P4", "gonarthrose", "FSSA P4 : arthroplasty NOS"),
    ("M15", "P4", "polyarthrose", "FSSA P4 : arthroplasty NOS"),
    ("M18", "P4", "rhizarthrose", "FSSA P4 : hand NOS"),
    ("M19", "P4", "autre arthrose", "FSSA P4 : arthroplasty NOS"),
    ("G56", "P4", "canal carpien ou autre compression nerveuse du membre supérieur",
     "FSSA P4 (P2 si amyotrophie ou déficit)"),
    ("G57", "P4", "compression nerveuse du membre inférieur", "FSSA P4 (P2 si déficit)"),
    ("M65", "P4", "doigt à ressaut, ténosynovite", "FSSA P4 : hand NOS"),
    ("M72", "P4", "maladie de Dupuytren", "FSSA P4 : hand NOS"),
    ("M20", "P4", "hallux valgus, déformation des orteils", "FSSA P4"),
    ("M21", "P4", "déformation acquise des membres", "FSSA P4"),
    ("Z47", "P4", "ablation de matériel d'ostéosynthèse", "FSSA P4 : metalwork removal"),
    ("Z45", "P3", "remplacement d'un dispositif implanté (neurostimulateur)", "FSSA P1b si panne, sinon P3"),
    ("I83", "P4", "varices des membres inférieurs", "FSSA P4 : vein surgery"),
    ("L60", "P4", "ongle incarné", "FSSA P4"),
    ("Q", "P4", "malformation congénitale", "FSSA P4 : corrective surgery"),
    ("K40", "P4", "hernie inguinale", "FSSA P4 : uncomplicated hernia"),
]

# classe par défaut d'un diagnostic sans règle, selon le chapitre CIM-10
DEFAUT_CHAPITRE = {"S": ("U2", "traumatisme sans règle précise : à réévaluer"),
                   "T": ("P2", "complication / traumatisme sans règle précise : à réévaluer")}
DEFAUT = ("P4", "pathologie chronique : programmé par défaut")

_REGLES_TRIEES = sorted(REGLES, key=lambda r: -len(r[0]))


def classer(code: str | None) -> dict:
    """Classe d'urgence d'UN diagnostic CIM-10 (ex. « S72.10 »)."""
    if not isinstance(code, str) or not code.strip():
        return {"classe_urgence": "P4", "motif_urgence": "diagnostic manquant", "reference_urgence": "", "regle": ""}
    c = code.strip().upper()
    # fracture OUVERTE : 5e caractère « 1 » dans la CIM-10 (ex. S82.21)
    if c.startswith("S") and c[1:3].isdigit() and int(c[1:3]) in {2, 12, 22, 32, 42, 52, 62, 72, 82, 92} \
            and len(c) >= 6 and c[5] == "1":
        return {"classe_urgence": "U1", "motif_urgence": "fracture ouverte", "reference_urgence": "FSSA P1a : open fractures",
                "regle": "fracture ouverte"}
    for prefixe, classe, motif, ref in _REGLES_TRIEES:
        if c.startswith(prefixe):
            return {"classe_urgence": classe, "motif_urgence": motif, "reference_urgence": ref, "regle": prefixe}
    classe, motif = DEFAUT_CHAPITRE.get(c[0], DEFAUT)
    return {"classe_urgence": classe, "motif_urgence": motif, "reference_urgence": "règle par défaut", "regle": "défaut"}


def classer_df(df: pd.DataFrame, colonne_diag: str = "diag_principal") -> pd.DataFrame:
    """Ajoute à une base les colonnes de classification et de priorité."""
    r = pd.DataFrame([classer(x) for x in df[colonne_diag]], index=df.index)
    out = pd.DataFrame(index=df.index)
    out["classe_urgence"] = pd.Categorical(r.classe_urgence, categories=list(CLASSES), ordered=True)
    out["classe_urgence_libelle"] = r.classe_urgence.map(lambda k: CLASSES[k]["libelle"])
    out["motif_urgence"] = r.motif_urgence
    out["reference_urgence"] = r.reference_urgence
    out["regle_urgence"] = r.regle
    out["delai_max_jours"] = r.classe_urgence.map(lambda k: CLASSES[k]["delai_max_jours"])
    out["urgence"] = r.classe_urgence.isin(["U1", "U2"])
    out["score_priorite"] = score_priorite(df, r.classe_urgence)
    return out


def score_priorite(df: pd.DataFrame, classes: pd.Series) -> pd.Series:
    """Ordre de passage. Plus le score est HAUT, plus le patient passe tôt.
    1. la classe d'urgence (U1 > U2 > P2 > P3 > P4) domine tout ;
    2. à classe égale, des facteurs inspirés de la matrice de repriorisation
       de la FSSA (risque lié à l'âge et aux comorbidités) départagent.
    Le temps déjà attendu, autre critère de la FSSA, s'ajoute au moment de
    programmer (+1 point par semaine d'attente, voir `ordre_de_passage`)."""
    base = classes.map(lambda k: (6 - CLASSES[k]["rang"]) * 100)
    age = df.get("age", pd.Series(0, index=df.index)).fillna(0)
    pts = (age >= 75) * 10 + ((age >= 65) & (age < 75)) * 5
    for col, p in (("comorbidite_anticoagulant", 5), ("comorbidite_diabete", 3), ("comorbidite_hypertension", 2),
                   ("comorbidite_obesite", 2)):
        if col in df:
            pts = pts + df[col].fillna(False).astype(int) * p
    return (base + pts).astype(int)


def ordre_de_passage(file: pd.DataFrame, aujourd_hui: pd.Timestamp, colonne_demande: str = "date_demande") -> pd.DataFrame:
    """Trie une file d'attente : score de priorité + 1 point par semaine
    d'attente ; indique la date limite d'opération et les retards."""
    f = file.copy()
    attente = (aujourd_hui - f[colonne_demande]).dt.days.clip(lower=0)
    f["attente_jours"] = attente
    f["score_avec_attente"] = f.score_priorite + attente // 7
    f["date_limite"] = f[colonne_demande] + pd.to_timedelta(f.delai_max_jours, unit="D")
    f["en_retard"] = f.date_limite < aujourd_hui
    return f.sort_values(["en_retard", "score_avec_attente"], ascending=[False, False])


def tableau_classes() -> pd.DataFrame:
    return pd.DataFrame([{"classe": k, **v} for k, v in CLASSES.items()]).set_index("classe")


def tableau_regles() -> pd.DataFrame:
    return pd.DataFrame(REGLES, columns=["prefixe_cim10", "classe", "motif", "reference"])
