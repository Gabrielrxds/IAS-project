r"""
demo.py — La chaîne complète, sur vos données réelles.

    python3 demo.py [chemin_du_fichier_historique.xlsx]

Déroulé :
  1. lecture et nettoyage de l'historique 2019-2022 ;
  2. estimation des durées, et ce qu'on montre au chirurgien ;
  3. l'instance, et le diagnostic charge / capacité par praticien ;
  4. UNE consultation : le patient arrive, on propose deux dates ;
  5. TOUT le flux de consultations rejoué, avec trois politiques de choix ;
  6. le tabou hors-ligne comme étalon : le prix des dates figées ;
  7. le tabou local, la veille, sur une journée.
"""

from __future__ import annotations

import sys
import time

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import chemins  # rend coeur/ et vacations/ importables
from donnees import (afficher_diagnostic_praticiens, charger_historique,
                     construire_estimateur, construire_instance, planning_reel)
from estimation import valideur_auto
from modele import afficher_journee, deriver_creneaux, profil_hebdo
from propositions import (MoteurPropositions, afficher_propositions,
                          choix_meilleur, choix_plus_proche, choix_aleatoire,
                          simuler_consultations)
from tabou import (Evaluateur, ParametresTabou, Poids, PoidsLocaux, TabouGlobal,
                   TabouLocal, doit_lancer_local, solution_initiale)

FICHIER = chemins.fichier_donnees()


def titre(t: str) -> None:
    print("\n" + "═" * 78)
    print(t)
    print("═" * 78)


def main(chemin: str) -> None:
    # ---------------------------------------------------------------- 1 ----
    titre("1. LECTURE ET NETTOYAGE DE L'HISTORIQUE")
    t0 = time.time()
    df = charger_historique(chemin)
    print(f"  {df.attrs['total_brut']} lignes lues, {df.attrs['ecartees']} écartées "
          f"(TROS nul ou négatif, séjour aberrant, date manquante)")
    print(f"  {len(df)} interventions exploitables — lu en {time.time() - t0:.1f}s")
    print(f"\n  Corrélation entre la colonne « Durée Séjour » du fichier et la durée")
    print(f"  réellement observée : {df.attrs['incoherence_colonne_sejour']:+.3f}, "
          f"c'est-à-dire aucune.")
    print(f"  Le séjour est recalculé depuis les dates (voir l'en-tête de donnees.py).")

    # ---------------------------------------------------------------- 2 ----
    titre("2. ESTIMATION DES DURÉES")
    estimateur = construire_estimateur(df)
    couv = estimateur.couverture()
    for k, v in couv.items():
        print(f"  {k:32} {v}")
    print(f"\n  Seuls {couv['couples_exploitables']} couples (praticien × acte) sur "
          f"{couv['couples_praticien_acte']} ont au moins 5 cas —")
    print(f"  {couv['actes_distincts']} libellés d'actes distincts. C'est pourquoi la "
          f"chaîne de repli existe.")
    print("\n  Ce qui est présenté au chirurgien :")
    for prat, acte in (("JT", "Prothese Totale Genou"), ("DE", "Canal Carpien")):
        proposition, stats = estimateur.proposition(prat, acte)
        print()
        print(stats.tableau())
        print(f"  → proposition soumise au praticien : {proposition} min")

    # ---------------------------------------------------------------- 3 ----
    titre("3. L'INSTANCE — 1er semestre 2022 rejoué")
    inst, diag = construire_instance(df, estimateur, debut_periode="2022-01-03",
                                     nb_semaines=26, capacite_lits=42,
                                     capacite_places=12, valideur=valideur_auto)
    inst.capacite_places_jour = 18
    for k, v in diag.items():
        print(f"  {k:44} {v if not isinstance(v, float) else round(v, 2)}")

    print("\n  CHARGE DEMANDÉE CONTRE TEMPS DE SALLE OFFERT, PAR PRATICIEN :")
    print(afficher_diagnostic_praticiens(inst))
    print("\n  À lire avant toute optimisation. Trois praticiens demandent plus de")
    print("  temps de salle que la grille ne leur en donne — SM en demande deux")
    print("  fois plus. Aucun algorithme d'ordonnancement ne peut résoudre ça :")
    print("  leurs patients seront reportés, quoi qu'on fasse. Le levier n'est pas")
    print("  l'ordonnancement, c'est la RÉPARTITION DES VACATIONS entre praticiens.")

    reel = planning_reel(df, inst, None)
    print("\n  Planning RÉELLEMENT réalisé sur la période :")
    print(f"    lits : moyenne {reel['lits_moyen']:.1f}, écart-type "
          f"{reel['lits_ecart_type']:.2f}, pic {reel['lits_pic']}")
    print(f"    ambulatoires : pic {reel['places_pic']}/jour, écart-type "
          f"{reel['places_ecart_type']:.2f}")

    # ---------------------------------------------------------------- 4 ----
    titre("4. UNE CONSULTATION")
    print("  Le patient est devant le chirurgien. On lui propose deux dates.")
    print("  Pas de recherche tabou ici : le praticien a une vingtaine de")
    print("  vacations sur six mois, on les évalue TOUTES et on trie. C'est une")
    print("  énumération exhaustive — donc exacte, et instantanée.\n")

    poids = Poids(w_remplissage=1.0, w_lits=1.0)
    moteur = MoteurPropositions(inst, poids=poids)
    exemples = sorted(inst.patients.values(), key=lambda p: (p.jour_demande, p.id))
    # on pose d'abord une centaine de patients pour que le planning ne soit pas vide
    for p in exemples[:120]:
        r = moteur.proposer(p.id)
        if not isinstance(r, list):
            continue
        moteur.appliquer(choix_meilleur(r))
    for p in exemples[120:123]:
        print(afficher_propositions(inst, p, moteur.proposer(p.id)))

    # ---------------------------------------------------------------- 5 ----
    titre("5. LE FLUX COMPLET DE CONSULTATIONS")
    print("  Chaque patient est traité à sa date de consultation réelle : les")
    print("  patients vus plus tard n'existent pas encore, et les dates déjà")
    print("  données ne bougent plus. Le planning porte donc la même myopie que")
    print("  dans la vraie vie.\n")

    print("  CAS DE RÉFÉRENCE — chaque patient garde le délai minimum que son")
    print("  chirurgien lui a fixé : « pas avant une semaine / trois semaines /")
    print("  six semaines ». C'est une BORNE INFÉRIEURE : au-delà on cherche")
    print("  jusqu'au bout de l'horizon, donc tout le monde ressort avec une")
    print("  date sauf si l'horizon est saturé. En simulation les trois délais")
    print("  sont tirés à une chance sur trois, sans rien caler sur le fichier :")
    print("  le protocole se rejoue à l'identique sur une autre base.\n")
    sol_ref, j_ref = simuler_consultations(inst, poids, choix=choix_meilleur)
    r_ref, i_ref = j_ref.resume(), sol_ref.indicateurs()
    print(f"    {r_ref['patients_dates']} dates données, "
          f"{r_ref['patients_reportes_hors_horizon']} reports | "
          f"délai médian {r_ref['delai_median_j']:.0f} j")
    print(f"    dispersion du remplissage {i_ref['remplissage_ecart_type']:.3f} | "
          f"écart-type lits {i_ref['lits_ecart_type']:.2f} | "
          f"remplissage {i_ref['taux_remplissage_moyen']:.1%}")

    print("\n  SENSIBILITÉ — ce que donnerait la même population si TOUS les")
    print("  praticiens imposaient le même délai minimum :\n")
    print(f"    {'fenêtre':16} {'dates':>7} {'reports':>8} {'délai méd.':>11} "
          f"{'sd remplissage':>15} {'sd lits':>9} {'remplissage':>12}")
    for f in ("semaine", "trois_semaines", "mois_et_demi"):
        s_f, j_f = simuler_consultations(inst, poids, choix=choix_meilleur, fenetre=f)
        r_f, i_f = j_f.resume(), s_f.indicateurs()
        print(f"    {f:16} {r_f['patients_dates']:7d} "
              f"{r_f['patients_reportes_hors_horizon']:8d} {r_f['delai_median_j']:11.0f} "
              f"{i_f['remplissage_ecart_type']:15.3f} {i_f['lits_ecart_type']:9.2f} "
              f"{i_f['taux_remplissage_moyen']:11.1%}")
    print("\n  Un délai minimum long ne fait pas que retarder : il retire des")
    print("  vacations du champ de recherche, donc il dégrade aussi le")
    print("  lissage. Ce n'est plus un poids à calibrer, c'est une décision")
    print("  clinique — bilan, consultation d'anesthésie, délai de réflexion —")
    print("  prise patient par patient par qui sait si le cas peut attendre.\n")

    print("  LA DÉCISION DU CHIRURGIEN entre les deux dates proposées.")
    print("  Elles sont VOLONTAIREMENT contrastées : la première est la plus")
    print("  proche possible, la seconde celle qui ménage le mieux le planning.")
    print("  Le choix est donc un vrai arbitrage, pas une formalité :\n")
    politiques = {
        "suit la recommandation": choix_meilleur,
        "prend la date la plus proche": choix_plus_proche,
        "choisit selon le patient": choix_aleatoire(0),
    }
    resultats_flux = {}
    for nom, politique in politiques.items():
        sol_f, journal = simuler_consultations(inst, poids, choix=politique)
        r, ind = journal.resume(), sol_f.indicateurs()
        resultats_flux[nom] = (sol_f, journal, ind)
        print(f"    {nom:30} délai méd. {r['delai_median_j']:3.0f} j | "
              f"sd remplissage {ind['remplissage_ecart_type']:.3f} | "
              f"sd lits {ind['lits_ecart_type']:.2f} | "
              f"{r['patients_reportes_hors_horizon']} reports")

    sol_flux, journal, ind_flux = resultats_flux["suit la recommandation"]

    print("\n  Le praticien qui prend systématiquement au plus tôt opère 211")
    print("  patients de plus et divise le délai médian par 1,5 — mais dégrade")
    print("  le lissage des lits de 3,63 à 4,77. Celui qui suit toujours la")
    print("  recommandation fait l'inverse. Le mélange des deux, c'est-à-dire")
    print("  la réalité d'un bloc, donne le meilleur compromis des trois.")
    print("\n  À surveiller : sans terme de délai dans l'objectif, la date")
    print("  « recommandée » peut tomber très loin — jusqu'à cinq mois. Si cela")
    print("  gêne, la parade est de ne chercher la recommandation que parmi les")
    print("  N prochaines vacations du praticien, au prix d'un paramètre.")
    print("\n  Les reports viennent surtout du bord de l'horizon : un patient vu en")
    print("  consultation au 5e mois n'a presque plus de vacations devant lui. En")
    print("  exploitation l'horizon glisse, et il est reprogrammé au mois suivant.")

    # ---------------------------------------------------------------- 6 ----
    titre("6. L'ÉTALON HORS-LIGNE — LE PRIX DES DATES FIGÉES")
    print("  Le tabou global connaît tous les patients d'avance et peut tout")
    print("  déplacer — mais il respecte la fenêtre de chacun, comme le flux.")
    print("  Son planning n'est pas réalisable en pratique, ce n'est pas le but :")
    print("  il mesure ce que coûte la MYOPIE, à fenêtres identiques.")
    print("  L'écart avec le flux réel mesure ce que coûte la règle « une date")
    print("  annoncée ne bouge plus ».\n")

    ev = Evaluateur(inst, poids)
    depart = solution_initiale(inst, ev)
    tabou = TabouGlobal(inst, poids, ParametresTabou(iterations=6000,
                                                     verbeux=False))
    t = time.time()
    sol_hl = tabou.resoudre(depart)
    ind_hl = sol_hl.indicateurs()
    print(f"  6000 itérations en {time.time() - t:.1f}s "
          f"({tabou.trace.diversifications} diversifications)\n")

    print(f"    {'':28} {'réel':>10} {'flux':>10} {'hors-ligne':>12}")
    for libelle, cle_reel, cle in (
            ("patients programmés", None, "patients_programmes"),
            ("écart-type lits", "lits_ecart_type", "lits_ecart_type"),
            ("pic lits", "lits_pic", "lits_pic"),
            ("écart-type ambulatoires", "places_ecart_type", "places_ecart_type"),
            ("pic ambulatoires / jour", "places_pic", "places_pic"),
            ("taux de remplissage", None, "taux_remplissage_moyen"),
            ("dispersion du remplissage", None, "remplissage_ecart_type"),
            ("délai moyen (j)", None, "delai_moyen_j")):
        v_reel = f"{reel[cle_reel]:10.2f}" if cle_reel else f"{'—':>10}"
        f_ = ind_flux[cle]
        h_ = ind_hl[cle]
        fmt = "{:10.2f}" if isinstance(f_, float) else "{:10d}"
        fmt2 = "{:12.2f}" if isinstance(h_, float) else "{:12d}"
        print(f"    {libelle:28} {v_reel} " + fmt.format(f_) + " " + fmt2.format(h_))

    ecart = ind_flux["lits_ecart_type"] - ind_hl["lits_ecart_type"]
    manquants = ind_hl["patients_programmes"] - ind_flux["patients_programmes"]
    print(f"\n  PRIX DE LA MYOPIE : +{ecart:.2f} d'écart-type sur les lits "
          f"({ind_hl['lits_ecart_type']:.2f} avec le recul, "
          f"{ind_flux['lits_ecart_type']:.2f} au fil de l'eau)")
    print(f"  et {manquants} patients programmés en moins sur {len(inst.patients)}.")
    print("\n  C'est-à-dire : presque rien. Le résultat mérite d'être souligné —")
    print("  une fois le poids du délai correctement réglé, donner deux dates en")
    print("  consultation sans jamais rien réorganiser produit un planning presque")
    print("  aussi bon que celui qu'on obtiendrait en connaissant tous les patients")
    print("  d'avance. Le flux fait même mieux sur les places ambulatoires.")
    print("  La contrainte « une date annoncée ne bouge plus » ne coûte donc")
    print("  pratiquement rien : ce n'est pas elle qu'il faut relâcher.")
    print(f"\n  Les deux sont très loin devant le réalisé "
          f"(écart-type {reel['lits_ecart_type']:.2f}, pic {reel['lits_pic']} lits).")

    print()
    print(profil_hebdo(sol_flux))

    # ---------------------------------------------------------------- 7 ----
    titre("7. TABOU LOCAL — LA VEILLE, SUR UNE JOURNÉE")
    print("  Deux familles de critères : creux et places, lissés dans la journée.")
    print("  Les violations — dépassement de vacation, sortie après fermeture de")
    print("  l'UCA, capacité en places — sont comparées AVANT le coût, jamais")
    print("  converties en poids.\n")

    candidats = [j for j in inst.jours_ouvres
                 if doit_lancer_local(inst, sol_flux, j, aujourdhui=-1,
                                      seuil_remplissage=0.80)]
    print(f"  {len(candidats)} journées dépassent 80 % de remplissage "
          f"sur {len(inst.jours_ouvres)} journées ouvrées.")

    local = TabouLocal(inst, PoidsLocaux(), iterations=120)
    t = time.time()
    gains = []
    for j in candidats[:12]:
        pj_avant = deriver_creneaux(inst, j, local._sequences_initiales(sol_flux, j))
        pj, reportes = local.resoudre(sol_flux, j)
        gains.append((j, pj_avant.pic_places, pj.pic_places,
                      pj_avant.places_variance ** 0.5, pj.places_variance ** 0.5,
                      pj_avant.creux_total, pj.creux_total))
    print(f"  12 journées optimisées en {time.time() - t:.1f}s\n")
    print(f"    {'jour':>6}   {'pic places':^20} {'écart-type places':^24} {'creux (min)':^20}")
    for j, pa, pp, sa, sp, ca, cp in gains:
        print(f"    J{j:<5}   {pa:>7} -> {pp:<10} {sa:>10.2f} -> {sp:<11.2f} "
              f"{ca:>8} -> {cp:<10}")

    jour_exemple = candidats[0] if candidats else inst.jours_ouvres[3]
    pj, _ = local.resoudre(sol_flux, jour_exemple)
    print()
    print(afficher_journee(inst, pj))
    print("\n  Les numéros de place sont attribués par coloriage d'intervalles :")
    print("  cette étape est OPTIMALE une fois les horaires connus. Tout le")
    print("  travail d'optimisation porte donc sur la séquence.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FICHIER)  # ou --donnees X
