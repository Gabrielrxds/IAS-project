# IAS-project
Planning optimization — optimisation de l'ordonnancement d'un bloc opératoire (projet IA et Santé, Centrale 2026).

**Commencer par [`CONTEXTE_PROJET.md`](CONTEXTE_PROJET.md)** : problème, architecture, chiffres de référence, décisions et suite du projet.

| Élément | Auteur | Contenu |
|---|---|---|
| [`ordonnancement/`](ordonnancement/) | Gabriel | Moteur complet : durée → jour (consultation) → heure/salle (tabou), planning des vacations (tabou, GRASP, path relinking), simulation annuelle, rapport |
| `modele.py` | Gabriel | Structures de données v1 (identique à `ordonnancement/coeur/modele.py`) |
| `modele_fn_cout.py`, `modele_fn_cout_1rdv.py` | Killian | Fonctions de coût, urgences, délai max 6 mois |
| `bdd_nettoyee.xlsx`, `bdd_nettoyee-2.csv` | Lucie | Base de données nettoyée |
| `Guide_planning_bloc_operatoire.pdf` | Solène | Guide du module de vérification `planning.py` |
