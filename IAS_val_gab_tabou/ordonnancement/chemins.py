"""Rend les modules de coeur/ et vacations/ importables depuis n'importe quel script.

    import chemins  # en tête d'un script de experiences/ ou tests/

Le fichier de données n'est pas versionné (dépôt public) : on le donne en
argument `--donnees chemin.xlsx`, ou par la variable d'environnement
BLOC_DONNEES, ou on le dépose dans ordonnancement/donnees/.
"""
import os, sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent
for sous in ("coeur", "vacations"):
    p = str(RACINE / sous)
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTATS = RACINE / "resultats"
NOM_DONNEES = "donees bloc anonyme pour centrale 2026.xlsx"


def fichier_donnees(argv=None) -> str:
    """--donnees X  >  $BLOC_DONNEES  >  ordonnancement/donnees/<nom d'origine>."""
    argv = sys.argv if argv is None else argv
    if "--donnees" in argv:
        i = argv.index("--donnees")
        chemin = argv[i + 1]
        del argv[i:i + 2]
        return chemin
    if os.environ.get("BLOC_DONNEES"):
        return os.environ["BLOC_DONNEES"]
    return str(RACINE / "donnees" / NOM_DONNEES)
