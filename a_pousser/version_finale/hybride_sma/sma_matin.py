"""Plans du matin pour le SMA : la Solution (dates données en ligne, définitives) et l'ordre
de passage de chaque journée de la période de test, figé 7 jours avant.

    python3 hybride_sma/sma_matin.py reference|hybride [tampon lits places]

  reference : grille actuelle + premier créneau + ordre glouton (référence du SMA de Kyllian)
  hybride   : grille hybride + règle hybride (Δ=14, fantômes uniformes) + journée tabou (graines AG)
              plans B : ordres génétique et recuit de la même journée
  tampon lits places : réserves pour les urgences (défaut : 0.05 2 2, réglages du groupe)

Sortie : cache/sma_matin_<nom>[_r<tampon>_<lits>_<places>].pkl
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import dataclasses
import pickle
import sys
import time

from protocole import charger, instance, grille_actuelle_neutre, lire_grille, Reglages, RACINE
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
from sma_aleas import optimiseur_glouton
import algos_journee as AJ

DEFAUT = (0.05, 2, 2)


def suffixe(reserves) -> str:
    if reserves is None or tuple(reserves) == DEFAUT:
        return ""
    tp, rl, rp = reserves
    return f"_r{round(100 * tp)}_{rl}_{rp}"


def construire(nom: str, reserves=None, budget: int = 1000, verbeux: bool = True) -> dict:
    """Plan du matin `nom` ('reference' ou 'hybride') avec des réserves (tampon, lits, places)."""
    assert nom in ("reference", "hybride"), nom
    df, don = charger()
    reg = Reglages()
    if reserves is not None:
        tp, rl, rp = reserves
        reg = dataclasses.replace(reg, tampon_urgence=float(tp), reserve_lits=int(rl), reserve_places=int(rp))
    t0, t1 = don.test
    JT = set(don.jours_ouvres(t0, t1))
    plans, plans_b = {}, {}

    def ordre(sol, j):
        if j not in JT:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        if nom == "reference":
            plans[j] = optimiseur_glouton(sol.inst, j, seq)
            return
        b = {}
        for algo in ("tabou", "génétique", "recuit"):
            try:
                s = AJ.ALGOS[algo](sol.inst, j, seq, Poids(), budget, 0, graines=True).sequences
            except JourneeInfaisable:
                s = optimiseur_glouton(sol.inst, j, seq)
            if algo == "tabou":
                plans[j] = s
            else:
                b[algo] = s
        plans_b[j] = b

    if nom == "reference":
        inst = instance(don, grille_actuelle_neutre(don), reg)
        regle = PremierCreneau()
    else:
        inst = instance(don, lire_grille("hybride")["vacations"], reg)
        regle = RegleHybride(don, delta=14, fantomes="uniformes")
    t = time.time()
    sol = simuler(inst, regle, apres_figer=ordre)
    if verbeux:
        print(f"plan du matin {nom}{suffixe(reserves)} : {time.time() - t:.0f} s, "
              f"{len(plans)} journées ordonnées", flush=True)
    return {"sol": sol, "plans": plans, "plans_b": plans_b, "jours": sorted(JT)}


def charger_ou_construire(nom: str, reserves=None) -> dict:
    """Plan du matin en cache (cache/sma_matin_*.pkl), construit au premier appel."""
    f = RACINE / "cache" / f"sma_matin_{nom}{suffixe(reserves)}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    m = construire(nom, reserves)
    f.write_bytes(pickle.dumps(m))
    return m


if __name__ == "__main__":
    nom = sys.argv[1] if len(sys.argv) > 1 else "hybride"
    res = (float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else None
    m = construire(nom, res)
    (RACINE / "cache" / f"sma_matin_{nom}{suffixe(res)}.pkl").write_bytes(pickle.dumps(m))
