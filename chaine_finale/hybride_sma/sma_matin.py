"""Plans du matin pour le SMA : la Solution (dates données en ligne) et l'ordre de chaque journée.
    python3 sma_matin.py reference|hybride [tampon lits places]
  reference : grille actuelle + premier créneau + ordre glouton (référence du SMA de Kyllian)
  hybride   : grille hybride + règle hybride (Δ=14, fantômes uniformes) + journée tabou (graines AG)
Plans B (hybride) : ordres génétique et recuit de la même journée, figés 7 jours avant.
Sortie : cache/sma_matin_<nom>.pkl

Depuis un autre module : charger_ou_construire(nom, reserves=None) relit le cache ou le construit."""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import dataclasses, pickle, sys, time
from protocole import charger, instance, grille_actuelle_neutre, lire_grille, Reglages, RACINE
from politiques import PremierCreneau, simuler
from regle_hybride import RegleHybride
from genetique_jour import Poids, sequences_du_jour, JourneeInfaisable
from sma_aleas import optimiseur_glouton
import algos_journee as AJ


def construire(nom: str, reserves=None) -> dict:
    """Plan du matin de la chaîne `nom` ; reserves = (part du TVO, lits, admissions) ou None."""
    df, don = charger(); reg = Reglages()
    if reserves is not None:      # réserves pour les urgences : part du TVO, lits, admissions
        tp, rl, rp = reserves
        reg = dataclasses.replace(reg, tampon_urgence=float(tp), reserve_lits=int(rl), reserve_places=int(rp))
    t0, t1 = don.test
    JT = set(don.jours_ouvres(t0, t1))
    plans, plans_b, temps = {}, {}, []

    def ordre(sol, j):
        if j not in JT:
            return
        seq = sequences_du_jour(sol, j)
        if not seq:
            return
        seq = {v: sorted(s, key=lambda p: (sol.inst.patients[p].jour_demande, p)) for v, s in seq.items()}
        if nom == "reference":
            plans[j] = optimiseur_glouton(sol.inst, j, seq); return
        t = time.time(); b = {}
        for algo, cle in (("tabou", None), ("génétique", "génétique"), ("recuit", "recuit")):
            try:
                s = AJ.ALGOS[algo](sol.inst, j, seq, Poids(), 1000, 0, graines=True).sequences
            except JourneeInfaisable:
                s = optimiseur_glouton(sol.inst, j, seq)
            if cle is None:
                plans[j] = s
            else:
                b[cle] = s
        plans_b[j] = b; temps.append(time.time() - t)

    if nom == "reference":
        inst = instance(don, grille_actuelle_neutre(don), reg); regle = PremierCreneau()
    else:
        inst = instance(don, lire_grille("hybride")["vacations"], reg)
        regle = RegleHybride(don, delta=14, fantomes="uniformes")
    t = time.time()
    sol = simuler(inst, regle, apres_figer=ordre)
    print(nom, f"{time.time()-t:.0f}s", len(plans), "jours ordonnés", flush=True)
    return {"sol": sol, "plans": plans, "plans_b": plans_b, "jours": sorted(JT)}


def _fichier(nom: str, reserves=None):
    suffixe = "" if reserves is None else f"_r{round(100 * float(reserves[0]))}_{int(reserves[1])}_{int(reserves[2])}"
    return RACINE / "cache" / f"sma_matin_{nom}{suffixe}.pkl"


def charger_ou_construire(nom: str, reserves=None) -> dict:
    f = _fichier(nom, reserves)
    if f.exists():
        return pickle.loads(f.read_bytes())
    m = construire(nom, reserves)
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(m))
    return m


if __name__ == "__main__":
    nom = sys.argv[1]
    reserves = tuple(sys.argv[2:5]) if len(sys.argv) > 4 else None
    m = construire(nom, reserves)
    f = _fichier(nom, reserves)
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(pickle.dumps(m))
