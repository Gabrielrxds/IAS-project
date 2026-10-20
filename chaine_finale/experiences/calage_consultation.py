r"""
calage_consultation.py — Quand chaque patient a-t-il consulté ?

La base ne donne que la date d'opération. On suppose que l'hôpital programme au premier créneau
libre de son chirurgien, au moins 7 jours après la consultation (pratique courante, et règle de
la référence). Pour un délai de consultation donné, rejouer cette pratique sur la grille réelle
redonne des dates d'opération ; on cherche, pour chaque chirurgien, le délai qui retombe le plus
près des vraies dates.

Modèle : délai D_p = clip(round(L_m · exp(σ z_p)), 7, 182), z_p ~ N(0, 1) fixé par patient.
L_m (délai typique du chirurgien m) et σ (dispersion, commune) sont calés en minimisant l'écart
absolu moyen |date simulée − date réelle|. Les chirurgiens n'échangent pas de créneaux : le
calage est séparable par chirurgien.
Sortie : resultats/calage_consultation.json
"""
import sys as _sys
from pathlib import Path as _Path
for _d in ("commun", "hybride", "hybride_sma", "aleas", "experiences"):
    _p = str(_Path(__file__).resolve().parents[1] / _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import json, copy
import numpy as np
from multiprocessing import Pool
import protocole
from protocole import charger, instance, grille_actuelle_neutre, Reglages, RACINE
from politiques import simuler, PremierCreneau

L_GRILLE = [7, 10, 14, 21, 28, 35, 42, 56, 70, 84, 105, 126, 150, 182]
SIGMAS = [0.25, 0.5, 0.75, 1.0]        # σ > 0 : tous les patients d'un chirurgien n'attendent pas pareil


def delais(don, L, sigma, graine=12345):
    rng = np.random.default_rng(graine)
    pids = sorted(don.patients)
    z = dict(zip(pids, rng.standard_normal(len(pids))))
    Lm = L if isinstance(L, dict) else None
    out = {}
    for p in pids:
        l = Lm[don.patients[p].med_id] if Lm else L
        out[p] = int(np.clip(round(l * np.exp(sigma * z[p])), 7, 182))
    return out


def rejouer(args):
    L, sigma = args
    _, don = charger(consultation="brut")
    D = delais(don, L, sigma)
    inst = instance(don, grille_actuelle_neutre(don), Reglages())
    for p, d in D.items():
        inst.patients[p].jour_demande = max(0, don.jour_reel[p] - d)
    sol = simuler(inst, PremierCreneau(), graine=0)
    par_med = {}
    for p, j in don.jour_reel.items():
        v = sol.affectation[p]
        m = don.patients[p].med_id
        e = 182 if v is None else inst.vacations[v].jour - j
        par_med.setdefault(m, []).append(e)
    return (L if not isinstance(L, dict) else "cale", sigma), {m: e for m, e in par_med.items()}


if __name__ == "__main__":
    protocole.CAPACITE_VACANCES = False          # pratique réelle : grille actuelle telle quelle
    _, don = charger(consultation="brut")
    with Pool() as P:
        R = dict(P.map(rejouer, [(L, s) for L in L_GRILLE for s in SIGMAS]))
    meds = sorted(don.med_codes)
    meilleur = None
    for s in SIGMAS:
        Lm, tot, n = {}, 0.0, 0
        for m in meds:
            mae = {L: np.mean(np.abs(R[(L, s)][m])) for L in L_GRILLE if m in R[(L, s)]}
            Lm[m] = min(mae, key=mae.get)
            tot += mae[Lm[m]] * len(R[(L_GRILLE[0], s)][m]); n += len(R[(L_GRILLE[0], s)][m])
        print(f"sigma {s}: écart absolu moyen {tot / n:.1f} j")
        if meilleur is None or tot / n < meilleur[0]:
            meilleur = (tot / n, s, Lm)
    _, s, Lm = meilleur
    # vérification : rejeu avec les délais calés
    _, E = rejouer((Lm, s))
    e = np.concatenate([np.array(v) for v in E.values()])
    D = delais(don, Lm, s); d = np.array(list(D.values()))
    res = dict(sigma=s, delai_type={don.med_codes[m]: Lm[m] for m in meds},
               patients_par_chirurgien={don.med_codes[m]: len(E[m]) for m in meds},
               ecart_abs_moyen=float(np.mean(np.abs(e))), ecart_median=float(np.median(e)),
               part_a_0j=float(np.mean(e == 0)), part_a_7j=float(np.mean(np.abs(e) <= 7)),
               delai_median=float(np.median(d)), delai_moyen=float(d.mean()),
               delai_p10=float(np.percentile(d, 10)), delai_p90=float(np.percentile(d, 90)),
               comparaison_30j={k: float(v) for k, v in zip(
                   ("ecart_abs_moyen", "part_a_7j"),
                   (lambda x: (np.mean(np.abs(x)), np.mean(np.abs(x) <= 7)))(
                       np.concatenate([np.array(v) for v in R[(28, SIGMAS[0])].values()])))})
    a = protocole.codes_anonymes()
    for k in ("delai_type", "patients_par_chirurgien"):
        if k in res:
            res[k] = {a.get(c, c): v for c, v in res[k].items()}
    (RACINE / "resultats" / "calage_consultation.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
