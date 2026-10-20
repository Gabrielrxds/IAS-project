# Modèle mathématique de la chaîne hybride

Version retenue : grille « croisement + recuit, poids lits ×4 », règle de consultation avec
fenêtre de 14 jours et patients fantômes uniformes, journée par graines génétiques + tabou.
Les formules ci-dessous sont celles du code (`chaine_finale/hybride/hybride_grille.py`, `chaine_finale/hybride/chaine_hybride.py`,
`chaine_finale/hybride/regle_hybride.py`, `chaine_finale/commun/algos_journee.py`, et `etudes_origine/genetique/` pour le modèle commun).

## Notations communes

| Symbole | Sens |
|---|---|
| $M$ | chirurgiens ; $m \in M$ |
| $p$ | patient ; $m(p)$ son chirurgien |
| $d_p$, $\mu_p$ | durée estimée (médiane du TROS de l'acte) et marge de risque (P90 − médiane), en minutes |
| $\theta = 15$, $\theta_= = 10$ | temps inter-salle, et temps réduit entre deux actes identiques |
| $e_p = d_p + \theta$ | minutes « engagées » sans marge (fantômes) |
| $n_p$, $a_p$ | nuits du séjour et nuits avant l'opération (par les dates) |
| $N_p(j) = \{\,j - a_p, \dots, j - a_p + n_p - 1\,\}$ | nuits où $p$ occupe un lit s'il est opéré le jour $j$ ($\emptyset$ si ambulatoire) |
| $c(p) \in \{A, C, L\}$ | classe de séjour : ambulatoire ($n_p = 0$), court ($1 \le n_p \le 2$), long ($n_p \ge 3$) |

---

## Étage 1 — La grille des vacations

### Variables

Cycle de $C = 4$ semaines, $L = 7C = 28$ jours, salles $R = \{2, 3, 4, 5\}$, demi-journées
$h \in \{\text{matin } 8\text{h}-13\text{h},\ \text{après-midi } 13\text{h}30-17\text{h}30\}$ :

$$X_{k,d,r,h} \in M \cup \{\text{vide}\}, \qquad k \in \{0..3\},\ d \in \{\text{lun..ven}\},\ r \in R$$

Si le même chirurgien a le matin et l'après-midi d'une même salle, les deux forment une
vacation de journée 8h-17h30 (570 min).

### Contraintes (dures, garanties par les mouvements)

1. **Quotas** : $\sum_{k,d,r,h} \mathbb 1[X_{k,d,r,h} = m] = q_m$ pour tout $m$.
2. **Pas de don d'ubiquité** : $\forall k,d,h,m$ : $\sum_r \mathbb 1[X_{k,d,r,h} = m] \le 1$.
3. **Urgences** : les cases occupées par un créneau URGENCES de la grille actuelle sont interdites.

### Quotas (volumes, fixés avant la recherche)

Avec $a_m$ les demi-journées actuelles de $m$, $D_m$ sa demande (minutes engagées par semaine
sur janvier-avril, $d_p + \mu_p + \theta$ par patient), et un budget en heures égal à la grille
actuelle :

$$Q = \operatorname{round}\!\Big(\frac{623\ \text{h}}{4{,}9\ \text{h}}\Big) = 127, \qquad
\underline q_m = \max(1, \lceil 0{,}75\, a_m \rceil), \quad \overline q_m = \max(1, \lfloor 1{,}5\, a_m \rfloor)$$

$$q_m \approx \min\!\Big(\overline q_m,\ \max\big(\underline q_m,\ Q\,\tfrac{D_m}{\sum_{m'} D_{m'}}\big)\Big),
\qquad \sum_m q_m = Q \text{ (plus grands restes)}$$

### Évaluation analytique d'une grille (régime permanent)

Pour chaque chirurgien $s$ et jour du cycle $j \in \{0..27\}$ : $\mathrm{TVO}_{s,j}$ minutes
offertes, $\nu_s$ vacations par cycle.

- Demande par cycle : $\Delta_s = C \cdot D_s$.
- Perte de fin de vacation : $\varphi_s$ = moitié de la durée engagée moyenne d'un acte de $s$.
- Capacité utile : $K^{\text{eff}}_s = \max\big(0,\ \sum_j \mathrm{TVO}_{s,j} - \nu_s \varphi_s\big)$.
- Demande servie : $S_s = \min(\Delta_s, K^{\text{eff}}_s)$ ; taux d'emploi $f_s = S_s / \sum_j \mathrm{TVO}_{s,j}$.
- Minutes opérées : $O_{s,j} = f_s\, \mathrm{TVO}_{s,j}$.
- Profil de lits de $s$ (convolution circulaire) :
  $\kappa_s(o) = \dfrac{\#\{\text{nuits à l'écart } o \text{ de l'opération}\}}{\sum \text{minutes engagées}}$
  sur ses patients d'apprentissage.

$$\boxed{\ \Lambda_n = \sum_s \sum_j \kappa_s\big((n - j) \bmod L\big)\, O_{s,j}\ } \quad \text{lits attendus la nuit } n$$

$$P_j = \sum_s \alpha_s\, O_{s,j} \quad \text{admissions ambulatoires attendues}
\quad (\alpha_s = \text{ambulatoires par minute engagée})$$

### Termes du coût

| Terme | Formule |
|---|---|
| Lits (toutes les nuits) | $T_{\text{lits}} = \operatorname{Var}_{n \in 0..27}(\Lambda_n) / 42^2$ |
| Places | $T_{\text{places}} = \operatorname{Var}_{j\ \text{ouvré}}(P_j) / 18^2$ |
| Remplissage | $T_{\text{rempl}} = \sum_s(\sum_j \mathrm{TVO}_{s,j} - S_s) \,/\, \sum_{s,j} \mathrm{TVO}_{s,j}$ (temps perdu) |
| Délai | $T_{\text{délai}} = \dfrac{1}{182}\,\dfrac{\sum_s \Delta_s\,\big(w_s + 182\,\tfrac{\Delta_s - S_s}{\Delta_s}\big)}{\sum_s \Delta_s}$, avec $w_s = \dfrac{\sum_i g_{s,i}^2}{2L}$ ($g_{s,i}$ : écarts entre jours opératoires successifs de $s$) |
| Salles | $T_{\text{salles}}$ = salles-jours ouvertes par semaine |
| Dépassement | $E = \sum_n (\Lambda_n - 40)^+ + \sum_{j\ \text{ouvré}} (P_j - 16)^+$ |

Chaque terme est normalisé par sa valeur $T^{\text{ref}}$ sur la grille actuelle, calculée avec
le même évaluateur :

$$\boxed{\ F(X) = 4\,\frac{T_{\text{lits}}}{T^{\text{ref}}_{\text{lits}}} + \frac{T_{\text{places}}}{T^{\text{ref}}_{\text{places}}}
+ \frac{T_{\text{rempl}}}{T^{\text{ref}}_{\text{rempl}}} + \frac{T_{\text{délai}}}{T^{\text{ref}}_{\text{délai}}}
+ \frac{T_{\text{salles}}}{T^{\text{ref}}_{\text{salles}}} + 100\,E\ }$$

Le poids 4 sur les lits est celui de la grille retenue ; les candidates au poids 1 ont aussi été
générées.

### Recherche

**Mouvements** (mutations de l'AG), qui conservent les contraintes 1-3 :
- échange de deux cases ;
- alignement du matin et de l'après-midi d'un chirurgien dans la même salle ;
- échange de deux jours entiers.

**Recuit simulé.** Un mouvement $X \to Y$ de variation $\delta = F(Y) - F(X)$ est accepté si
$\delta \le 0$, sinon avec la probabilité $e^{-\delta/T}$. Température :
- $T_0$ = 1er décile des dégradations positives d'un échantillon de mouvements ;
- $T_{\min} = 0{,}005\,T_0$ ;
- $T_{k+1} = \alpha T_k$ avec $\alpha = (T_{\min}/T_0)^{1/B}$, sur un budget $B = 15\,000$
  évaluations.

**Croisement + recuit froid.** L'enfant prend chaque jour $(k, d)$ entier de l'un des deux
parents (probabilité ½), puis les quotas sont réparés. Il est repris par un recuit avec
$T_0 = 0{,}3 \times$ le 1er décile et $B = 5\,000$.

**Sélection simheuristique.** Chaque candidate $g$ est rejouée sur les patients de janvier-avril
avec la règle de l'étage 2 :

$$g^\star = \arg\min_{g\,:\ \text{sans\_date}(g) \le \min_{g'} \text{sans\_date}(g') + 2}\ \operatorname{Var}_{\text{nuits fév.-avr.}}\big(L^{\text{sim}}_n(g)\big)$$

---

## Étage 2 — La date à la consultation

### Ce qui est appris (janvier-avril)

Pour chaque chirurgien $m$ et classe $c$ :

$$K_{m,c}(o) = \frac{\sum_{p:\ m(p)=m,\ c(p)=c} \mathbb 1[o \in N_p(0)]}{\sum_{p:\ m(p)=m,\ c(p)=c} e_p}
\quad\text{(lits par minute opérée, à l'écart } o\text{)}$$

$$\pi_{m,c} = \frac{\sum_{p:\,m,c} e_p}{\sum_{p:\,m} e_p} \ \ \text{(mélange des classes)}, \qquad
\rho_m = \min\!\Big(0{,}95,\ \frac{\text{minutes } e_p \text{ demandées par semaine}}{\text{capacité programmée par semaine dans la grille}}\Big)$$

### Contraintes dures d'une affectation $p \to v$

Une vacation $v$ du chirurgien $m(p)$, le jour $j_v$, est réalisable si :

- **Fenêtre de délai** : $t_p + 7 \le j_v \le t_p + 182$ ($t_p$ = jour de consultation).
- **Journée non figée** : $j_v > t + 7$ ($t$ = aujourd'hui).
- **Capacité de la vacation** (marges cumulées quadratiquement, 5 % du TVO réservé aux urgences) :

$$\sum_{p' \in v} d_{p'} + \sqrt{\textstyle\sum_{p' \in v} \mu_{p'}^2} + \theta\,(|v| - 1) \;\le\; 0{,}95\,\mathrm{TVO}_v$$

- **Lits** : chaque nuit $j \in N_p(j_v)$ reste sous $42 - 2 = 40$ lits programmés.
- **Places** : le jour $j_v$ reste sous $18 - 2 = 16$ admissions ambulatoires programmées.

### Patients fantômes

Chaque jour $t$, pour chaque vacation $v$ de $m$ telle que $t + 7 < j_v \le t + 63$ :

$$r_{v,c} = \pi_{m,c}\ \max\!\big(0,\ \rho_m\, \mathrm{cap}_v - \mathrm{charge}_v\big)$$

(minutes de fantômes de la classe $c$), où $\mathrm{cap}_v$ est la capacité programmée de $v$ et
$\mathrm{charge}_v$ la charge déjà réservée. Les fantômes d'une journée disparaissent quand elle
est figée.

**Courbe des lits attendus**, avec $R_j$ les lits des vrais patients déjà datés :

$$E_j = R_j + G_j, \qquad G_j = \sum_{v}\sum_{c} r_{v,c}\; K_{m(v),c}(j - j_v)$$

### Coût d'une candidate

Le patient $p$ (classe $c$, minutes $e = e_p$) entre dans $v$ et remplace des fantômes : d'abord
ceux de sa classe, puis, pour le reste, ceux des autres classes au prorata.

$$x_c = \min(e, r_{v,c}), \qquad x_{c'} = \min\!\Big(r_{v,c'},\ (e - x_c)\,\frac{r_{v,c'}}{\sum_{c'' \ne c} r_{v,c''}}\Big)\quad (c' \ne c)$$

Variation de la courbe : $\delta_v(j) = \mathbb 1[j \in N_p(j_v)] - \sum_{c} x_c\, K_{m,c}(j - j_v)$.

Sur la fenêtre de mesure $W_t = [t+1,\ t+63+30)$, $|W_t| = n$ :

$$\boxed{\ \Phi_p(v) = \operatorname{Var}_{j \in W_t}\big(E_j + \delta_v(j)\big) - \operatorname{Var}_{j \in W_t}(E_j)\ }$$

calculé en $O(|N_p| + |\text{support de } K|)$ grâce aux sommes $S_1 = \sum E_j$ et $S_2 = \sum E_j^2$ :

$$\Phi_p(v) = \frac{S_2 + \sum_j (2E_j\delta_j + \delta_j^2)}{n} - \Big(\frac{S_1 + \sum_j \delta_j}{n}\Big)^2 - \Big(\frac{S_2}{n} - \frac{S_1^2}{n^2}\Big)$$

### Décision

- $j_1$ = premier jour réalisable ; candidates $V_p = \{\, v \text{ réalisable} : j_1 \le j_v \le j_1 + 14 \,\}$.
- Pour chaque jour, la meilleure vacation $v^\star(j) = \arg\min_{v : j_v = j} \Phi_p(v)$.
- On propose les deux jours de plus petit $\Phi_p$ ; **le patient prend le plus proche des deux**.
- $V_p = \emptyset$ → patient sans date.

Puis mise à jour : $R \leftarrow R + \mathbb 1_{N_p(j_v)}$, $r_{v,\cdot} \leftarrow r_{v,\cdot} - x_\cdot$.

---

## Étage 3 — La journée (à J−7)

### Variables

Pour les $n$ patients figés du jour : une permutation $\sigma$ et une vacation $u_i$ pour chacun,
parmi celles de **son** chirurgien ce jour-là. Les heures s'en déduisent au plus tôt :

$$\text{début}_{i} = \text{fin}_{\text{préc}(i)} + \begin{cases}\theta_= & \text{même acte que le précédent}\\ \theta & \text{sinon}\end{cases}, \qquad \text{fin}_i = \text{début}_i + d_i$$

Un ambulatoire occupe une place de $\max(7\text{h}30,\ \text{début}_i - 120)$ à $\text{fin}_i + 180$ min.
Les numéros de place sont ensuite attribués par coloriage d'intervalles, ce qui est optimal.

### Coût (celui de l'AG journalier)

Deux scénarios : durées estimées $d_i$, et pessimiste $d_i + \mu_i$ (P90).

$$\begin{aligned}
J(\sigma, u) =\ & 10^7\,\mathrm{dep} + 10^5\,\#\text{reportés} + 100\,\mathrm{dep} + 2000\,\mathrm{UCA} + 3000\,(\mathrm{pic} - 12)^+ \\
&+ 3\,\mathrm{dep}^{90} + 150\,\mathrm{UCA}^{90} + 15\,\mathrm{pic} + 30\,\sigma_{\text{places}} + 1\cdot\mathrm{creux} + 200\,\mathrm{disp}
\end{aligned}$$

| Terme | Sens |
|---|---|
| $\mathrm{dep}$ | minutes de dépassement des vacations, durées estimées (le terme à $10^7$ en fait une contrainte dure) |
| $\mathrm{UCA}$ | ambulatoires qui sortent après 20 h |
| $\mathrm{pic}$, $\sigma_{\text{places}}$ | pic et écart-type temporel du nombre de places ambulatoires occupées ($12$ places simultanées) |
| $\mathrm{dep}^{90}$, $\mathrm{UCA}^{90}$ | mêmes mesures si chaque acte dure son P90 (robustesse) |
| $\mathrm{creux}$ | minutes de vacation non utilisées |
| $\mathrm{disp}$ | écart-type des taux de remplissage entre les vacations d'un même chirurgien ce jour-là |

### Recherche

- **Départ** : la meilleure des graines heuristiques de l'AG :
  - ordre d'inscription ;
  - ambulatoires d'abord, groupés par acte ;
  - ambulatoires d'abord, plus courts d'abord ;
  - ambulatoires prévisibles d'abord ;
  - hospitalisés d'abord ;
  - plus leurs variantes avec un équilibrage des vacations (plus long d'abord dans la moins
    remplie).
- **Tabou** :
  - 20 voisins tirés par itération : insertion d'un patient ailleurs dans $\sigma$ (40 %),
    échange de deux patients (35 %), changement de vacation (25 %) ;
  - on prend le meilleur voisin non tabou ;
  - un patient déplacé est tabou pendant 7 itérations, sauf s'il améliore le meilleur coût
    connu (aspiration) ;
  - budget : 1 000 évaluations distinctes.
