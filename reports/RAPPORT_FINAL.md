# Rapport final — Agents d’apprentissage par renforcement sur Taxi-v3

**Projet** : comparaison de Q-Learning, SARSA, Monte Carlo et Deep Q-Learning (Gymnasium Taxi-v3).  
**Sources de résultats figés** : fichiers `report-*.html` dans le dépôt (dates de génération indiquées ci-dessous) et [`best_params.json`](best_params.json) pour la persistance des meilleurs hyperparamètres par algorithme.

---

## 1. Introduction

**Environnement.** Taxi-v3 est un MDP à espace d’états fini (500 états) et 6 actions. Les récompenses structurent le comportement : +20 pour une dépose réussie, −1 par pas, −10 pour une action illégale.

**Objectif.** Pour chaque famille d’algorithmes, identifier une configuration performante par **grid-search** (ε, γ, α), puis comparer les agents entre eux sur des métriques de test communes.

**Critère de sélection du « meilleur » modèle (benchmark).** Aligné sur l’implémentation ([`report.py`](report.py), [`main.py`](main.py)) :

1. Taux de succès au test **≥ 95 %** (seuil de fiabilité).
2. Parmi les configurations qualifiées, **récompense moyenne** la plus élevée.
3. En cas d’équivalence statistique (écart ≤ incertitude σ/√n), **temps d’entraînement** le plus court.

Les graphiques détaillés (courbes d’apprentissage, barres Top/Bottom 5) se trouvent dans les rapports HTML listés en annexe A.

---

## 2. Méthodologie

### 2.1 Grid-search et early stopping

Pour chaque algorithme, le **mode Benchmark** parcourt un produit cartésien de valeurs pour ε, γ et le learning rate. Chaque configuration est entraînée puis évaluée sur un nombre fixe d’épisodes de test. L’**early stopping** peut interrompre l’entraînement lorsque la moyenne glissante du reward stagne (détails dans les sections analytiques des fichiers HTML générés).

### 2.2 Persistance des meilleurs paramètres

Les meilleurs jeux de paramètres par agent sont enregistrés dans [`best_params.json`](best_params.json) avec un **horodatage** et des métriques associées (reward moyen, écart-type, succès, temps d’entraînement, etc.). Cela permet au **mode Temps limité** de réutiliser les mêmes hyperparamètres sans nouveau tuning pendant la comparaison chronométrée.

### 2.3 Deux campagnes SARSA

- **Première grille** ([`report-sarsa.html`](report-sarsa.html), généré le **29/04/2026 à 21:32:14**) : 24 configurations (ε ∈ {0.7,0.8,0.9}, γ ∈ {0.95,0.99}, lr ∈ {0.05,…}). **Aucune configuration n’atteint 95 % de succès** ; la meilleure du classement reste autour de **92 %** (ex. SARSA #19). Le message d’avertissement du programme (« règle qualité non respectée ») est donc attendu.
- **Seconde grille** ([`report-sarsa-2.html`](report-sarsa-2.html), **29/04/2026 à 21:33:54**) : grille recentrée (ε plus bas, lr {0.2,0.3,0.4}, etc.). **SARSA #1** (ε=0.5, γ=0.99, lr=0.2) atteint **100 %** de succès avec un reward moyen d’environ **7.91**, ce qui valide l’intérêt d’**affiner la recherche** après un premier échec au seuil.

### 2.4 Mode « Comparaison finale » (multi-agents) vs `best_params.json`

Le rapport [`report-comparaison-finale.html`](report-comparaison-finale.html) (**29/04/2026 à 21:53:57**) provient du **mode manuel multi-sélection** dans [`main.py`](main.py) : les hyperparamètres sont ceux **saisis dans les invites** (valeurs par défaut proposées par l’interface), **et non** les entrées relues automatiquement depuis `best_params.json`.  

**Conséquence méthodologique.** Les scores SARSA et Monte Carlo y sont nettement inférieurs au Q-Learning et au DQN **pour ce run précis** (SARSA ≈ 88,8 %, MC ≈ 89,2 %), alors que les benchmarks isolés montrent des configs SARSA et MC à 100 % ([`report-sarsa-2.html`](report-sarsa-2.html), [`report-monte-carlo.html`](report-monte-carlo.html)). Ce n’est pas une contradiction : ce sont **deux protocoles différents** (grilles optimisées vs une session manuelle avec paramètres par défaut ou personnalisés). Pour une comparaison « tous agents au meilleur de chaque grille », il faudrait soit saisir manuellement les mêmes ε, γ, lr que dans `best_params.json`, soit ajouter un mode qui charge ces fichiers pour le mode manuel.

### 2.5 Mode Temps limité

D’après [`main.py`](main.py) (lignes 114–185), le mode **Temps limité** :

1. Charge [`best_params.json`](best_params.json) pour chaque agent.
2. **Ré-entraîne** l’agent avec ces paramètres (avec early stopping possible).
3. Lance un **test chronométré** avec une **seed d’environnement inconnue** (`unseen_seed`) et un budget en secondes par agent.
4. Affiche des barres (reward moyen, succès %, nombre d’épisodes testés dans le temps imparti) et applique la même règle de classement **succès ≥ 95 % → reward → temps d’entraînement**.

Sur l’exécution documentée par capture d’écran (barres + tableau), le **gagnant** annoncé est **Q-Learning** (reward et succès en tête ; DQN proche en succès mais moins d’épisodes testés dans le même budget — coût par pas plus élevé). Les figures interactives de ce run peuvent être jointes en annexe (fichiers image locaux).

---

## 3. Résultats par algorithme (benchmarks unitaires)

Les tableaux complets figurent dans les HTML correspondants ; on synthétise ici **meilleure** et **pire** configuration (au sens du reward moyen au test dans le rapport) et l’**interprétation**.

### 3.1 Q-Learning

**Source :** [`report-Q-Learning.html`](report-Q-Learning.html) (29/04/2026 à 21:29:49), grille 3×2×4 = **24** configs.

| Rôle | Config (extrait rapport) | Reward | Succès |
|------|--------------------------|--------|--------|
| Meilleure (Top 1) | **#18** — ε=0.9, γ=0.99, lr=0.3 | 8.54 | 100 % |
| Pire (Bottom) | **#1** — ε=0.7, γ=0.99, lr=0.1 | −0.14 | 96 % |

**Interprétation.** Avec ε relativement bas et lr faible (#1), l’exploration et la vitesse de mise à jour peuvent être insuffisantes pour atteindre rapidement la politique optimale sur tous les tirages de test, d’où un reward moyen faible malgré un succès encore élevé. La config #18 combine un **γ élevé** (prise en compte de la récompense terminale +20), un **ε élevé** (meilleure couverture) et un **lr modéré** (0.3) qui offre un bon compromis stabilité / convergence dans cette grille.

### 3.2 SARSA

**Grille 1 —** [`report-sarsa.html`](report-sarsa.html) (29/04/2026 à 21:32:14).

| Rôle | Exemple (rapport) | Reward | Succès |
|------|-------------------|--------|--------|
| « Meilleure » du classement | **#19** — ε=0.9, γ=0.95, lr=0.2 | −8.07 | 92 % |
| Pire (Bottom) | **#21** — ε=0.9, γ=0.99, lr=0.05 | −145.15 | 26 % |

**Interprétation.** SARSA étant **on-policy**, de faibles learning rates avec une exploration inadaptée peuvent figer une politique sous-optimale ; les pires lignes combinent souvent **lr très bas** et mauvaise synergie ε/γ, d’où des succès effondrés.

**Grille 2 —** [`report-sarsa-2.html`](report-sarsa-2.html) (29/04/2026 à 21:33:54).

| Rôle | Config | Reward | Succès |
|------|--------|--------|--------|
| Meilleure | **#1** — ε=0.5, γ=0.99, lr=0.2 | 7.91 | 100 % |
| Pire (ex. Bottom) | **#2** — ε=0.5, γ=0.99, lr=0.3 | −85.62 | 55 % |

**Interprétation.** À **γ=0.99** fixé, augmenter lr de 0.2 à 0.3 dégrade fortement la stabilité (mises à jour trop agressives sur la politique suivie), typique de SARSA sensible au bruit d’exploration.

### 3.3 Monte Carlo

**Source :** [`report-monte-carlo.html`](report-monte-carlo.html) (29/04/2026 à 21:36:32).

| Rôle | Config | Reward | Succès |
|------|--------|--------|--------|
| Meilleure | **#7** — ε=0.8, γ=0.95, lr=0.05 | 7.28 | 100 % |
| Pire | **#18** — ε=0.9, γ=0.99, lr=0.2 | −749.06 | 12 % |

**Interprétation.** MC met à jour à la fin d’épisode : un **lr élevé** amplifie la variance et peut déstabiliser totalement l’estimation ; à l’inverse, **lr=0.05** avec γ=0.95 limite les oscillations et permet une politique proche de l’optimum tabulaire.

### 3.4 Deep Q-Learning

**Source :** [`report-Deep-Q.html`](report-Deep-Q.html) (29/04/2026 à 21:51:04).

| Rôle | Config | Reward | Succès |
|------|--------|--------|--------|
| Meilleure | **#7** — ε=0.9, γ=0.95, lr=0.0005 | 8.12 | 100 % |
| Pire | **#12** — ε=0.9, γ=0.99, lr=0.005 | −396.83 | 0 % |

**Interprétation.** Sur un problème tabulaire, le réseau reste sensible : un **lr trop grand** (0.005) avec γ=0.99 mène à divergence / effondrement du succès ; les meilleures configs utilisent un **lr d’ordre 5×10⁻⁴** et un γ légèrement plus bas (0.95) pour stabiliser la cible bootstrap.

---

## 4. Synthèse inter-agents (comparaison finale HTML)

**Source :** [`report-comparaison-finale.html`](report-comparaison-finale.html), généré le **29/04/2026 à 21:53:57**.

| Agent | Reward moyen | Pas moyens | Succès | Épisodes train | Temps train |
|-------|--------------|------------|--------|----------------|---------------|
| Q-Learning | 7.90 | 13.1 | 100.0 % | 2400 | 0.81 s |
| Deep Q-Learning | 7.13 | 13.8 | 99.6 % | 2000 | 46.34 s |
| Monte Carlo | −14.49 | 33.2 | 89.2 % | 7900 | 3.41 s |
| SARSA | −15.51 | 34.2 | 88.8 % | 7800 | 2.01 s |

**Lecture.** Pour ce run **manuel**, Q-Learning domine à la fois le reward et le succès ; le DQN reste compétitif en qualité de politique mais avec un **coût d’entraînement** beaucoup plus élevé. SARSA et MC sont en dessous du seuil 95 % **sur ce jeu de paramètres d’entraînement manuel**, sans remettre en cause les excellents résultats obtenus pour ces mêmes algorithmes dans les grilles dédiées (sections 3.2–3.3 et `best_params.json`).

---

## 5. Mode temps limité — conclusion

Avec les paramètres issus de [`best_params.json`](best_params.json), entraînement puis **tests chronométrés** sur seed inconnue, la règle **succès ≥ 95 % → reward → temps** désigne **Q-Learning** comme meilleur agent sur la capture d’excution analysée (barres reward / succès / débit d’épisodes + message console « Meilleur agent : Q-Learning »). Le DQN reste une bonne politique mais moins efficace **par seconde** dans ce protocole.

---

## 6. SARSA vs Q-Learning (synthèse)

- **Q-Learning (off-policy)** estime la valeur du meilleur choix futur **indépendamment** de la politique d’exploration : il peut converger plus directement vers la politique optimale du MDP.
- **SARSA (on-policy)** met à jour en fonction de l’action **réellement** prise au pas suivant, ce qui intègre le risque lié à l’exploration (important quand les erreurs coûtent cher, ex. −10 à Taxi-v3).

**Chiffres.** Sur la comparaison finale HTML, Q-Learning affiche **100 %** / reward **7.90** contre SARSA **88.8 %** / **−15.51** pour les hyperparamètres **de ce run manuel**. Sur la grille affinée, SARSA #1 atteint **100 %** / **7.91** ([`report-sarsa-2.html`](report-sarsa-2.html)), très proche du meilleur Q-Learning en reward, ce qui illustre que l’**écart algorithmique** se confond souvent avec l’**écart de réglage** et de budget d’épisodes.

---

## 7. Hyperparamètres — rappel (exigence minimale Q-Learning)

- **ε (epsilon)** : trade-off exploration / exploitation ; souvent décroissant au cours de l’entraînement.
- **γ (gamma)** : actualisation des gains futurs ; proche de 1 pour valoriser la +20 terminale.
- **α / lr (learning rate)** : amplitude des mises à jour ; trop grand → oscillations, trop petit → lenteur ou minimums locaux.

Les mêmes notions s’appliquent à SARSA et Monte Carlo ; le DQN utilise typiquement des **lr plus petits** (ordre 10⁻³–10⁻⁴).

---

## 8. Conclusion

1. Les **méthodes tabulaires** bien réglées (Q-Learning, SARSA #1, MC #7) atteignent des succès proches de 100 % sur Taxi-v3 avec des rewards compétitifs.
2. Le **grid-search** + **persistance** + éventuelle **deuxième vague** (cas SARSA) est essentiel lorsque le seuil 95 % n’est pas atteint.
3. Le **DQN** peut rivaliser en reward ([`report-Deep-Q.html`](report-Deep-Q.html)) mais avec un **surcoût de calcul** important sur ce petit MDP.
4. Les **rapports HTML** fournissent les **graphiques** et courbes demandés par le suiveur ; ce document **interprète** et **croise** les sources.

---

## Annexe A — Fichiers sources (rapports figés)

| Fichier | Date (génération HTML) |
|---------|-------------------------|
| [report-Q-Learning.html](report-Q-Learning.html) | 29/04/2026 21:29:49 |
| [report-sarsa.html](report-sarsa.html) | 29/04/2026 21:32:14 |
| [report-sarsa-2.html](report-sarsa-2.html) | 29/04/2026 21:33:54 |
| [report-monte-carlo.html](report-monte-carlo.html) | 29/04/2026 21:36:32 |
| [report-Deep-Q.html](report-Deep-Q.html) | 29/04/2026 21:51:04 |
| [report-comparaison-finale.html](report-comparaison-finale.html) | 29/04/2026 21:53:57 |

Figures : chaque fichier contient les sections « Évolution de la récompense », « Steps », « Temps d’entraînement », « Taux de succès », « Top 5 / Bottom 5 » avec images intégrées.

---

## Annexe B — Meilleurs hyperparamètres persistés (`best_params.json`)

Extrait au moment de la rédaction (voir fichier pour horodatages exacts par agent) :

| Agent | ε | γ | lr | Épisodes | Reward (métrique sauvée) | Succès |
|-------|---|---|-----|----------|--------------------------|--------|
| Q-Learning | 0.7 | 0.99 | 0.7 | 10000 | 8.39 | 100 % |
| SARSA | 0.5 | 0.99 | 0.2 | 50000 | 7.91 | 100 % |
| Monte Carlo | 0.8 | 0.95 | 0.05 | 50000 | 7.28 | 100 % |
| Deep Q-Learning | 0.9 | 0.95 | 0.0005 | 2000 | 8.12 | 100 % |

*Note.* L’entrée Q-Learning dans le JSON peut différer du « vainqueur » d’un rapport de grille antérieur (#18 dans [`report-Q-Learning.html`](report-Q-Learning.html)) si un benchmark ultérieur a **écrasé** le fichier — d’où l’importance de conserver les HTML datés comme preuve d’expérience.

---

## Annexe C — Checklist exigences encadrant ([FOLLOW_UP_1.md](FOLLOW_UP_1.md))

| Exigence | Où c’est couvert |
|----------|------------------|
| Graphiques | Annexes A (HTML) ; courbes et barres par run |
| Analyse & conclusions | §3–6, §8 |
| Impact des paramètres | §3 par algorithme, §7 |
| Meilleure / pire config expliquée | §3 (tableaux + interprétation) |
| SARSA vs Q-Learning | §6 |
| Protocole benchmark | §2.1–2.3 |
| Temps limité + meilleur modèle | §2.5, §5 |
| ε, γ, lr (min. Q-Learning) | §7 |

---

*Document généré pour remise / rapport écrit — cohérent avec le code [`main.py`](main.py), [`report.py`](report.py) et les artefacts listés.*
