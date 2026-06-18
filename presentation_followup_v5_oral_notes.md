# Présentation follow-up RL Taxi-v4 — v5 (notes orales)

> Ordre suivi : **Environnement → Algo → Métriques → Benchmarks → Conclusion & limites.**
> Fil conducteur de la partie Algo : un seul squelette de mise à jour, et **seule la cible change** d'un agent à l'autre.

---

## PARTIE 1 — ENVIRONNEMENT

### Slide 1 — Introduction

- Notre projet porte sur `Taxi-v4`, un environnement de reinforcement learning discret.
- Le but : apprendre à un taxi à récupérer un passager, puis à le déposer à la bonne destination.
- Environnement petit et structuré : `500 états`, `6 actions`, récompense terminale positive (`+20`), pénalité par étape.
- Nous comparons quatre agents : `Q-Learning`, `SARSA`, `Monte Carlo`, `Deep Q-Learning`.
- Plan en cinq temps, affiché dès l'intro : **Environnement → Algo → Métriques → Benchmarks → Conclusion & limites.**

Message simple :

> L'idée n'est pas seulement de dire qui gagne, mais de montrer comment on construit une comparaison propre.

---

## PARTIE 2 — ALGO

### Slide 2 — Le squelette commun (apprendre par l'erreur)

C'est la slide la plus importante de la partie Algo : elle pose le cadre des trois suivantes.

Formule affichée :

> `Q(s, a) ← Q(s, a) + α [ cible − Q(s, a) ]`

À expliquer lentement, morceau par morceau :

- `Q(s, a)` : ce que l'agent croit aujourd'hui sur l'action `a` dans l'état `s`.
- `cible` : une meilleure estimation de cette valeur, construite **après** avoir agi.
- `cible − Q(s, a)` : l'écart entre ce qui était prévu et ce qui est observé. C'est **l'erreur**, la « surprise ».
- `α` (learning rate) : la fraction de cette erreur qu'on applique à chaque mise à jour.

Le point clé à marteler :

> Nos trois agents tabulaires utilisent **exactement cette formule**. La seule chose qui change, c'est **la cible**.

Si on te demande pourquoi commencer par ça :

- Parce que ça transforme « trois algorithmes à mémoriser » en « une formule + trois façons de définir la cible ». C'est beaucoup plus simple à présenter et à comprendre.

---

### Slide 3 — Q-Learning vs SARSA : une seule différence

On présente les deux **en même temps**, car ils ne diffèrent que par un seul terme.

Les deux formules, côte à côte :

- Q-Learning : `Q(s,a) ← Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]`
- SARSA :      `Q(s,a) ← Q(s,a) + α [ r + γ     Q(s',a') − Q(s,a) ]`

La seule différence (à pointer du doigt à l'écran) :

- Q-Learning utilise `max Q(s',a')` → **la meilleure action possible** dans l'état suivant.
- SARSA utilise `Q(s',a')` → **l'action réellement choisie** au pas suivant (exploration comprise).

Ce que ça veut dire concrètement :

- `Q-Learning` est **off-policy** : il apprend la meilleure politique théorique, même s'il ne joue pas cette action ensuite. → plus agressif, vise l'optimal.
- `SARSA` est **on-policy** : il apprend la politique qu'il suit vraiment, exploration incluse. → plus prudent, plus stable.

Formulation orale simple :

> Q-Learning apprend à partir du **meilleur futur possible**. SARSA apprend à partir du **futur réellement suivi**.

Lien avec les résultats (anticipe la partie Benchmarks) :

- Sur Taxi-v4, Q-Learning obtient un reward plus élevé, SARSA est plus stable mais moins efficace en reward moyen.

---

### Slide 4 — Monte Carlo : observer le futur au lieu de l'estimer

> ⚠️ C'est la slide pointée par les encadrants. Le cœur de l'explication est le **contraste avec le bootstrapping**, pas la formule seule.

Formules affichées :

- `G_t = r_t + γ r_(t+1) + γ² r_(t+2) + ...`  (le **retour réel**)
- `Q(s,a) ← Q(s,a) + α [ G_t − Q(s,a) ]`

La différence essentielle, à dire explicitement :

- Q-Learning et SARSA **estiment** le futur avec une autre Q-value (`γ Q(s',a')`). Ils s'appuient sur leur propre estimation : c'est le **bootstrapping**.
- Monte Carlo **n'estime rien**. Il attend la **fin complète de l'épisode**, puis utilise `G`, la **somme réelle des récompenses** effectivement obtenues.

La conséquence directe (et c'est ça qui explique tous les résultats) :

- **Non biaisé** : la cible est le vrai retour observé, pas une approximation.
- **Mais très bruité** : `G` dépend de *toutes* les actions (souvent aléatoires) prises jusqu'à la fin de l'épisode.
- Donc **très sensible au learning rate** : un `α` élevé fait sur-réagir l'agent à des retours très variables → effondrement. C'est exactement ce qu'on verra au benchmark.

Phrase à retenir :

> Q-Learning et SARSA mettent à jour **à chaque pas** avec une cible *estimée*. Monte Carlo met à jour **en fin d'épisode** avec le retour *réel*. Même squelette, cible radicalement différente.

Si on te demande « pourquoi ne pas toujours utiliser le retour réel, plus exact ? » :

- Parce qu'il faut attendre la fin de l'épisode (pas d'apprentissage en cours de route) et qu'il est beaucoup plus bruité. L'estimation des méthodes TD est biaisée mais bien plus stable.

---

### Slide 5 — Deep Q-Learning

Idée centrale :

- Le DQN ne met pas à jour une Q-table, mais les **poids** d'un réseau de neurones qui approxime `Q(s,a)`.

Formules :

- `cible = r + γ max Q_target(s', a')`
- `perte = (Q_policy(s, a) − cible)²`

Comment le réseau apprend :

- `policy_net` prédit les Q-values de l'état courant ; on extrait celle de l'action jouée `a`.
- `target_net` calcule une cible plus stable sur `s'`.
- La perte MSE mesure l'écart prédiction / cible.
- L'optimizer `Adam` ajuste les poids par descente de gradient.

Ce qui stabilise l'apprentissage (à citer) :

- `replay buffer` : rejoue des expériences passées au lieu d'apprendre sur une seule transition.
- `target network` : mis à jour périodiquement, pas à chaque pas.

Phrase simple :

> Les agents tabulaires mettent à jour des cases. Le DQN met à jour les paramètres d'une fonction approchée.

---

## PARTIE 3 — MÉTRIQUES

### Slide 6 — Protocole : objectif du benchmark

- Le benchmark sert à comparer plusieurs configurations d'un même agent de façon reproductible.
- Avant de comparer les algorithmes entre eux, il faut d'abord s'assurer que chacun est bien réglé.
- On sauvegarde les meilleures configurations dans `best_params.json`.

À bien expliquer :

- Un **hyperparamètre** est fixé avant l'entraînement : `ε`, `γ`, `α`, nombre d'épisodes.
- Le benchmark ne cherche pas encore le meilleur agent global, mais **la meilleure version de chaque agent**.

---

### Slide 7 — Epsilon (ε)

- `ε` contrôle l'équilibre **exploration / exploitation** : la probabilité de jouer une action au hasard.
- Trop élevé → l'agent reste aléatoire trop longtemps, apprend lentement.
- Trop faible → l'agent se bloque trop tôt sur une stratégie imparfaite.

Lien avec les résultats :

- Pour Q-Learning, `ε = 0.9` a bien marché : il faut beaucoup explorer au départ pour découvrir les bonnes trajectoires dans Taxi-v4.

---

### Slide 8 — Gamma (γ)

- `γ` mesure l'importance donnée aux **récompenses futures**.
- Proche de 1 → l'agent raisonne à long terme ; faible → l'agent devient myope.
- Sur Taxi-v4, la grosse récompense (`+20`) arrive en fin d'épisode, donc le long terme compte.

Lien avec les résultats :

- `γ = 0.99` ressort très souvent. Exception : Monte Carlo a mieux marché avec `γ = 0.95`.

---

### Slide 9 — Learning rate (α)

- `α` contrôle l'amplitude de la mise à jour (cf. le squelette commun, slide 2).
- Trop faible → apprentissage lent. Trop élevé → apprentissage instable.

Lien avec les résultats :

- Q-Learning supporte `α = 0.7`, Monte Carlo exige `α = 0.05`, le DQN travaille entre `0.0005` et `0.005`.

Idée à dire :

> Chaque algorithme a sa propre tolérance au learning rate, parce qu'ils ne mettent pas à jour leur connaissance de la même manière (rappel du squelette commun et des trois cibles).

---

### Slide 10 — Grid-search

- Principe : on choisit plusieurs valeurs pour chaque hyperparamètre, puis on teste **toutes les combinaisons** (produit cartésien).
- Méthode exhaustive, simple à expliquer et cohérente pour une soutenance.
- Exemple Q-Learning : `ε ∈ {0.7, 0.8, 0.9} × γ ∈ {0.95, 0.99} × lr ∈ {0.1, 0.3, 0.5, 0.7}` = `24 configurations`.

Pourquoi les grilles diffèrent selon les agents :

- Le DQN met à jour un réseau → learning rates beaucoup plus faibles.
- Monte Carlo est très sensible au learning rate.
- SARSA a nécessité une grille recentrée car la première était trop instable.

---

### Slide 11 — Critère de sélection

Règle en trois étapes :

1. taux de succès `≥ 95%` (filtre de fiabilité) ;
2. parmi les configs fiables, meilleur `reward moyen` ;
3. en cas d'égalité, temps d'entraînement le plus court.

Pourquoi cet ordre :

- Un agent peut réussir sans être efficace. Le reward moyen départage ceux qui réussissent vite de ceux qui réussissent avec trop de pas ou trop de pénalités.

> On filtre d'abord la fiabilité, puis on départage sur l'efficacité.

---

## PARTIE 4 — BENCHMARKS

### Slide 12 — Benchmark Q-Learning

- Agent tabulaire le plus performant au benchmark.
- Meilleure config : `ε=0.9 · γ=0.99 · lr=0.7 · 10k ép.` → `reward 8.51`, `100% succès`, `≈ 1.38s`.
- Lecture : `lr = 0.1` concentre les échecs ; dès `0.3–0.7`, les scores montent fortement.

> Q-Learning apprend vite et très bien dès qu'on lui donne assez d'exploration et un learning rate élevé.

---

### Slide 13 — Benchmark SARSA

- Recherche plus ciblée. Meilleure config : `ε=0.5 · γ=0.99 · lr=0.2 · 50k ép.` → `reward 0.03`, `96% succès`, `≈ 2.02s`.
- Une seule config dépasse 95%. SARSA franchit le seuil de fiabilité mais reste peu efficace en reward.
- Rappel du « pourquoi » (slide 3) : étant on-policy, il intègre son exploration → plus prudent, moins agressif dans l'optimisation.

Si on te demande pourquoi il a l'air « moins bon » malgré 96 % :

- Parce qu'il réussit souvent, mais avec davantage de pénalités ou de détours.

---

### Slide 14 — Benchmark Monte Carlo

- Très sensible aux hyperparamètres ; il a fallu monter à `100 000 épisodes`.
- Meilleure config : `ε=0.7 · γ=0.95 · lr=0.05 · 100k ép.` → `reward 5.83`, `99% succès`, `≈ 4.82s`.
- Lecture : dès que `lr` dépasse `0.05`, les performances s'effondrent. **C'est la preuve concrète de la slide 4** : ses retours sont bruités, donc il ne tolère pas un grand learning rate.

---

### Slide 15 — Benchmark Deep Q-Learning

- Meilleure config : `ε=0.9 · γ=0.99 · lr=0.005 · 2k ép.` → `reward 8.29`, `100% succès`, `≈ 50.83s`.
- En performance brute, proche de Q-Learning. Mais coût d'entraînement sans commune mesure (40–60 s par config).

> Le gain n'est pas suffisant pour justifier un modèle beaucoup plus lourd sur un problème aussi discret.

---

### Slide 16 — Synthèse benchmark

- Chaque agent a sa meilleure configuration sauvegardée :
  - `Q-Learning` : meilleur compromis performance / coût (8.51, 100%, 1.38s).
  - `Deep Q-Learning` : très bon score (8.29) mais très coûteux (50.83s).
  - `Monte Carlo` : bon potentiel (5.83) mais très sensible.
  - `SARSA` : fiable (96%) mais peu efficace en reward (0.03).

Transition :

> Le benchmark donne les meilleurs candidats. La vraie comparaison finale consiste à les tester dans le même cadre.

---

### Slide 17 — Mode temps limité : règles

- Plus de tuning. On recharge les meilleurs hyperparamètres **et surtout les checkpoints entraînés**.
- Si un checkpoint existe, on ne repart pas d'un agent vierge.
- Ce qui est chronométré : seulement le **test**, ici `5 secondes par agent`.
- Commun à tous : même `seed inconnu`, même environnement, même budget de temps.

> Cette fois, on compare les meilleurs agents déjà appris, pas leur capacité à être réentraînés depuis zéro.

---

### Slide 18 — Résultat final en mode temps limité

- `Q-Learning` : `reward 7.92`, `100.0%`, `56 521 épisodes`.
- `Monte Carlo` : `reward 6.37`, `99.3%`, `51 614 épisodes`.
- `SARSA` : `reward 3.77`, `98.0%`, `46 850 épisodes`.
- `Deep Q-Learning` : `reward 4.55`, `98.4%`, `9 011 épisodes`.

Conclusion :

- `Q-Learning` reste premier sur les trois dimensions.
- `Monte Carlo` est la vraie surprise de la version checkpointée.
- `Deep Q-Learning` garde une bonne politique, mais son inférence reste trop coûteuse (9 011 épisodes seulement).

> Q-Learning garde la première place, mais le protocole checkpointé montre aussi que Monte Carlo généralise bien mieux que ce qu'on aurait pu croire au premier regard.

---

## PARTIE 5 — CONCLUSION & LIMITES

### Slide 19 — Conclusion

- Le meilleur agent final reste `Q-Learning` : meilleur compromis performance / robustesse / coût sur Taxi-v4.
- `Deep Q-Learning` : très performant mais trop coûteux pour ce problème.
- `Monte Carlo` : ressort beaucoup mieux avec le protocole checkpointé.
- `SARSA` : fiable mais moins efficace.

Message final :

> Sur Taxi-v4, le meilleur algorithme n'est pas le plus sophistiqué, mais le plus **adapté** au problème.

---

### Slide 20 — Limites & ouvertures

> Demande explicite des encadrants : finir sur les limites. À présenter avec recul, sans défensive — montrer qu'on connaît les angles morts de l'étude.

Limites à assumer :

- **Un seul seed de test** : robustesse statistique non garantie, les classements pourraient bouger.
- **Grille discrète choisie à la main** : grid-search exhaustif uniquement sur les valeurs fixées ; l'optimum réel est peut-être entre deux.
- **Un seul environnement** : Taxi-v4 est petit et discret, ce qui avantage structurellement les méthodes tabulaires.
- **Débit dépendant du matériel** : le mode temps limité mesure aussi la machine, pas seulement l'algorithme.
- **DQN possiblement sous-exploité** : peu d'épisodes, architecture simple.

Ouvertures :

- Rejouer le protocole sur **plusieurs seeds** (moyenne ± écart-type).
- Remplacer le grid-search par une **optimisation bayésienne**.
- Tester un **environnement plus complexe** (états continus, image) pour voir quand le DQN redevient pertinent.
- Mesurer le coût d'inférence indépendamment du matériel.

Phrase de clôture possible :

> Si l'environnement grandissait, Deep Q-Learning reprendrait l'avantage : c'est précisément là que l'approximation par réseau devient indispensable.
