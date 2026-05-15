# Présentation follow-up RL Taxi-v4

## Slide 1 — Introduction

- Notre projet porte sur `Taxi-v4`, un environnement de reinforcement learning discret.
- Le but est d'apprendre à un taxi à récupérer un passager puis à le déposer à la bonne destination.
- L'environnement reste petit et structuré : `500 états`, `6 actions`, récompense positive à la réussite, pénalité par étape.
- Nous avons comparé quatre agents :
  - `Q-Learning`
  - `SARSA`
  - `Monte Carlo`
  - `Deep Q-Learning`
- Le fil directeur est en deux temps :
  1. benchmarker chaque agent pour trouver ses meilleurs hyperparamètres ;
  2. comparer ces meilleurs agents en mode temps limité.

Message simple :

> L'idée n'est pas seulement de dire qui gagne, mais de montrer comment on construit une comparaison propre.

---

## Slide 2 — Protocole : objectif du benchmark

- Le benchmark sert à comparer plusieurs configurations d'un même agent de façon reproductible.
- Avant de comparer les algorithmes entre eux, il faut déjà s'assurer que chacun est bien réglé.
- Nous sauvegardons les meilleures configurations dans `best_params.json`.

À bien expliquer :

- Un **hyperparamètre** est fixé avant l'entraînement : `epsilon`, `gamma`, `learning rate`, nombre d'épisodes.
- Le benchmark ne cherche pas encore le meilleur agent global.
- Il cherche d'abord la meilleure version de chaque agent.

---

## Slide 3 — Grid-search

- Nous avons utilisé un `grid-search`.
- Le principe est simple : on choisit plusieurs valeurs pour chaque hyperparamètre, puis on teste toutes les combinaisons possibles.
- C'est une méthode exhaustive, simple à expliquer et cohérente pour une soutenance.

Exemple utile :

- Pour `Q-Learning` :
  - `epsilon ∈ {0.7, 0.8, 0.9}`
  - `gamma ∈ {0.95, 0.99}`
  - `lr ∈ {0.1, 0.3, 0.5, 0.7}`
  - donc `3 × 2 × 4 = 24 configurations`

Pourquoi les grilles diffèrent selon les agents :

- `Deep Q-Learning` met à jour un réseau de neurones, donc il a besoin de learning rates beaucoup plus faibles.
- `Monte Carlo` est très sensible au learning rate.
- `SARSA` a nécessité une seconde grille plus ciblée car la première était trop instable.

---

## Slide 4 — Critère de sélection

- Nous avons utilisé une règle en trois étapes :
  1. taux de succès `≥ 95%`
  2. parmi les configurations fiables, meilleur `reward moyen`
  3. en cas de résultats proches, temps d'entraînement le plus court

Pourquoi ce choix est logique :

- Un agent peut réussir sans être efficace.
- Le `reward moyen` permet de distinguer un agent qui réussit vite d'un agent qui réussit avec trop de pas ou trop de pénalités.

Formule orale simple :

> On filtre d'abord la fiabilité, puis on départage sur l'efficacité.

---

## Slide 5 — Benchmark Q-Learning

- `Q-Learning` est l'agent tabulaire le plus performant au benchmark.
- Meilleure configuration :
  - `episodes = 10000`
  - `epsilon = 0.9`
  - `gamma = 0.99`
  - `lr = 0.7`
- Résultats :
  - `reward_mean = 8.51`
  - `success_rate = 100%`
  - `train_time ≈ 1.38s`

Lecture intéressante :

- Les meilleures configs utilisent souvent un `epsilon` élevé.
- `lr = 0.1` concentre beaucoup de mauvais résultats.
- Donc ici, Q-Learning a besoin d'un apprentissage assez agressif.

Message à retenir :

> Sur Taxi-v4, Q-Learning apprend vite et très bien dès qu'on lui donne assez d'exploration et un learning rate élevé.

---

## Slide 6 — Benchmark SARSA

- `SARSA` a demandé une recherche plus ciblée.
- Meilleure configuration :
  - `episodes = 50000`
  - `epsilon = 0.5`
  - `gamma = 0.99`
  - `lr = 0.2`
- Résultats :
  - `reward_mean = 0.03`
  - `success_rate = 96%`
  - `train_time ≈ 2.02s`

Commentaire important :

- SARSA franchit le seuil de fiabilité, mais son reward moyen reste bien inférieur à Q-Learning.
- Il est plus prudent, plus sensible à la politique réellement suivie, donc souvent moins agressif dans l'optimisation.

Si on te demande pourquoi il a l'air “moins bon” malgré 96% :

- Parce qu'il réussit souvent, mais avec davantage de pénalités ou de détours.

---

## Slide 7 — Benchmark Monte Carlo

- `Monte Carlo` s'est révélé très sensible aux hyperparamètres.
- Il a fallu monter à `100000 épisodes`.
- Meilleure configuration :
  - `episodes = 100000`
  - `epsilon = 0.7`
  - `gamma = 0.95`
  - `lr = 0.05`
- Résultats :
  - `reward_mean = 5.83`
  - `success_rate = 99%`
  - `train_time ≈ 4.82s`

Point important :

- Monte Carlo peut très bien marcher, mais seulement avec un learning rate très prudent.
- Dès que `lr` monte à `0.1` ou `0.2`, les performances se dégradent fortement.

---

## Slide 8 — Benchmark Deep Q-Learning

- `Deep Q-Learning` utilise un réseau de neurones à la place d'une Q-table.
- Meilleure configuration :
  - `episodes = 2000`
  - `epsilon = 0.9`
  - `gamma = 0.99`
  - `lr = 0.005`
- Résultats :
  - `reward_mean = 8.29`
  - `success_rate = 100%`
  - `train_time ≈ 50.83s`

Lecture :

- En performance brute, DQN est proche de Q-Learning.
- Mais son coût d'entraînement est sans commune mesure.

Message simple :

> Le gain n'est pas suffisant pour justifier un modèle beaucoup plus lourd sur un problème aussi discret.

---

## Slide 9 — Synthèse benchmark

- À ce stade, chaque agent a sa meilleure configuration sauvegardée.
- On peut résumer :
  - `Q-Learning` : meilleur compromis performance / coût
  - `Deep Q-Learning` : très bon score, mais très coûteux
  - `Monte Carlo` : bon potentiel, mais très sensible
  - `SARSA` : fiable, mais moins efficace en reward

Transition à faire :

> Le benchmark nous donne les meilleurs candidats. La vraie comparaison finale consiste maintenant à les tester dans le même cadre.

---

## Slide 10 — Mode temps limité : règles

- Le mode temps limité ne refait plus un tuning.
- On recharge :
  - les meilleurs hyperparamètres
  - et surtout les `checkpoints` entraînés sauvegardés pendant le benchmark
- Donc si un checkpoint existe, on ne repart pas d'un agent vierge.

Ce qui est chronométré :

- seulement le `test`
- ici `5 secondes par agent`

Ce qui est commun :

- même `seed inconnu`
- même environnement de test
- même budget de temps

Message important :

> Cette fois, on compare les meilleurs agents déjà appris, pas leur capacité à être réentraînés depuis zéro.

---

## Slide 11 — Résultat final en mode temps limité

- Résultats obtenus :
  - `Q-Learning` : `reward 7.92`, `100.0%`, `56 521 épisodes`
  - `Monte Carlo` : `reward 6.37`, `99.3%`, `51 614 épisodes`
  - `SARSA` : `reward 3.77`, `98.0%`, `46 850 épisodes`
  - `Deep Q-Learning` : `reward 4.55`, `98.4%`, `9 011 épisodes`

Conclusion :

- `Q-Learning` reste premier.
- `Monte Carlo` est la vraie surprise de la version checkpointée.
- `Deep Q-Learning` garde une bonne politique, mais son inférence reste trop coûteuse.

Formulation possible :

> Q-Learning garde la première place, mais le protocole checkpointé montre aussi que Monte Carlo généralise bien mieux que ce qu'on aurait pu croire au premier regard.

---

## Slide 12 — Résultat final : lecture comparative

- `Q-Learning` gagne sur les trois dimensions importantes :
  - meilleur reward
  - 100% de succès
  - meilleur débit en nombre d'épisodes testés
- `Monte Carlo` devient un très bon second.
- `SARSA` reste solide mais moins efficace.
- `Deep Q-Learning` souffre surtout du coût d'inférence.

Point méthodologique utile :

- Le résultat final ne dit pas que DQN est “mauvais”.
- Il dit qu'il est moins adapté ici, compte tenu de la taille du problème et du coût calculatoire.

---

## Slide 13 — Epsilon

- `Epsilon` contrôle l'équilibre entre exploration et exploitation.
- Si `epsilon` est trop élevé, l'agent explore trop longtemps.
- S'il est trop faible, il risque de se bloquer trop tôt sur une stratégie imparfaite.

Lien avec les résultats :

- Pour `Q-Learning`, un `epsilon = 0.9` a bien marché.
- Cela suggère qu'il faut beaucoup explorer au départ pour découvrir les bonnes trajectoires dans Taxi-v4.

---

## Slide 14 — Gamma

- `Gamma` mesure l'importance donnée aux récompenses futures.
- Plus `gamma` est proche de `1`, plus l'agent raisonne à long terme.
- Sur Taxi-v4, c'est pertinent car la grosse récompense arrive à la fin de l'épisode.

Lien avec les résultats :

- `gamma = 0.99` ressort très souvent.
- Exception notable : `Monte Carlo` a mieux marché avec `gamma = 0.95`.

---

## Slide 15 — Learning rate

- Le `learning rate` contrôle l'amplitude de la mise à jour.
- Trop faible :
  - apprentissage lent
- Trop élevé :
  - apprentissage instable

Lien avec vos résultats :

- `Q-Learning` supporte bien `lr = 0.7`
- `Monte Carlo` exige `lr = 0.05`
- `Deep Q-Learning` travaille sur une autre échelle, ici `0.0005` à `0.005`

Idée à dire :

> Chaque algorithme a sa propre tolérance au learning rate, parce qu'ils ne mettent pas à jour leur connaissance de la même manière.

---

## Slide 16 — Formule Q-Learning

Formule :

`Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') - Q(s, a) ]`

Comment l'expliquer :

- L'agent met à jour la case `Q(s,a)` après chaque transition.
- Il regarde la meilleure action possible dans l'état suivant.
- Il corrige donc sa valeur actuelle vers une cible optimiste.

Point clé :

- `Q-Learning` est `off-policy`.
- Il apprend la meilleure politique théorique, même si l'action réellement jouée au pas suivant est différente.

Formulation orale :

> Q-Learning apprend à partir du meilleur futur possible.

---

## Slide 17 — Formule SARSA

Formule :

`Q(s, a) ← Q(s, a) + α [ r + γ Q(s', a') - Q(s, a) ]`

Comment l'expliquer :

- La structure ressemble à Q-Learning.
- Mais ici, `a'` est l'action réellement choisie dans l'état suivant.
- La mise à jour dépend donc de la politique effectivement suivie.

Point clé :

- `SARSA` est `on-policy`.
- Il apprend une politique plus prudente, car il intègre son propre comportement exploratoire.

Formulation orale :

> SARSA apprend à partir du futur réellement suivi, pas du meilleur futur théorique.

---

## Slide 18 — Formule Monte Carlo

Formules :

- `G = r_t + γr_(t+1) + γ²r_(t+2) + ...`
- `Q(s, a) ← Q(s, a) + α [ G - Q(s, a) ]`

Comment l'expliquer :

- Monte Carlo n'actualise pas la table à chaque pas.
- Il attend la fin de l'épisode complet.
- Puis il calcule le retour total `G` et corrige les couples état-action rencontrés.

Point clé :

- L'information est riche car elle résume tout l'épisode.
- Mais elle est aussi plus bruitée, donc plus instable si le learning rate est trop grand.

---

## Slide 19 — Formule Deep Q-Learning

Idée centrale :

- DQN ne met pas à jour une Q-table.
- Il met à jour les `poids` d'un réseau de neurones qui approxime `Q(s,a)`.

Formules simples :

- `cible = r + γ max Q_target(s', a')`
- `perte = (Q_policy(s, a) - cible)²`

Explication :

- `policy_net` prédit les Q-values de l'état courant.
- `target_net` calcule une cible plus stable.
- La différence entre prédiction et cible donne une perte.
- L'optimizer `Adam` ajuste ensuite les poids par descente de gradient.

Éléments de stabilité à citer :

- `replay buffer`
- `target network`

Phrase simple :

> Les agents tabulaires mettent à jour des cases. DQN met à jour les paramètres d'une fonction approchée.

---

## Slide 20 — Conclusion

- Le meilleur agent final reste `Q-Learning`.
- C'est le meilleur compromis entre performance, robustesse et coût de calcul sur Taxi-v4.
- `Deep Q-Learning` est très performant, mais trop coûteux pour ce problème.
- `Monte Carlo` ressort finalement beaucoup mieux avec le protocole checkpointé.
- `SARSA` reste fiable, mais moins efficace.

Message final :

> Sur Taxi-v4, le meilleur algorithme n'est pas le plus sophistiqué, mais le plus adapté au problème.

Ouvertures possibles :

- tester plusieurs `seeds`
- affiner encore les grilles
- essayer une autre méthode de recherche d'hyperparamètres, comme une optimisation bayésienne
- complexifier l'environnement pour voir à partir de quand `Deep Q-Learning` devient vraiment plus pertinent
