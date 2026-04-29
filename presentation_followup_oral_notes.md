# Guide d'oral — présentation follow-up RL Taxi-v3

## Slide 1 — Benchmark et comparaison finale des agents RL sur Taxi-v3

Ce que tu peux dire :

- Notre projet porte sur `Taxi-v3`, un environnement où un taxi doit récupérer un passager puis le déposer au bon endroit.
- Nous avons comparé quatre approches : `Q-Learning`, `SARSA`, `Monte Carlo` et `Deep Q-Learning`.
- Le fil de la présentation est simple :
  1. montrer comment nous avons benchmarké chaque agent ;
  2. puis montrer quel agent gagne à la fin en mode temps limité.

Concepts à bien comprendre :

- Un **agent** est le programme qui prend des décisions.
- Un **environnement** est le problème dans lequel l’agent agit.
- En **reinforcement learning**, l’agent apprend par essais/erreurs à partir de récompenses.

Phrase courte possible :

> Aujourd’hui, on va montrer comment on a choisi les meilleurs paramètres pour chaque agent, puis comment on les a comparés dans un cadre final commun.

---

## Slide 2 — Protocole de benchmark : objectif

Ce que tu peux dire :

- Le benchmark sert à comparer des configurations de manière reproductible.
- L’idée est d’éviter une comparaison “au feeling”.
- Avant de dire quel algorithme est le meilleur, il faut d’abord vérifier qu’on l’a testé avec de bons paramètres.

Concepts à bien comprendre :

- Un **benchmark** est un protocole de test.
- Un **hyperparamètre** est une valeur fixée avant l’entraînement, par exemple `epsilon`, `gamma` ou `learning rate`.

Message important :

- Le benchmark ne cherche pas directement le meilleur modèle global.
- Il cherche d’abord la meilleure configuration pour chaque modèle.

---

## Slide 3 — Protocole de benchmark : espace de recherche

Ce que tu peux dire :

- Nous avons utilisé un **grid-search**.
- Cela veut dire qu’on choisit plusieurs valeurs pour chaque paramètre, puis qu’on teste toutes les combinaisons possibles.
- Les grilles ne sont pas exactement les mêmes pour tous les agents, car ils n’ont pas les mêmes sensibilités.

Concepts à bien comprendre :

- **Grid-search** : test exhaustif d’une grille de paramètres.
- **Épisodes d’entraînement** : nombre de parties jouées pour apprendre.

Point pédagogique utile :

- On n’a pas mis la même grille partout, car un `learning rate` adapté à Q-Learning ne l’est pas forcément à DQN.

---

## Slide 4 — Protocole de benchmark : critère de succès

Ce que tu peux dire :

- Nous avons défini une règle claire pour choisir la meilleure configuration.
- D’abord, on demande un taux de succès d’au moins `95%`.
- Ensuite, parmi les configurations fiables, on choisit celle avec le meilleur `reward moyen`.
- Enfin, si deux résultats sont très proches, on regarde le temps d’entraînement.

Concepts à bien comprendre :

- **Taux de succès** : proportion d’épisodes réussis.
- **Reward moyen** : score moyen obtenu sur les épisodes de test.
- **Temps d’entraînement** : temps nécessaire pour apprendre.

Pourquoi ce choix est logique :

- Un agent peut réussir souvent mais être peu efficace.
- Le reward moyen permet de distinguer un agent qui réussit vite d’un agent qui réussit en faisant trop d’étapes ou trop d’erreurs.

---

## Slide 5 — Résultats du benchmark : meilleure configuration par agent

Ce que tu peux dire :

- Ici, on ne compare pas encore les agents directement entre eux.
- On résume seulement la meilleure configuration trouvée pour chacun.
- `Q-Learning` ressort déjà très fort.
- `SARSA` a eu besoin d’un second benchmark plus ciblé pour devenir vraiment compétitif.
- `Monte Carlo` a été plus sensible aux paramètres.
- `Deep Q-Learning` a bien marché, mais avec un coût plus élevé.

Message à faire passer :

- Le benchmark nous donne un bon point de départ pour la comparaison finale.

Si on te demande pourquoi SARSA a eu un second benchmark :

- Le premier benchmark n’atteignait pas le seuil de `95%` de succès.
- Nous avons donc affiné la grille et augmenté le nombre d’épisodes.

---

## Slide 6 — Mode temps limité : règles du jeu

Ce que tu peux dire :

- Après le benchmark, on passe à la comparaison finale.
- Cette fois, les paramètres sont figés : on recharge les meilleures configurations déjà enregistrées.
- Le temps limité s’applique seulement à la phase de test.
- Tous les agents sont testés dans un environnement inconnu avec le même budget de temps.

Concepts à bien comprendre :

- **Phase d’entraînement** : l’agent apprend.
- **Phase de test** : on évalue ce qu’il a appris.
- **Environnement inconnu** : l’agent ne connaît pas à l’avance la situation exacte qu’il va rencontrer.

Message important :

- Cette étape répond à la vraie question finale : quel agent gagne dans un cadre commun ?

---

## Slide 7 — Mode temps limité : résultat final

Ce que tu peux dire :

- Le meilleur agent final est `Q-Learning`.
- Il a obtenu `100%` de succès et le meilleur reward moyen.
- `SARSA` et `Deep Q-Learning` sont proches, mais restent un peu en dessous.
- `Monte Carlo` est plus faible dans ce cadre final.

Comment l’expliquer simplement :

- `Q-Learning` combine fiabilité et efficacité.
- Il réussit plus souvent et avec de meilleurs scores moyens.

Formulation simple possible :

> La conclusion finale, c’est que Q-Learning est le meilleur compromis entre réussite et efficacité sur Taxi-v3.

---

## Slide 8 — Paramètres de Q-Learning : epsilon

Ce que tu peux dire :

- `Epsilon` contrôle l’équilibre entre exploration et exploitation.
- Explorer, c’est essayer des actions nouvelles.
- Exploiter, c’est choisir l’action qui semble déjà la meilleure.
- Si epsilon est trop haut, l’agent explore trop longtemps.
- S’il est trop bas, il risque de se bloquer trop tôt sur une stratégie imparfaite.

Lien avec vos résultats :

- Dans nos résultats, un `epsilon` initial élevé a bien marché pour Q-Learning.
- Cela suggère qu’au début, il faut beaucoup explorer pour découvrir les bonnes trajectoires.

---

## Slide 9 — Paramètres de Q-Learning : gamma

Ce que tu peux dire :

- `Gamma` mesure l’importance donnée aux récompenses futures.
- Si gamma est élevé, l’agent valorise davantage les gains qui arrivent plus tard.
- Si gamma est faible, il privilégie plus le court terme.

Lien avec Taxi-v3 :

- Dans Taxi-v3, la grosse récompense arrive à la fin, quand le passager est bien déposé.
- Donc un `gamma` élevé est souvent pertinent.

Formulation simple :

> Gamma répond à la question : est-ce que l’agent pense plutôt à tout de suite, ou à la récompense de fin d’épisode ?

---

## Slide 10 — Paramètres de Q-Learning : learning rate

Ce que tu peux dire :

- Le `learning rate` indique à quelle vitesse l’agent modifie ce qu’il croit savoir.
- S’il est trop faible, l’apprentissage est lent.
- S’il est trop élevé, l’apprentissage peut devenir instable.

Lien avec vos résultats :

- Q-Learning supporte ici un learning rate assez élevé.
- Monte Carlo et DQN demandent au contraire des learning rates plus prudents.

Concept à retenir :

- Le learning rate règle l’amplitude des mises à jour pendant l’apprentissage.

---

## Slide 11 — Lien avec nos résultats, synthèse et limites

Ce que tu peux dire :

- Les résultats montrent que les hyperparamètres changent réellement le comportement des agents.
- Q-Learning a été le plus efficace au final.
- SARSA est plus prudent et a demandé plus d’épisodes.
- Monte Carlo est plus sensible au learning rate.
- DQN fonctionne, mais il est coûteux pour un environnement aussi petit.

Conclusion à dire clairement :

> Pour Taxi-v3, qui est un environnement discret et de petite taille, les méthodes tabulaires, en particulier Q-Learning, restent les plus adaptées.

Limites que tu peux citer si on te pose la question :

- On pourrait encore tester plusieurs seeds pour renforcer la robustesse.
- On pourrait affiner certaines grilles.
- On pourrait étudier d’autres métriques ou d’autres environnements.

---

## Conseils généraux pour l’oral

- Ne récite pas tout le texte : garde l’idée principale de chaque slide.
- Répète souvent la logique globale :
  1. benchmark pour bien régler chaque agent ;
  2. temps limité pour comparer les agents entre eux.
- Si tu ne veux pas entrer dans un détail trop technique, ramène toujours la réponse à une intuition simple :
  - `epsilon` = explorer ou exploiter ;
  - `gamma` = court terme ou long terme ;
  - `learning rate` = vitesse d’apprentissage.
