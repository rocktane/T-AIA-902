# Présentation follow-up RL Taxi-v3

## Slide 1 — Benchmark et comparaison finale des agents RL sur Taxi-v3


- Notre projet porte sur `Taxi-v3`, un environnement où un taxi doit récupérer un passager puis le déposer au bon endroit.
- Nous avons comparé quatre approches : `Q-Learning`, `SARSA`, `Monte Carlo` et `Deep Q-Learning`.
- Le fil de la présentation est simple :
  1. montrer comment nous avons benchmarké chaque agent ;
  2. puis montrer quel agent gagne à la fin en mode temps limité.

Petit point possible sur Taxi-v3 :

- Taxi-v3 est un environnement classique de reinforcement learning proposé dans Gymnasium.
- Le taxi se déplace sur une grille, doit récupérer un passager, puis le déposer à la bonne destination.
- L’agent reçoit une pénalité à chaque étape, ce qui l’encourage à être efficace, et une récompense positive quand la mission est réussie.
- C’est donc un bon environnement pour comparer des stratégies d’apprentissage, car on peut mesurer à la fois la réussite et l’efficacité.

Concepts à bien comprendre :

- Un **agent** est le programme qui prend des décisions.
- Un **environnement** est le problème dans lequel l’agent agit.
- En **reinforcement learning**, l’agent apprend par essais/erreurs à partir de récompenses.
- `Taxi-v3` est un environnement **discret** : il y a un nombre fini d’états et un nombre fini d’actions.

---

## Slide 2 — Protocole de benchmark : objectif

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

- Nous avons utilisé un **grid-search**.
- Cela veut dire qu’on choisit plusieurs valeurs pour chaque paramètre, puis qu’on teste toutes les combinaisons possibles.
- Les grilles ne sont pas exactement les mêmes pour tous les agents, car ils n’ont pas les mêmes sensibilités.

Pourquoi ces valeurs ?

 **Q-Learning**.
    * 0.7, 0.8, 0.9 testent trois niveaux d’exploration initiale.
    * Sur Taxi-v4, l’agent doit beaucoup explorer au début pour découvrir les bonnes séquences d’actions.
    * Le code applique un epsilon decay pendant l’entraînement, on veut surtout mesurer si une exploration initiale plus ou moins forte aide la convergence
  **Sarsa**
  * episodes = 20000
    * SARSA converge souvent un peu plus lentement que Q-Learning.
  * epsilon = on garde la même logique que pour Q-Learning.
    * SARSA dépend encore plus du comportement d’exploration, parce qu’il apprend à partir des actions réellement prises.
    * un epsilon élevé peut donc changer fortement la qualité de la politique apprise.
  * gamma = 0.95, 0.99
    * même raison que pour Q-Learning : Taxi-v4 récompense surtout à la fin.
    * il est donc pertinent de comparer une vision long terme assez forte (0.95) à très forte (0.99).
  * lr = 0.05, 0.1, 0.2, 0.3
    * SARSA est souvent plus sensible aux mises à jour agressives.
    * comme il apprend la politique réellement suivie, il “subit” davantage les effets de l’exploration. Un lr trop haut peut le rendre plus instable.

Puis :
  * train_episodes = 50000
      * le premier benchmark suggère que 20000 n’étaient pas suffisants.
      * SARSA a besoin de plus de temps pour stabiliser sa politique.
  * epsilon = 0.5, 0.6, 0.7
      * donc on explore vers le bas, pour réduire l’impact négatif d’une exploration trop forte.
  * gamma = 0.99
      * c’est clairement la meilleure direction dans ce benchmark.
      * inutile de gaspiller du temps sur 0.95 pour l’instant.
  * lr = 0.2, 0.3, 0.4
      * 0.3 est le meilleur point observé.
      * on affine autour.
      * 0.05 et 0.1 semblent trop faibles ici.

  **Monte-Carlo**
  * Monte Carlo converge plus lentement, donc 50000 épisodes minimum
  * gamma élevé reste pertinent
  * lr doit rester plus prudent que Q-Learning
  * epsilon peut rester assez élevé car il faut bien explorer avant que les retours complets d’épisode soient utiles

  **Deep Q-Learning**
  * DQN n’utilise pas du tout la même échelle de lr
      * 0.0005 à 0.005 est une plage raisonnable pour Adam (l'optimiseur utilisé pour ajuster les poids) car le réseau de neurones est plus sensible qu’une Q-table
  * 2000 épisodes suffisent pour un premier benchmark sur Taxi-v3
  * epsilon et gamma restent dans la même logique que les autres agents


Concepts à bien comprendre :

- **Grid-search** : test exhaustif d’une grille de paramètres.
- **Épisodes d’entraînement** : nombre de parties jouées pour apprendre.

Point pédagogique utile :

- On n’a pas mis la même grille partout, car un `learning rate` adapté à Q-Learning ne l’est pas forcément à DQN.

---

## Slide 4 — Protocole de benchmark : critère de succès


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


- Ici, on ne compare pas encore les agents directement entre eux.
- On résume seulement la meilleure configuration trouvée pour chacun.
- `Q-Learning` ressort déjà très fort.
- `SARSA` a eu besoin d’un second benchmark plus ciblé pour devenir vraiment compétitif.
- `Monte Carlo` a été plus sensible aux paramètres.
- `Deep Q-Learning` a bien marché, mais avec un coût plus élevé.

- Le benchmark nous donne un bon point de départ pour la comparaison finale.

Si on te demande pourquoi SARSA a eu un second benchmark :

- Le premier benchmark n’atteignait pas le seuil de `95%` de succès.
- Nous avons donc affiné la grille et augmenté le nombre d’épisodes.

---

## Slide 6 — Mode temps limité : règles du jeu


- Après le benchmark, on passe à la comparaison finale.
- Cette fois, les paramètres sont figés : on recharge les meilleures configurations déjà enregistrées.
- Le temps limité s’applique seulement à la phase de test.
- Tous les agents sont testés dans un environnement inconnu avec le même budget de temps.

Nous avons choisi 5 secondes par agent comme compromis pratique : c’est assez court pour garder une comparaison rapide, mais assez long pour tester un grand nombre d’épisodes et obtenir des résultats stables.

Concepts à bien comprendre :

- **Phase d’entraînement** : l’agent apprend.
- **Phase de test** : on évalue ce qu’il a appris.
- **Environnement inconnu** : l’agent ne connaît pas à l’avance la situation exacte qu’il va rencontrer.

Message important :

- Cette étape répond à la vraie question finale : quel agent gagne dans un cadre commun ?

---

## Slide 7 — Mode temps limité : résultat final


- Le meilleur agent final est `Q-Learning`.
- Il a obtenu `100%` de succès et le meilleur reward moyen.
- `SARSA` et `Deep Q-Learning` sont proches, mais restent un peu en dessous.
- `Monte Carlo` est plus faible dans ce cadre final.

Comment l’expliquer simplement :

- `Q-Learning` combine fiabilité et efficacité.
- Il réussit plus souvent et avec de meilleurs scores moyens.
- Pour `SARSA`, il faut expliquer un peu plus : cet algorithme apprend en tenant compte de l’action qu’il va réellement faire ensuite, donc il intègre davantage l’effet de l’exploration. En pratique, cela le rend souvent plus prudent que Q-Learning : il réussit bien la tâche, mais il converge moins agressivement vers la stratégie la plus efficace. Dans nos résultats, cela se voit avec un très bon taux de succès, mais un reward moyen un peu plus faible.
- Pour `Deep Q-Learning`, le point important n’est pas qu’il soit “mauvais”, au contraire : il obtient lui aussi un très bon taux de succès. Le vrai sujet, c’est le rapport coût / gain. Il demande beaucoup plus de temps d’entraînement qu’un algorithme tabulaire, alors qu’au final il ne fait pas mieux que Q-Learning sur Taxi-v3. Cela suggère qu’un réseau de neurones est ici plus lourd que nécessaire.
- Pour `Monte Carlo`, l’explication intéressante est qu’il s’est montré plus sensible aux hyperparamètres et moins robuste dans le cadre final. Son reward négatif signifie qu’en moyenne il accumule encore trop de pénalités ou fait des trajets trop peu efficaces. Son taux de succès plus faible montre qu’il s’adapte moins bien que les autres agents dans cette comparaison finale.


> Q-Learning est le meilleur agent final, car il obtient à la fois le meilleur taux de succès et le meilleur reward moyen. SARSA et Deep Q-Learning restent solides, mais sont légèrement moins efficaces. Monte Carlo est plus en difficulté dans ce cadre final.

---

## Slide 8 — Paramètres de Q-Learning : epsilon

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

- Le `learning rate` indique à quelle vitesse l’agent modifie ce qu’il croit savoir.
- S’il est trop faible, l’apprentissage est lent.
- S’il est trop élevé, l’apprentissage peut devenir instable.

Lien avec vos résultats :

- Q-Learning supporte ici un learning rate assez élevé.
- Monte Carlo et DQN demandent au contraire des learning rates plus prudents.

Concept à retenir :

- Le learning rate règle l’amplitude des mises à jour pendant l’apprentissage.

---

## Slide 11 — Conclusion et ouverture


- Le point principal, c’est que `Q-Learning` est le meilleur modèle sur `Taxi-v3`, mais surtout que cela nous dit quelque chose de plus général.
- Ici, un algorithme plus complexe comme `Deep Q-Learning` n’apporte pas un meilleur résultat final, alors qu’il coûte beaucoup plus cher à entraîner.
- Cela confirme une idée classique en reinforcement learning : la complexité de l’algorithme doit rester proportionnée à la complexité du problème.


> Pour Taxi-v3, le meilleur algorithme n’est pas le plus sophistiqué, mais le plus adapté au problème.

Limites :

- On pourrait encore tester plusieurs seeds pour renforcer la robustesse.
- On pourrait affiner certaines grilles d’hyperparamètres.
- On pourrait utiliser une autre méthode de recherche que le grid-search.
- On pourrait idéalement complexifier l’environnement pour voir à partir de quand `Deep Q-Learning` devient vraiment plus intéressant.

Ouverture possible :

> Une suite naturelle serait de rendre le problème plus difficile, par exemple avec un environnement plus grand ou moins facilement représentable sous forme de table. Dans ce cas, un algorithme comme DQN pourrait devenir plus pertinent.
