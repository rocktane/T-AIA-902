# RECAP — Reinforcement Learning sur Taxi-v3

## L'environnement Taxi-v3

Taxi-v3 est un environnement **déterministe** de Gymnasium. Un taxi doit récupérer un passager et le déposer à la bonne destination sur une grille 5x5.

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

- **500 états** : 25 positions du taxi × 5 positions du passager (4 emplacements + dans le taxi) × 4 destinations
- **6 actions** : Sud, Nord, Est, Ouest, Prendre passager, Déposer passager
- **Récompenses** :
  - `-1` par action (pénalité de temps → incite à être rapide)
  - `+20` quand le passager est déposé au bon endroit
  - `-10` pour une action illégale (prendre/déposer au mauvais endroit)
- **Épisode optimal** : ~13 steps → récompense de `+7` (20 - 13)
- **Déterministe** : contrairement à FrozenLake, une action donne toujours le même résultat (pas de glissement aléatoire)

---

## Concepts fondamentaux du RL

Le Reinforcement Learning repose sur un cycle :

```
    ┌──────────┐
    │          │
    │   Agent  │──── action ────►┌──────────────┐
    │          │                 │              │
    │          │◄── état ───────│ Environnement│
    │          │◄── récompense ─│              │
    └──────────┘                └──────────────┘
```

L'agent interagit avec l'environnement en boucle :
1. Il observe l'**état** actuel (sa position, celle du passager, etc.)
2. Il choisit une **action**
3. L'environnement lui renvoie un **nouvel état** et une **récompense**
4. L'agent utilise cette récompense pour **ajuster sa stratégie**

L'objectif n'est pas de maximiser la récompense immédiate, mais la **récompense cumulée** sur tout l'épisode. C'est pour ça qu'il accepte des -1 à chaque step : il sait que le +20 final compensera, à condition d'y arriver vite.

---

## Exploration vs Exploitation (Epsilon-Greedy)

C'est le dilemme central du RL : **explorer** (essayer des actions au hasard pour découvrir) ou **exploiter** (utiliser ce qu'on sait déjà pour maximiser la récompense).

La stratégie **epsilon-greedy** résout ce dilemme avec un paramètre `epsilon` (entre 0 et 1) :

```
    Tirage aléatoire entre 0 et 1
              │
              ▼
    ┌─────────────────────┐
    │  > epsilon ?         │
    │                     │
    ├── OUI ──► Exploitation (meilleure action connue)
    │
    └── NON ──► Exploration (action aléatoire)
```

- `epsilon = 0.9` → 90% d'exploration, 10% d'exploitation (début d'entraînement)
- `epsilon = 0.05` → 5% d'exploration, 95% d'exploitation (fin d'entraînement)

### Piège rencontré

La condition `if random > epsilon` signifie **exploitation**. C'est contre-intuitif : un epsilon élevé (0.9) donne PLUS d'exploration car il est plus difficile de dépasser 0.9 avec un tirage aléatoire. Inverser cette condition fait que l'agent exploite sans avoir exploré → il ne converge jamais.

---

## Epsilon Decay

L'epsilon diminue progressivement au fil de l'entraînement. Au début, l'agent explore beaucoup (il ne sait rien). À la fin, il exploite ses connaissances.

On utilise un **decay exponentiel** :

```
    epsilon_nouveau = epsilon × decay_rate

    avec decay_rate = (epsilon_final / epsilon_initial) ^ (1 / n_episodes)
```

Cela garantit que epsilon passe de `0.9` à `0.05` en exactement `n_episodes` épisodes, avec une décroissance douce.

### Point important

Le decay se fait **une fois par épisode**, pas à chaque step. Si on le fait à chaque step, l'epsilon chute beaucoup trop vite (un épisode peut faire 200 steps) et l'agent arrête d'explorer bien trop tôt.

---

## Le Discount Factor (Gamma)

Gamma (γ) détermine l'importance que l'agent accorde aux **récompenses futures** par rapport aux récompenses immédiates.

```
    γ = 0.99 → "Les récompenses futures comptent presque autant que les immédiates"
    γ = 0.5  → "Les récompenses lointaines sont fortement diminuées"
    γ = 0    → "Seule la récompense immédiate compte"
```

Concrètement, gamma intervient dans le calcul de la valeur d'un état. Pour un épisode de 13 steps suivi d'une récompense de +20 :

```
    Avec γ = 0.99 : la récompense de +20 vue depuis le 1er step vaut 20 × 0.99^13 ≈ 17.5
    Avec γ = 0.5  : la récompense de +20 vue depuis le 1er step vaut 20 × 0.5^13  ≈ 0.002
```

Pour Taxi-v3, un **gamma élevé (0.99)** est nécessaire car la grosse récompense (+20) arrive seulement au dernier step. Avec un gamma trop bas, cette récompense ne se propage pas assez vers les premiers steps et l'agent n'apprend pas à planifier son trajet.

---

## Le Learning Rate (Alpha)

Le learning rate (α, ou `lr`) contrôle la **vitesse d'apprentissage** : à quel point l'agent ajuste ses connaissances quand il reçoit une nouvelle information.

```
    Q(s,a) = Q(s,a) + α × (nouvelle_info - Q(s,a))
                       ▲
                       │
                   learning rate
```

- `α = 1.0` → L'agent remplace totalement l'ancienne valeur par la nouvelle. Il oublie tout ce qu'il savait.
- `α = 0.01` → L'agent ne bouge presque pas. Il est très conservateur et apprend très lentement.
- `α = 0.7` (Q-Learning Taxi) → Bon compromis pour un environnement déterministe. Comme Taxi-v3 n'a pas d'aléatoire, on peut se permettre un learning rate élevé.
- `α = 0.2` (SARSA Taxi) → SARSA est on-policy et plus sensible aux fluctuations, donc un learning rate plus bas stabilise l'apprentissage.
- `α = 0.1` (Monte Carlo) → Monte Carlo utilise le retour complet de l'épisode (beaucoup d'information d'un coup), donc un petit learning rate évite des mises à jour trop brutales.

### Le learning rate pour les réseaux de neurones (DQN)

En DQN, le learning rate est celui de l'**optimizer** (Adam, lr=0.001). Il contrôle la taille des ajustements des poids du réseau. Les réseaux de neurones sont beaucoup plus sensibles au learning rate que les Q-tables :
- Trop élevé → le réseau oscille et n'apprend pas
- Trop bas → l'apprentissage est trop lent
- Adam adapte automatiquement le learning rate pour chaque poids, ce qui le rend plus robuste que d'autres optimizers

---

## La Q-Table

La Q-table est un tableau qui stocke une **valeur estimée** pour chaque paire (état, action). Cette valeur représente "la récompense totale espérée si je fais cette action dans cet état, puis que je joue de manière optimale ensuite".

```
    Q-Table pour Taxi-v3 : 500 lignes × 6 colonnes

              Sud    Nord    Est   Ouest  Prendre  Déposer
    État 0  [ 2.1,   5.3,   1.0,   0.5,   -8.2,    -7.1  ]
    État 1  [ 3.4,   2.1,   4.5,   1.2,   -9.0,    -8.5  ]
    ...
    État 499 [...]

    Pour l'état 0, l'agent choisit "Nord" (valeur max = 5.3)
```

Au début, toutes les valeurs sont à 0 (l'agent ne sait rien). Elles se mettent à jour progressivement grâce à la formule de Q-Learning, SARSA, ou Monte Carlo.

---

## Q-Learning (Off-Policy)

Q-Learning est un algorithme **off-policy** et **model-free**. "Off-policy" signifie que l'agent apprend la stratégie **optimale** indépendamment de ce qu'il fait réellement (il peut explorer tout en apprenant la meilleure politique).

### Formule de mise à jour

```
    Q(s, a) ← Q(s, a) + α × [ R + γ × max Q(s', :) - Q(s, a) ]
                                         ▲
                                         │
                                    Meilleure action
                                    possible dans s'
```

La clé est `max Q(s', :)` : on prend la **meilleure valeur possible** dans l'état suivant, **peu importe l'action qu'on va réellement prendre**. C'est ce qui rend Q-Learning off-policy — il apprend la politique optimale même en explorant.

### Pourquoi ça marche pour Taxi-v3

Taxi-v3 est déterministe et sans risque : il n'y a pas de pénalité catastrophique pour l'exploration (juste -1 par step). Q-Learning est donc idéal car il converge plus vite que SARSA en apprenant directement la politique optimale.

---

## SARSA (On-Policy)

SARSA est un algorithme **on-policy**. Son nom vient de la séquence qu'il utilise : **S**tate, **A**ction, **R**eward, **S**tate, **A**ction.

### Formule de mise à jour

```
    Q(s, a) ← Q(s, a) + α × [ R + γ × Q(s', a') - Q(s, a) ]
                                              ▲
                                              │
                                    Action réellement
                                    choisie dans s'
```

La différence avec Q-Learning : au lieu de `max Q(s', :)`, SARSA utilise `Q(s', a')` — la valeur de l'action **qu'il va réellement prendre** dans le prochain état. Il apprend donc la politique qu'il **suit réellement**, exploration comprise.

### Le pattern next_action

En SARSA, l'action `a'` choisie pour le prochain état doit être **la même** que celle effectivement jouée au step suivant. Cela implique un pattern particulier :

```
    Début d'épisode : next_action = None

    À chaque step :
        action = next_action (si existe) OU choisir nouvelle action
        → jouer action → obtenir s', r
        next_action = choisir action pour s'     ← on choisit MAINTENANT
        → mise à jour Q(s, action) avec Q(s', next_action)

    Au step suivant :
        action = next_action                     ← on RÉUTILISE ce choix
```

Si on ne réutilise pas `next_action`, on choisit deux actions différentes pour le même état : une pour la mise à jour Q et une pour le step suivant. SARSA perd alors sa propriété on-policy.

---

## Q-Learning vs SARSA — La vraie différence

```
    Situation : l'agent est près d'un "bord dangereux"

    Q-Learning (off-policy) :
    ┌──────┐     ┌──────┐     ┌──────┐
    │  s   │────►│  s'  │     │DANGER│
    │      │     │max=5 │     │ -100 │
    └──────┘     └──────┘     └──────┘
    → Met à jour avec max Q(s') = 5
    → "La meilleure action dans s' vaut 5, je fonce"
    → Ignore le risque d'explorer vers DANGER

    SARSA (on-policy) :
    ┌──────┐     ┌──────┐     ┌──────┐
    │  s   │────►│  s'  │────►│DANGER│
    │      │     │a'→-2 │     │ -100 │
    └──────┘     └──────┘     └──────┘
    → Met à jour avec Q(s', a') = -2  (l'action réelle, qui peut être exploratoire)
    → "L'action que je vais vraiment faire dans s' vaut -2, prudence"
    → Tient compte du risque d'exploration
```

- **Q-Learning** est **optimiste** : il suppose qu'on jouera toujours la meilleure action. Converge plus vite mais peut surestimer.
- **SARSA** est **réaliste** : il tient compte du fait qu'on explore encore. Plus prudent, plus stable, mais converge plus lentement.

Pour **Taxi-v3**, Q-Learning est préférable car l'environnement est déterministe et les pénalités d'exploration sont faibles (-1 par step, -10 pour une action illégale — pas catastrophique).

---

## Monte Carlo

Monte Carlo est fondamentalement différent de Q-Learning et SARSA. Au lieu d'apprendre **à chaque step**, il attend la **fin de l'épisode** et utilise les récompenses réellement obtenues.

### Le retour G

Le retour G est la récompense cumulée pondérée par gamma, calculée **en remontant** depuis la fin de l'épisode :

```
    Épisode : s₁,a₁,r₁ → s₂,a₂,r₂ → s₃,a₃,r₃ → FIN

    Calcul en partant de la fin :
        G₃ = r₃
        G₂ = r₂ + γ × G₃
        G₁ = r₁ + γ × G₂

    Mise à jour :
        Q(s₃, a₃) += α × (G₃ - Q(s₃, a₃))
        Q(s₂, a₂) += α × (G₂ - Q(s₂, a₂))
        Q(s₁, a₁) += α × (G₁ - Q(s₁, a₁))
```

### Différence avec Q-Learning/SARSA

```
    Q-Learning/SARSA (TD learning) :
    ┌────┐    ┌────┐
    │ s  │───►│ s' │    Met à jour Q(s,a) IMMÉDIATEMENT
    └────┘    └────┘    en utilisant une ESTIMATION de Q(s')
                        → Apprentissage step-by-step

    Monte Carlo :
    ┌────┐    ┌────┐    ┌────┐    ┌────┐
    │ s₁ │───►│ s₂ │───►│ s₃ │───►│FIN │    Met à jour APRÈS l'épisode
    └────┘    └────┘    └────┘    └────┘    en utilisant les VRAIES récompenses
                                            → Pas de biais, mais plus de variance
```

- **Avantage** : pas de biais (utilise les vraies récompenses, pas des estimations)
- **Inconvénient** : haute variance (un épisode malchanceux peut donner un G très éloigné de la vraie valeur), et nécessite beaucoup plus d'épisodes pour converger

Monte Carlo a besoin de ~100 000 épisodes sur Taxi-v3 alors que Q-Learning converge en ~10 000.

---

## Deep Q-Learning (DQN)

DQN remplace la Q-table par un **réseau de neurones**. Au lieu de stocker une valeur par case du tableau, le réseau **apprend une fonction** qui prédit les Q-values.

### Pourquoi un réseau de neurones ?

La Q-table fonctionne bien quand l'espace d'états est petit (500 pour Taxi-v3). Mais pour un jeu vidéo avec des millions d'états possibles (pixels d'un écran), une table devient impossible. Un réseau de neurones peut **généraliser** : il apprend des patterns et peut estimer les Q-values d'états qu'il n'a jamais vus.

### Architecture du réseau

```
    Entrée (500)          Couche cachée 1 (64)     Couche cachée 2 (64)     Sortie (6)
    ┌─────────┐           ┌──────────────┐         ┌──────────────┐         ┌────────┐
    │ one-hot │──── × ───►│    ReLU      │── × ──►│    ReLU      │── × ──►│ Q-vals │
    │ [0,0,1, │  poids    │  64 neurones │ poids  │  64 neurones │ poids  │ 6 vals │
    │  0,...] │           └──────────────┘         └──────────────┘         └────────┘
    └─────────┘
```

- **Entrée** : l'état encodé en one-hot (un vecteur de 500 zéros avec un seul 1 à la position de l'état)
- **ReLU** : fonction d'activation qui garde les valeurs positives et met les négatives à 0. Sans elle, le réseau serait juste une multiplication de matrices (linéaire) et ne pourrait pas apprendre des relations complexes
- **Sortie** : 6 valeurs Q, une par action possible. Pas de ReLU en sortie car les Q-values peuvent être négatives

### One-Hot Encoding

L'état dans Taxi-v3 est un entier (0 à 499). Le réseau a besoin d'un vecteur en entrée. Le one-hot transforme l'entier en vecteur :

```
    État 3 → [0, 0, 0, 1, 0, 0, ..., 0]    (500 dimensions)
                      ▲
                  position 3 = 1, tout le reste = 0
```

C'est nécessaire car les états ne sont pas ordinaux : l'état 200 n'est pas "plus grand" que l'état 50. Le one-hot évite que le réseau interprète les numéros comme des valeurs numériques.

### Le Replay Buffer

Sans replay buffer, le réseau apprend les expériences dans l'ordre. Problème : les expériences consécutives sont très **corrélées** (steps successifs du même épisode). Le réseau sur-apprend les patterns récents et oublie les anciens.

```
    Sans replay buffer :
    exp1 → exp2 → exp3 → exp4    (corrélées, même épisode)
    Le réseau sur-apprend le trajet actuel

    Avec replay buffer :
    Stocker : [exp1, exp2, ..., exp10000]
    Piocher au hasard un batch de 64 :
    [exp42, exp8391, exp103, exp7722, ...]    (décorrélées)
    Le réseau apprend de manière équilibrée
```

Le buffer a une taille maximale (10000). Quand il est plein, la plus ancienne expérience est supprimée.

### Le Target Network (réseau cible)

DQN utilise **deux réseaux identiques** :
- Le **policy network** : celui qui apprend (ses poids sont mis à jour à chaque batch)
- Le **target network** : une copie figée, utilisée uniquement pour calculer les cibles

```
    Avec un seul réseau :
    ┌──────────┐
    │ Réseau   │──► Prédiction Q(s,a)      ← change à chaque update
    │          │──► Cible r + γ×maxQ(s')   ← change AUSSI à chaque update
    └──────────┘
    → La cible bouge en même temps que la prédiction
    → Le réseau "court après sa propre ombre"
    → Instabilité, oscillations

    Avec deux réseaux :
    ┌──────────┐
    │ Policy   │──► Prédiction Q(s,a)      ← change à chaque update
    └──────────┘
    ┌──────────┐
    │ Target   │──► Cible r + γ×maxQ(s')   ← FIGÉ pendant N steps
    └──────────┘
    → La cible est stable pendant 100 steps
    → Le policy network peut converger vers cette cible
    → Tous les 100 steps : copier policy → target
```

C'est comme tirer sur une cible fixe vs une cible qui bouge : beaucoup plus facile de viser quand elle ne bouge pas.

### Le processus d'entraînement (batch)

```
    1. Jouer un step → stocker (s, a, r, s', done) dans le buffer
    2. Si buffer assez rempli (≥ 64) :
       a. Piocher 64 expériences au hasard
       b. Pour chaque expérience, calculer :
          - Prédiction : policy_net(s)[a]
          - Cible : r + γ × max(target_net(s')) × (1 - done)
       c. Calculer la loss (MSE entre prédictions et cibles)
       d. Backpropagation : ajuster les poids du policy_net
    3. Tous les 100 steps : copier policy_net → target_net
```

Le `(1 - done)` est important : si l'épisode est terminé, il n'y a pas de futur, donc la cible est simplement `r` sans le terme γ × max Q(s').

---

## Pièges rencontrés et leçons apprises

### 1. Epsilon-greedy inversé

```
    ❌  if random < epsilon → exploitation
    ✅  if random > epsilon → exploitation
```

Quand epsilon est élevé (0.9), il faut que l'exploration soit **fréquente**. Avec `random > epsilon`, un tirage doit dépasser 0.9 pour exploiter → seulement 10% du temps → 90% d'exploration. Logique.

### 2. Confusion state / next_state dans la mise à jour Q

```
    ❌  state, reward, ... = env.step(action)
        Q[state, action] = ...  ← state a DÉJÀ été écrasé par le nouveau !

    ✅  current_state = state
        state, reward, ... = env.step(action)
        Q[current_state, action] = ...  ← on utilise l'ancien état
```

`env.step()` retourne le **nouvel** état. Si on utilise la même variable, on perd l'état d'origine. La mise à jour Q a besoin de l'état **avant** l'action.

### 3. Instances séparées pour train et test

```
    ❌  QLearning().train(10000)
        QLearning().test(10)     ← NOUVEL objet, Q-table vide !

    ✅  agent = QLearning()
        agent.train(10000)
        agent.test(10)           ← même objet, Q-table entraînée
```

`QLearning()` crée un **nouvel objet** à chaque appel, avec une Q-table initialisée à zéro. Il faut garder l'agent dans une variable pour que le test utilise la Q-table entraînée.

### 4. Métriques par step vs par épisode

```
    ❌  while not done:
            ...
            reward_history.append(total_reward)    ← 200 entrées par épisode !

    ✅  while not done:
            ...
        reward_history.append(total_reward)        ← 1 entrée par épisode
```

L'indentation compte. Les métriques d'un épisode (récompense totale, nombre de steps) ne sont connues qu'**après** la fin de l'épisode. Les append doivent être **en dehors** de la boucle `while not done`.

### 5. Epsilon decay dans la boucle de steps

```
    ❌  while not done:
            ...
            epsilon *= decay_rate    ← decay à chaque STEP (200× trop rapide)

    ✅  while not done:
            ...
        epsilon *= decay_rate        ← decay à chaque ÉPISODE
```

Même problème d'indentation. Un épisode peut durer 200 steps. Si le decay est par step, l'epsilon atteint son minimum dès les premiers épisodes.

### 6. Variable qui écrase un module

```
    ❌  import time
        time = 0                    ← écrase le module time !
        time.time()                 ← erreur : int n'a pas d'attribut time

    ✅  import time
        training_time = 0           ← nom différent
        time.time()                 ← fonctionne
```

En Python, les noms de variables et de modules partagent le même espace. Nommer une variable comme un module importé écrase ce module.

### 7. SARSA : ne pas réutiliser next_action

```
    ❌  while not done:
            action = choose_action(state)        ← nouvelle action à chaque step
            next_action = choose_action(s')      ← utilisé pour Q mais IGNORÉ au step suivant
            Q[s,a] += α × (r + γ×Q[s',a'] - Q[s,a])

    ✅  while not done:
            action = next_action or choose_action(state)   ← RÉUTILISE a'
            next_action = choose_action(s')
            Q[s,a] += α × (r + γ×Q[s',a'] - Q[s,a])
```

SARSA est on-policy : l'action utilisée dans la mise à jour Q **doit être la même** que celle jouée au step suivant. Sinon, on choisit deux actions différentes et l'algorithme perd sa cohérence.

### 8. Oublier `episode += 1`

Sans incrémenter le compteur d'épisodes, la condition `episode >= n_episodes` n'est jamais vraie. La boucle `while True` ne s'arrête jamais (sauf en mode `time_limit`).

### 9. Le test doit être en epsilon = 0

Pendant le test, l'agent doit **toujours exploiter** (epsilon = 0). Si l'agent explore encore pendant le test, ses résultats ne reflètent pas ce qu'il a appris. Le test mesure la **performance réelle** de la politique apprise, sans aléatoire.
