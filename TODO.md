# TODO - Taxi Driver (T-AIA-902)

## Points manquants obligatoires

- [x] **Tuning des parametres en mode user** : permettre a l'utilisateur de modifier `epsilon`, `gamma`, `learning_rate` (pas seulement le nombre d'episodes)
- [x] **Nb d'episodes de test en mode temps limite** : demander le nombre d'episodes de test a l'utilisateur (actuellement hardcode a 10)
- [~] **Affichage d'episodes aleatoires** : afficher le deroulement d'episodes en mode texte (`render_mode="ansi"`) apres le test
  - [x] Mode Manuel
  - [ ] Mode Temps limite
- [ ] **Commentaires analytiques dans le rapport** : ajouter dans le rapport HTML des justifications sur les choix d'algorithmes, la strategie d'optimisation des parametres et rewards
- [ ] **Benchmark multi-parametres** : comparer les performances d'un meme algo avec differentes configurations de parametres (epsilon, gamma, lr)

## Bonus

- [ ] **Extension 2 passagers** : creer un environnement custom ou le taxi doit collecter 2 passagers avec 4 destinations chacun, et optimiser la route
