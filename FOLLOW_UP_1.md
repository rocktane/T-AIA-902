# FOLLOW UP 1 - Point intermédiaire encadrant

**Date** : 2 avril 2026

## Résumé des retours

### Mode Temps Limité
- Le mode temps limité doit utiliser les **meilleurs paramètres déjà trouvés** (pas de tuning pendant le chrono).
- Le temps limité ne concerne que la phase de **test**, en **environnement inconnu**.
- Doit inclure la **visualisation des résultats**.

### Persistance des meilleurs paramètres
- **Sauvegarder les meilleurs paramètres** trouvés pour chaque modèle de manière persistante (fichier JSON ou similaire).
- Ajouter un **timestamp** pour savoir quand la meilleure configuration a été trouvée.
- Toujours conserver les meilleurs résultats pour améliorer les benchmarks et comparaisons futures.
- **Question ouverte** : définir sur quelles **métriques** se baser pour déterminer ce qu'est un "meilleur" résultat (taux de succès ? reward moyen ? temps d'entraînement ? combinaison pondérée ?).

### Entraînement
- Ajouter un **early stopping** quand le modèle stagne sur une métrique.
- **Question ouverte** : quelle métrique utiliser pour détecter la stagnation ? (reward moyen sur N derniers épisodes ? taux de succès glissant ?)

### Affichage & Graphiques
- Afficher quel est le **meilleur modèle** global (basé sur quelles métriques ? à définir).
- Utiliser une **échelle logarithmique** pour les graphiques.
- Garder les **5 meilleurs et 5 pires** configurations dans le report et les graphes terminal.
- **Question ouverte** : sur quelles métriques classer meilleurs/pires ?

### Mode Manuel
- Il manque le **nom du modèle** dans l'output du mode manuel.

### Rapport écrit
- Inclure des **graphiques**, **conclusions** et **analyse des résultats**.
- Expliquer l'**impact des paramètres** sur les résultats.
- Conclure sur pourquoi la pire config est la pire et pourquoi la meilleure est la meilleure.
- Exemple : expliquer pourquoi SARSA est mieux/moins bien que Q-Learning.

### Présentation (et rapport)
- **Benchmark** : expliquer le protocole pour trouver les meilleurs paramètres (grid-search, analyse).
- **Temps limité** : conclure sur quel modèle est le meilleur au final.
- **Explications des paramètres** : au minimum sur Q-Learning (epsilon, gamma, learning rate).

---

## TODO List

### Priorité haute - Fonctionnalités

- [ ] **Définir les métriques de référence** : choisir les métriques pour classer les modèles (ex: taux de succès, reward moyen, temps d'entraînement) et documenter ce choix
- [ ] **Persistance des meilleurs paramètres** : sauvegarder les meilleurs params par modèle dans un fichier JSON avec timestamp
- [ ] **Early stopping** : implémenter l'arrêt anticipé de l'entraînement quand stagnation détectée sur la métrique choisie
- [ ] **Mode temps limité** : refactorer pour utiliser les meilleurs paramètres sauvegardés, limiter le chrono aux tests, en environnement inconnu, avec visualisation des résultats
- [ ] **Afficher le meilleur modèle** : après benchmark/comparaison, afficher clairement quel modèle est le meilleur et pourquoi

### Priorité moyenne - Affichage & UX

- [ ] **Nom du modèle en mode manuel** : ajouter le nom du modèle dans l'output du mode manuel
- [ ] **Échelle logarithmique** : passer les graphiques en échelle log
- [ ] **Top 5 / Bottom 5** : ne garder que les 5 meilleurs et 5 pires dans le report HTML et les graphes terminal

### Priorité basse - Rapport & Présentation

- [ ] **Graphiques dans le rapport** : ajouter des graphiques pertinents dans le rapport HTML
- [ ] **Analyse d'impact des paramètres** : expliquer dans le rapport comment chaque paramètre influence les résultats
- [ ] **Conclusions comparatives** : expliquer pourquoi le meilleur est le meilleur et le pire est le pire (ex: SARSA vs Q-Learning)
- [ ] **Section benchmark dans le rapport** : documenter le protocole de recherche des meilleurs paramètres (grid-search, etc.)
- [ ] **Section temps limité dans le rapport** : conclure sur le meilleur modèle final
- [ ] **Explications des paramètres** : section dédiée expliquant epsilon, gamma, learning rate (au moins pour Q-Learning)

### Questions ouvertes à trancher

- [ ] Quelles métriques pour définir "meilleur résultat" ? (taux de succès, reward moyen, combinaison ?)
- [ ] Quelle métrique pour l'early stopping ? (reward moyen glissant sur N épisodes ?)
- [ ] Quelles métriques pour le classement top 5 / bottom 5 ?
