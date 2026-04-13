import numpy as np


class EarlyStopping:
    """
    Détecte la stagnation du reward moyen glissant.

    - window_size : taille de la fenêtre glissante (défaut 100)
    - patience : nombre de fenêtres consécutives sans amélioration avant stop
    - min_episodes : interdit l'arrêt avant ce nombre d'épisodes
    - min_improvement (ε) : seuil d'amélioration — auto-calculé comme std(window)/sqrt(n)
                           si non fourni, pour être cohérent avec la règle de sélection.
    """

    def __init__(self, window_size=100, patience=3, min_episodes=0, min_improvement=None):
        self.window_size = window_size
        self.patience = patience
        self.min_episodes = min_episodes
        self.min_improvement = min_improvement
        self.rewards_buffer = []
        self.best_window_mean = -float("inf")
        self.episodes_without_improvement = 0
        self.triggered_at = None

    def should_stop(self, episode, episode_reward):
        self.rewards_buffer.append(episode_reward)

        if episode < self.min_episodes:
            return False
        if len(self.rewards_buffer) < self.window_size:
            return False
        if episode % self.window_size != 0:
            return False

        window = self.rewards_buffer[-self.window_size:]
        current_mean = float(np.mean(window))
        epsilon = (
            self.min_improvement
            if self.min_improvement is not None
            else float(np.std(window)) / np.sqrt(self.window_size)
        )

        if current_mean > self.best_window_mean + epsilon:
            self.best_window_mean = current_mean
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

        if self.episodes_without_improvement >= self.patience:
            self.triggered_at = episode
            return True
        return False
