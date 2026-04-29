"""Headless time-limited test: train each agent with its best_params, then test
for `time_limit` seconds on an unknown seed shared across agents.
"""
import random
import time
from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.monte_carlo import MonteCarlo
from agents.deep_q_learning import DeepQLearning
from best_params import load_best_params

AGENT_CLASSES = {
    "Q-Learning": QLearning,
    "SARSA": Sarsa,
    "Monte Carlo": MonteCarlo,
    "Deep Q-Learning": DeepQLearning,
}

TIME_LIMIT = 5
SEED = 42424242

stored = load_best_params()
print(f"Seed test commun : {SEED}  ·  Budget par agent : {TIME_LIMIT}s")
print()

for name in ["Q-Learning", "SARSA", "Monte Carlo", "Deep Q-Learning"]:
    entry = stored.get(name)
    if not entry:
        print(f"{name}: pas de best_params, skip")
        continue
    p = entry["params"]
    print(f"\n>> {name}  params={p}")
    agent = AGENT_CLASSES[name](
        epsilon=float(p["epsilon"]),
        gamma=float(p["gamma"]),
        lr=float(p["lr"]),
    )
    train_data = agent.train(int(p["episodes"]), early_stopping=True)
    print(f"   train: {train_data['training_time']:.2f}s, {train_data['n_episodes']} ép.")
    stats = agent.test_time_limited(TIME_LIMIT, seed=SEED)
    print(
        f"   test:  reward={stats['reward_mean']:.2f}  "
        f"succ={stats['success_rate']*100:.1f}%  "
        f"épisodes={stats['test_episodes']}"
    )
