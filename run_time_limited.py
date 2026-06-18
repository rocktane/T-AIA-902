"""Headless time-limited test: load each agent from its best checkpoint when
available, otherwise retrain with best_params, then test for `time_limit`
seconds on an unknown seed shared across agents.
"""
import random
import os
from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.monte_carlo import MonteCarlo
from agents.deep_q_learning import DeepQLearning
from best_params import load_best_params, resolve_checkpoint_path

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
    p = entry.get("params") if isinstance(entry, dict) else None
    if p is None or entry.get("metrics") is None:
        print(f"{name}: best_params incomplet, skip")
        continue
    print(f"\n>> {name}  params={p}")
    agent = AGENT_CLASSES[name](
        epsilon=float(p["epsilon"]),
        gamma=float(p["gamma"]),
        lr=float(p["lr"]),
    )
    checkpoint_path = resolve_checkpoint_path(entry)
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print(f"   checkpoint: {checkpoint_path}")
    else:
        print("   checkpoint: absent, ré-entraînement")
        train_data = agent.train(int(p["episodes"]), early_stopping=True)
        print(f"   train: {train_data['training_time']:.2f}s, {train_data['n_episodes']} ép.")
    stats = agent.test_time_limited(TIME_LIMIT, seed=SEED)
    print(
        f"   test:  reward={stats['reward_mean']:.2f}  "
        f"succ={stats['success_rate']*100:.1f}%  "
        f"épisodes={stats['test_episodes']}"
    )
