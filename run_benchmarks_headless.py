"""Headless benchmark runner. Runs the grid-search for one or more agents
without the questionary CLI, and persists top5/bottom5 via best_params.

Usage:
    python run_benchmarks_headless.py q
    python run_benchmarks_headless.py sarsa
    python run_benchmarks_headless.py mc
    python run_benchmarks_headless.py dqn
    python run_benchmarks_headless.py all
"""
import sys
import time
from itertools import product

from agents.q_learning import QLearning
from agents.sarsa import Sarsa
from agents.monte_carlo import MonteCarlo
from agents.deep_q_learning import DeepQLearning
from best_params import save_best_params, save_benchmark_runs


GRIDS = {
    "q": {
        "name": "Q-Learning",
        "cls": QLearning,
        "train_episodes": 10000,
        "test_episodes": 100,
        "epsilons": [0.7, 0.8, 0.9],
        "gammas": [0.95, 0.99],
        "lrs": [0.1, 0.3, 0.5, 0.7],
    },
    "sarsa": {
        "name": "SARSA",
        "cls": Sarsa,
        "train_episodes": 50000,
        "test_episodes": 100,
        "epsilons": [0.5, 0.6, 0.7],
        "gammas": [0.99],
        "lrs": [0.2, 0.3, 0.4],
    },
    "mc": {
        "name": "Monte Carlo",
        "cls": MonteCarlo,
        "train_episodes": 50000,
        "test_episodes": 100,
        "epsilons": [0.7, 0.8, 0.9],
        "gammas": [0.95, 0.99],
        "lrs": [0.05, 0.1, 0.2],
    },
    "dqn": {
        "name": "Deep Q-Learning",
        "cls": DeepQLearning,
        "train_episodes": 2000,
        "test_episodes": 100,
        "epsilons": [0.8, 0.9],
        "gammas": [0.95, 0.99],
        "lrs": [0.0005, 0.001, 0.005],
    },
}


def run_grid(key):
    grid = GRIDS[key]
    name = grid["name"]
    cls = grid["cls"]
    train_ep = grid["train_episodes"]
    test_ep = grid["test_episodes"]
    configs = list(product(grid["epsilons"], grid["gammas"], grid["lrs"]))

    print(f"\n=== {name} : {len(configs)} configurations · {train_ep} ép. train ===")
    runs = []
    grid_start = time.time()
    for i, (eps, gam, lr) in enumerate(configs, 1):
        cfg_start = time.time()
        agent = cls(epsilon=eps, gamma=gam, lr=lr)
        train_data = agent.train(train_ep)
        agent.test(test_ep)
        stats = agent.last_test_stats
        params = {
            "episodes": train_ep,
            "epsilon": eps,
            "gamma": gam,
            "lr": lr,
        }
        metrics = {
            "reward_mean": stats["reward_mean"],
            "reward_std": stats["reward_std"],
            "success_rate": stats["success_rate"],
            "train_time": train_data["training_time"],
            "test_episodes": stats["test_episodes"],
            "epsilon_tolerance": stats["epsilon_tolerance"],
        }
        label = f"{name} #{i} (e={eps}, g={gam}, lr={lr})"
        runs.append({"label": label, "params": params, "metrics": metrics})
        save_best_params(name, params, metrics)
        elapsed = time.time() - cfg_start
        print(
            f"  [{i:>2}/{len(configs)}] e={eps} g={gam} lr={lr}  "
            f"reward={stats['reward_mean']:>7.2f}  "
            f"succ={stats['success_rate']*100:>5.1f}%  "
            f"train={train_data['training_time']:>5.2f}s  "
            f"({elapsed:.1f}s)"
        )

    save_benchmark_runs(name, runs, train_ep)
    total = time.time() - grid_start
    print(f"=== {name} terminé en {total:.1f}s. top5/bottom5 sauvegardés ===")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    arg = sys.argv[1].lower()
    if arg == "all":
        for key in ["q", "sarsa", "mc", "dqn"]:
            run_grid(key)
    elif arg in GRIDS:
        run_grid(arg)
    else:
        print(f"Agent inconnu : {arg}. Choix : q, sarsa, mc, dqn, all")
        sys.exit(1)


if __name__ == "__main__":
    main()
