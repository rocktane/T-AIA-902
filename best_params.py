import json
import os
from datetime import datetime

BEST_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
QUALITY_THRESHOLD = 0.95


def load_best_params():
    if not os.path.exists(BEST_PARAMS_FILE):
        return {}
    try:
        with open(BEST_PARAMS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data):
    with open(BEST_PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_best_params(agent_name):
    return load_best_params().get(agent_name)


def is_better(new_metrics, old_metrics):
    """
    Règle de sélection :
      1. success_rate >= 95% (filtre qualité)
      2. reward moyen supérieur (au-delà de l'ε)
      3. en cas d'ex-æquo : temps d'entraînement minimum
    """
    new_success = new_metrics["success_rate"]
    if new_success < QUALITY_THRESHOLD:
        return False
    if old_metrics is None:
        return True
    old_success = old_metrics["success_rate"]
    if old_success < QUALITY_THRESHOLD:
        return True

    new_reward = new_metrics["reward_mean"]
    old_reward = old_metrics["reward_mean"]
    epsilon = new_metrics.get("epsilon_tolerance", 0.0)

    if new_reward > old_reward + epsilon:
        return True
    if new_reward < old_reward - epsilon:
        return False
    return new_metrics.get("train_time", float("inf")) < old_metrics.get("train_time", float("inf"))


def save_best_params(agent_name, params, metrics):
    """
    Sauvegarde les params uniquement s'ils sont meilleurs que l'existant.
    metrics : {reward_mean, reward_std, success_rate, train_time, test_episodes, epsilon_tolerance}
    Retourne True si écrit, False sinon.
    """
    data = load_best_params()
    current = data.get(agent_name)
    current_metrics = current["metrics"] if current else None
    if not is_better(metrics, current_metrics):
        return False
    existing_last_benchmark = current.get("last_benchmark") if current else None
    data[agent_name] = {
        "params": params,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if existing_last_benchmark is not None:
        data[agent_name]["last_benchmark"] = existing_last_benchmark
    _write(data)
    return True


def save_benchmark_runs(agent_name, runs, train_episodes):
    """
    Persiste le top 5 et bottom 5 du dernier benchmark de cet agent.
    Ne touche pas au winner (params/metrics/timestamp).

    runs : liste de {"label": str, "params": {...}, "metrics": {...}}
    """
    if not runs:
        return False

    data = load_best_params()
    entry = data.get(agent_name, {})

    sorted_desc = sorted(runs, key=lambda r: r["metrics"]["reward_mean"], reverse=True)
    qualified = [r for r in sorted_desc if r["metrics"]["success_rate"] >= QUALITY_THRESHOLD]
    top_pool = qualified if len(qualified) >= 5 else sorted_desc
    top5 = [{"params": r["params"], "metrics": r["metrics"]} for r in top_pool[:5]]

    sorted_asc = sorted(runs, key=lambda r: r["metrics"]["reward_mean"])
    bottom5 = [{"params": r["params"], "metrics": r["metrics"]} for r in sorted_asc[:5]]

    entry["last_benchmark"] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_configs": len(runs),
        "train_episodes": train_episodes,
        "top5": top5,
        "bottom5": bottom5,
    }
    data[agent_name] = entry
    _write(data)
    return True


def load_benchmark_runs(agent_name):
    entry = get_best_params(agent_name)
    if not entry:
        return None
    return entry.get("last_benchmark")
