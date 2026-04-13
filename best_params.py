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
    data[agent_name] = {
        "params": params,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    _write(data)
    return True
