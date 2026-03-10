import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


def rolling_mean(data, window=100):
    if len(data) < window:
        window = max(1, len(data) // 5)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_reward_history(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        train = data["train"]
        if train is None:
            continue
        smoothed = rolling_mean(train["reward_history"])
        ax.plot(smoothed, label=name)
    ax.set_title("Évolution de la récompense par épisode")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Récompense (moyenne glissante)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def plot_steps_history(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        train = data["train"]
        if train is None:
            continue
        smoothed = rolling_mean(train["steps_history"])
        ax.plot(smoothed, label=name)
    ax.set_title("Évolution du nombre de steps par épisode")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Nombre de steps (moyenne glissante)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def plot_training_time(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    times = []
    for name, data in results.items():
        train = data["train"]
        if train is None:
            continue
        names.append(name)
        times.append(train["training_time"])
    bars = ax.bar(names, times, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.2f}s", ha='center', va='bottom', fontsize=10)
    ax.set_title("Comparaison du temps d'entraînement")
    ax.set_ylabel("Temps (secondes)")
    ax.grid(True, alpha=0.3, axis='y')
    return fig_to_base64(fig)


def plot_test_success(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = []
    rates = []
    for name, data in results.items():
        test = data["test"]
        rate_str = test[3]
        rate = float(rate_str.replace('%', ''))
        names.append(name)
        rates.append(rate)
    bars = ax.bar(names, rates, color=plt.cm.Set2(np.linspace(0, 1, len(names))))
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r:.1f}%", ha='center', va='bottom', fontsize=10)
    ax.set_title("Comparaison du taux de succès au test")
    ax.set_ylabel("Taux de succès (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    return fig_to_base64(fig)


def plot_success_over_training(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        train = data["train"]
        if train is None:
            continue
        success = train["success_history"]
        smoothed = rolling_mean(success) * 100
        window = 100 if len(success) >= 100 else max(1, len(success) // 5)
        time_axis = np.linspace(0, train["training_time"], len(smoothed))
        ax.plot(time_axis, smoothed, label=name)
    ax.set_title("Évolution du taux de succès au fil du temps d'entraînement")
    ax.set_xlabel("Temps d'entraînement (secondes)")
    ax.set_ylabel("Taux de succès (%, moyenne glissante)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig_to_base64(fig)


def generate_report(results):
    """
    Génère un rapport HTML avec des graphiques comparatifs.

    Args:
        results: dict au format:
            {
                "Agent Name": {
                    "train": { "reward_history", "steps_history", "training_time",
                               "n_episodes", "success_history" } ou None,
                    "test": [name, mean_reward, mean_steps, "XX.X%"]
                }
            }

    Returns:
        Le chemin du fichier HTML généré.
    """
    img_reward = plot_reward_history(results)
    img_steps = plot_steps_history(results)
    img_time = plot_training_time(results)
    img_success = plot_test_success(results)
    img_success_training = plot_success_over_training(results)

    # Table des résultats de test
    test_rows = ""
    for name, data in results.items():
        test = data["test"]
        test_rows += f"""
            <tr>
                <td>{test[0]}</td>
                <td>{test[1]:.2f}</td>
                <td>{test[2]:.1f}</td>
                <td>{test[3]}</td>
            </tr>"""

    # Table des infos d'entraînement
    train_rows = ""
    for name, data in results.items():
        train = data["train"]
        if train is None:
            train_rows += f"""
            <tr>
                <td>{name}</td>
                <td>-</td>
                <td>-</td>
            </tr>"""
        else:
            train_rows += f"""
            <tr>
                <td>{name}</td>
                <td>{train['n_episodes']}</td>
                <td>{train['training_time']:.2f}s</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Rapport - Agents RL sur Taxi-v3</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            text-align: center;
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 40px;
        }}
        .chart {{
            text-align: center;
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .chart img {{
            max-width: 100%;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: center;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .footer {{
            text-align: center;
            color: #888;
            margin-top: 40px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>Rapport comparatif des agents RL - Taxi-v3</h1>
    <p style="text-align:center; color:#666;">Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}</p>

    <h2>Résumé des résultats de test</h2>
    <table>
        <tr>
            <th>Agent</th>
            <th>Récompense moyenne</th>
            <th>Nombre de pas moyen</th>
            <th>Taux de succès</th>
        </tr>
        {test_rows}
    </table>

    <h2>Informations d'entraînement</h2>
    <table>
        <tr>
            <th>Agent</th>
            <th>Nombre d'épisodes</th>
            <th>Temps d'entraînement</th>
        </tr>
        {train_rows}
    </table>

    <h2>1. Évolution de la récompense par épisode</h2>
    <div class="chart">
        <img src="data:image/png;base64,{img_reward}" alt="Reward history">
    </div>

    <h2>2. Évolution du nombre de steps par épisode</h2>
    <div class="chart">
        <img src="data:image/png;base64,{img_steps}" alt="Steps history">
    </div>

    <h2>3. Comparaison du temps d'entraînement</h2>
    <div class="chart">
        <img src="data:image/png;base64,{img_time}" alt="Training time">
    </div>

    <h2>4. Comparaison du taux de succès au test</h2>
    <div class="chart">
        <img src="data:image/png;base64,{img_success}" alt="Test success rate">
    </div>

    <h2>5. Évolution du taux de succès pendant l'entraînement</h2>
    <div class="chart">
        <img src="data:image/png;base64,{img_success_training}" alt="Success rate over training">
    </div>

    <div class="footer">
        <p>Environnement : Taxi-v3 (Gymnasium)</p>
    </div>
</body>
</html>"""

    output_path = "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
