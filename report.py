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
        ax.plot(range(1, len(smoothed) + 1), smoothed, label=name)
    ax.set_title("Évolution de la récompense par épisode (échelle log X)")
    ax.set_xlabel("Épisode (log)")
    ax.set_ylabel("Récompense (moyenne glissante)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    return fig_to_base64(fig)


def plot_steps_history(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, data in results.items():
        train = data["train"]
        if train is None:
            continue
        smoothed = rolling_mean(train["steps_history"])
        ax.plot(range(1, len(smoothed) + 1), smoothed, label=name)
    ax.set_title("Évolution du nombre de steps par épisode (échelle log X)")
    ax.set_xlabel("Épisode (log)")
    ax.set_ylabel("Nombre de steps (moyenne glissante)")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
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


def _ranked(results):
    ranked = []
    for name, data in results.items():
        test = data["test"]
        reward_mean = float(test[1])
        success = float(test[3].replace('%', ''))
        train_time = data["train"]["training_time"] if data.get("train") else None
        ranked.append({
            "name": name,
            "reward": reward_mean,
            "steps": float(test[2]),
            "success": success,
            "train_time": train_time,
        })
    ranked.sort(key=lambda r: r["reward"], reverse=True)
    return ranked


def plot_top_bottom(results):
    ranked = _ranked(results)
    if len(ranked) <= 1:
        return None
    top = ranked[:5]
    bottom = ranked[-5:] if len(ranked) > 5 else []
    fig, axes = plt.subplots(1, 2 if bottom else 1, figsize=(14, 5))
    if not bottom:
        axes = [axes]
    axes[0].barh([r["name"] for r in top][::-1], [r["reward"] for r in top][::-1], color="#2ecc71")
    axes[0].set_title("Top 5 (reward moyen)")
    axes[0].set_xlabel("Reward moyen")
    axes[0].grid(True, alpha=0.3, axis="x")
    if bottom:
        axes[1].barh([r["name"] for r in bottom][::-1], [r["reward"] for r in bottom][::-1], color="#e74c3c")
        axes[1].set_title("Bottom 5 (reward moyen)")
        axes[1].set_xlabel("Reward moyen")
        axes[1].grid(True, alpha=0.3, axis="x")
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


def generate_analysis(results):
    """
    Génère des commentaires analytiques basés sur les résultats.
    """
    sections = []

    # --- Analyse du choix des algorithmes ---
    sections.append("""
    <h2>Analyse des choix d'algorithmes</h2>
    <div class="analysis">
        <p><strong>Taxi-v3</strong> est un environnement à espace d'états discret (500 états) et espace d'actions discret (6 actions).
        Ce contexte justifie le choix de méthodes tabulaires (Q-Learning, SARSA, Monte Carlo) qui peuvent
        représenter explicitement la Q-table complète, garantissant une convergence optimale sans approximation.</p>

        <p><strong>Bruteforce</strong> sert de baseline : il agit aléatoirement sans apprentissage, ce qui permet
        de quantifier le gain réel apporté par chaque algorithme d'apprentissage.</p>

        <p><strong>Q-Learning</strong> (off-policy, TD) apprend la politique optimale indépendamment de la politique
        d'exploration. C'est l'algorithme de référence pour les environnements tabulaires grâce à sa convergence
        prouvée et sa simplicité d'implémentation.</p>

        <p><strong>SARSA</strong> (on-policy, TD) apprend la valeur de la politique effectivement suivie. Il est
        généralement plus conservateur que Q-Learning car il tient compte du risque lié à l'exploration
        (pénalités de -10 pour les actions illégales dans Taxi-v3).</p>

        <p><strong>Monte Carlo</strong> met à jour les Q-values uniquement en fin d'épisode à partir du retour
        cumulé complet. Il ne souffre pas du biais d'amorçage (bootstrapping) mais converge plus lentement
        car il nécessite des épisodes terminés et présente une variance plus élevée.</p>

        <p><strong>Deep Q-Learning (DQN)</strong> utilise un réseau de neurones pour approximer la Q-function.
        Sur un espace d'états aussi petit que Taxi-v3, le DQN est surdimensionné par rapport aux méthodes
        tabulaires — il est inclus à titre de démonstration et comparaison, car ses avantages se manifestent
        surtout sur des espaces d'états continus ou très grands.</p>
    </div>""")

    # --- Analyse dynamique des résultats ---
    best_agent = None
    best_rate = -1
    fastest_agent = None
    fastest_time = float('inf')
    for name, data in results.items():
        test = data["test"]
        rate = float(test[3].replace('%', ''))
        if rate > best_rate:
            best_rate = rate
            best_agent = name
        train = data["train"]
        if train is not None and train["training_time"] < fastest_time:
            fastest_time = train["training_time"]
            fastest_agent = name

    sections.append(f"""
    <h2>Analyse des résultats</h2>
    <div class="analysis">
        <p><strong>Meilleur taux de succès :</strong> <em>{best_agent}</em> avec {best_rate:.1f}%.
        {"Ce résultat confirme l'efficacité des méthodes tabulaires sur un espace d'états discret de petite taille." if best_rate > 90 else "Un taux inférieur à 90% suggère qu'un entraînement plus long ou un ajustement des hyperparamètres pourrait améliorer les performances."}</p>

        <p><strong>Entraînement le plus rapide :</strong> <em>{fastest_agent}</em> en {fastest_time:.2f}s.
        Les méthodes TD (Q-Learning, SARSA) convergent généralement plus vite que Monte Carlo car elles
        mettent à jour les Q-values à chaque pas de temps, sans attendre la fin de l'épisode.</p>
    </div>""")

    # --- Analyse des récompenses et stratégie d'optimisation ---
    sections.append("""
    <h2>Stratégie d'optimisation des paramètres et récompenses</h2>
    <div class="analysis">
        <p><strong>Structure des récompenses de Taxi-v3 :</strong></p>
        <ul>
            <li><strong>+20</strong> pour une dépose réussie du passager à destination</li>
            <li><strong>-1</strong> par pas de temps (encourage l'efficacité)</li>
            <li><strong>-10</strong> pour une prise en charge ou dépose illégale</li>
        </ul>
        <p>Cette structure de récompense pénalise fortement les actions illégales et incite l'agent à
        trouver le chemin le plus court. La récompense finale optimale dépend de la distance entre
        le passager et la destination, typiquement entre +5 et +15 pour un trajet réussi.</p>

        <h3>Rôle des hyperparamètres</h3>
        <p><strong>Epsilon (ε)</strong> contrôle le compromis exploration/exploitation. Une valeur initiale
        élevée (ex: 0.1) favorise l'exploration des actions inconnues, essentielle en début d'entraînement.
        Une décroissance progressive vers 0 permet ensuite d'exploiter la politique apprise.</p>

        <p><strong>Gamma (γ)</strong> est le facteur d'actualisation. Une valeur proche de 1 (ex: 0.99)
        donne plus de poids aux récompenses futures, ce qui est important dans Taxi-v3 où la récompense
        principale (+20) n'arrive qu'en fin d'épisode. Une valeur trop basse rendrait l'agent myope,
        incapable de planifier le trajet complet.</p>

        <p><strong>Learning rate (α)</strong> détermine l'amplitude des mises à jour. Une valeur modérée
        (ex: 0.1) offre un bon compromis entre vitesse de convergence et stabilité. Trop élevée,
        l'apprentissage oscille ; trop faible, il stagne.</p>
    </div>""")

    # --- Méthodologie (benchmark & règle de sélection) ---
    sections.append("""
    <h2>Méthodologie et protocole de benchmark</h2>
    <div class="analysis">
        <p><strong>Protocole de recherche des meilleurs paramètres.</strong> Le mode benchmark effectue un
        <em>grid-search</em> sur le produit cartésien des valeurs fournies pour epsilon, gamma et learning rate.
        Chaque configuration est entraînée indépendamment (avec <em>early stopping</em> actif pour couper quand la
        métrique stagne), puis testée sur un nombre fixe d'épisodes.</p>
        <p><strong>Early stopping.</strong> Pendant l'entraînement, on surveille la moyenne glissante du reward
        sur une fenêtre de 100 épisodes. Si aucune amélioration significative (plus grande que
        σ / √n) n'apparaît pendant 3 fenêtres consécutives (après un minimum de 15% des épisodes),
        l'entraînement est interrompu.</p>
        <p><strong>Règle de sélection du « meilleur » modèle.</strong> Une configuration est retenue si :</p>
        <ol>
            <li>elle franchit le seuil de qualité <strong>taux de succès ≥ 95%</strong> ;</li>
            <li>parmi les candidates qualifiées, celle au plus haut <strong>reward moyen</strong> gagne ;</li>
            <li>si l'écart est plus petit que l'incertitude ε = σ / √n, on départage par <strong>temps d'entraînement minimum</strong>.</li>
        </ol>
        <p>On privilégie <em>reward moyen</em> plutôt que <em>taux de succès</em> car, une fois le seuil de fiabilité
        atteint, le reward distingue les politiques efficaces (chemins courts, peu de pénalités) des politiques
        seulement correctes.</p>
    </div>""")

    # --- Top / Bottom 5 ---
    ranked = _ranked(results)
    if len(ranked) > 1:
        top = ranked[:5]
        bottom = ranked[-5:] if len(ranked) > 5 else []
        rows_top = "".join(
            f"<tr><td>{r['name']}</td><td>{r['reward']:.2f}</td><td>{r['success']:.1f}%</td><td>"
            + (f"{r['train_time']:.2f}s" if r['train_time'] is not None else "-") + "</td></tr>"
            for r in top
        )
        rows_bot = "".join(
            f"<tr><td>{r['name']}</td><td>{r['reward']:.2f}</td><td>{r['success']:.1f}%</td><td>"
            + (f"{r['train_time']:.2f}s" if r['train_time'] is not None else "-") + "</td></tr>"
            for r in bottom
        )
        sections.append(f"""
        <h2>Classement — Top 5 / Bottom 5 (par reward moyen)</h2>
        <div class="analysis">
            <h3>Top 5</h3>
            <table><tr><th>Agent</th><th>Reward</th><th>Succès</th><th>Temps d'entraînement</th></tr>{rows_top}</table>
            {("<h3>Bottom 5</h3><table><tr><th>Agent</th><th>Reward</th><th>Succès</th><th>Temps d'entraînement</th></tr>" + rows_bot + "</table>") if rows_bot else ""}
        </div>""")

    # --- Conclusions comparatives ---
    best = ranked[0] if ranked else None
    worst = ranked[-1] if len(ranked) > 1 else None
    if best and worst:
        sections.append(f"""
        <h2>Conclusions comparatives</h2>
        <div class="analysis">
            <p><strong>Meilleure config : {best['name']}</strong> — reward {best['reward']:.2f}, succès {best['success']:.1f}%.
            Ce qui la distingue : un bon compromis exploration/exploitation (epsilon assez haut pour découvrir,
            décroissance suffisante pour exploiter), un gamma élevé (≈0.99) qui permet de valoriser la récompense
            terminale de +20, et un learning rate adapté à la famille d'algorithme.</p>
            <p><strong>Pire config : {worst['name']}</strong> — reward {worst['reward']:.2f}.
            Typiquement : exploration insuffisante (l'agent reste bloqué sur une politique sous-optimale),
            ou learning rate trop élevé provoquant des oscillations, ou encore trop peu d'épisodes pour que la
            Q-table se stabilise.</p>
            <p><strong>SARSA vs Q-Learning.</strong> Sur Taxi-v3, Q-Learning (off-policy) tend à converger plus
            vite vers la politique optimale car il apprend la Q-value du « meilleur choix futur » indépendamment
            de l'exploration. SARSA (on-policy) apprend la valeur de la politique effectivement suivie, donc
            intègre le coût de l'exploration — il est plus prudent mais plus lent à converger, et plus sensible
            à la stratégie de décroissance d'epsilon.</p>
        </div>""")

    return "\n".join(sections)


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
    img_top_bottom = plot_top_bottom(results)
    analysis_html = generate_analysis(results)

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
        .analysis {{
            background: white;
            padding: 20px 25px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            line-height: 1.7;
        }}
        .analysis ul {{
            margin: 10px 0;
            padding-left: 25px;
        }}
        .analysis li {{
            margin: 5px 0;
        }}
        .analysis h3 {{
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
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

    {analysis_html}

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

    {"<h2>6. Top 5 / Bottom 5 (reward moyen)</h2><div class='chart'><img src='data:image/png;base64," + img_top_bottom + "' alt='Top/Bottom 5'></div>" if img_top_bottom else ""}

    <div class="footer">
        <p>Environnement : Taxi-v3 (Gymnasium)</p>
    </div>
</body>
</html>"""

    output_path = "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
