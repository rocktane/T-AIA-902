from itertools import product
import random
import time
import questionary
import plotext as plt
from tqdm import tqdm
from agents.bruteforce import Bruteforce
from agents.monte_carlo import MonteCarlo
from agents.q_learning import QLearning
from agents.deep_q_learning import DeepQLearning
from agents.sarsa import Sarsa
from tabulate import tabulate
from report import generate_report, rolling_mean
from best_params import save_best_params, get_best_params, load_best_params

results = {}
agents = ["Bruteforce", "Q-Learning", "SARSA", "Monte Carlo", "Deep Q-Learning"]
benchmark_agents = ["Q-Learning", "SARSA", "Monte Carlo", "Deep Q-Learning"]

benchmark_defaults = {
    "Q-Learning": {"episodes": 10000, "epsilon": "0.9", "gamma": "0.99", "lr": "0.7", "max_episodes": 100000},
    "SARSA": {"episodes": 20000, "epsilon": "0.9", "gamma": "0.99", "lr": "0.2", "max_episodes": 100000},
    "Monte Carlo": {"episodes": 100000, "epsilon": "0.9", "gamma": "0.99", "lr": "0.1", "max_episodes": 100000},
    "Deep Q-Learning": {"episodes": 1000, "epsilon": "0.9", "gamma": "0.99", "lr": "0.001", "max_episodes": 10000},
}

mode = questionary.select(
    "Choisir le mode :",
    choices=["Manuel", "Temps limité", "Benchmark", "Battle"]
).ask()

tab = [["Target", 7, 13, "100.0%"]]


def validate_float_0_1(x):
    try:
        return 0 < float(x) <= 1
    except ValueError:
        return False


def parse_float_list(raw_values):
    values = [value.strip() for value in raw_values.split(",")]
    if not values or any(value == "" for value in values):
        raise ValueError
    return [float(value) for value in values]


def validate_float_list_0_1(raw_values):
    try:
        values = parse_float_list(raw_values)
        return all(0 < value <= 1 for value in values)
    except ValueError:
        return False


def create_benchmark_configs(epsilons, gammas, lrs):
    return [
        {"epsilon": epsilon, "gamma": gamma, "lr": lr}
        for epsilon, gamma, lr in product(epsilons, gammas, lrs)
    ]


agent_classes = {
    "Q-Learning": QLearning,
    "SARSA": Sarsa,
    "Monte Carlo": MonteCarlo,
    "Deep Q-Learning": DeepQLearning,
}


def render_battle_graphs(name_a, name_b, train_a, train_b, test_a, test_b):
    # Graphique 1 : Progression du taux de succès pendant l'entraînement
    success_a = [float(s) * 100 for s in train_a["success_history"]]
    success_b = [float(s) * 100 for s in train_b["success_history"]]
    smooth_a = rolling_mean(success_a)
    smooth_b = rolling_mean(success_b)

    plt.clear_figure()
    plt.plotsize(80, 15)
    plt.plot(list(range(len(smooth_a))), list(smooth_a), label=name_a, color="red")
    plt.plot(list(range(len(smooth_b))), list(smooth_b), label=name_b, color="blue")
    plt.title("Taux de succès pendant l'entraînement (%)")
    plt.xlabel("Épisodes")
    plt.ylabel("Succès (%)")
    plt.show()
    print()

    # Graphique 2 : Taux de succès final au test
    rate_a = float(test_a[3].replace("%", ""))
    rate_b = float(test_b[3].replace("%", ""))

    plt.clear_figure()
    plt.plotsize(80, 15)
    plt.bar([name_a, name_b], [rate_a, rate_b], color=["blue", "red"])
    plt.title("Taux de succès au test (%)")
    plt.ylabel("Succès (%)")
    plt.show()
    print()

    # Graphique 3 : Temps d'entraînement
    time_a = train_a["training_time"]
    time_b = train_b["training_time"]

    plt.clear_figure()
    plt.plotsize(80, 15)
    plt.bar([name_a, name_b], [time_a, time_b], color=["blue", "red"])
    plt.title("Temps d'entraînement (secondes)")
    plt.ylabel("Secondes")
    plt.show()
    print()


if mode == "Temps limité":
    time_limit = questionary.text(
        "Temps alloué à CHAQUE agent pour le test (secondes)",
        default="5",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 60
    ).ask()
    time_limit = int(time_limit)

    stored = load_best_params()
    time_limited_stats = {}
    unseen_seed = random.randint(10_000_000, 99_999_999)
    print(f"\n{'─' * 80}")
    print("  MODE TEMPS LIMITÉ — Paramètres sauvegardés + test chronométré (env inconnu)")
    print(f"  Seed inconnu : {unseen_seed}  |  Budget test par agent : {time_limit}s")
    print(f"{'─' * 80}\n")

    for name in benchmark_agents:
        entry = stored.get(name)
        if entry is None:
            print(f"⚠️  {name} : aucun best_params enregistré. Agent ignoré (lance un benchmark d'abord).")
            continue
        params = entry["params"]
        print(f"\n▶ {name} — params {params}")
        agent = agent_classes[name](
            epsilon=float(params["epsilon"]),
            gamma=float(params["gamma"]),
            lr=float(params["lr"]),
        )
        n_train = int(params["episodes"])
        pbar = tqdm(total=n_train, desc=f"  {name} (train)", unit="ep", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        train_data = agent.train(n_train, on_episode=lambda ep: pbar.update(1), early_stopping=True)
        pbar.close()
        print(f"  Entraîné en {train_data['training_time']:.2f}s ({train_data['n_episodes']} ép.)")

        test_stats = agent.test_time_limited(time_limit, seed=unseen_seed)
        p_success = f"{(100 * test_stats['success_rate']):.1f}%"
        tab.append([name, test_stats["reward_mean"], test_stats["steps_mean"], p_success])
        results[name] = {"train": train_data, "test": [name, test_stats["reward_mean"], test_stats["steps_mean"], p_success]}
        time_limited_stats[name] = {
            "reward_mean": test_stats["reward_mean"],
            "reward_std": test_stats["reward_std"],
            "success_rate": test_stats["success_rate"],
            "episodes_tested": test_stats["test_episodes"],
            "epsilon_tolerance": test_stats["epsilon_tolerance"],
            "train_time": train_data["training_time"],
        }
        print(f"  Test : {test_stats['test_episodes']} ép. | reward {test_stats['reward_mean']:.2f} | succès {p_success}")

    if time_limited_stats:
        print(f"\n{'─' * 80}\n  VISUALISATION\n{'─' * 80}\n")
        names_tl = list(time_limited_stats.keys())
        rewards_tl = [time_limited_stats[n]["reward_mean"] for n in names_tl]
        success_tl = [time_limited_stats[n]["success_rate"] * 100 for n in names_tl]
        episodes_tl = [time_limited_stats[n]["episodes_tested"] for n in names_tl]

        plt.clear_figure(); plt.plotsize(80, 12)
        plt.bar(names_tl, rewards_tl); plt.title("Reward moyen (test chronométré)"); plt.show(); print()

        plt.clear_figure(); plt.plotsize(80, 12)
        plt.bar(names_tl, success_tl); plt.title("Taux de succès % (test chronométré)"); plt.show(); print()

        plt.clear_figure(); plt.plotsize(80, 12)
        plt.bar(names_tl, episodes_tl); plt.title("Épisodes testés dans le temps imparti"); plt.show(); print()

        qualified = [(n, s) for n, s in time_limited_stats.items() if s["success_rate"] >= 0.95]
        pool = qualified if qualified else list(time_limited_stats.items())
        winner = max(
            pool,
            key=lambda kv: (kv[1]["success_rate"] >= 0.95, kv[1]["reward_mean"], -kv[1]["train_time"]),
        )
        print(f"🏆 Meilleur agent (règle : succès≥95% → reward → temps) : {winner[0]}")
        print(f"   reward {winner[1]['reward_mean']:.2f} ± {winner[1]['epsilon_tolerance']:.3f}  |  succès {winner[1]['success_rate']*100:.1f}%")
elif mode == "Benchmark":
    benchmark_agent = questionary.select(
        "Quel agent voulez-vous benchmarker ?",
        choices=benchmark_agents
    ).ask()

    defaults = benchmark_defaults[benchmark_agent]
    benchmark_answers = questionary.form(
        train_episodes=questionary.text(
            "Combien d'épisodes à entraîner par configuration ?",
            default=str(defaults["episodes"]),
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= defaults["max_episodes"]
        ),
        test_episodes=questionary.text(
            "Combien d'épisodes à tester par configuration ?",
            default="100",
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) < 1000
        ),
        epsilons=questionary.text(
            "Valeurs de epsilon séparées par des virgules",
            default=defaults["epsilon"],
            validate=validate_float_list_0_1
        ),
        gammas=questionary.text(
            "Valeurs de gamma séparées par des virgules",
            default=defaults["gamma"],
            validate=validate_float_list_0_1
        ),
        lrs=questionary.text(
            "Valeurs de learning rate séparées par des virgules",
            default=defaults["lr"],
            validate=validate_float_list_0_1
        ),
    ).ask()

    train_episodes = int(benchmark_answers["train_episodes"])
    test_episodes = int(benchmark_answers["test_episodes"])
    configs = create_benchmark_configs(
        parse_float_list(benchmark_answers["epsilons"]),
        parse_float_list(benchmark_answers["gammas"]),
        parse_float_list(benchmark_answers["lrs"]),
    )

    previous_best = get_best_params(benchmark_agent)
    new_best_saved = False

    for i, config in enumerate(configs, 1):
        agent = agent_classes[benchmark_agent](
            epsilon=config["epsilon"],
            gamma=config["gamma"],
            lr=config["lr"],
        )
        train_data = agent.train(train_episodes)
        t = agent.test(test_episodes)
        label = (
            f"{benchmark_agent} #{i} "
            f"(e={config['epsilon']}, g={config['gamma']}, lr={config['lr']})"
        )
        tab.append([label, t[1], t[2], t[3]])
        results[label] = {"train": train_data, "test": [label, t[1], t[2], t[3]]}
        stats = agent.last_test_stats
        metrics = {
            "reward_mean": stats["reward_mean"],
            "reward_std": stats["reward_std"],
            "success_rate": stats["success_rate"],
            "train_time": train_data["training_time"],
            "test_episodes": stats["test_episodes"],
            "epsilon_tolerance": stats["epsilon_tolerance"],
        }
        params = {
            "episodes": train_episodes,
            "epsilon": config["epsilon"],
            "gamma": config["gamma"],
            "lr": config["lr"],
        }
        if save_best_params(benchmark_agent, params, metrics):
            new_best_saved = True
elif mode == "Battle":
    battle_choices = []
    while len(battle_choices) != 2:
        battle_choices = questionary.checkbox(
            "Choisir exactement 2 agents à affronter :",
            choices=benchmark_agents
        ).ask()
        if len(battle_choices) != 2:
            print("Veuillez sélectionner exactement 2 agents.")

    default_episodes = str(max(
        benchmark_defaults[battle_choices[0]]["episodes"],
        benchmark_defaults[battle_choices[1]]["episodes"],
    ))
    default_lr = benchmark_defaults[battle_choices[0]]["lr"]

    battle_answers = questionary.form(
        train_episodes=questionary.text(
            "Combien d'épisodes à entraîner ?",
            default=default_episodes,
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
        ),
        test_episodes=questionary.text(
            "Combien d'épisodes à tester ?",
            default="100",
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) < 1000
        ),
        epsilon=questionary.text(
            "Valeur de epsilon ?",
            default="0.9",
            validate=validate_float_0_1
        ),
        gamma=questionary.text(
            "Valeur de gamma ?",
            default="0.99",
            validate=validate_float_0_1
        ),
        lr=questionary.text(
            "Valeur de learning rate ?",
            default=default_lr,
            validate=validate_float_0_1
        ),
    ).ask()

    train_ep = int(battle_answers["train_episodes"])
    test_ep = int(battle_answers["test_episodes"])
    epsilon = float(battle_answers["epsilon"])
    gamma = float(battle_answers["gamma"])
    lr = float(battle_answers["lr"])

    name_a, name_b = battle_choices
    agent_a = agent_classes[name_a](epsilon=epsilon, gamma=gamma, lr=lr)
    agent_b = agent_classes[name_b](epsilon=epsilon, gamma=gamma, lr=lr)

    print(f"\n{'─' * 80}")
    print(f"  BATTLE : {name_a} vs {name_b}")
    print(f"{'─' * 80}\n")

    pbar_a = tqdm(total=train_ep, desc=f"  {name_a}", unit="ep", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    train_a = agent_a.train(train_ep, on_episode=lambda ep: pbar_a.update(1))
    pbar_a.close()
    speed_a = train_ep / train_a["training_time"]
    test_a = agent_a.test(test_ep)
    tab.append(test_a)
    results[name_a] = {"train": train_a, "test": [name_a, test_a[1], test_a[2], test_a[3]]}
    print(f"  Terminé en {train_a['training_time']:.2f}s — {speed_a:.2f}ep/s\n")

    pbar_b = tqdm(total=train_ep, desc=f"  {name_b}", unit="ep", ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
    train_b = agent_b.train(train_ep, on_episode=lambda ep: pbar_b.update(1))
    pbar_b.close()
    speed_b = train_ep / train_b["training_time"]
    test_b = agent_b.test(test_ep)
    tab.append(test_b)
    results[name_b] = {"train": train_b, "test": [name_b, test_b[1], test_b[2], test_b[3]]}
    print(f"  Terminé en {train_b['training_time']:.2f}s — {speed_b:.2f}ep/s")

    print(f"\n{'─' * 80}")
    print(f"  RÉSULTATS")
    print(f"{'─' * 80}\n")
    render_battle_graphs(name_a, name_b, train_a, train_b, test_a, test_b)

    for name, ag, train_d in ((name_a, agent_a, train_a), (name_b, agent_b, train_b)):
        s = ag.last_test_stats
        metrics = {
            "reward_mean": s["reward_mean"],
            "reward_std": s["reward_std"],
            "success_rate": s["success_rate"],
            "train_time": train_d["training_time"],
            "test_episodes": s["test_episodes"],
            "epsilon_tolerance": s["epsilon_tolerance"],
        }
        params = {"episodes": train_ep, "epsilon": epsilon, "gamma": gamma, "lr": lr}
        if save_best_params(name, params, metrics):
            print(f"  ✓ Nouveau best_params enregistré pour {name}")
else:
    choices = questionary.checkbox(
        "Choisir les agents à tester :",
        choices=agents
    ).ask()

    ep_q_learning = 10000
    ep_sarsa = 20000
    ep_monte_carlo = 100000
    ep_deep_q_learning = 1000

    for choice in choices:
        if choice == "Q-Learning":
            ep_q_learning = questionary.text(
                "Combien d'épisodes à entraîner pour Q-Learning ?",
                default=str(ep_q_learning),
                validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
            ).ask()
            epsilon_q_learning = questionary.text(
                "Quelle valeur pour epsilon pour Q-Learning ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_q_learning = questionary.text(
                "Quelle valeur pour gamma pour Q-Learning ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_q_learning = questionary.text(
                "Quelle valeur pour learning rate pour Q-Learning ?",
                default=str(0.7),
                validate=validate_float_0_1
            ).ask()
        elif choice == "SARSA":
            ep_sarsa = questionary.text(
                "Combien d'épisodes à entraîner pour SARSA ?",
                default=str(ep_sarsa),
                validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
            ).ask()
            epsilon_sarsa = questionary.text(
                "Quelle valeur pour epsilon pour SARSA ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_sarsa = questionary.text(
                "Quelle valeur pour gamma pour SARSA ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_sarsa = questionary.text(
                "Quelle valeur pour learning rate pour SARSA ?",
                default=str(0.2),
                validate=validate_float_0_1
            ).ask()
        elif choice == "Monte Carlo":
            ep_monte_carlo = questionary.text(
                "Combien d'épisodes à entraîner pour Monte Carlo ?",
                default=str(ep_monte_carlo),
                validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
            ).ask()
            epsilon_monte_carlo = questionary.text(
                "Quelle valeur pour epsilon pour Monte Carlo ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_monte_carlo = questionary.text(
                "Quelle valeur pour gamma pour Monte Carlo ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_monte_carlo = questionary.text(
                "Quelle valeur pour learning rate pour Monte Carlo ?",
                default=str(0.1),
                validate=validate_float_0_1
            ).ask()
        elif choice == "Deep Q-Learning":
            ep_deep_q_learning = questionary.text(
                "Combien d'épisodes à entraîner pour Deep Q-Learning ?",
                default=str(ep_deep_q_learning),
                validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 10000
            ).ask()
            epsilon_deep_q_learning = questionary.text(
                "Quelle valeur pour epsilon pour Deep Q-Learning ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_deep_q_learning = questionary.text(
                "Quelle valeur pour gamma pour Deep Q-Learning ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_deep_q_learning = questionary.text(
                "Quelle valeur pour learning rate pour Deep Q-Learning ?",
                default=str(0.001),
                validate=validate_float_0_1
            ).ask()

    episodes_test = questionary.text(
        "Combien d'épisodes à tester ?",
        default="100",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) < 1000
    ).ask()

    display_episode = questionary.confirm(
        "Voulez-vous afficher des épisodes ?",
        default=True
    ).ask()

    for choice in choices:
        if choice == "Bruteforce":
            agent = Bruteforce()
            t = agent.test(int(episodes_test))
            tab.append(t)
            results["Bruteforce"] = {"train": None, "test": ["Bruteforce", t[1], t[2], t[3]]}
        elif choice == "Q-Learning":
            agent = QLearning(epsilon=float(epsilon_q_learning), gamma=float(gamma_q_learning), lr=float(lr_q_learning))
            train_data = agent.train(int(ep_q_learning))
            t = agent.test(int(episodes_test))
            tab.append(t)
            results["Q-Learning"] = {"train": train_data, "test": ["Q-Learning", t[1], t[2], t[3]]}
        elif choice == "SARSA":
            agent = Sarsa(epsilon=float(epsilon_sarsa), gamma=float(gamma_sarsa), lr=float(lr_sarsa))
            train_data = agent.train(int(ep_sarsa))
            t = agent.test(int(episodes_test))
            tab.append(t)
            results["SARSA"] = {"train": train_data, "test": ["SARSA", t[1], t[2], t[3]]}
        elif choice == "Monte Carlo":
            agent = MonteCarlo(epsilon=float(epsilon_monte_carlo), gamma=float(gamma_monte_carlo), lr=float(lr_monte_carlo))
            train_data = agent.train(int(ep_monte_carlo))
            t = agent.test(int(episodes_test))
            tab.append(t)
            results["Monte Carlo"] = {"train": train_data, "test": ["Monte Carlo", t[1], t[2], t[3]]}
        elif choice == "Deep Q-Learning":
            agent = DeepQLearning(epsilon=float(epsilon_deep_q_learning), gamma=float(gamma_deep_q_learning), lr=float(lr_deep_q_learning))
            train_data = agent.train(int(ep_deep_q_learning))
            t = agent.test(int(episodes_test))
            tab.append(t)
            results["Deep Q-Learning"] = {"train": train_data, "test": ["Deep Q-Learning", t[1], t[2], t[3]]}

        if display_episode:
            print(f"\n🤖 Modèle : {choice}\n")
            agent.display_episode(3)

print(tabulate(tab, headers=["Agent", "Récompense moyenne", "Nombre de pas moyen", "Taux de succès"], tablefmt="rounded_outline"))

if mode == "Benchmark" and new_best_saved:
    current_best = get_best_params(benchmark_agent)
    best_params_values = current_best["params"]
    best_metrics = current_best["metrics"]
    print(f"\n✓ Nouvelle meilleure configuration pour {benchmark_agent}")
    print(
        f"  params : epsilon={best_params_values['epsilon']}, "
        f"gamma={best_params_values['gamma']}, "
        f"lr={best_params_values['lr']}, "
        f"episodes={best_params_values['episodes']}"
    )
    print(
        f"  reward {best_metrics['reward_mean']:.2f} ± {best_metrics['epsilon_tolerance']:.3f}  |  "
        f"succès {best_metrics['success_rate']*100:.1f}%  |  "
        f"train_time {best_metrics['train_time']:.2f}s"
    )
    if previous_best is None:
        print("  Raison : premier enregistrement, aucun best antérieur.")
    else:
        prev_metrics = previous_best["metrics"]
        tol = best_metrics["epsilon_tolerance"]
        if prev_metrics["success_rate"] < 0.95:
            print(
                f"  Raison : l'ancien best n'atteignait pas 95% de succès "
                f"({prev_metrics['success_rate']*100:.1f}%), celui-ci oui."
            )
        elif best_metrics["reward_mean"] > prev_metrics["reward_mean"] + tol:
            delta = best_metrics["reward_mean"] - prev_metrics["reward_mean"]
            print(
                f"  Raison : récompense moyenne supérieure "
                f"({prev_metrics['reward_mean']:.2f} → {best_metrics['reward_mean']:.2f}, "
                f"Δ={delta:+.2f} > ε_tol={tol:.3f})."
            )
        else:
            print(
                f"  Raison : récompense équivalente "
                f"({prev_metrics['reward_mean']:.2f} → {best_metrics['reward_mean']:.2f}, "
                f"|Δ| ≤ ε_tol={tol:.3f}), mais entraînement plus rapide "
                f"({prev_metrics['train_time']:.2f}s → {best_metrics['train_time']:.2f}s)."
            )

if results and mode != "Battle":
    report_path = generate_report(results)
    print(f"\nRapport généré : {report_path}")
