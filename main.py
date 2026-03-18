from itertools import product
import questionary
from agents.bruteforce import Bruteforce
from agents.monte_carlo import MonteCarlo
from agents.q_learning import QLearning
from agents.deep_q_learning import DeepQLearning
from agents.sarsa import Sarsa
from tabulate import tabulate
from report import generate_report

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
    choices=["Manuel", "Temps limité", "Benchmark"]
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


if mode == "Temps limité":
    time_limit = questionary.text(
        "Combien de temps à entraîner (en secondes) ?",
        default="10",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 60
    ).ask()
    time_limit = int(time_limit)

    test_episodes = questionary.text(
        "Combien d'épisodes à tester ?",
        default="10",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100
    ).ask()
    test_episodes = int(test_episodes)

    # Bruteforce
    t = Bruteforce().test(test_episodes)
    tab.append(t)
    results["Bruteforce"] = {"train": None, "test": ["Bruteforce", t[1], t[2], t[3]]}
    # Q-Learning
    agent = QLearning()
    train_data = agent.train(time_limit=time_limit)
    t = agent.test(test_episodes)
    tab.append(t)
    results["Q-Learning"] = {"train": train_data, "test": ["Q-Learning", t[1], t[2], t[3]]}
    # SARSA
    agent = Sarsa()
    train_data = agent.train(time_limit=time_limit)
    t = agent.test(test_episodes)
    tab.append(t)
    results["SARSA"] = {"train": train_data, "test": ["SARSA", t[1], t[2], t[3]]}
    # Deep Q-Learning
    agent = DeepQLearning()
    train_data = agent.train(time_limit=time_limit)
    t = agent.test(test_episodes)
    tab.append(t)
    results["Deep Q-Learning"] = {"train": train_data, "test": ["Deep Q-Learning", t[1], t[2], t[3]]}
    # Monte Carlo
    agent = MonteCarlo()
    train_data = agent.train(time_limit=time_limit)
    t = agent.test(test_episodes)
    tab.append(t)
    results["Monte Carlo"] = {"train": train_data, "test": ["Monte Carlo", t[1], t[2], t[3]]}
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

    agent_classes = {
        "Q-Learning": QLearning,
        "SARSA": Sarsa,
        "Monte Carlo": MonteCarlo,
        "Deep Q-Learning": DeepQLearning,
    }

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
            agent.display_episode(3)

print(tabulate(tab, headers=["Agent", "Récompense moyenne", "Nombre de pas moyen", "Taux de succès"], tablefmt="rounded_outline"))

if results:
    report_path = generate_report(results)
    print(f"\nRapport généré : {report_path}")
