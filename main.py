import questionary
from agents.bruteforce import Bruteforce
from agents.monte_carlo import MonteCarlo
from agents.q_learning import QLearning
from agents.deep_q_learning import DeepQLearning
from agents.sarsa import Sarsa
from tabulate import tabulate
from report import generate_report

results = {}
mode = questionary.select(
    "Choisir le mode :",
    choices=["Manuel", "Temps limité"]
).ask()

tab = [["Target", 7, 13, "100.0%"]]

def validate_float_0_1(x):
    try:
        return 0 < float(x) <= 1
    except ValueError:
        return False

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
else:
    choices = questionary.checkbox(
        "Choisir les agents à tester :",
        choices=["Bruteforce", "Q-Learning", "SARSA", "Monte Carlo", "Deep Q-Learning"]
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
                "Quelle valeur pour epsilon ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_q_learning = questionary.text(
                "Quelle valeur pour gamma ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_q_learning = questionary.text(
                "Quelle valeur pour learning rate ?",
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
                "Quelle valeur pour epsilon ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_sarsa = questionary.text(
                "Quelle valeur pour gamma ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_sarsa = questionary.text(
                "Quelle valeur pour learning rate ?",
                default=str(0.7),
                validate=validate_float_0_1
            ).ask()
        elif choice == "Monte Carlo":
            ep_monte_carlo = questionary.text(
                "Combien d'épisodes à entraîner pour Monte Carlo ?",
                default=str(ep_monte_carlo),
                validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
            ).ask()
            epsilon_monte_carlo = questionary.text(
                "Quelle valeur pour epsilon ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_monte_carlo = questionary.text(
                "Quelle valeur pour gamma ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_monte_carlo = questionary.text(
                "Quelle valeur pour learning rate ?",
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
                "Quelle valeur pour epsilon ?",
                default=str(0.9),
                validate=validate_float_0_1
            ).ask()
            gamma_deep_q_learning = questionary.text(
                "Quelle valeur pour gamma ?",
                default=str(0.99),
                validate=validate_float_0_1
            ).ask()
            lr_deep_q_learning = questionary.text(
                "Quelle valeur pour learning rate ?",
                default=str(0.001),
                validate=validate_float_0_1
            ).ask()

    episodes_test = questionary.text(
        "Combien d'épisodes à tester ?",
        default="100",
        validate=lambda x: x.isdigit() and int(x) > 0 and int(x) < 1000
    ).ask()

    for choice in choices:
        if choice == "Bruteforce":
            t = Bruteforce().test(int(episodes_test))
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

print(tabulate(tab, headers=["Agent", "Récompense moyenne", "Nombre de pas moyen", "Taux de succès"], tablefmt="rounded_outline"))

if results:
    report_path = generate_report(results)
    print(f"\nRapport généré : {report_path}")
