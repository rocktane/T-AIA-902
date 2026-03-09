import questionary
from agents.bruteforce import Bruteforce
from agents.monte_carlo import MonteCarlo
from agents.q_learning import QLearning
from agents.deep_q_learning import DeepQLearning
from agents.sarsa import Sarsa
from tabulate import tabulate

tab = [["Target", 7, 13, "100.0%"]]

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
    elif choice == "SARSA":
        ep_sarsa = questionary.text(
            "Combien d'épisodes à entraîner pour SARSA ?",
            default=str(ep_sarsa),
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
        ).ask()
    elif choice == "Monte Carlo":
        ep_monte_carlo = questionary.text(
            "Combien d'épisodes à entraîner pour Monte Carlo ?",
            default=str(ep_monte_carlo),
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 100000
        ).ask()
    elif choice == "Deep Q-Learning":
        ep_deep_q_learning = questionary.text(
            "Combien d'épisodes à entraîner pour Deep Q-Learning ?",
            default=str(ep_deep_q_learning),
            validate=lambda x: x.isdigit() and int(x) > 0 and int(x) <= 10000
        ).ask()

episodes_test = questionary.text(
    "Combien d'épisodes à tester ?",
    default="100",
    validate=lambda x: x.isdigit() and int(x) > 0 and int(x) < 1000
).ask()

for choice in choices:
    if choice == "Bruteforce":
        tab.append(Bruteforce().test(episodes_test))
    elif choice == "Q-Learning":
        agent = QLearning()
        agent.train(int(ep_q_learning))
        tab.append(agent.test(int(episodes_test)))
    elif choice == "SARSA":
        agent = Sarsa()
        agent.train(int(ep_sarsa))
        tab.append(agent.test(int(episodes_test)))
    elif choice == "Monte Carlo":
        agent = MonteCarlo()
        agent.train(int(ep_monte_carlo))
        tab.append(agent.test(int(episodes_test)))
    elif choice == "Deep Q-Learning":
        agent = DeepQLearning()
        agent.train(int(ep_deep_q_learning))
        tab.append(agent.test(int(episodes_test)))

print(tabulate(tab, headers=["Agent", "Récompense moyenne", "Nombre de pas moyen", "Taux de succès"], tablefmt="rounded_outline"))
