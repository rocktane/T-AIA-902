import numpy as np
import torch
import torch.nn as nn
from agents import BaseAgent
import random
import time

class DQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(500, 64)   # 500 entrées → 64 neurones
        self.layer2 = nn.Linear(64, 64)    # 64 → 64
        self.layer3 = nn.Linear(64, 6)     # 64 → 6 sorties (une par action)

    def forward(self, x):                  # comment les données traversent le réseau
        x = torch.relu(self.layer1(x))     # relu = garder les valeurs positives, mettre les négatives à 0
        x = torch.relu(self.layer2(x))
        return self.layer3(x)              # pas de relu sur la sortie (les Q-values peuvent être négatives)

class DeepQLearning(BaseAgent):
    def __init__(self):
        super().__init__()
        self.policy_net = DQNetwork()           # réseau qui apprend
        self.target_net = DQNetwork()           # copie figée
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copier les poids

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.001) # Optimizer
        self.loss_fn = nn.MSELoss()             # Loss function

        self.replay_buffer = []                 # mémoire des expériences
        self.buffer_size = 10000
        self.batch_size = 64
        self.epsilon = 0.9
        self.gamma = 0.99

        self.update_target_every = 100
        self.step_count = 0

    def encode_state(self, state):
        one_hot = torch.zeros(500)
        one_hot[state] = 1.0
        return one_hot

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            with torch.no_grad():              # pas besoin de calculer les gradients ici
                q_values = self.policy_net(self.encode_state(state))
                action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, n_episodes=None, time_limit=None):
        episode = 0
        start_time = time.time()
        self.decay_rate = (0.05 / 0.9) ** (1 / (n_episodes or 10000))
        reward_history = []
        steps_history = []
        training_time = 0
        success_history = []
        while True:
            total_reward = 0
            total_steps = 0
            if n_episodes is not None and episode >= n_episodes:
                break
            if time_limit is not None and time.time() - start_time >= time_limit:
                break
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 1. stocker l'expérience
                self.replay_buffer.append((state, action, reward, next_state, terminated))
                if len(self.replay_buffer) > self.buffer_size:
                    self.replay_buffer.pop(0)

                # 2. si assez d'expériences → piocher un batch → entraîner
                if len(self.replay_buffer) >= self.batch_size:
                    batch = random.sample(self.replay_buffer, self.batch_size)
                    states = torch.stack([self.encode_state(s) for s, _, _, _, _ in batch])
                    actions = torch.tensor([a for _, a, _, _, _ in batch])
                    rewards = torch.tensor([float(r) for _, _, r, _, _ in batch])
                    next_states = torch.stack([self.encode_state(s) for _, _, _, s, _ in batch])
                    dones = torch.tensor([float(d) for _, _, _, _, d in batch])

                    # Prédictions : Q(s, a) pour chaque expérience du batch
                    predictions = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

                    # Cibles : r + gamma * max(Q_target(s')) * (1 - done)
                    with torch.no_grad():
                        targets = rewards + self.gamma * self.target_net(next_states).max(1)[0] * (1 - dones)

                    loss = self.loss_fn(predictions, targets)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                state = next_state
                total_reward += float(reward)
                total_steps += 1
                self.step_count += 1
                if self.step_count % self.update_target_every == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            success_history.append(1 if terminated else 0)
            reward_history.append(total_reward)
            steps_history.append(total_steps)
            episode += 1
            self.epsilon = max(0.05, self.epsilon * self.decay_rate)
        training_time = time.time() - start_time
        return {
            "reward_history": reward_history,
            "steps_history": steps_history,
            "training_time": training_time,
            "n_episodes": episode,
            "success_history": success_history,
        }
