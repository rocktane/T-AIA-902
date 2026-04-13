from agents import BaseAgent
from agents.early_stopping import EarlyStopping
import numpy as np
import time

class Sarsa(BaseAgent):

    def __init__(self, epsilon=0.9, gamma=0.99, lr=0.2):
        super().__init__()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # type: ignore
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, n_episodes=None, time_limit=None, on_episode=None, early_stopping=True):
        self.decay_rate = (0.05 / self.epsilon) ** (1 / (n_episodes or 10000))
        episode = 0
        start_time = time.time()
        training_time = 0
        steps_history = []
        reward_history = []
        success_history = []
        es = None
        if early_stopping and n_episodes is not None:
            es = EarlyStopping(window_size=100, patience=3, min_episodes=int(0.15 * n_episodes))
        while True:
            total_reward = 0
            total_steps = 0
            if n_episodes is not None and episode >= n_episodes:
                break
            if time_limit is not None and time.time() - start_time >= time_limit:
                break
            state, info = self.env.reset()
            next_action = None
            done = False
            while not done:
                action = next_action if next_action is not None else self.choose_action(state)
                current_q = self.q_table[state, action]
                current_state = state
                next_state, reward, terminated, truncated, info = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.q_table[current_state, action] = current_q + self.lr * (float(reward) + self.gamma * self.q_table[next_state, next_action] - current_q)
                total_reward += float(reward)
                total_steps += 1
                state = next_state
                done = terminated or truncated
            reward_history.append(total_reward)
            steps_history.append(total_steps)
            episode += 1
            success_history.append(1 if terminated else 0)
            self.epsilon = max(0.05, self.epsilon * self.decay_rate)
            if on_episode:
                on_episode(episode)
            if es is not None and es.should_stop(episode, total_reward):
                break
        training_time = time.time() - start_time
        return {
            "reward_history": reward_history,
            "steps_history": steps_history,
            "training_time": training_time,
            "n_episodes": episode,
            "success_history": success_history,
            "early_stopped_at": es.triggered_at if es else None,
        }
