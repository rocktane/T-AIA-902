from agents import BaseAgent
import numpy as np
import time

class Sarsa(BaseAgent):

    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # type: ignore
        self.epsilon = 0.9
        self.gamma = 0.99
        self.lr = 0.2

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, n_episodes=None, time_limit=None):
        self.decay_rate = (0.05 / 0.9) ** (1 / (n_episodes or 10000))
        episode = 0
        start_time = time.time()
        while True:
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
                state = next_state
                done = terminated or truncated
            self.epsilon = max(0.05, self.epsilon * self.decay_rate)
