from agents import BaseAgent
import numpy as np

class QLearning(BaseAgent):

    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # type: ignore
        self.epsilon = 0.9
        self.gamma = 0.99
        self.lr = 0.7

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
        else:
            action = self.env.action_space.sample()
        return action

    def train(self, n_episodes):
        self.decay_rate = (0.05 / 0.9) ** (1 / n_episodes)
        for episode in range(n_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                current_q = self.q_table[state, action]
                current_state = state
                state, reward, terminated, truncated, info = self.env.step(action)
                self.q_table[current_state, action] = current_q + self.lr * (float(reward) + self.gamma * np.max(self.q_table[state]) - current_q)
                done = terminated or truncated
            self.epsilon = max(0.05, self.epsilon * self.decay_rate)
