from agents import BaseAgent
import numpy as np
import time

class MonteCarlo(BaseAgent):

    def __init__(self):
        super().__init__()
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n)) # type: ignore
        self.epsilon = 0.9
        self.gamma = 0.99
        self.lr = 0.1

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            action = np.argmax(self.q_table[state])
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
            episodes = []
            while not done:
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                episodes.append((state, action, reward))
                state = next_state
                total_reward += float(reward)
                total_steps += 1
                done = terminated or truncated
            G = 0
            for state, action, reward in reversed(episodes):
                G = float(reward) + self.gamma * G
                self.q_table[state, action] += self.lr * (G - self.q_table[state, action])
            reward_history.append(total_reward)
            steps_history.append(total_steps)
            success_history.append(1 if terminated else 0)
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
