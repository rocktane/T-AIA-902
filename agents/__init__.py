from statistics import mean
import gymnasium as gym
import time

class BaseAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")

    def choose_action(self, state):
        pass

    def train(self, n_episodes):
        pass

    def test(self, n_episodes):
        self.epsilon = 0
        reward_list = []
        steps_list = []
        total_success = 0
        for episode in range(n_episodes):
            total_reward = 0
            total_steps = 0
            state, info = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                state, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                total_steps += 1
                done = terminated or truncated
            if terminated:
                total_success += 1
            reward_list.append(total_reward)
            steps_list.append(total_steps)
            p_success = f"{(100 * total_success / n_episodes):.1f}%"
        return [self.__class__.__name__, mean(reward_list), mean(steps_list), p_success]

    def display_episode(self, episode):
        env = gym.make("Taxi-v3", render_mode="ansi")
        self.epsilon = 0
        for i in range(episode):
            print(f"--- {self.__class__.__name__} - Épisode {i+1}/{episode} ---")
            state, info = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                print(env.render())
                time.sleep(0.1)
