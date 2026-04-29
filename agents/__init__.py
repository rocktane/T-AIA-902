from statistics import mean
import gymnasium as gym
import numpy as np
import time

class BaseAgent:
    def __init__(self):
        self.env = gym.make("Taxi-v3")
        self.last_test_stats = None

    def choose_action(self, state):
        pass

    def train(self, n_episodes):
        pass

    def test(self, n_episodes, seed=None):
        self.epsilon = 0
        reward_list = []
        steps_list = []
        total_success = 0
        start = time.time()
        for episode in range(n_episodes):
            total_reward = 0
            total_steps = 0
            reset_seed = (seed + episode) if seed is not None else None
            state, info = self.env.reset(seed=reset_seed)
            done = False
            terminated = False
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
        success_rate = total_success / n_episodes
        p_success = f"{(100 * success_rate):.1f}%"
        reward_std = float(np.std(reward_list)) if reward_list else 0.0
        self.last_test_stats = {
            "reward_list": reward_list,
            "reward_mean": float(mean(reward_list)),
            "reward_std": reward_std,
            "steps_mean": float(mean(steps_list)),
            "success_rate": success_rate,
            "test_episodes": n_episodes,
            "epsilon_tolerance": reward_std / np.sqrt(n_episodes) if n_episodes > 0 else 0.0,
            "test_time": time.time() - start,
        }
        return [self.__class__.__name__, mean(reward_list), mean(steps_list), p_success]

    def test_time_limited(self, time_limit, seed=None):
        self.epsilon = 0
        reward_list = []
        steps_list = []
        total_success = 0
        start = time.time()
        episode = 0
        while time.time() - start < time_limit:
            total_reward = 0
            total_steps = 0
            reset_seed = (seed + episode) if seed is not None else None
            state, info = self.env.reset(seed=reset_seed)
            done = False
            terminated = False
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
            episode += 1
        n = max(1, episode)
        reward_std = float(np.std(reward_list)) if reward_list else 0.0
        self.last_test_stats = {
            "reward_list": reward_list,
            "reward_mean": float(mean(reward_list)) if reward_list else 0.0,
            "reward_std": reward_std,
            "steps_mean": float(mean(steps_list)) if steps_list else 0.0,
            "success_rate": total_success / n,
            "test_episodes": episode,
            "epsilon_tolerance": reward_std / np.sqrt(n) if n > 0 else 0.0,
            "test_time": time.time() - start,
        }
        return self.last_test_stats

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
