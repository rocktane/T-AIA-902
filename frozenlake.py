import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", render_mode="ansi")
env.reset()
print(env.render())

n_episodes = 60100
q_table = np.zeros((16, 4))
lr = 0.05
gamma = 0.99
epsilon = 0.9
decay_rate = (0.05 / 0.9) ** (1 / n_episodes)
counter = 0
visit_count = np.zeros((16, 4))

for episode in range(n_episodes):
    state, info = env.reset()
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
        current_q = q_table[state, action]
        current_state = state
        visit_count[state, action] += 1
        lr = 1.0 / (1 + 0.001 * visit_count[state, action])
        state, reward, terminated, truncated, info = env.step(action)
        next_max = np.max(q_table[state])
        q_table[current_state, action] = current_q + lr * (float(reward) + gamma * next_max - current_q)
        if episode > n_episodes - 100 and reward == 1:
            counter += 1
        done = terminated or truncated

    if episode > 60000:
        epsilon = 0
    else:
        epsilon = max(0.05, epsilon * decay_rate)

print(f"Taux de succès: {100 * counter / 100:.1f}%")
