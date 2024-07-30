# idqn.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt

from environment import MultiAgentGridEnv

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class IDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
    
    def save(self, path):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']



    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_network(next_state)).item()
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            q_values[0][action] = target

            loss = nn.MSELoss()(q_values, self.q_network(state).detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_idqn(num_episodes=4, batch_size=32, update_freq=1, save_freq=2):
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)],
        reward_type='global'
    )
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    agents = [IDQNAgent(state_size, action_size) for _ in range(env.num_agents)]

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_reward = float('-inf')
    best_episode_actions = None

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            actions = [agent.act(state[i]) for i, agent in enumerate(agents)]
            episode_actions.append(actions)
            next_state, reward, done = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(state[i], actions[i], reward, next_state[i], done)
                agent.replay(batch_size)

            state = next_state
            total_reward += reward

        if episode % update_freq == 0:
            for agent in agents:
                agent.update_target_network()

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_actions = episode_actions

        if episode % save_freq == 0:
            for i, agent in enumerate(agents):
                agent.save(f'models/agent_{i}_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")

    for i, agent in enumerate(agents):
        agent.save(f'models/best_agent_{i}.pth')

    return agents, best_episode_actions




def visualize_best_strategy(env, best_episode_actions):
    fig, ax = plt.subplots(figsize=(10, 10))
    state = env.reset()
    
    for step, actions in enumerate(best_episode_actions):
        state, _, done = env.step(actions)
        env.render(ax, actions=actions, step=step)
        plt.pause(0.5)
        if done:
            break
    
    plt.show()


if __name__ == "__main__":
    trained_agents, best_episode_actions = train_idqn()
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=2,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)],
        reward_type='global'
    )
    visualize_best_strategy(env, best_episode_actions)

