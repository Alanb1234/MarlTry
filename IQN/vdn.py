import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
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

class VDNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = action_size
        self.memory = deque(maxlen=10000)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network(state)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size, other_agents):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                own_next_q = torch.max(self.target_network(next_state)).item()
                other_next_q = sum(torch.max(agent.target_network(next_state)).item() for agent in other_agents)
                target = reward + self.gamma * (own_next_q + other_next_q)
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            q_values[0][action] = target

            loss = nn.MSELoss()(q_values, self.q_network(state).detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

def train_vdn(num_episodes=1000, batch_size=32, update_freq=10):
    env = MultiAgentGridEnv('grid_world.json', 2, 100, 4, 'vdn')
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    agents = [VDNAgent(state_size, action_size) for _ in range(env.num_agents)]

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            actions = [agent.act(state[i]) for i, agent in enumerate(agents)]
            next_state, rewards, done = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(state[i], actions[i], rewards[i], next_state[i], done)
                agent.replay(batch_size, [a for j, a in enumerate(agents) if j != i])

            state = next_state
            total_reward += sum(rewards)

        if episode % update_freq == 0:
            for agent in agents:
                agent.update_target_network()

        print(f"Episode {episode}, Total Reward: {total_reward}")

    return agents

if __name__ == "__main__":
    trained_agents = train_vdn()
    # You can add code here to save the trained agents or perform immediate evaluation