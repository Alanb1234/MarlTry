import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter


from environment import MultiAgentGridEnv
import json

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class IDQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)

    def act(self, state, sensor_reading):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action_values = self.q_network(state).squeeze(0)

            mask = np.zeros(self.action_size, dtype=float)
            for i, reading in enumerate(sensor_reading):
                if reading == 1:
                    mask[i] = float('-inf')

            masked_action_values = action_values.cpu().numpy() + mask
            valid_action_indices = np.where(mask == 0)[0]

            if len(valid_action_indices) == 0:
                return self.action_size - 1  # "stay" action if no valid actions
            
            best_action_index = valid_action_indices[np.argmax(masked_action_values[valid_action_indices])]
            return best_action_index

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

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

def train_idqn(num_episodes=600, batch_size=32, update_freq=50, save_freq=100, epsilon_start=1.0, epsilon_min=0.00, epsilon_decay=0.005):
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    agents = [IDQNAgent(state_size, action_size, epsilon=epsilon_start) for _ in range(env.num_agents)]

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None  # Add this line

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            sensor_readings = env.get_sensor_readings()
            actions = [agent.act(state[i], sensor_readings[i]) for i, agent in enumerate(agents)]
            next_state, reward, done, actual_actions = env.step(actions)
            episode_actions.append(actual_actions)

            for i, agent in enumerate(agents):
                agent.remember(state[i], actual_actions[i], reward, next_state[i], done)
                agent.replay(batch_size)

            state = next_state
            total_reward += reward

        if episode % update_freq == 0:
            for agent in agents:
                agent.update_target_network()

        for agent in agents:
            agent.epsilon = max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode))

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agents[0].epsilon}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode  # Add this line

        if episode % save_freq == 0:
            for i, agent in enumerate(agents):
                agent.save(f'models/agent_{i}_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")

    for i, agent in enumerate(agents):
        agent.save(f'models/best_agent_{i}.pth')

    save_best_episode(env.initial_positions, best_episode_actions, best_episode_number)  # Modify this line
    save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, best_episode_actions)
    return agents, best_episode_actions, best_episode_number  # Modify this line




def save_best_episode(initial_positions, best_episode_actions, best_episode_number, filename='idqn_best_strategy.json'):
    action_map = ['forward', 'backward', 'left', 'right', 'stay']
    
    best_episode = {
        "episode_number": best_episode_number
    }
    
    for i in range(len(initial_positions)):
        best_episode[f'agent_{i}'] = {
            'actions': [action_map[action[i]] for action in best_episode_actions],
            'initial_position': initial_positions[i]
        }
    
    with open(filename, 'w') as f:
        json.dump(best_episode, f, indent=4)

    print(f"Best episode actions and initial positions saved to {filename}")




def save_final_positions(env, best_episode_actions, filename='idqn_final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    for actions in best_episode_actions:
        env.step(actions)
    
    env.render(ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")






def visualize_and_record_best_strategy(env, best_episode_actions, filename='idqn_best_episode.mp4'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    # Set up the video writer
    writer = FFMpegWriter(fps=2)
    
    with writer.saving(fig, filename, dpi=100):
        for step, actions in enumerate(best_episode_actions):
            env.step(actions)
            ax.clear()
            env.render(ax, actions=actions, step=step)
            writer.grab_frame()
            plt.pause(0.1)
    
    plt.close(fig)
    print(f"Best episode visualization saved as {filename}")




if __name__ == "__main__":
    trained_agents, best_episode_actions, best_episode_number = train_idqn()
    print(f"Best episode: {best_episode_number}")
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )