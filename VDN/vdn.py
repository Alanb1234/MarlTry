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

class VDNQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(VDNQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class VDNMixer(nn.Module):
    def __init__(self, num_agents):
        super(VDNMixer, self).__init__()
        self.num_agents = num_agents

    def forward(self, agent_qs):
        # agent_qs shape: [batch_size, num_agents]
        return torch.sum(agent_qs, dim=1, keepdim=True)


class VDNAgent:
    def __init__(self, state_size, action_size, num_agents, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.q_networks = [VDNQNetwork(state_size, action_size) for _ in range(num_agents)]
        self.target_networks = [VDNQNetwork(state_size, action_size) for _ in range(num_agents)]
        for i in range(num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.mixer = VDNMixer(num_agents)
        self.target_mixer = VDNMixer(num_agents)
        self.optimizers = [optim.Adam(net.parameters(), lr=learning_rate) for net in self.q_networks]
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)

    def act(self, states, sensor_readings):
        actions = []
        for i in range(self.num_agents):
            if random.random() < self.epsilon:
                actions.append(random.randrange(self.action_size))
            else:
                with torch.no_grad():
                    state = torch.FloatTensor(states[i]).unsqueeze(0)
                    action_values = self.q_networks[i](state).squeeze(0)
                    mask = np.zeros(self.action_size, dtype=float)
                    for j, reading in enumerate(sensor_readings[i]):
                        if reading == 1:
                            mask[j] = float('-inf')
                    masked_action_values = action_values.cpu().numpy() + mask
                    valid_action_indices = np.where(mask == 0)[0]
                    if len(valid_action_indices) == 0:
                        actions.append(self.action_size - 1)
                    else:
                        best_action_index = valid_action_indices[np.argmax(masked_action_values[valid_action_indices])]
                        actions.append(best_action_index)
        return actions

    def remember(self, states, actions, reward, next_states, done):
        self.memory.append((states, actions, reward, next_states, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = [torch.FloatTensor(np.array(state)) for state in zip(*states)]
        next_states = [torch.FloatTensor(np.array(next_state)) for next_state in zip(*next_states)]
        actions = [torch.LongTensor(action) for action in zip(*actions)]
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Add this line
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Add this line

        current_q_values = [self.q_networks[i](states[i]).gather(1, actions[i].unsqueeze(1)).squeeze(1) 
                            for i in range(self.num_agents)]
        current_q_total = self.mixer(torch.stack(current_q_values, dim=1))  # Change this line

        next_q_values = [self.target_networks[i](next_states[i]).max(1)[0] for i in range(self.num_agents)]
        next_q_total = self.target_mixer(torch.stack(next_q_values, dim=1))  # Change this line

        target_q_total = rewards + (1 - dones) * self.gamma * next_q_total

        loss = nn.MSELoss()(current_q_total, target_q_total)
        
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()


    def update_target_network(self):
        for i in range(self.num_agents):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def save(self, path):
        torch.save({
            'q_networks_state_dict': [net.state_dict() for net in self.q_networks],
            'target_networks_state_dict': [net.state_dict() for net in self.target_networks],
            'mixer_state_dict': self.mixer.state_dict(),
            'target_mixer_state_dict': self.target_mixer.state_dict(),
            'optimizers_state_dict': [opt.state_dict() for opt in self.optimizers],
            'epsilon': self.epsilon,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        for i, net in enumerate(self.q_networks):
            net.load_state_dict(checkpoint['q_networks_state_dict'][i])
        for i, net in enumerate(self.target_networks):
            net.load_state_dict(checkpoint['target_networks_state_dict'][i])
        self.mixer.load_state_dict(checkpoint['mixer_state_dict'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer_state_dict'])
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizers_state_dict'][i])
        self.epsilon = checkpoint['epsilon']

def train_vdn(num_episodes=600, batch_size=32, update_freq=50, save_freq=100, epsilon_start=1.0, epsilon_min=0.00, epsilon_decay=0.005):
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )
    state_size = env.get_obs_size()
    action_size = env.get_total_actions()
    vdn_agent = VDNAgent(state_size, action_size, env.num_agents, epsilon=epsilon_start)

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    episode_rewards = []
    best_reward = float('-inf')
    best_episode_actions = None
    best_episode_number = None  
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        episode_actions = []

        while not done:
            sensor_readings = env.get_sensor_readings()
            actions = vdn_agent.act(state, sensor_readings)
            next_state, reward, done, actual_actions = env.step(actions)
            episode_actions.append(actual_actions)

            vdn_agent.remember(state, actual_actions, reward, next_state, done)
            vdn_agent.replay(batch_size)

            state = next_state
            total_reward += reward

        if episode % update_freq == 0:
            vdn_agent.update_target_network()

        vdn_agent.epsilon = max(epsilon_min, epsilon_start * np.exp(-epsilon_decay * episode))

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {vdn_agent.epsilon}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_episode_actions = episode_actions
            best_episode_number = episode  

        if episode % save_freq == 0:
            vdn_agent.save(f'models/vdn_agent_episode_{episode}.pth')

        with open('logs/rewards.txt', 'a') as f:
            f.write(f"{episode},{total_reward}\n")

    vdn_agent.save('models/best_vdn_agent.pth')

    save_best_episode(env.initial_positions, best_episode_actions, best_episode_number)  
    save_final_positions(env, best_episode_actions)
    visualize_and_record_best_strategy(env, best_episode_actions)
    return vdn_agent, best_episode_actions, best_episode_number  

# The helper functions save_best_episode, save_final_positions, and visualize_and_record_best_strategy 


def save_best_episode(initial_positions, best_episode_actions, best_episode_number, filename='vdn_best_strategy.json'):
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



def save_final_positions(env, best_episode_actions, filename='vdn_final_positions.png'):
    fig, ax = plt.subplots(figsize=(10, 10))
    env.reset()
    
    for actions in best_episode_actions:
        env.step(actions)
    
    env.render(ax, actions=best_episode_actions[-1], step=len(best_episode_actions)-1)
    plt.title("Final Positions")
    plt.savefig(filename)
    plt.close(fig)
    print(f"Final positions saved as {filename}")



def visualize_and_record_best_strategy(env, best_episode_actions, filename='vdn_best_episode.mp4'):
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
    trained_vdn_agent, best_episode_actions, best_episode_number = train_vdn()
    print(f"Best episode: {best_episode_number}")
    env = MultiAgentGridEnv(
        grid_file='grid_world.json',
        coverage_radius=7,
        max_steps_per_episode=100,
        num_agents=4,
        initial_positions=[(1, 1), (2, 1), (1, 2), (2, 2)]
    )