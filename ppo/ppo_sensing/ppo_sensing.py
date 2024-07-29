# In this file we develope and explore the sensing aspect#

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json

class MultiAgentGridEnv(gym.Env):
    def __init__(self, grid_file, coverage_radius=3, max_steps_per_episode=100, num_agents=4, initial_positions=None):
        super(MultiAgentGridEnv, self).__init__()
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[0]
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents
        self.current_step = 0

        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        self.observation_space = spaces.Box(0, 1, shape=(self.grid_size * self.grid_size + 4 * num_agents,), dtype=np.float32)

        self.initial_positions = initial_positions if initial_positions is not None else self.generate_valid_positions(num_agents)
        self.agent_positions = list(self.initial_positions)  # Use a copy to avoid modifying the original
        self.coverage_grid = np.copy(self.grid)
        self.reset()

    def load_grid(self, filename):
        with open(filename, 'r') as f:
            grid = np.array(json.load(f))
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_positions = list(self.initial_positions)  # Reset to initial positions
        self.coverage_grid = np.copy(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)
        self.current_step = 0
        return self.get_observation().astype(np.float32), {}

    def generate_valid_positions(self, num_agents):
        positions = []
        while len(positions) < num_agents:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if self.grid[pos[0], pos[1]] == 0 and pos not in positions:
                positions.append(pos)
        return positions

    def step(self, actions):
        self.coverage_grid = np.copy(self.grid)
        rewards = np.zeros(self.num_agents)
        for i, action in enumerate(actions):
            next_state = self.move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = next_state
            self.cover_area(next_state)
            rewards[i] = self.calculate_individual_reward(next_state)
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode
        truncated = False
        return self.get_observation().astype(np.float32), float(np.sum(rewards)), done, truncated, {}

    def move_agent(self, position, action):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        next_state = (position[0] + actions[action][0], position[1] + actions[action][1])
        next_state = (max(0, min(self.grid_size - 1, next_state[0])), max(0, min(self.grid_size - 1, next_state[1])))
        if self.grid[next_state[0], next_state[1]] == 0:  # Check if the next state is not an obstacle
            return next_state
        else:
            return position  # Stay in the same position if the next state is an obstacle

    def cover_area(self, state):
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                x, y = state[0] + dx, state[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[x, y] == 0:
                    self.coverage_grid[x, y] = 1

    def calculate_individual_reward(self, state):
        reward = 0
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                x, y = state[0] + dx, state[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    reward += self.coverage_grid[x, y]
        return reward

    def get_observation(self):
        flattened_grid = self.coverage_grid.flatten()
        sensor_readings = self.get_sensor_readings()
        observation = np.concatenate([flattened_grid, np.concatenate(sensor_readings)])
        return observation

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            readings.append(self.sense_obstacles(pos))
        return readings

    def sense_obstacles(self, position):
        x, y = position
        up = 1 if x == 0 or self.grid[x-1, y] == 1 else 0
        down = 1 if x == self.grid_size-1 or self.grid[x+1, y] == 1 else 0
        left = 1 if y == 0 or self.grid[x, y-1] == 1 else 0
        right = 1 if y == self.grid_size-1 or self.grid[x, y+1] == 1 else 0
        return [up, down, left, right]

    def render(self, ax=None, actions=None, episode=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        for (i, j) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:  # Obstacles are black
                rect = plt.Rectangle((j, i), 1, 1, color='black')
            elif self.coverage_grid[i, j] == 1:  # Covered areas are blue
                rect = plt.Rectangle((j, i), 1, 1, color='blue', alpha=0.5)
            else:
                continue
            ax.add_patch(rect)
        for pos in self.agent_positions:
            rect = plt.Rectangle((pos[1], pos[0]), 1, 1, color='red')
            ax.add_patch(rect)
        plt.grid(True)
        if actions is not None:
            action_texts = ['left', 'right', 'forward', 'back', 'Stay']
            action_display = ', '.join([action_texts[action] for action in actions])
            title = f'Actions: {action_display}'
            if episode is not None:
                title += f' | Episode: {episode}'
            ax.set_title(title)
        plt.draw()
        plt.pause(0.001)

def main():
    grid_file = 'grid_world.json'
    coverage_radius = 2
    max_steps_per_episode = 10
    num_agents = 4

    # Define initial positions for the agents
    initial_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    env = MultiAgentGridEnv(grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions)

    actions_list = [
        [0, 1, 2, 3],  # Up, Down, Left, Right
        [1, 0, 3, 2],  # Down, Up, Right, Left
        [2, 3, 0, 1],  # Left, Right, Up, Down
        [3, 2, 1, 0],  # Right, Left, Down, Up
        [4, 4, 4, 4]   # Stay, Stay, Stay, Stay
    ]

    fig, ax = plt.subplots()
    for step, actions in enumerate(actions_list):
        observation, reward, done, truncated, _ = env.step(actions)
        sensor_readings = env.get_sensor_readings()
        print(f"Step {step + 1}")
        print(f"Actions: {actions}")
        for agent_idx, readings in enumerate(sensor_readings):
            print(f"Agent {agent_idx + 1} Sensor Readings: {readings}")
        env.render(ax=ax, actions=actions, episode=step + 1)
    
    plt.show()

if __name__ == "__main__":
    main()
