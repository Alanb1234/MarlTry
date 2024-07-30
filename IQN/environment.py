import numpy as np
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque

class MultiAgentGridEnv:
    def __init__(self, grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions, reward_type='global'):
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[1]
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents
        self.initial_positions = initial_positions
        self.reward_type = reward_type
        self.reset()



    def load_grid(self, filename):
        with open(filename, 'r') as f:
            return np.array(json.load(f))

    def reset(self):
        self.agent_positions = list(self.initial_positions)
        self.coverage_grid = np.zeros_like(self.grid)
        self.current_step = 0
        self.update_coverage()
        return self.get_observations()

    def step(self, actions):
        self.current_step += 1
        for i, action in enumerate(actions):
            new_pos = self.get_new_position(self.agent_positions[i], action)
            if self.is_valid_position(new_pos):
                self.agent_positions[i] = new_pos
        self.update_coverage()
        reward = self.calculate_global_reward()
        done = self.current_step >= self.max_steps_per_episode
        return self.get_observations(), reward, done
    
    def update_coverage(self):
        self.coverage_grid = np.zeros_like(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)



    def get_new_position(self, position, action):
        x, y = position
        if action == 0:  # up
            return (x, min(y + 1, self.grid_size - 1))
        elif action == 1:  # down
            return (x, max(y - 1, 0))
        elif action == 2:  # left
            return (max(x - 1, 0), y)
        elif action == 3:  # right
            return (min(x + 1, self.grid_size - 1), y)
        else:  # stay
            return (x, y)

    def is_valid_position(self, position):
        x, y = position
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[y, x] == 0

    def cover_area(self, state):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    self.coverage_grid[ny, nx] = 1




    def calculate_reward(self, state):
        reward = 0
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                x, y = state[0] + dx, state[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    reward += self.coverage_grid[y, x]
        return reward



    def calculate_global_reward(self):
        total_area = np.sum(self.coverage_grid)
        overlap_area = self.calculate_overlap()
        num_components = self.count_connected_components()
        penalty = num_components if num_components == self.num_agents else num_components - 1
        penalty_score = penalty * (total_area / self.num_agents)
        reward = total_area - overlap_area - penalty_score
        return reward
    
    def calculate_overlap(self):
        overlap_grid = np.zeros_like(self.coverage_grid)
        for pos in self.agent_positions:
            temp_grid = np.zeros_like(self.coverage_grid)
            self.cover_area_on_grid(pos, temp_grid)
            overlap_grid += temp_grid
        return np.sum(overlap_grid > 1)

    def cover_area_on_grid(self, state, grid):
        x, y = state
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                    grid[ny, nx] = 1

    def count_connected_components(self):
        graph = self.build_graph()
        visited = set()
        components = 0
        for node in range(self.num_agents):
            if node not in visited:
                components += 1
                self.bfs(node, graph, visited)
        return components
    
    def build_graph(self):
        graph = defaultdict(list)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if self.areas_overlap(self.agent_positions[i], self.agent_positions[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph
    
    def areas_overlap(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) <= 2 * self.coverage_radius and abs(y1 - y2) <= 2 * self.coverage_radius

    def bfs(self, start, graph, visited):
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)






    def calculate_individual_reward(self, position):
        x, y = position
        reward = 0
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    reward += self.coverage_grid[ny, nx]
        return reward / ((2 * self.coverage_radius + 1) ** 2)

    def get_observations(self):
        observations = []
        for pos in self.agent_positions:
            obs = np.zeros((self.grid_size, self.grid_size, 3))
            obs[:,:,0] = self.grid
            obs[:,:,1] = self.coverage_grid
            x, y = pos
            obs[y, x, 2] = 1
            observations.append(obs.flatten())
        return observations

    def get_obs_size(self):
        return self.grid_size * self.grid_size * 3

    def get_total_actions(self):
        return 5  # up, down, left, right, stay

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            x, y = pos
            reading = [
                1 if y == self.grid_size - 1 or self.grid[y + 1, x] == 1 else 0,  # up
                1 if y == 0 or self.grid[y - 1, x] == 1 else 0,  # down
                1 if x == 0 or self.grid[y, x - 1] == 1 else 0,  # left
                1 if x == self.grid_size - 1 or self.grid[y, x + 1] == 1 else 0  # right
            ]
            readings.append(reading)
        return readings


    def render(self, ax=None, actions=None, step=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        
        # Draw the grid and obstacles
        for (j, i) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:  # Obstacles are black
                rect = plt.Rectangle((j, i), 1, 1, color='black')
                ax.add_patch(rect)
        
        # Draw the coverage area for each agent
        for pos in self.agent_positions:
            x, y = pos
            for dx in range(-self.coverage_radius, self.coverage_radius + 1):
                for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.grid[ny, nx] == 0:
                        rect = plt.Rectangle((nx, ny), 1, 1, color='blue', alpha=0.3)
                        ax.add_patch(rect)
        
        # Draw the agents
        for pos in self.agent_positions:
            x, y = pos
            rect = plt.Rectangle((x, y), 1, 1, color='red')
            ax.add_patch(rect)

        ax.grid(True)
        if actions is not None:
            action_texts = ['up', 'down', 'left', 'right', 'stay']
            action_display = ', '.join([action_texts[action] for action in actions])
            title = f'Actions: {action_display}'
            if step is not None:
                title += f' | Step: {step}'
            ax.set_title(title)
        plt.draw()
        plt.pause(0.001)




