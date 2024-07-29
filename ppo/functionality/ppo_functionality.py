import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict, deque

class MultiAgentGridEnv(gym.Env):
    def __init__(self, grid_file, coverage_radius=3, max_steps_per_episode=100, num_agents=4, initial_positions=None):
        super(MultiAgentGridEnv, self).__init__()
        self.grid = self.load_grid(grid_file)
        self.grid_size = self.grid.shape[1]  # Use width for grid size
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.num_agents = num_agents
        self.current_step = 0

        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        obs_shape = (self.grid_size * self.grid_size + 4 * num_agents,)
        self.observation_space = spaces.Box(0, 1, shape=obs_shape, dtype=np.float32)

        self.initial_positions = initial_positions if initial_positions is not None else self.generate_valid_positions(num_agents)
        self.agent_positions = list(self.initial_positions)
        self.coverage_grid = np.copy(self.grid)
        self.reset()

    def load_grid(self, filename):
        with open(filename, 'r') as f:
            grid = np.array(json.load(f))
        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_positions = list(self.initial_positions)
        self.coverage_grid = np.copy(self.grid)
        for pos in self.agent_positions:
            self.cover_area(pos)
        self.current_step = 0
        return self.get_observation().astype(np.float32), {}

    def generate_valid_positions(self, num_agents):
        positions = []
        while len(positions) < num_agents:
            pos = (np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size))
            if self.grid[pos[1], pos[0]] == 0 and pos not in positions:
                positions.append(pos)
        return positions

    def step(self, actions):
        self.coverage_grid = np.copy(self.grid)
        rewards = np.zeros(self.num_agents)
        sensor_readings = self.get_sensor_readings()

        for i, action in enumerate(actions):
            if not self.is_move_valid(self.agent_positions[i], action, sensor_readings[i]):
                action = 4  # Stay

            next_state = self.move_agent(self.agent_positions[i], action)
            self.agent_positions[i] = next_state
            self.cover_area(next_state)

        reward, total_area, overlap_area, penalty, num_robots = self.calculate_global_reward()
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode
        truncated = False
        return self.get_observation().astype(np.float32), reward, done, truncated, {"reward": reward}


    def is_move_valid(self, position, action, sensor_reading):
        action_mapping = {0: 'forward', 1: 'backward', 2: 'left', 3: 'right', 4: 'stay'}
        action_direction = action_mapping[action]

        # Check sensor reading based on action direction
        if action_direction == 'forward' and sensor_reading[0] == 1:
            return False
        if action_direction == 'backward' and sensor_reading[1] == 1:
            return False
        if action_direction == 'left' and sensor_reading[2] == 1:
            return False
        if action_direction == 'right' and sensor_reading[3] == 1:
            return False

        # Check if the target position is occupied by another agent
        target_position = self.get_target_position(position, action)
        if target_position in self.agent_positions:
            return False

        return True


    def get_target_position(self, position, action):
        # Actions: 0 -> forward (x+1), 1 -> backward (x-1), 2 -> left (y+1), 3 -> right (y-1), 4 -> stay
        actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
        next_state = (position[0] + actions[action][0], position[1] + actions[action][1])
        next_state = (max(0, min(self.grid_size - 1, next_state[0])), max(0, min(self.grid_size - 1, next_state[1])))
        return next_state


    def move_agent(self, position, action):
        return self.get_target_position(position, action)

    def cover_area(self, state):
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                x, y = state[0] + dx, state[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size and self.grid[y, x] == 0:
                    self.coverage_grid[y, x] = 1



    def calculate_global_reward(self):
        total_area = 0
        overlap_area = 0
        fovs = []
        num_robots = self.num_agents

        for pos in self.agent_positions:
            fov = {
                'x_min': pos[0] - self.coverage_radius,
                'x_max': pos[0] + self.coverage_radius,
                'y_min': pos[1] - self.coverage_radius,
                'y_max': pos[1] + self.coverage_radius,
                'area': (2 * self.coverage_radius + 1) ** 2
            }
            fovs.append(fov)
            total_area += fov['area']

        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                overlap_x = max(0, min(fovs[i]['x_max'], fovs[j]['x_max']) - max(fovs[i]['x_min'], fovs[j]['x_min']))
                overlap_y = max(0, min(fovs[i]['y_max'], fovs[j]['y_max']) - max(fovs[i]['y_min'], fovs[j]['y_min']))
                overlap_area += overlap_x * overlap_y

        graph = self.build_graph(fovs)
        num_components = self.count_connected_components(graph, num_robots)
        penalty = num_components if num_components == num_robots else num_components - 1
        penalty_score = penalty * (total_area / num_robots)

        reward = total_area - overlap_area - penalty_score
        return reward, total_area, overlap_area, penalty, num_robots

    def build_graph(self, fovs):
        graph = defaultdict(list)
        for i in range(len(fovs)):
            for j in range(i + 1, len(fovs)):
                if self.fovs_overlap(fovs[i], fovs[j]):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph

    def count_connected_components(self, graph, num_nodes):
        visited = set()
        components = 0
        for node in range(num_nodes):
            if node not in visited:
                components += 1
                queue = deque([node])
                while queue:
                    current_node = queue.popleft()
                    if current_node not in visited:
                        visited.add(current_node)
                        queue.extend(graph[current_node])
        return components

    def fovs_overlap(self, fov1, fov2):
        overlap_x = max(0, min(fov1['x_max'], fov2['x_max']) - max(fov1['x_min'], fov2['x_min']))
        overlap_y = max(0, min(fov1['y_max'], fov2['y_max']) - max(fov1['y_min'], fov2['y_min']))
        return overlap_x * overlap_y > 0

    def get_observation(self):
        flattened_grid = self.coverage_grid.flatten()
        sensor_readings = np.concatenate(self.get_sensor_readings())
        observation = np.concatenate([flattened_grid, sensor_readings])
        return observation

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            readings.append(self.sense_obstacles(pos))
        return readings

    def sense_obstacles(self, position):
        x, y = position
        forward = 1 if x == self.grid_size - 1 or self.grid[y, x + 1] == 1 or (x + 1, y) in self.agent_positions else 0
        backward = 1 if x == 0 or self.grid[y, x - 1] == 1 or (x - 1, y) in self.agent_positions else 0
        left = 1 if y == self.grid_size - 1 or self.grid[y + 1, x] == 1 or (x, y + 1) in self.agent_positions else 0
        right = 1 if y == 0 or self.grid[y - 1, x] == 1 or (x, y - 1) in self.agent_positions else 0
        return [forward, backward, left, right]

    def render(self, ax=None, actions=None, episode=None, reward=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        for (j, i) in np.ndindex(self.grid.shape):
            if self.grid[i, j] == 1:  # Obstacles are black
                rect = plt.Rectangle((j, i), 1, 1, color='black')
            elif self.coverage_grid[i, j] == 1:  # Covered areas are blue
                rect = plt.Rectangle((j, i), 1, 1, color='blue', alpha=0.5)
            else:
                continue
            ax.add_patch(rect)
        for pos in self.agent_positions:
            rect = plt.Rectangle((pos[0], pos[1]), 1, 1, color='red')
            ax.add_patch(rect)

        # Display sensor readings
        sensor_readings = self.get_sensor_readings()
        for agent_idx, pos in enumerate(self.agent_positions):
            readings = sensor_readings[agent_idx]
            ax.text(pos[0] + 0.5, pos[1] + 0.5, f'{readings}', color='white', ha='center', va='center')

        plt.grid(True)
        if actions is not None:
            action_texts = ['forward', 'backward', 'left', 'right', 'stay']
            action_display = ', '.join([action_texts[action] for action in actions])
            title = f'Actions: {action_display}'
            if episode is not None:
                title += f' | Episode: {episode}'
            if reward is not None:
                title += f' | Reward: {reward:.2f}'
            ax.set_title(title)
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot



def main():
    grid_file = 'grid_world.json'
    coverage_radius = 2
    max_steps_per_episode = 10
    num_agents = 4

    # Define initial positions for the agents
    initial_positions = [(1, 1), (2, 1), (1, 2), (2, 2)]

    env = MultiAgentGridEnv(grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions)

    
    #[0, 0, 0, 0],   All agents move forward
    #[1, 1, 1, 1],   All agents move back
    #[2, 2, 2, 2],   All agents move left
    #[3, 3, 3, 3],   All agents move right
    #[4, 4, 4, 4],   All agents stay


    predefined_actions = [
        [2, 2, 2, 2],  
        [2, 2, 2, 2],  
        [2, 2, 2, 2],  
        [3, 3, 3, 3],  
        [3, 3, 3, 3],  
        [4, 4, 4, 4],
    ]

    fig, ax = plt.subplots()

    for step, actions in enumerate(predefined_actions):
        print(f"Step {step + 1}")
        print("Actions taken:", actions)
        observation, reward, done, truncated, info = env.step(actions)
        print("Agent positions:", env.agent_positions)
        print("Sensor readings:", env.get_sensor_readings())
        print("Reward:", reward)
        env.render(ax=ax, actions=actions)
        
        input("Press Enter to proceed to the next step...")  # Wait for user input to proceed
        
        if done:
            break


if __name__ == "__main__":
    main()
