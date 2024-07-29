import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
import os
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
        observation = np.concatenate([flattened_grid, sensor_readings])
        return observation

    def get_sensor_readings(self):
        readings = []
        for pos in self.agent_positions:
            readings.extend(self.sense_walls(pos))
        return np.array(readings)

    def sense_walls(self, position):
        x, y = position
        up = 1 if x == 0 or self.coverage_grid[x-1, y] == 1 else 0
        down = 1 if x == self.grid_size-1 or self.coverage_grid[x+1, y] == 1 else 0
        left = 1 if y == 0 or self.coverage_grid[x, y-1] == 1 else 0
        right = 1 if y == self.grid_size-1 or self.coverage_grid[x, y+1] == 1 else 0
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







def visualize_multi_agent_model(model, env, steps=50):
    unwrapped_env = env.envs[0].env
    state, _ = unwrapped_env.reset()
    done = False
    episode = 0

    fig, ax = plt.subplots()
    for step in range(steps):
        actions, _ = model.predict(state, deterministic=True)
        state, reward, done, truncated, _ = unwrapped_env.step(actions)
        unwrapped_env.render(ax=ax, actions=actions, episode=episode)
        if done:
            episode += 1
            state, _ = unwrapped_env.reset()
        plt.title(f'Step: {step + 1}, Actions: {actions}')
        plt.draw()
        plt.pause(0.5)

class LoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LoggingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            reward_mean = np.mean(self.locals['rewards'])
            print(f"Step: {self.num_timesteps}, Average Reward: {reward_mean}")
        return True

class RenderCallback(BaseCallback):
    def __init__(self, env, ax, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.env = env
        self.ax = ax
        self.episode = 0

    def _on_step(self) -> bool:
        actions = self.locals['actions'][0]
        if self.locals['dones'][0]:
            self.episode += 1
        self.env.get_attr('env')[0].render(ax=self.ax, actions=actions, episode=self.episode)
        return True

def train_multi_agent_model(env, eval_env, total_timesteps=10000):
    os.makedirs('./models/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)

    model = PPO('MlpPolicy', env, verbose=1)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='ppo_multi_agent_grid')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_multi_agent_model', log_path='./logs/', eval_freq=500, n_eval_episodes=10)
    fig, ax = plt.subplots()
    render_callback = RenderCallback(env, ax)
    logging_callback = LoggingCallback()

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, logging_callback, render_callback])
    model.save('ppo_multi_agent_grid_final')
    return model

def create_env(grid_file, coverage_radius=3, max_steps_per_episode=100, num_agents=4, initial_positions=None):
    env = MultiAgentGridEnv(grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions)
    env = Monitor(env)
    return env

def main():
    grid_file = 'grid_world.json'
    coverage_radius = 2
    max_steps_per_episode = 10
    num_agents = 4

    # Define initial positions for the agents
    initial_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    env = create_env(grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions)
    env = DummyVecEnv([lambda: env])
    eval_env = create_env(grid_file, coverage_radius, max_steps_per_episode, num_agents, initial_positions)
    eval_env = DummyVecEnv([lambda: eval_env])

    check_env(env.envs[0])

    model = train_multi_agent_model(env, eval_env)
    model = PPO.load('ppo_multi_agent_grid_final')

    visualize_multi_agent_model(model, env)

if __name__ == "__main__":
    main()
