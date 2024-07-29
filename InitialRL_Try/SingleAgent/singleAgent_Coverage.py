import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

class GridEnv(gym.Env):
    def __init__(self, grid_size=10, coverage_radius=3, max_steps_per_episode=10):
        super(GridEnv, self).__init__()
        self.grid_size = grid_size
        self.coverage_radius = coverage_radius
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.action_space = spaces.Discrete(5)  # 0=up, 1=down, 2=left, 3=right, 4=stay
        self.observation_space = spaces.Box(0, 1, shape=(grid_size * grid_size,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0, 0)
        self.coverage_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.cover_area(self.state)
        self.current_step = 0
        return self.coverage_grid.flatten(), {}


    def step(self, action):
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        next_state = (self.state[0] + actions[action][0], self.state[1] + actions[action][1])
        next_state = (max(0, min(self.grid_size - 1, next_state[0])), max(0, min(self.grid_size - 1, next_state[1])))

        previous_coverage = np.sum(self.coverage_grid)
        self.cover_area(next_state)
        new_coverage = np.sum(self.coverage_grid)

        reward = float(new_coverage - previous_coverage)  # Ensure reward is a float

        self.state = next_state
        self.current_step += 1
        done = self.current_step >= self.max_steps_per_episode
        truncated = False

        if done:
            self.reset()

        return self.coverage_grid.flatten(), reward, done, truncated, {}

    def cover_area(self, state):
        for dx in range(-self.coverage_radius, self.coverage_radius + 1):
            for dy in range(-self.coverage_radius, self.coverage_radius + 1):
                x, y = state[0] + dx, state[1] + dy
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    self.coverage_grid[x, y] = 1

    def render(self, ax=None, action=None, episode=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.clear()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        for (i, j) in np.ndindex(self.coverage_grid.shape):
            if self.coverage_grid[i, j] == 1:
                rect = plt.Rectangle((j, i), 1, 1, color='blue', alpha=0.5)
                ax.add_patch(rect)
        rect = plt.Rectangle((self.state[1], self.state[0]), 1, 1, color='red')
        ax.add_patch(rect)
        plt.grid(True)
        if action is not None:
            action_text = ['Up', 'Down', 'Left', 'Right', 'Stay'][action]
            title = f'Action: {action_text}'
            if episode is not None:
                title += f' | Episode: {episode}'
            ax.set_title(title)
        plt.draw()
        plt.pause(0.001)

def create_env(grid_size=10, coverage_radius=3, max_steps_per_episode=10):
    env = GridEnv(grid_size, coverage_radius, max_steps_per_episode)
    env = Monitor(env)
    return env

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
        action = self.locals['actions'][0]
        if self.locals['dones'][0]:
            self.episode += 1
        self.env.get_attr('env')[0].render(ax=self.ax, action=action, episode=self.episode)
        return True

def train_model(env, eval_env, total_timesteps=100000):
    model = DQN('MlpPolicy', env, verbose=1, exploration_fraction=0.1, exploration_final_eps=0.02,
                learning_rate=1e-3, buffer_size=50000, learning_starts=1000, target_update_interval=500,
                train_freq=1, batch_size=32, gamma=0.99)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='dqn_grid')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model', log_path='./logs/',
                                 eval_freq=500, n_eval_episodes=10)
    fig, ax = plt.subplots()
    render_callback = RenderCallback(env, ax)
    logging_callback = LoggingCallback()

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, logging_callback, render_callback])
    model.save('dqn_grid_final')
    return model

def visualize_model(model, env, steps=50):
    unwrapped_env = env.envs[0].env  # Access the underlying environment
    state, _ = unwrapped_env.reset()  # Unpack the observation
    done = False
    episode = 0

    fig, ax = plt.subplots()
    for step in range(steps):
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, truncated, _ = unwrapped_env.step(action)
        unwrapped_env.render(ax=ax, action=action, episode=episode)
        if done:
            episode += 1
            state, _ = unwrapped_env.reset()
        plt.title(f'Step: {step + 1}, Action: {action}')
        plt.draw()
        plt.pause(0.5)





def main():
    grid_size = 10
    coverage_radius = 3
    max_steps_per_episode = 10
    env = create_env(grid_size, coverage_radius, max_steps_per_episode)
    env = DummyVecEnv([lambda: env])
    eval_env = create_env(grid_size, coverage_radius, max_steps_per_episode)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Check the environment
    check_env(env.envs[0])

    #model = train_model(env, eval_env)
    model = DQN.load('dqn_grid_final')
    visualize_model(model, env)

if __name__ == "__main__":
    main()
