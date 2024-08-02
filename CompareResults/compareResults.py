import matplotlib.pyplot as plt
import numpy as np

def read_rewards(filename):
    episodes = []
    rewards = []
    with open(filename, 'r') as f:
        for line in f:
            episode, reward = line.strip().split(',')
            episodes.append(int(episode))
            rewards.append(float(reward))
    return episodes, rewards

def running_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

# Read the reward data
idqn_episodes, idqn_rewards = read_rewards('rewards_idqn.txt')
vdn_episodes, vdn_rewards = read_rewards('rewards_vdn.txt')

# Calculate running averages
window_size = 20
idqn_avg = running_average(idqn_rewards, window_size)
vdn_avg = running_average(vdn_rewards, window_size)

# Create the plot
plt.figure(figsize=(12, 6))

# Plot raw data
plt.plot(idqn_episodes, idqn_rewards, alpha=0.3, color='blue', label='IDQN')
plt.plot(vdn_episodes, vdn_rewards, alpha=0.3, color='red', label='VDN')

# Plot running averages
plt.plot(idqn_episodes[window_size-1:], idqn_avg, color='blue', linewidth=2, label='IDQN 20-ep avg')
plt.plot(vdn_episodes[window_size-1:], vdn_avg, color='red', linewidth=2, label='VDN 20-ep avg')

plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('IDQN vs VDN Rewards')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('idqn_vs_vdn_rewards.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'idqn_vs_vdn_rewards.png'")