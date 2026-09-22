"""
Builds the training reward/cost convergence plot from the saved raw arrays
produced by report_plots_5000_real.py, without needing to rerun training.
"""
import numpy as np
from matplotlib import pyplot as plt

train_rewards = np.load('train_rewards_5000_real.npy')
train_costs = np.load('train_costs_5000_real.npy')
n = len(train_rewards)

window = 50
reward_ma = np.convolve(train_rewards, np.ones(window) / window, mode='valid')
cost_ma = np.convolve(train_costs, np.ones(window) / window, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(train_rewards, alpha=0.3, color='tab:blue', label='Episode reward')
axes[0].plot(np.arange(window - 1, n), reward_ma, color='tab:orange', linewidth=2,
             label=f'{window}-episode moving average')
axes[0].set_title('Training Reward Convergence — Reward v2 (Projection-Aware), 5000 episodes, real price data')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_costs, alpha=0.3, color='tab:red', label='Episode operating cost')
axes[1].plot(np.arange(window - 1, n), cost_ma, color='tab:green', linewidth=2,
             label=f'{window}-episode moving average')
axes[1].set_title('Training Operating Cost Convergence')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Cost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_convergence_5000_real.png', dpi=150)
print("Saved: training_convergence_5000_real.png")
print(f"Final 50-ep avg reward: {train_rewards[-50:].mean():.2f}, cost: {train_costs[-50:].mean():.2f}")
