"""
Reproduces train.py's exact config (projection reward, same hyperparameters,
seed 42) for n_episodes=5000, and additionally persists the raw per-episode
reward/cost arrays to .npy so the run doesn't need to be repeated for future
plots.

Usage:
    python train_5000.py
"""
import numpy as np
from matplotlib import pyplot as plt
from src.mg_env import MicroGridEnv
from src.q_agent import QAgent
from src.loops import train
from src.utils import set_seed

# ── Setup (identical to train.py) ───────────────────────────────────────────
env = MicroGridEnv()
set_seed(env, 42)

agent = QAgent(
    env,
    learning_rate=0.0002,
    discount_factor=0.99,
    batch_size=128,
    memory_size=10000,
    freq_steps_train=16,
    freq_steps_update_target=10,
    n_steps_warm_up_memory=1000,
    n_gradient_steps=16,
    nn_hidden_layers=[256, 256],
    max_grad_norm=1,
    normalize_state=False,
    epsilon_start=0.9,
    epsilon_end=0.15,
    steps_epsilon_decay=10000,
)

# ── Train ──────────────────────────────────────────────────────────────────
n_episodes = 5000
rewards, costs = train(agent, env, n_episodes=n_episodes)
rewards = np.asarray(rewards, dtype=np.float64)
costs = np.asarray(costs, dtype=np.float64)

# ── Persist raw arrays ───────────────────────────────────────────────────────
np.save('train_rewards_5000.npy', rewards)
np.save('train_costs_5000.npy', costs)
print(f"Saved train_rewards_5000.npy, train_costs_5000.npy (shape {rewards.shape})")

# ── Plot ───────────────────────────────────────────────────────────────────
window = 50
reward_ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
cost_ma = np.convolve(costs, np.ones(window) / window, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(rewards, alpha=0.3, color='tab:blue', label='Episode reward')
axes[0].plot(np.arange(window - 1, n_episodes), reward_ma, color='tab:orange', linewidth=2, label=f'{window}-ep moving average')
axes[0].set_title('Training Reward — Projection Reward Function (5000 episodes)')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(costs, alpha=0.3, color='tab:red', label='Episode cost')
axes[1].plot(np.arange(window - 1, n_episodes), cost_ma, color='tab:green', linewidth=2, label=f'{window}-ep moving average')
axes[1].set_title('Training Operating Cost (5000 episodes)')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Cost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curve_projection_5000.png', dpi=150)
print("Done. Plot saved to training_curve_projection_5000.png")
print(f"Final 50-ep avg reward: {rewards[-50:].mean():.2f}, cost: {costs[-50:].mean():.2f}")
