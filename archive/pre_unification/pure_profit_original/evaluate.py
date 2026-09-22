import numpy as np
import random
from pathlib import Path
from matplotlib import pyplot as plt
from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed

# ── Train the agent first (days 0-299) ────────────────────────────────────
from src.loops import train

print("Training agent on days 0-299...")
env_train = MicroGridEnv(day0=0, dayn=300)
set_seed(env_train, 42)

agent = QAgent(
    env_train,
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

train_rewards, train_costs = train(agent, env_train, n_episodes=1000)
print(f"Training done. Final 50-ep avg reward: {np.mean(train_rewards[-50:]):.2f}")

# ── Evaluate on unseen days 300-364 ───────────────────────────────────────
print("\nEvaluating on unseen days 300-364...")

env_eval = MicroGridEnv(day0=300, dayn=365)
set_seed(env_eval, 0)

eval_rewards = []
eval_costs   = []
eval_socs    = []   # full SOC trajectory for one sample day
eval_prices  = []   # price trajectory for same sample day
eval_actions = []   # charge/discharge for same sample day

SAMPLE_DAY = 310    # pick one day to plot in detail

for day in range(300, 365):
    state = env_eval.reset(day=day)
    ep_reward = 0.0
    ep_cost   = 0.0
    day_socs    = []
    day_prices  = []
    day_actions = []

    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)   # pure exploitation
        next_state, reward, done, _ = env_eval.step(action)
        op_cost = env_eval.render()

        ep_reward += reward
        ep_cost   += op_cost

        day_socs.append(env_eval.battery.SOC)
        day_prices.append(
            env_eval.grid.sell_prices[
                env_eval.grid.time % len(env_eval.grid.sell_prices)
            ]
        )
        day_actions.append(env_eval.battery.energy_change)
        state = next_state

    eval_rewards.append(ep_reward)
    eval_costs.append(ep_cost)

    if day == SAMPLE_DAY:
        eval_socs    = day_socs
        eval_prices  = day_prices
        eval_actions = day_actions

eval_rewards = np.array(eval_rewards)
eval_costs   = np.array(eval_costs)

print(f"\n=== Evaluation Results (days 300-364) ===")
print(f"Mean reward  : {eval_rewards.mean():.4f}")
print(f"Std  reward  : {eval_rewards.std():.4f}")
print(f"Mean cost    : {eval_costs.mean():.4f}  (negative = profit)")
print(f"Std  cost    : {eval_costs.std():.4f}")
print(f"Best day cost: {eval_costs.min():.4f}")
print(f"Worst day cost: {eval_costs.max():.4f}")

# ── Plot 1: Evaluation reward and cost per day ─────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

days = np.arange(300, 365)

axes[0].bar(days, eval_rewards, color='steelblue', alpha=0.7, label='Episode reward')
axes[0].axhline(eval_rewards.mean(), color='orange', linewidth=2,
                linestyle='--', label=f'Mean = {eval_rewards.mean():.2f}')
axes[0].set_title('Evaluation Reward — Unseen Days 300-364')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(days, eval_costs, color='tomato', alpha=0.7, label='Operation cost')
axes[1].axhline(eval_costs.mean(), color='green', linewidth=2,
                linestyle='--', label=f'Mean = {eval_costs.mean():.2f}')
axes[1].set_title('Evaluation Cost — Unseen Days 300-364')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Cost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluation_results.png', dpi=150)
plt.show()
print("Saved: evaluation_results.png")

# ── Plot 2: SOC + Price + Charge/Discharge for sample day ──────────────────
hours = np.arange(1, len(eval_socs) + 1)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

# Price
axes[0].plot(hours, eval_prices, color='dodgerblue', linewidth=2, marker='o', markersize=4)
axes[0].set_title(f'Grid Price — Day {SAMPLE_DAY}')
axes[0].set_ylabel('Price (€/kWh)')
axes[0].grid(True, alpha=0.3)

# SOC
axes[1].plot(hours, eval_socs, color='mediumseagreen', linewidth=2, marker='o', markersize=4)
axes[1].axhline(0.8, color='red',   linestyle='--', linewidth=1.5, label='Max SOC (0.8)')
axes[1].axhline(0.2, color='red',   linestyle='--', linewidth=1.5, label='Min SOC (0.2)')
axes[1].axhline(0.6, color='gray',  linestyle=':',  linewidth=1,   label='Target SOC (0.6)')
axes[1].set_title(f'Battery SOC — Day {SAMPLE_DAY}')
axes[1].set_ylabel('SOC')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Charge / Discharge
charge    = np.clip(eval_actions, 0, None)
discharge = np.clip(eval_actions, None, 0)
axes[2].bar(hours, charge,    color='seagreen', label='Charge (kW)')
axes[2].bar(hours, discharge, color='tomato',   label='Discharge (kW)')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_title(f'Battery Charge / Discharge — Day {SAMPLE_DAY}')
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('Power (kW)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('soc_analysis.png', dpi=150)
plt.show()
print("Saved: soc_analysis.png")

# ── Plot 3: Training curve ─────────────────────────────────────────────────
window = 50
reward_ma = np.convolve(train_rewards, np.ones(window)/window, mode='valid')
cost_ma   = np.convolve(train_costs,   np.ones(window)/window, mode='valid')
n = len(train_rewards)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(train_rewards, alpha=0.3, color='tab:blue', label='Episode reward')
axes[0].plot(np.arange(window-1, n), reward_ma, color='tab:orange',
             linewidth=2, label=f'{window}-ep moving average')
axes[0].set_title('Training Reward — Projection Reward Function')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_costs, alpha=0.3, color='tab:red', label='Episode cost')
axes[1].plot(np.arange(window-1, n), cost_ma, color='tab:green',
             linewidth=2, label=f'{window}-ep moving average')
axes[1].set_title('Training Cost')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Cost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curve_projection.png', dpi=150)
plt.show()
print("Saved: training_curve_projection.png")

print("\n=== All done ===")
print("Files saved:")
print("  training_curve_projection.png")
print("  evaluation_results.png")
print("  soc_analysis.png")