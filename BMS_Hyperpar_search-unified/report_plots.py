import numpy as np
from matplotlib import pyplot as plt
from src.mg_env import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed
from src.loops import train

# ── Train agent on days 0-299 ───────────────────────────────────────────────
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

# ── Evaluate on unseen days 300-364 ─────────────────────────────────────────
print("\nEvaluating on unseen days 300-364...")

env_eval = MicroGridEnv(day0=300, dayn=365)
set_seed(env_eval, 0)

eval_costs = []
sample_socs, sample_prices, sample_actions = [], [], []

SAMPLE_DAY = 310  # representative day for the price/power/SOC panel

for day in range(300, 365):
    state = env_eval.reset(day=day)
    ep_cost = 0.0
    day_socs, day_prices, day_actions = [], [], []

    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env_eval.step(action)
        op_cost = env_eval.render()
        ep_cost += op_cost

        day_socs.append(env_eval.battery.SOC)
        day_prices.append(
            env_eval.grid.sell_prices[env_eval.grid.time % len(env_eval.grid.sell_prices)]
        )
        day_actions.append(env_eval.battery.energy_change)
        state = next_state

    eval_costs.append(ep_cost)

    if day == SAMPLE_DAY:
        sample_socs, sample_prices, sample_actions = day_socs, day_prices, day_actions

eval_costs = np.array(eval_costs)
daily_profit = -eval_costs  # cost convention: positive = spend, negative = revenue
days = np.arange(300, 365)

# ── Persist raw data so future plot tweaks don't require retraining ─────────
np.savez('eval_data.npz',
         days=days, daily_profit=daily_profit,
         sample_day=SAMPLE_DAY, sample_socs=sample_socs,
         sample_prices=sample_prices, sample_actions=sample_actions)

print(f"\n=== Evaluation Results (days 300-364) ===")
print(f"Mean daily profit : {daily_profit.mean():.4f}")
print(f"Std  daily profit : {daily_profit.std():.4f}")
print(f"Best day profit   : {daily_profit.max():.4f}")
print(f"Worst day profit  : {daily_profit.min():.4f}")
print(f"Total profit      : {daily_profit.sum():.4f}")

# ── Figure 1: Price / Power / SOC for the sample day ────────────────────────
hours = np.arange(1, len(sample_socs) + 1)
charge = np.clip(sample_actions, 0, None)
discharge = np.clip(sample_actions, None, 0)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(hours, sample_prices, color='dodgerblue', linewidth=2, marker='o', markersize=4)
axes[0].set_title(f'Grid Price — Day {SAMPLE_DAY}')
axes[0].set_ylabel('Price (€/kWh)')
axes[0].grid(True, alpha=0.3)

axes[1].bar(hours, charge, color='seagreen', label='Charge (kW)')
axes[1].bar(hours, discharge, color='tomato', label='Discharge (kW)')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title(f'Battery Power — Day {SAMPLE_DAY}')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(hours, sample_socs, color='mediumseagreen', linewidth=2, marker='o', markersize=4)
axes[2].axhline(0.8, color='red', linestyle='--', linewidth=1.5, label='Max SOC (0.8)')
axes[2].axhline(0.2, color='red', linestyle='--', linewidth=1.5, label='Min SOC (0.2)')
axes[2].set_title(f'Battery SOC — Day {SAMPLE_DAY}')
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('SOC')
axes[2].set_ylim(0, 1)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('price_power_soc.png', dpi=150)
print("Saved: price_power_soc.png")

# ── Figure 2: Daily profit, days 300-365 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['seagreen' if p >= 0 else 'tomato' for p in daily_profit]
ax.bar(days, daily_profit, color=colors, alpha=0.8)
ax.axhline(daily_profit.mean(), color='orange', linewidth=2, linestyle='--',
           label=f'Mean = {daily_profit.mean():.2f}')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Daily Profit — Days 300-365')
ax.set_xlabel('Day')
ax.set_ylabel('Profit')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('daily_profit_300_364.png', dpi=150)
print("Saved: daily_profit_300_364.png")

# ── Figure 3: Battery operation on Day 310 (power + SOC) ────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.suptitle(f'Battery Operation — Day {SAMPLE_DAY}', fontsize=14)

axes[0].bar(hours, charge, color='seagreen', label='Charge (kW)')
axes[0].bar(hours, discharge, color='tomato', label='Discharge (kW)')
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].set_ylabel('Power (kW)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hours, sample_socs, color='mediumseagreen', linewidth=2, marker='o', markersize=4)
axes[1].axhline(0.8, color='red', linestyle='--', linewidth=1.5, label='Max SOC (0.8)')
axes[1].axhline(0.2, color='red', linestyle='--', linewidth=1.5, label='Min SOC (0.2)')
axes[1].set_xlabel('Hour')
axes[1].set_ylabel('SOC')
axes[1].set_ylim(0, 1)
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('battery_operation_day310.png', dpi=150)
print("Saved: battery_operation_day310.png")

# ── Figure 4: Daily trading profit on unseen test days 300-364 ─────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(days, daily_profit, color=colors, alpha=0.8)
ax.axhline(daily_profit.mean(), color='orange', linewidth=2, linestyle='--',
           label=f'Mean = {daily_profit.mean():.2f}')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Daily Trading Profit — Unseen Test Days 300–364')
ax.set_xlabel('Day')
ax.set_ylabel('Profit')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('daily_trading_profit_300_364.png', dpi=150)
print("Saved: daily_trading_profit_300_364.png")

print("\n=== Done ===")
