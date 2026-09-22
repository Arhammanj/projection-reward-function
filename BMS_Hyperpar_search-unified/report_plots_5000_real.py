"""
Corrected run: 5000 episodes, REAL price data (src/Prices.csv, now that the
hardcoded-broken-path bug in mg_env.py is fixed), same hyperparameters/seeds
as train.py. Trains on days 0-299, evaluates on unseen days 300-364.

Saves per-timestep trace (profit_traces_real.npz) and per-day profit values
(daily_profit_300_364_real.csv) so results are independently verifiable.
"""
import numpy as np
from matplotlib import pyplot as plt
from src.mg_env import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed
from src.loops import train

print("Training agent on days 0-299 (5000 episodes, real price data)...")
env_train = MicroGridEnv(day0=0, dayn=300)
set_seed(env_train, 42)

SOC_MIN, SOC_MAX = env_train.battery.min_soc, env_train.battery.max_soc
print(f"Battery SOC limits in use: {SOC_MIN} - {SOC_MAX}")

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

N_EPISODES = 5000
train_rewards, train_costs = train(agent, env_train, n_episodes=N_EPISODES)
train_rewards = np.asarray(train_rewards, dtype=np.float64)
train_costs = np.asarray(train_costs, dtype=np.float64)
print(f"Training done. Final 50-ep avg reward: {train_rewards[-50:].mean():.2f}")

np.save('train_rewards_5000_real.npy', train_rewards)
np.save('train_costs_5000_real.npy', train_costs)

# ── Figure: Training reward & operating cost convergence ───────────────────
window = 50
reward_ma = np.convolve(train_rewards, np.ones(window) / window, mode='valid')
cost_ma = np.convolve(train_costs, np.ones(window) / window, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
axes[0].plot(train_rewards, alpha=0.3, color='tab:blue', label='Episode reward')
axes[0].plot(np.arange(window - 1, N_EPISODES), reward_ma, color='tab:orange', linewidth=2,
             label=f'{window}-episode moving average')
axes[0].set_title('Training Reward Convergence — Reward v2 (Projection-Aware), 5000 episodes, real price data')
axes[0].set_ylabel('Reward')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_costs, alpha=0.3, color='tab:red', label='Episode operating cost')
axes[1].plot(np.arange(window - 1, N_EPISODES), cost_ma, color='tab:green', linewidth=2,
             label=f'{window}-episode moving average')
axes[1].set_title('Training Operating Cost Convergence')
axes[1].set_xlabel('Episode')
axes[1].set_ylabel('Cost')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_convergence_5000_real.png', dpi=150)
print("Saved: training_convergence_5000_real.png")

# ── Evaluate on unseen days 300-364 (pure exploitation) ─────────────────────
print("\nEvaluating on unseen days 300-364 (real price data)...")
env_eval = MicroGridEnv(day0=300, dayn=365)
set_seed(env_eval, 0)

DAY0, DAYN = 300, 365
records = []  # (day, hour, p_ch, p_dch, price, profit, soc)
SAMPLE_DAY = 310

for day in range(DAY0, DAYN):
    state = env_eval.reset(day=day)
    t = 0
    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env_eval.step(action)
        env_eval.render()

        signed = env_eval.battery.energy_change
        p_ch = max(signed, 0.0)
        p_dch = max(-signed, 0.0)
        price = env_eval.grid.sell_prices[env_eval.grid.time % len(env_eval.grid.sell_prices)]
        profit = price * p_dch - price * p_ch
        soc = env_eval.battery.SOC

        records.append((day, t, p_ch, p_dch, price, profit, soc))
        state = next_state
        t += 1

records = np.array(records)
np.savez('profit_traces_real.npz',
         day=records[:, 0].astype(int), hour=records[:, 1].astype(int),
         p_ch=records[:, 2], p_dch=records[:, 3],
         sell_price=records[:, 4], profit=records[:, 5], soc=records[:, 6])

day_arr = records[:, 0].astype(int)
hour_arr = records[:, 1].astype(int)
price_arr, p_ch_arr, p_dch_arr = records[:, 4], records[:, 2], records[:, 3]
profit_arr, soc_arr = records[:, 5], records[:, 6]

days = np.arange(300, 365)
profit_per_day = np.array([profit_arr[day_arr == d].sum() for d in days])
energy_sold = p_dch_arr.sum()
energy_bought = p_ch_arr.sum()

np.savetxt('daily_profit_300_364_real.csv',
           np.column_stack([days, profit_per_day]),
           delimiter=',', header='day,profit', comments='', fmt=['%d', '%.6f'])

print(f"\n=== Evaluation Results (days 300-364, REAL price data) ===")
print(f"Total profit      : {profit_per_day.sum():.2f}")
print(f"Average profit/day: {profit_per_day.mean():.2f}")
print(f"Std/day (ddof=0)  : {profit_per_day.std():.2f}")
print(f"Best day          : {profit_per_day.max():.2f} (day {days[profit_per_day.argmax()]})")
print(f"Worst day         : {profit_per_day.min():.2f} (day {days[profit_per_day.argmin()]})")
print(f"Energy sold (kWh) : {energy_sold:.2f}")
print(f"Energy bought(kWh): {energy_bought:.2f}")
print("\nPer-day values:")
for d, p in zip(days, profit_per_day):
    print(f"  day {d}: {p:.4f}")

# ── Figure: Battery operation, Day 310 (price / power / SOC) ───────────────
mask = day_arr == SAMPLE_DAY
order = np.argsort(hour_arr[mask])
hours = hour_arr[mask][order] + 1
price_day = price_arr[mask][order]
soc_day = soc_arr[mask][order]
charge_day = p_ch_arr[mask][order]
discharge_day = -p_dch_arr[mask][order]

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
axes[0].plot(hours, price_day, color='dodgerblue', linewidth=2, marker='o', markersize=4)
axes[0].set_title(f'Grid Price — Day {SAMPLE_DAY} (real data)')
axes[0].set_ylabel('Price')
axes[0].grid(True, alpha=0.3)

axes[1].bar(hours, charge_day, color='seagreen', label='Charge (kW)')
axes[1].bar(hours, discharge_day, color='tomato', label='Discharge (kW)')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_title(f'Battery Power — Day {SAMPLE_DAY}')
axes[1].set_ylabel('Power (kW)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(hours, soc_day, color='mediumseagreen', linewidth=2, marker='o', markersize=4)
axes[2].axhline(SOC_MAX, color='red', linestyle='--', linewidth=1.5, label=f'Max SOC ({SOC_MAX})')
axes[2].axhline(SOC_MIN, color='red', linestyle='--', linewidth=1.5, label=f'Min SOC ({SOC_MIN})')
axes[2].set_title(f'Battery SOC — Day {SAMPLE_DAY}')
axes[2].set_xlabel('Hour')
axes[2].set_ylabel('SOC')
axes[2].set_ylim(0, 1)
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('battery_operation_day310_real.png', dpi=150)
print("\nSaved: battery_operation_day310_real.png")

# ── Figure: Daily trading profit, days 300-364 ───────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['seagreen' if p >= 0 else 'tomato' for p in profit_per_day]
ax.bar(days, profit_per_day, color=colors, alpha=0.85)
ax.axhline(profit_per_day.mean(), color='orange', linewidth=2, linestyle='--',
           label=f'Mean = {profit_per_day.mean():.2f}')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Daily Trading Profit — Unseen Test Days 300-364 (real price data, 5000 episodes)')
ax.set_xlabel('Day')
ax.set_ylabel('Profit')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('daily_profit_300_364_real.png', dpi=150)
print("Saved: daily_profit_300_364_real.png")

print("\n=== Done ===")
