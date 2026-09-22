"""
Full pipeline for the 5000-episode projection-reward run:
  1. Train (same config as train.py / evaluate.py, just n_episodes=5000).
  2. Save the trained model + raw train reward/cost arrays to saved_eval_5000/.
  3. Evaluate on unseen test days 300-364 (pure exploitation), capturing
     per-timestep SOC / price / charge-discharge / profit traces, same
     convention as evaluate.py + profit_analysis.py.
  4. Generate 3 styled figures (matching the v2_*_STYLED look):
       - v2_train_convergence_5000_STYLED.png  (Fig. 2 equivalent, 5000 ep)
       - v2_day310_5000_STYLED.png              (price / SOC / power, Day 310)
       - v2_daily_profit_5000_STYLED.png        (daily trading profit, days 300-364)

Usage:
    python train_5000_and_plot.py
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.loops import train
from src.utils import set_seed

SAVE_DIR = Path('saved_eval_5000')
SAVE_DIR.mkdir(exist_ok=True)

# ── Shared style spec (same as make_v2_figures_styled.py) ──────────────────
plt.style.use('seaborn-v0_8-whitegrid')
COLOR_POS = "#2E7D32"
COLOR_NEG = "#C62828"
COLOR_REF = "#1565C0"
COLOR_NEUTRAL = "#546E7A"
DPI = 200
SOC_MIN, SOC_MAX = 0.2, 0.8


def style_title(ax, text):
    ax.set_title(text, fontsize=14, fontweight='bold')


def style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)
    ax.tick_params(axis='both', labelsize=9)


def bar_colors(values):
    return [COLOR_POS if v >= 0 else COLOR_NEG for v in values]


def add_legend(ax):
    ax.legend(loc='upper right', frameon=True, fontsize=10)


# ══════════════════════════════════════════════════════════════════════════
# 1. Train (days 0-299, projection reward, n_episodes=5000)
# ══════════════════════════════════════════════════════════════════════════
print("Training agent on days 0-299 (5000 episodes, projection reward)...")
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

N_EPISODES = 5000
train_rewards, train_costs = train(agent, env_train, n_episodes=N_EPISODES)
train_rewards = np.asarray(train_rewards, dtype=np.float64)
train_costs = np.asarray(train_costs, dtype=np.float64)
print(f"Training done. Final 50-ep avg reward: {train_rewards[-50:].mean():.2f}")

agent.save_to_disk(SAVE_DIR)
np.save(SAVE_DIR / 'train_rewards.npy', train_rewards)
np.save(SAVE_DIR / 'train_costs.npy', train_costs)
print(f"Model + train arrays saved to {SAVE_DIR}/")

# ══════════════════════════════════════════════════════════════════════════
# 2. Evaluate on unseen days 300-364 (pure exploitation)
# ══════════════════════════════════════════════════════════════════════════
print("\nEvaluating on unseen days 300-364...")
env_eval = MicroGridEnv(day0=300, dayn=365)
set_seed(env_eval, 0)

DAY0, DAYN = 300, 365
records = []  # (day, hour, p_ch, p_dch, price, profit, soc)
eval_rewards = []
eval_costs = []

for day in range(DAY0, DAYN):
    state = env_eval.reset(day=day)
    ep_reward, ep_cost, t = 0.0, 0.0, 0
    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env_eval.step(action)
        op_cost = env_eval.render()

        signed = env_eval.battery.energy_change
        p_ch = max(signed, 0.0)
        p_dch = max(-signed, 0.0)
        price = env_eval.grid.sell_prices[env_eval.grid.time % len(env_eval.grid.sell_prices)]
        profit = price * p_dch - price * p_ch
        soc = env_eval.battery.SOC

        records.append((day, t, p_ch, p_dch, price, profit, soc))
        ep_reward += reward
        ep_cost += op_cost

        state = next_state
        t += 1

    eval_rewards.append(ep_reward)
    eval_costs.append(ep_cost)

records = np.array(records)
eval_rewards = np.array(eval_rewards)
eval_costs = np.array(eval_costs)

np.savez(SAVE_DIR / 'profit_traces.npz',
         day=records[:, 0].astype(int), hour=records[:, 1].astype(int),
         p_ch=records[:, 2], p_dch=records[:, 3],
         sell_price=records[:, 4], profit=records[:, 5], soc=records[:, 6])
np.save(SAVE_DIR / 'eval_rewards.npy', eval_rewards)
np.save(SAVE_DIR / 'eval_costs.npy', eval_costs)
print(f"Eval traces saved to {SAVE_DIR}/")

day_arr, hour_arr = records[:, 0].astype(int), records[:, 1].astype(int)
price_arr, p_ch_arr, p_dch_arr = records[:, 4], records[:, 2], records[:, 3]
profit_arr, soc_arr = records[:, 5], records[:, 6]

days = np.arange(300, 365)
profit_per_day = np.array([profit_arr[day_arr == d].sum() for d in days])
avg_profit = profit_per_day.mean()
print(f"Average trading profit: £{avg_profit:.2f}/day")

# ══════════════════════════════════════════════════════════════════════════
# Fig A — Training convergence, full 5000 episodes
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding training convergence figure (5000 episodes)...")
WINDOW = 50
reward_ma = np.convolve(train_rewards, np.ones(WINDOW) / WINDOW, mode='valid')
cost_ma = np.convolve(train_costs, np.ones(WINDOW) / WINDOW, mode='valid')
ma_x = np.arange(WINDOW - 1, N_EPISODES)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(train_rewards, color=COLOR_POS, alpha=0.35, linewidth=1, label='Episode reward')
axes[0].plot(ma_x, reward_ma, color=COLOR_REF, linestyle='--', linewidth=2, label=f'{WINDOW}-episode moving average')
axes[0].set_ylabel("Reward")
add_legend(axes[0])

axes[1].plot(train_costs, color=COLOR_NEG, alpha=0.35, linewidth=1, label='Episode cost')
axes[1].plot(ma_x, cost_ma, color=COLOR_REF, linestyle='--', linewidth=2, label=f'{WINDOW}-episode moving average')
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Cost (£)")
add_legend(axes[1])

for ax in axes:
    style_axes(ax)

fig.suptitle("Fig. 2. Training reward and operating cost convergence — Reward v2 (Projection-Aware), 5000 episodes",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = Path('v2_train_convergence_5000_STYLED.png')
fig.savefig(out1, dpi=DPI)
plt.close(fig)
print(f"Saved: {out1.resolve()}")

# ══════════════════════════════════════════════════════════════════════════
# Fig B — Single-day battery operation, Day 310 (price / SOC / power)
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding battery operation figure (Day 310)...")
DAY = 310
mask = day_arr == DAY
order = np.argsort(hour_arr[mask])
hours = hour_arr[mask][order] + 1
price_day = price_arr[mask][order]
soc_day = soc_arr[mask][order]
charge_day = p_ch_arr[mask][order]
discharge_day = -p_dch_arr[mask][order]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(hours, price_day, color=COLOR_NEUTRAL, linewidth=2, marker='o', markersize=4)
axes[0].set_ylabel("Price (£/kWh)")

axes[1].plot(hours, soc_day, color=COLOR_NEUTRAL, linewidth=2, marker='o', markersize=4, label='SOC')
axes[1].axhline(SOC_MAX, color=COLOR_REF, linestyle='--', linewidth=2, label=f'SOC limits ({SOC_MIN}-{SOC_MAX})')
axes[1].axhline(SOC_MIN, color=COLOR_REF, linestyle='--', linewidth=2, label='_nolegend_')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("SOC")
add_legend(axes[1])

axes[2].bar(hours, charge_day, color=COLOR_POS, label='Charge')
axes[2].bar(hours, discharge_day, color=COLOR_NEG, label='Discharge')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel("Power (kW)")
axes[2].set_xlabel("Hour")
add_legend(axes[2])

for ax in axes:
    style_axes(ax)

fig.suptitle(f"Fig. 6. Battery operation on Day {DAY} — Reward v2 (Projection-Aware), 5000-episode training",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out2 = Path('v2_day310_5000_STYLED.png')
fig.savefig(out2, dpi=DPI)
plt.close(fig)
print(f"Saved: {out2.resolve()}")

# ══════════════════════════════════════════════════════════════════════════
# Fig C — Daily trading profit, all 65 test days
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding daily trading profit figure...")
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(days, profit_per_day, color=bar_colors(profit_per_day))
ax.axhline(avg_profit, color=COLOR_REF, linestyle='--', linewidth=2, label=f'Average = £{avg_profit:.2f}/day')
style_title(ax, "Fig. 9. Daily trading profit on unseen test days 300-364 — Reward v2 (Projection-Aware), 5000-episode training")
ax.set_xlabel("Day")
ax.set_ylabel("Profit (£)")
style_axes(ax)
add_legend(ax)
fig.tight_layout()
out3 = Path('v2_daily_profit_5000_STYLED.png')
fig.savefig(out3, dpi=DPI)
plt.close(fig)
print(f"Saved: {out3.resolve()}")

print("\n=== All done ===")
for p in (out1, out2, out3):
    print(p.resolve())
