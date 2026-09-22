"""
Regenerate the Reward v2 (Projection-Aware) figure set using the shared style
spec agreed for the parallel Reward v1 figure set (seaborn-v0_8-whitegrid,
fixed color roles, exact titles, dpi=200 PNGs).

Plotting-only script: loads the already-trained agent from saved_eval/ and
reuses the existing per-timestep test traces (saved_eval/profit_traces.npz,
saved_eval/eval_rewards.npy, saved_eval/eval_costs.npy) produced by
evaluate.py / profit_analysis.py for the projection reward function
(mg_env.py MicroGridEnv, default reward_mode="projection" -> _projection_reward).
No reward function or environment logic is modified here.

Fig. 2 (training convergence) requires per-episode reward/cost history that
was never persisted to disk, so this script reproduces it by re-running the
exact training procedure used in evaluate.py to build saved_eval/model
(same env days 0-299, seed 42, hyperparameters, n_episodes=1000) purely to
capture the learning curve for plotting.

Usage:
    python make_v2_figures_styled.py
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

# ── Shared style spec ────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')

COLOR_POS = "#2E7D32"   # profit / charge / positive
COLOR_NEG = "#C62828"   # loss / discharge / negative
COLOR_REF = "#1565C0"   # average / reference line
COLOR_NEUTRAL = "#546E7A"  # price line (not a profit/loss/reference quantity)

DPI = 200
SAVE_DIR = Path('saved_eval')
SOC_MIN, SOC_MAX = 0.2, 0.8


def style_title(ax_or_fig, text):
    ax_or_fig.set_title(text, fontsize=14, fontweight='bold')


def style_axes(ax):
    ax.set_xlabel(ax.get_xlabel(), fontsize=11)
    ax.set_ylabel(ax.get_ylabel(), fontsize=11)
    ax.tick_params(axis='both', labelsize=9)


def bar_colors(values):
    return [COLOR_POS if v >= 0 else COLOR_NEG for v in values]


def add_legend(ax):
    ax.legend(loc='upper right', frameon=True, fontsize=10)


# ══════════════════════════════════════════════════════════════════════════
# Load shared data
# ══════════════════════════════════════════════════════════════════════════
print("Loading trading profit traces and evaluation arrays from saved_eval/...")
traces = np.load(SAVE_DIR / 'profit_traces.npz')
day_arr = traces['day']
hour_arr = traces['hour']
price_arr = traces['sell_price']
p_ch_arr = traces['p_ch']
p_dch_arr = traces['p_dch']
profit_arr = traces['profit']
soc_arr = traces['soc']

eval_rewards = np.load(SAVE_DIR / 'eval_rewards.npy')
eval_costs = np.load(SAVE_DIR / 'eval_costs.npy')

days = np.arange(300, 365)
profit_per_day = np.array([profit_arr[day_arr == d].sum() for d in days])
avg_profit = profit_per_day.mean()
print(f"Average trading profit: £{avg_profit:.2f}/day")


# ══════════════════════════════════════════════════════════════════════════
# Fig. 9 — Daily trading profit, all 65 days
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding Fig. 9 (daily trading profit)...")
fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(days, profit_per_day, color=bar_colors(profit_per_day))
ax.axhline(avg_profit, color=COLOR_REF, linestyle='--', linewidth=2,
           label=f'Average = £{avg_profit:.2f}/day')

style_title(ax, "Fig. 9. Daily trading profit on unseen test days 300-364 — Reward v2 (Projection-Aware)")
ax.set_xlabel("Day")
ax.set_ylabel("Profit (£)")
style_axes(ax)
add_legend(ax)

fig.tight_layout()
out1 = Path('v2_daily_profit_STYLED.png')
fig.savefig(out1, dpi=DPI)
plt.close(fig)
print(f"Saved: {out1.resolve()}")


# ══════════════════════════════════════════════════════════════════════════
# Fig. 6 — Single-day battery operation, Day 310
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding Fig. 6 (Day 310 battery operation)...")
DAY = 310
mask = day_arr == DAY
order = np.argsort(hour_arr[mask])
hours = hour_arr[mask][order] + 1   # 1-24 to match repo convention
price_day = price_arr[mask][order]
soc_day = soc_arr[mask][order]
p_ch_day = p_ch_arr[mask][order]
p_dch_day = p_dch_arr[mask][order]
charge_day = p_ch_day
discharge_day = -p_dch_day

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Price
axes[0].plot(hours, price_day, color=COLOR_NEUTRAL, linewidth=2, marker='o', markersize=4)
axes[0].set_ylabel("Price (£/kWh)")

# SOC with boundary lines
axes[1].plot(hours, soc_day, color=COLOR_NEUTRAL, linewidth=2, marker='o', markersize=4, label='SOC')
axes[1].axhline(SOC_MAX, color=COLOR_REF, linestyle='--', linewidth=2, label=f'SOC limits ({SOC_MIN}-{SOC_MAX})')
axes[1].axhline(SOC_MIN, color=COLOR_REF, linestyle='--', linewidth=2, label='_nolegend_')
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("SOC")
add_legend(axes[1])

# Charge / discharge
axes[2].bar(hours, charge_day, color=COLOR_POS, label='Charge')
axes[2].bar(hours, discharge_day, color=COLOR_NEG, label='Discharge')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].set_ylabel("Power (kW)")
axes[2].set_xlabel("Hour")
add_legend(axes[2])

for ax in axes:
    style_axes(ax)

fig.suptitle(f"Fig. 6. Battery operation on Day {DAY} — Reward v2 (Projection-Aware)",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out2 = Path('v2_day310_STYLED.png')
fig.savefig(out2, dpi=DPI)
plt.close(fig)
print(f"Saved: {out2.resolve()}")


# ══════════════════════════════════════════════════════════════════════════
# Fig. 7 — Multi-day battery operation, Days 300-306 (3 rows x 7 day-columns)
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding Fig. 7 (multi-day battery operation, days 300-306)...")
MULTI_DAYS = list(range(300, 307))

fig, axes = plt.subplots(3, 7, figsize=(10, 8), sharex=True)

for j, d in enumerate(MULTI_DAYS):
    m = day_arr == d
    o = np.argsort(hour_arr[m])
    h = hour_arr[m][o] + 1
    price_d = price_arr[m][o]
    soc_d = soc_arr[m][o]
    charge_d = p_ch_arr[m][o]
    discharge_d = -p_dch_arr[m][o]

    ax0, ax1, ax2 = axes[0, j], axes[1, j], axes[2, j]

    # Row 1: Price
    ax0.plot(h, price_d, color=COLOR_NEUTRAL, linewidth=2)
    ax0.set_title(f"Day {d}", fontsize=11, fontweight='bold')

    # Row 2: SOC with boundary lines
    ax1.plot(h, soc_d, color=COLOR_NEUTRAL, linewidth=2,
             label='SOC' if j == 0 else '_nolegend_')
    ax1.axhline(SOC_MAX, color=COLOR_REF, linestyle='--', linewidth=2,
                label=f'SOC limits ({SOC_MIN}-{SOC_MAX})' if j == 0 else '_nolegend_')
    ax1.axhline(SOC_MIN, color=COLOR_REF, linestyle='--', linewidth=2, label='_nolegend_')
    ax1.set_ylim(0, 1)

    # Row 3: Charge / discharge
    ax2.bar(h, charge_d, color=COLOR_POS, label='Charge' if j == 0 else '_nolegend_')
    ax2.bar(h, discharge_d, color=COLOR_NEG, label='Discharge' if j == 0 else '_nolegend_')
    ax2.axhline(0, color='black', linewidth=0.6)
    ax2.set_xlabel("Hour", fontsize=11)

    for ax in (ax0, ax1, ax2):
        ax.tick_params(axis='both', labelsize=9)

# Y-axis labels (with units) on leftmost column only
axes[0, 0].set_ylabel("Price (£/kWh)", fontsize=11)
axes[1, 0].set_ylabel("SOC", fontsize=11)
axes[2, 0].set_ylabel("Power (kW)", fontsize=11)

add_legend(axes[1, 0])
add_legend(axes[2, 0])

fig.suptitle(f"Fig. 7. Battery operation on Days {MULTI_DAYS[0]}-{MULTI_DAYS[-1]} — Reward v2 (Projection-Aware)",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
out3 = Path('v2_multiday_STYLED.png')
fig.savefig(out3, dpi=DPI)
plt.close(fig)
print(f"Saved: {out3.resolve()}")


# ══════════════════════════════════════════════════════════════════════════
# Fig. 2 — Training reward and operating cost convergence (1000 episodes)
# ══════════════════════════════════════════════════════════════════════════
print("\nReproducing training run for Fig. 2 (1000 episodes, same config as evaluate.py)...")
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

N_EPISODES = 1000
train_rewards, train_costs = train(agent, env_train, n_episodes=N_EPISODES)
train_rewards = np.asarray(train_rewards, dtype=np.float64)
train_costs = np.asarray(train_costs, dtype=np.float64)
print(f"Training done. Final 50-ep avg reward: {train_rewards[-50:].mean():.2f}")

WINDOW = 50
reward_ma = np.convolve(train_rewards, np.ones(WINDOW) / WINDOW, mode='valid')
cost_ma = np.convolve(train_costs, np.ones(WINDOW) / WINDOW, mode='valid')
ma_x = np.arange(WINDOW - 1, N_EPISODES)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].plot(train_rewards, color=COLOR_POS, alpha=0.35, linewidth=1, label='Episode reward')
axes[0].plot(ma_x, reward_ma, color=COLOR_REF, linestyle='--', linewidth=2,
             label=f'{WINDOW}-episode moving average')
axes[0].set_ylabel("Reward")
add_legend(axes[0])

axes[1].plot(train_costs, color=COLOR_NEG, alpha=0.35, linewidth=1, label='Episode cost')
axes[1].plot(ma_x, cost_ma, color=COLOR_REF, linestyle='--', linewidth=2,
             label=f'{WINDOW}-episode moving average')
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Cost (£)")
add_legend(axes[1])

for ax in axes:
    style_axes(ax)

fig.suptitle("Fig. 2. Training reward and operating cost convergence — Reward v2 (Projection-Aware)",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out4 = Path('v2_train_convergence_STYLED.png')
fig.savefig(out4, dpi=DPI)
plt.close(fig)
print(f"Saved: {out4.resolve()}")


# ══════════════════════════════════════════════════════════════════════════
# Fig. 4 — Evaluation convergence: per-day reward/cost, days 300-364
# ══════════════════════════════════════════════════════════════════════════
print("\nBuilding Fig. 4 (evaluation reward/cost per test day)...")
mean_reward = eval_rewards.mean()
mean_cost = eval_costs.mean()

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axes[0].bar(days, eval_rewards, color=bar_colors(eval_rewards))
axes[0].axhline(mean_reward, color=COLOR_REF, linestyle='--', linewidth=2,
                label=f'Average = {mean_reward:.2f}')
axes[0].set_ylabel("Reward")
add_legend(axes[0])

axes[1].bar(days, eval_costs, color=bar_colors(eval_costs))
axes[1].axhline(mean_cost, color=COLOR_REF, linestyle='--', linewidth=2,
                label=f'Average = £{mean_cost:.2f}')
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Cost (£)")
add_legend(axes[1])

for ax in axes:
    style_axes(ax)

fig.suptitle("Fig. 4. Evaluation reward and cost per test day — Reward v2 (Projection-Aware)",
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.96])
out5 = Path('v2_eval_convergence_STYLED.png')
fig.savefig(out5, dpi=DPI)
plt.close(fig)
print(f"Saved: {out5.resolve()}")


print("\n=== All done ===")
for p in (out1, out2, out3, out4, out5):
    print(p.resolve())
