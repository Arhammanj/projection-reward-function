"""
Generate the four figures for the Projection Reward Function paper.

Data sources (existing, no training / no new evaluation runs):
  - saved_eval/profit_traces.npz   -> per-timestep P_ch, P_dch, prices, SOC, profit
  - saved_eval/eval_rewards.npy    -> per-test-day reward (used for Fig. 1)
  - saved_eval/eval_costs.npy      -> per-test-day operating cost (used for Fig. 1)

Outputs (PNG, 150 dpi):
  fig1_training_convergence.png
  fig2_battery_operation.png
  fig3_seven_day_battery_operation.png
  fig4_daily_trading_profit.png
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAVE_DIR = Path('saved_eval')
OUT_DIR  = Path('.')
DAY0, DAYN = 300, 365
REPRESENTATIVE_DAY = 310
SEVEN_DAY_START = 300
SEVEN_DAYS = list(range(SEVEN_DAY_START, SEVEN_DAY_START + 7))

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})

# ── Load data ─────────────────────────────────────────────────────────────
tr = np.load(SAVE_DIR / 'profit_traces.npz')
days   = tr['day'].astype(int)
hours  = tr['hour'].astype(int)
p_ch   = tr['p_ch']
p_dch  = tr['p_dch']
buy_p  = tr['buy_price']
soc    = tr['soc']
profit = tr['profit']

eval_rewards = np.load(SAVE_DIR / 'eval_rewards.npy')
eval_costs   = np.load(SAVE_DIR / 'eval_costs.npy')
test_days    = np.arange(DAY0, DAYN)


def day_slice(day):
    m = days == day
    return m


def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode='valid')


# ── Figure 1: Training Convergence ────────────────────────────────────────
W = 7
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(test_days, eval_rewards, color='tab:blue', alpha=0.35, lw=1.2,
        label='Reward per test day')
ax.plot(test_days[W - 1:], moving_average(eval_rewards, W), color='tab:orange',
        lw=2.5, label=f'{W}-day moving average')
ax.set_title('Reward — Unseen Test Days 300-364')
ax.set_xlabel('Test day')
ax.set_ylabel('Reward')
ax.legend()

ax = axes[1]
ax.plot(test_days, eval_costs, color='tab:red', alpha=0.35, lw=1.2,
        label='Cost per test day')
ax.plot(test_days[W - 1:], moving_average(eval_costs, W), color='tab:green',
        lw=2.5, label=f'{W}-day moving average')
ax.set_title('Operating Cost — Unseen Test Days 300-364')
ax.set_xlabel('Test day')
ax.set_ylabel('Operating cost (£)')
ax.legend()

fig.suptitle('Figure 1 — Evaluation Convergence on Unseen Days', fontsize=13, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / 'fig1_training_convergence.png', dpi=150)
plt.close(fig)
print("Saved: fig1_training_convergence.png")


# ── Helper: 3-row battery operation on one day (paper Fig. 3 layout) ──────
def draw_day_operation(axs, day):
    m = day_slice(day)
    h = hours[m]
    price = buy_p[m]
    soc_ = soc[m]
    ch = p_ch[m]
    dch = -p_dch[m]

    axs[0].plot(h, price, color='dodgerblue', lw=2, marker='o', ms=4)
    axs[0].set_ylabel('Price\n(£/kWh)')
    axs[0].tick_params(labelbottom=False)

    axs[1].plot(h, soc_, color='mediumseagreen', lw=2, marker='o', ms=4)
    axs[1].axhline(0.8, color='red', ls='--', lw=1, label='SOC max (0.8)')
    axs[1].axhline(0.2, color='red', ls='--', lw=1, label='SOC min (0.2)')
    axs[1].axhline(0.6, color='gray', ls=':', lw=1, label='Target (0.6)')
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('Battery SOC')
    axs[1].tick_params(labelbottom=False)
    axs[1].legend(loc='upper right', fontsize=7)

    axs[2].bar(h, ch, color='seagreen', label='Charge (P_ch)')
    axs[2].bar(h, dch, color='tomato', label='Discharge (P_dch)')
    axs[2].axhline(0, color='black', lw=0.8)
    axs[2].set_ylabel('Power (kW)')
    axs[2].set_xlabel('Hour')
    axs[2].set_xlim(-0.5, 23.5)
    axs[2].legend(loc='upper left', fontsize=7)


# ── Figure 2: Battery Operation — one representative day ──────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
draw_day_operation(axes, REPRESENTATIVE_DAY)
axes[0].set_title(f'Figure 2 — Battery Operation on Test Day {REPRESENTATIVE_DAY}',
                  fontsize=13, fontweight='bold')
axes[2].set_xticks(np.arange(0, 24))
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig2_battery_operation.png', dpi=150)
plt.close(fig)
print(f"Saved: fig2_battery_operation.png (day {REPRESENTATIVE_DAY})")


# ── Figure 3: Seven-Day Battery Operation ─────────────────────────────────
fig, axes = plt.subplots(3, 7, figsize=(21, 9), sharex=True, sharey='row')
for j, day in enumerate(SEVEN_DAYS):
    draw_day_operation(axes[:, j], day)
    axes[0, j].set_title(f'Day {day}', fontsize=10, fontweight='bold')
    axes[0, j].tick_params(labelbottom=False)
for j in range(7):
    axes[2, j].set_xlabel('Hour')
    axes[2, j].set_xticks(np.arange(0, 24, 4))
fig.suptitle(f'Figure 3 — Battery Operation on Test Days {SEVEN_DAY_START}-{SEVEN_DAY_START + 6}',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / 'fig3_seven_day_battery_operation.png', dpi=150)
plt.close(fig)
print(f"Saved: fig3_seven_day_battery_operation.png (days {SEVEN_DAY_START}-{SEVEN_DAY_START + 6})")


# ── Figure 4: Daily Trading Profit ────────────────────────────────────────
profit_per_day = np.array([profit[day_slice(d)].sum() for d in test_days])
avg_profit = profit_per_day.mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(test_days, profit_per_day, color='steelblue', alpha=0.8, label='Daily profit')
ax.axhline(avg_profit, color='darkorange', lw=2, ls='--',
           label=f'Average = {avg_profit:.2f} £/day')
ax.set_title(f'Figure 4 — Daily Trading Profit on Unseen Test Days {DAY0}-{DAYN - 1}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Test day')
ax.set_ylabel('Trading profit (£)')
ax.legend()
ax.set_xlim(DAY0 - 1, DAYN)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig4_daily_trading_profit.png', dpi=150)
plt.close(fig)
print(f"Saved: fig4_daily_trading_profit.png (total {profit_per_day.sum():,.2f} £, "
      f"avg {avg_profit:.2f} £/day)")
