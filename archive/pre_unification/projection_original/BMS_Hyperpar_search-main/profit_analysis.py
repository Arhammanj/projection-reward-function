"""
Energy trading profit analysis using the exact equation:

    Profit = (Sell Price x P_dch) - (Buy Price x P_ch)

Run ONCE over test days 300-364 with the saved agent (pure exploitation),
dump per-timestep charge/discharge power + prices to disk, and report
total / average trading profit. Environment reward values are ignored.

Usage:
    python profit_analysis.py
"""
import numpy as np
from pathlib import Path
from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed

SAVE_DIR = Path('saved_eval')
DAY0, DAYN = 300, 365

# ── Load the trained agent (no training) ──────────────────────────────────
print(f"Loading trained agent from {SAVE_DIR}/...")
env = MicroGridEnv(day0=DAY0, dayn=DAYN)
agent = QAgent.load_from_disk(env, SAVE_DIR)
set_seed(env, 0)

# ── Single evaluation run, capturing per-timestep traces ──────────────────
#   P_ch  (kW)  : charging power (energy bought from grid)
#   P_dch (kW)  : discharging power (energy sold to grid)
#   buy_price, sell_price (£/kWh) : ToU prices used in the env
#   profit (£)  : Sell_Price*P_dch - Buy_Price*P_ch   (per timestep)
#   soc (-)     : battery state of charge after the action
#   hour (h)    : hour of day 0-23

n_steps_per_day = env.iterations
records = []          # (day, hour, p_ch, p_dch, buy_price, sell_price, profit, soc)
profit_per_day = np.zeros(DAYN - DAY0)

for day in range(DAY0, DAYN):
    state = env.reset(day=day)
    done = False
    t = 0
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env.step(action)

        signed  = env.battery.energy_change          # + = charge (buy), - = discharge (sell)
        p_ch    = max(signed, 0.0)
        p_dch   = max(-signed, 0.0)
        price   = env.grid.sell_prices[env.grid.time % len(env.grid.sell_prices)]

        buy_price  = price
        sell_price = price
        profit = sell_price * p_dch - buy_price * p_ch
        soc    = env.battery.SOC

        records.append((day, t, p_ch, p_dch, buy_price, sell_price, profit, soc))
        profit_per_day[day - DAY0] += profit

        state = next_state
        t += 1

records = np.array(records)
print(f"Captured {len(records)} timesteps ({DAYN - DAY0} days x {n_steps_per_day} steps).")

# ── Save per-timestep traces for future comparison of reward functions ────
np.savez(SAVE_DIR / 'profit_traces.npz',
         day=records[:, 0].astype(int),
         hour=records[:, 1].astype(int),
         p_ch=records[:, 2],
         p_dch=records[:, 3],
         buy_price=records[:, 4],
         sell_price=records[:, 5],
         profit=records[:, 6],
         soc=records[:, 7],
         reward_function='projection',
         days=(DAY0, DAYN))
print(f"Per-timestep traces saved to {SAVE_DIR / 'profit_traces.npz'}")

# ── Report ─────────────────────────────────────────────────────────────────
total_profit = records[:, 6].sum()
avg_profit   = total_profit / (DAYN - DAY0)
n_days       = DAYN - DAY0

print()
print("=== Trading Profit (ignoring environment reward) ===")
print(f"Reward function : projection")
print(f"Test period     : days {DAY0}-{DAYN - 1} ({n_days} days, {len(records)} timesteps)")
print(f"Total profit    : {total_profit:,.2f} £")
print(f"Average profit  : {avg_profit:,.2f} £/day")
print(f"Profit std      : {profit_per_day.std():,.2f} £/day")
print(f"Best day        : {profit_per_day.max():,.2f} £")
print(f"Worst day       : {profit_per_day.min():,.2f} £")
print(f"Total energy sold  : {records[:, 3].sum():,.2f} kWh")
print(f"Total energy bought: {records[:, 2].sum():,.2f} kWh")

# ── Comparison table template (extend with other reward functions) ────────
print()
print("=== Comparison table: total / average profit per reward function ===")
print(f"{'Reward function':<20} {'Total profit (£)':>18} {'Avg profit (£/day)':>18}")
print('-' * 58)
print(f"{'projection':<20} {total_profit:>18,.2f} {avg_profit:>18,.2f}")
print()
print("Add rows by dumping profit_traces.npz for other agents "
      "(same days/seed) and re-running this table.")
