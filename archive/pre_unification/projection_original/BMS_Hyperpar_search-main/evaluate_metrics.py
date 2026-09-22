"""
Detailed operational metrics for the trained agent (no training).

Loads the agent saved in saved_eval/ and evaluates it on unseen days 300-364
with pure exploitation. For every day it reports:
  reward, operating cost, profit, energy bought/sold (kWh),
  peak import/export (kW), over-request (SOC-clamp) count, end-of-day SOC.
"""
import numpy as np
from pathlib import Path
from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed

SAVE_DIR = Path('saved_eval')
DAY0, DAYN = 300, 365

# ── Load the trained agent ─────────────────────────────────────────────────
print(f"Loading trained agent from {SAVE_DIR}/...")
env = MicroGridEnv(day0=DAY0, dayn=DAYN)
agent = QAgent.load_from_disk(env, SAVE_DIR)
set_seed(env, 0)

soc_min, soc_max = env.battery.min_soc, env.battery.max_soc

# ── Collect per-day metrics ────────────────────────────────────────────────
rows = []
for day in range(DAY0, DAYN):
    state = env.reset(day=day)
    ep_reward = 0.0
    ep_op_cost = 0.0          # includes 0.01 * degradation cost
    buy_cost_sum = 0.0        # £ paid to grid
    sell_rev_sum  = 0.0       # £ earned from grid
    energy_bought = 0.0       # kWh imported (charging)
    energy_sold   = 0.0       # kWh exported (discharging)
    energy_sq_sum = 0.0       # sum of energy^2 (for degradation cost)
    peak_import = 0.0         # kW max charging power
    peak_export = 0.0         # kW max discharging power
    n_over_request = 0        # times intended action was clamped by SOC limits
    n_soc_violation = 0       # times SOC fell outside [0.2, 0.8]
    end_soc = None

    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env.step(action)

        signed = env.battery.energy_change
        price  = env.grid.sell_prices[env.grid.time % len(env.grid.sell_prices)]
        buy  = max(signed, 0.0)
        sell = max(-signed, 0.0)

        ep_reward    += reward
        ep_op_cost   += env.operation_cost
        buy_cost_sum += price * buy
        sell_rev_sum += price * sell
        energy_bought += buy
        energy_sold   += sell
        energy_sq_sum += signed ** 2
        peak_import = max(peak_import, buy)
        peak_export = max(peak_export, sell)

        if env.last_gap > 1e-6:
            n_over_request += 1
        if env.battery.SOC < soc_min - 1e-6 or env.battery.SOC > soc_max + 1e-6:
            n_soc_violation += 1

        end_soc = env.battery.SOC
        state = next_state

    rows.append({
        'day': day,
        'reward': ep_reward,
        'op_cost': ep_op_cost,
        'energy_bought': energy_bought,
        'energy_sold': energy_sold,
        'peak_import': peak_import,
        'peak_export': peak_export,
        'n_over_request': n_over_request,
        'n_soc_violation': n_soc_violation,
        'end_soc': end_soc,
    })

r = np.array([[x['reward'], x['op_cost'], x['energy_bought'], x['energy_sold'],
               x['peak_import'], x['peak_export'], x['n_over_request'],
               x['n_soc_violation'], x['end_soc']] for x in rows])

# ── Report ─────────────────────────────────────────────────────────────────
print("\n=== Evaluation Results (unseen days 300-364, pure exploitation) ===\n")
hdr = f"{'Day':>4} {'Reward':>10} {'OpCost £':>10} {'Buy kWh':>8} {'Sell kWh':>8} {'PkImp kW':>8} {'PkExp kW':>8} {'Clamps':>6} {'SOCend':>6}"
print(hdr)
print('-' * len(hdr))
for i, x in enumerate(rows):
    print(f"{x['day']:>4} {x['reward']:>10.2f} {x['op_cost']:>10.2f} "
          f"{x['energy_bought']:>8.2f} {x['energy_sold']:>8.2f} "
          f"{x['peak_import']:>8.1f} {x['peak_export']:>8.1f} "
          f"{x['n_over_request']:>6} {x['end_soc']:>6.3f}")

n_days = len(rows)
print('-' * len(hdr))
def summ(a, name, fmt='.4f'):
    print(f"{name:<28} mean={np.mean(a):{fmt}}  std={np.std(a):{fmt}}  "
          f"min={np.min(a):{fmt}}  max={np.max(a):{fmt}}")

print("\n--- Averages over all 65 test days ---")
summ(r[:, 0], 'Reward')
summ(r[:, 1], 'Operating cost (£)')
profit = -r[:, 1]
summ(profit, 'Net profit (£)  [= -op cost]')
print()
print("--- Totals over 65 test days ---")
print(f"Total reward         : {r[:, 0].sum():.2f}")
print(f"Total operating cost : {r[:, 1].sum():.2f} £")
print(f"Total net profit     : {-r[:, 1].sum():.2f} £")
print(f"Total energy bought  : {r[:, 2].sum():.2f} kWh")
print(f"Total energy sold    : {r[:, 3].sum():.2f} kWh")
print(f"Avg daily energy     : bought {r[:, 2].mean():.2f} kWh, sold {r[:, 3].mean():.2f} kWh")
print(f"Peak load  (import)  : {r[:, 4].max():.1f} kW  (grid exchange limit = {env.grid.exchange_ability} kW)")
print(f"Peak load  (export)  : {r[:, 5].max():.1f} kW  (grid exchange limit = {env.grid.exchange_ability} kW)")
print(f"Over-requests/clamps : {r[:, 6].sum()} of {n_days * 24} decisions "
      f"({100.0 * r[:, 6].sum() / (n_days * 24):.2f}%)")
print(f"SOC bound violations : {r[:, 7].sum()}")
print()
print("--- End-of-day battery SOC ---")
summ(r[:, 8], 'End-of-day SOC')
target = 0.6
pct_close = 100.0 * np.mean(np.abs(r[:, 8] - target) <= 0.1)
print(f"% days ending within 0.1 of target SOC {target}: {pct_close:.1f}%")
