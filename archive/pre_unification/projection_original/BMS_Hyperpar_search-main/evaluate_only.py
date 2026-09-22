"""
Evaluate an already-trained agent (no training).

Loads the agent saved in saved_eval/ and tests it on unseen days 300-364
with pure exploitation (epsilon = 0). Prints metrics and saves plots.

Usage:
    python evaluate_only.py
"""
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed
from src.config import SAVED_AGENTS_DIR

SAVE_DIR = Path('saved_eval')

# ── Load the trained agent from disk ───────────────────────────────────────
print(f"Loading trained agent from {SAVE_DIR}/...")
env_eval = MicroGridEnv(day0=300, dayn=365)
agent = QAgent.load_from_disk(env_eval, SAVE_DIR)
print("Agent loaded.")

# ── Evaluate on unseen days 300-364 ───────────────────────────────────────
print("\nEvaluating on unseen days 300-364...")
set_seed(env_eval, 0)

eval_rewards = []
eval_costs   = []
eval_gaps    = []   # normalised projection gap per episode
eval_socs    = []   # SOC trajectories for 7 sample days
eval_prices  = []   # price trajectories for 7 sample days
eval_actions = []   # charge/discharge for 7 sample days

SAMPLE_START = 310   # first day of 7-day window
SAMPLE_DAYS  = list(range(SAMPLE_START, SAMPLE_START + 7))

for day in range(300, 365):
    state = env_eval.reset(day=day)
    ep_reward = 0.0
    ep_cost   = 0.0
    ep_gap    = 0.0
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
        ep_gap    += env_eval.last_gap

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
    eval_gaps.append(ep_gap)

    if day in SAMPLE_DAYS:
        eval_socs.append(day_socs)
        eval_prices.append(day_prices)
        eval_actions.append(day_actions)

eval_rewards = np.array(eval_rewards)
eval_costs   = np.array(eval_costs)
eval_gaps    = np.array(eval_gaps)

# ── Save eval data to disk ─────────────────────────────────────────────────
np.save(SAVE_DIR / 'eval_rewards.npy', eval_rewards)
np.save(SAVE_DIR / 'eval_costs.npy',   eval_costs)
np.save(SAVE_DIR / 'eval_gaps.npy',    eval_gaps)
print(f"\nEval data saved to {SAVE_DIR}/")

# ── Report metrics ─────────────────────────────────────────────────────────
print(f"\n=== Evaluation Results (days 300-364) ===")
print(f"Mean reward  : {eval_rewards.mean():.4f}")
print(f"Std  reward  : {eval_rewards.std():.4f}")
print(f"Mean cost    : {eval_costs.mean():.4f}  (negative = profit)")
print(f"Std  cost    : {eval_costs.std():.4f}")
print(f"Best day cost: {eval_costs.min():.4f}")
print(f"Worst day cost: {eval_costs.max():.4f}")
print(f"--- Projection Gap (normalised, cumulative per episode) ---")
print(f"Mean gap     : {eval_gaps.mean():.4f}")
print(f"Std  gap     : {eval_gaps.std():.4f}")
print(f"Max gap      : {eval_gaps.max():.4f}")
print(f"Min gap      : {eval_gaps.min():.4f}")
print(f"--- Projection Gap (per decision = cumulative / 24) ---")
print(f"Mean gap     : {(eval_gaps / 24).mean():.4f}")
print(f"Std  gap     : {(eval_gaps / 24).std():.4f}")
print(f"Max gap      : {(eval_gaps / 24).max():.4f}")
print(f"Min gap      : {(eval_gaps / 24).min():.4f}")

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

# ── Plot 2: SOC + Price + Charge/Discharge for 7 sample days ──────────────
fig, axes = plt.subplots(7, 3, figsize=(18, 18), sharex=True)
hours = np.arange(1, 25)

for i, day in enumerate(SAMPLE_DAYS):
    day_socs    = eval_socs[i]
    day_prices  = eval_prices[i]
    day_actions = eval_actions[i]

    # Price
    axes[i, 0].plot(hours, day_prices, color='dodgerblue', linewidth=2, marker='o', markersize=3)
    axes[i, 0].set_ylabel('Price (£/kWh)')
    axes[i, 0].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 0].set_title('Grid Price')
    if i == 6:
        axes[i, 0].set_xlabel('Hour')

    # SOC
    axes[i, 1].plot(hours, day_socs, color='mediumseagreen', linewidth=2, marker='o', markersize=3)
    axes[i, 1].axhline(0.8, color='red',   linestyle='--', linewidth=1)
    axes[i, 1].axhline(0.2, color='red',   linestyle='--', linewidth=1)
    axes[i, 1].axhline(0.6, color='gray',  linestyle=':',  linewidth=0.8)
    axes[i, 1].set_ylabel('SOC')
    axes[i, 1].set_ylim(0, 1)
    axes[i, 1].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 1].set_title('Battery SOC')
    if i == 6:
        axes[i, 1].set_xlabel('Hour')

    # Charge / Discharge
    charge    = np.clip(day_actions, 0, None)
    discharge = np.clip(day_actions, None, 0)
    axes[i, 2].bar(hours, charge,    color='seagreen', label='Charge')
    axes[i, 2].bar(hours, discharge, color='tomato',   label='Discharge')
    axes[i, 2].axhline(0, color='black', linewidth=0.8)
    axes[i, 2].set_ylabel('Power (kW)')
    axes[i, 2].grid(True, alpha=0.3)
    if i == 0:
        axes[i, 2].set_title('Charge / Discharge')
        axes[i, 2].legend(loc='upper right', fontsize=8)
    if i == 6:
        axes[i, 2].set_xlabel('Hour')

    # Day label on left
    axes[i, 0].annotate(f'Day {day}', xy=(0, 0.5), xycoords='axes fraction',
                        xytext=(-40, 0), textcoords='offset points',
                        fontsize=10, fontweight='bold', rotation=90,
                        va='center', ha='center')

plt.suptitle(f'7-Day Charge/Discharge Profile (Days {SAMPLE_DAYS[0]}-{SAMPLE_DAYS[-1]})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('charge_discharge_7day.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: charge_discharge_7day.png")

print("\n=== All done ===")
print("Files saved:")
print("  evaluation_results.png")
print("  charge_discharge_7day.png")
