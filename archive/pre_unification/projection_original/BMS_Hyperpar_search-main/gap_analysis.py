"""
Standalone script to report projection gap statistics.
Run after evaluate.py has saved data to saved_eval/.
"""
import numpy as np
from pathlib import Path

SAVE_DIR = Path('saved_eval')

eval_rewards = np.load(SAVE_DIR / 'eval_rewards.npy')
eval_costs   = np.load(SAVE_DIR / 'eval_costs.npy')
eval_gaps    = np.load(SAVE_DIR / 'eval_gaps.npy')

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
