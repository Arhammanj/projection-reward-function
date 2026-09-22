"""
Training convergence comparison: Projection reward vs Baseline reward.

Reproduces the "Fig. 2 style" panel from the base paper, but with both
reward-function training curves overlaid so convergence behaviour can be
compared directly, not just final numbers.

Data sources (no retraining):
  saved_eval_baseline/train_rewards.npy, train_costs.npy   (baseline run)
  saved_eval/eval_rewards.npy, eval_costs.npy              (projection run,
      test-day granularity only -- see note below)

NOTE: evaluate.py (projection) did not save per-episode TRAINING curves to
disk, only per-test-day eval arrays. If saved_eval/train_rewards.npy /
train_costs.npy are not found, this script falls back to plotting the
projection curve from training_curve_projection.png's underlying run is not
recoverable after the fact -- so the projection panel uses eval_rewards /
eval_costs (per test day) for a fair, apples-to-apples x-axis with the
baseline eval curve, in addition to the baseline's own training curve.
"""
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJ_DIR = Path('saved_eval')
BASE_DIR = Path('saved_eval_baseline')

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
                     'axes.spines.top': False, 'axes.spines.right': False})


def moving_average(x, window):
    return np.convolve(x, np.ones(window) / window, mode='valid')


# ── Panel A: training-episode convergence (baseline has it; projection doesn't) ─
base_train_r = np.load(BASE_DIR / 'train_rewards.npy')
base_train_c = np.load(BASE_DIR / 'train_costs.npy')

# ── Panel B: test-day evaluation convergence (both runs have this, same days) ──
proj_eval_r = np.load(PROJ_DIR / 'eval_rewards.npy')
proj_eval_c = np.load(PROJ_DIR / 'eval_costs.npy')
base_eval_r = np.load(BASE_DIR / 'eval_rewards.npy')
base_eval_c = np.load(BASE_DIR / 'eval_costs.npy')

test_days = np.arange(300, 300 + len(proj_eval_r))
W = 7

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# -- Top-left: training reward (baseline only has this saved) --
ax = axes[0, 0]
W2 = 50
n = len(base_train_r)
ax.plot(base_train_r, alpha=0.25, color='tab:red', lw=1)
ax.plot(np.arange(W2 - 1, n), moving_average(base_train_r, W2), color='tab:red',
        lw=2.5, label='Baseline (50-ep MA)')
ax.set_title('Training Reward per Episode (Baseline)')
ax.set_xlabel('Training episode')
ax.set_ylabel('Reward')
ax.legend()

# -- Top-right: training cost (baseline only) --
ax = axes[0, 1]
ax.plot(base_train_c, alpha=0.25, color='tab:orange', lw=1)
ax.plot(np.arange(W2 - 1, n), moving_average(base_train_c, W2), color='tab:orange',
        lw=2.5, label='Baseline (50-ep MA)')
ax.set_title('Training Operating Cost per Episode (Baseline)')
ax.set_xlabel('Training episode')
ax.set_ylabel('Cost')
ax.legend()

# -- Bottom-left: reward on unseen test days, both reward functions --
ax = axes[1, 0]
ax.plot(test_days, proj_eval_r, alpha=0.25, color='tab:blue', lw=1)
ax.plot(test_days[W - 1:], moving_average(proj_eval_r, W), color='tab:blue',
        lw=2.5, label='Projection')
ax.plot(test_days, base_eval_r, alpha=0.25, color='tab:red', lw=1)
ax.plot(test_days[W - 1:], moving_average(base_eval_r, W), color='tab:red',
        lw=2.5, label='Baseline')
ax.set_title('Reward — Unseen Test Days 300-364')
ax.set_xlabel('Test day')
ax.set_ylabel('Reward')
ax.legend()

# -- Bottom-right: operating cost on unseen test days, both reward functions --
ax = axes[1, 1]
ax.plot(test_days, proj_eval_c, alpha=0.25, color='tab:blue', lw=1)
ax.plot(test_days[W - 1:], moving_average(proj_eval_c, W), color='tab:blue',
        lw=2.5, label='Projection')
ax.plot(test_days, base_eval_c, alpha=0.25, color='tab:red', lw=1)
ax.plot(test_days[W - 1:], moving_average(base_eval_c, W), color='tab:red',
        lw=2.5, label='Baseline')
ax.set_title('Operating Cost — Unseen Test Days 300-364')
ax.set_xlabel('Test day')
ax.set_ylabel('Operating cost (EUR)')
ax.legend()

fig.suptitle('Figure 2 — Training Convergence: Projection vs Baseline Reward',
             fontsize=14, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig('fig2_training_convergence_comparison.png', dpi=150)
print("Saved: fig2_training_convergence_comparison.png")

print()
print(f"Projection  -- mean eval reward: {proj_eval_r.mean():.2f}, mean eval cost: {proj_eval_c.mean():.2f}")
print(f"Baseline    -- mean eval reward: {base_eval_r.mean():.2f}, mean eval cost: {base_eval_c.mean():.2f}")
