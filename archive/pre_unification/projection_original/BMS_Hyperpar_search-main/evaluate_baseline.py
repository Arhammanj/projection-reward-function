"""
Train + evaluate the BASELINE reward function (paper reward, no gap penalty /
price-alignment bonus), mirroring evaluate.py exactly so the two runs are
comparable: same hyperparameters, same seeds, same day splits.

Saves to saved_eval_baseline/ (does not touch saved_eval/, which holds the
projection-reward results).
"""
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from src.mg_env_projection import MicroGridEnv
from src.q_agent import QAgent
from src.utils import set_seed

SAVE_DIR = Path('saved_eval_baseline')

# ── Train the agent first (days 0-299) ────────────────────────────────────
from src.loops import train

print("Training agent on days 0-299 (BASELINE reward)...")
env_train = MicroGridEnv(day0=0, dayn=300, reward_mode='baseline')
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

# ── Evaluate on unseen days 300-364 ───────────────────────────────────────
print("\nEvaluating on unseen days 300-364 (BASELINE reward)...")

env_eval = MicroGridEnv(day0=300, dayn=365, reward_mode='baseline')
set_seed(env_eval, 0)

eval_rewards = []
eval_costs   = []
eval_gaps    = []

for day in range(300, 365):
    state = env_eval.reset(day=day)
    ep_reward = 0.0
    ep_cost   = 0.0
    ep_gap    = 0.0

    done = False
    while not done:
        action = agent.act(state, epsilon=0.0)   # pure exploitation
        next_state, reward, done, _ = env_eval.step(action)
        op_cost = env_eval.render()

        ep_reward += reward
        ep_cost   += op_cost
        ep_gap    += env_eval.last_gap

        state = next_state

    eval_rewards.append(ep_reward)
    eval_costs.append(ep_cost)
    eval_gaps.append(ep_gap)

eval_rewards = np.array(eval_rewards)
eval_costs   = np.array(eval_costs)
eval_gaps    = np.array(eval_gaps)

# ── Save agent + eval data + training curves to disk ──────────────────────
if not SAVE_DIR.exists():
    SAVE_DIR.mkdir(parents=True)
agent.save_to_disk(SAVE_DIR)
np.save(SAVE_DIR / 'eval_rewards.npy', eval_rewards)
np.save(SAVE_DIR / 'eval_costs.npy',   eval_costs)
np.save(SAVE_DIR / 'eval_gaps.npy',    eval_gaps)
np.save(SAVE_DIR / 'train_rewards.npy', np.array(train_rewards))
np.save(SAVE_DIR / 'train_costs.npy',   np.array(train_costs))
print(f"\nAgent and eval/train data saved to {SAVE_DIR}/")

print(f"\n=== Evaluation Results (days 300-364, BASELINE reward) ===")
print(f"Mean reward  : {eval_rewards.mean():.4f}")
print(f"Std  reward  : {eval_rewards.std():.4f}")
print(f"Mean cost    : {eval_costs.mean():.4f}  (negative = profit)")
print(f"Std  cost    : {eval_costs.std():.4f}")
print(f"Best day cost: {eval_costs.min():.4f}")
print(f"Worst day cost: {eval_costs.max():.4f}")

# ── Per-timestep profit traces (same equation/format as profit_analysis.py) ─
print("\nCapturing per-timestep profit traces (pure exploitation)...")
env_profit = MicroGridEnv(day0=300, dayn=365, reward_mode='baseline')
set_seed(env_profit, 0)

records = []
for day in range(300, 365):
    state = env_profit.reset(day=day)
    done = False
    t = 0
    while not done:
        action = agent.act(state, epsilon=0.0)
        next_state, reward, done, _ = env_profit.step(action)

        signed = env_profit.battery.energy_change
        p_ch   = max(signed, 0.0)
        p_dch  = max(-signed, 0.0)
        price  = env_profit.grid.sell_prices[env_profit.grid.time % len(env_profit.grid.sell_prices)]

        profit = price * p_dch - price * p_ch
        soc    = env_profit.battery.SOC

        records.append((day, t, p_ch, p_dch, price, price, profit, soc))
        state = next_state
        t += 1

records = np.array(records)
np.savez(SAVE_DIR / 'profit_traces.npz',
         day=records[:, 0].astype(int),
         hour=records[:, 1].astype(int),
         p_ch=records[:, 2],
         p_dch=records[:, 3],
         buy_price=records[:, 4],
         sell_price=records[:, 5],
         profit=records[:, 6],
         soc=records[:, 7],
         reward_function='baseline',
         days=(300, 365))

total_profit = records[:, 6].sum()
avg_profit   = total_profit / 65
print(f"Total profit (BASELINE): {total_profit:,.2f} (avg {avg_profit:,.2f}/day)")
print(f"Saved: {SAVE_DIR / 'profit_traces.npz'}")

print("\n=== All done ===")
