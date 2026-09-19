from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mg_env import MicroGridEnv, DEFAULT_DAY0_TRAIN, DEFAULT_DAYN_TRAIN
from src.q_agent import QAgent
from src.loops import train, plot_reward_cost_curves
from src.utils import set_seed

# Set up environment and agent
env = MicroGridEnv(
	day0=DEFAULT_DAY0_TRAIN,
	dayn=DEFAULT_DAYN_TRAIN,
	reward_mode="pure_profit",
)

# v1 hyperparameter set:
# - nn_hidden_layers=[256, 256]
# - memory_size=10_000, n_steps_warm_up_memory=1_000
# - freq_steps_update_target=10 (target net synced every 10 steps)
# - freq_steps_train=16, n_gradient_steps=16 (16 Adam updates every 16 env steps)
# - epsilon: 0.90 -> 0.15 over 10,000 steps
#
# normalize_state=True: raw features (price window [-0.91, 1], time-step [0,1],
# SOC/max_soc [0,1.25]) have mismatched scales. Normalizing puts them on a
# comparable footing so the net can use the price trajectory, not just time-of-day.
#
# reward_mode="pure_profit" + the Battery safety shield: reward is profit only,
# violations are impossible by construction (SOC clipped to the safe band).
#
# optimizer is fixed to Adam inside QAgent (src/q_agent.py), not configurable here.
agent = QAgent(
    env,
    nn_hidden_layers=[256, 256],
    memory_size=10_000,
    n_steps_warm_up_memory=1_000,
    freq_steps_update_target=10,
    freq_steps_train=16,
    n_gradient_steps=16,
    epsilon_start=0.90,
    epsilon_end=0.15,
    steps_epsilon_decay=10_000,
    normalize_state=True,
)

# Greedy evaluation callback (epsilon forced to 0). Re-seeding each checkpoint
# means every eval evaluates the same set of days, so points are comparable.
def greedy_eval(agent, env, n_episodes=10, seed=0):
    set_seed(env, seed)
    ep_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, _ = env.step(action)
            ep_r += float(reward)
            state = next_state
        ep_rewards.append(ep_r)
    return float(np.mean(ep_rewards))

# Train
n_episodes = 5000
eval_freq = 100
ckpt_dir = Path('results') / 'best_checkpoint'
print(f"Training for {n_episodes} episodes (epsilon 0.90 -> 0.15 over 10k steps, greedy eval every {eval_freq} eps)...")
print(f"normalize_state = {agent.normalize_state} | hidden_layers = [256, 256] | best-eval checkpoint dir = {ckpt_dir}")
rewards, costs = train(agent, env, n_episodes=n_episodes, log_dir=None,
                       eval_fn=greedy_eval, freq_episodes_eval=eval_freq,
                       checkpoint_dir=ckpt_dir)

# Save plot
out_path = Path('results') / f'reward_cost_{n_episodes}eps.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
fig, axs = plot_reward_cost_curves(rewards, costs, moving_avg_window=50, save_path=str(out_path),
                                   data_label="Prices (3).csv")

# Overlay greedy eval line on the reward panel
if len(agent.eval_episodes) > 0:
    axs[0].plot(agent.eval_episodes, agent.eval_rewards, marker='o', markersize=5,
                color='black', linewidth=2, label='Greedy eval (eps=0)')
    axs[0].legend(loc='upper left', frameon=False)

fig.tight_layout()
fig.savefig(str(out_path), dpi=220)
print(f"Plot saved to {out_path}")
print(f"Greedy eval checkpoints: {agent.eval_episodes}")
print(f"Greedy eval rewards    : {[round(r, 4) for r in agent.eval_rewards]}")
print(f"Best greedy eval       : episode {agent.best_eval_episode}, reward={agent.best_eval_reward:.4f} -> saved in {ckpt_dir}")

# Log per-episode profit / safety metrics (populated by the profit_safety reward mode)
if len(agent.profit_per_episode) > 0:
    profits = np.asarray(agent.profit_per_episode)
    viols = np.asarray(agent.safety_violation_per_episode)
    print(f"Profit per episode   : total={profits.sum():.2f} mean={profits.mean():.3f}")
    print(f"Safety violation peak: mean={viols.mean():.4f} max={viols.max():.4f}")

    # Save metrics to CSV for later reporting (thesis plots etc.)
    csv_path = Path('results') / f'profit_safety_{n_episodes}eps.csv'
    with open(csv_path, 'w') as f:
        f.write("episode,reward,operation_cost,profit,safety_violation\n")
        for i, (r, c, p, v) in enumerate(zip(rewards, costs, agent.profit_per_episode, agent.safety_violation_per_episode)):
            f.write(f"{i + 1},{r:.6f},{c:.6f},{p:.6f},{v:.6f}\n")
    print(f"Metrics CSV saved to {csv_path}")
