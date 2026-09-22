"""
Training convergence plot for Reward v1 (Stability-Enhanced) — the
mg_env.py MicroGridEnv config (reward_mode="projection", the default) that
produced the €6,605.01 / 65-day evaluation result in profit_results.txt.

Mirrors evaluate.py's training setup exactly (same env days, seed, and
QAgent hyperparameters) but trains for 2000 episodes and only produces the
training convergence figure — it does not re-run the day 300-364 evaluation
or overwrite saved_eval/.
"""
from pathlib import Path

from src.mg_env import MicroGridEnv
from src.q_agent import QAgent
from src.loops import train, plot_reward_cost_curves
from src.utils import set_seed

N_EPISODES = 2000
SAVE_PATH = Path(__file__).parent / 'v1_convergence.png'

print(f"Training agent on days 0-299 (reward_mode='projection', {N_EPISODES} episodes)...")
env = MicroGridEnv(day0=0, dayn=300)
set_seed(env, 42)

agent = QAgent(
    env,
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

reward_per_episode, operation_cost_per_episode = train(agent, env, n_episodes=N_EPISODES)

print(f"Training done. Final 50-ep avg reward: "
      f"{sum(reward_per_episode[-50:]) / 50:.2f}")

plot_reward_cost_curves(
    reward_per_episode,
    operation_cost_per_episode,
    moving_avg_window=10,
    save_path=str(SAVE_PATH),
    data_label="Reward v1 (Stability-Enhanced)",
)

print(f"Saved: {SAVE_PATH.resolve()}")
