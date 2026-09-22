from typing import Tuple, List, Callable, Union, Optional
import random
from pathlib import Path
from collections import deque
from pdb import set_trace as stop

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def train(
    agent,
    env,
    n_episodes: int,
    log_dir: Optional[Path] = None,
    max_steps: Optional[int] = float("inf"),
    n_episodes_evaluate_agent: Optional[int] = 100,
    freq_episodes_evaluate_agent: int = 200,
) -> None:

    # Tensorborad log writer
    logging = False
    if log_dir is not None:
        writer = SummaryWriter(log_dir)
        logging = True

    reward_per_episode = []
    # r_unbalance_per_episode = []
    operation_cost_per_episode = []
    steps_per_episode = []
    #global_step_counter = 0

    for i in tqdm(range(0, n_episodes)):

        state = env.reset()
        rewards = 0
        # r_unbal = 0
        op_cost = 0
        #steps = 0
        done = False
        while not done:

            action = agent.act(state)

            # agents takes a step and the environment throws out a new state and
            # a reward
            next_state, reward, done, info = env.step(action)

            # r_ub, o_co = env.render()
            o_co = env.render()
            # agent observes transition and stores it for later use
            agent.observe(state, action, reward, next_state, done)
            #env.render()
            # learning happens here, through experience replay
            agent.replay()

            #global_step_counter += 1
            #steps += 1
            rewards += reward
            # r_unbal += r_ub
            op_cost += o_co

            state = next_state
        
        # log to Tensorboard
        if logging:
            writer.add_scalar('train/rewards', rewards, i)
            #writer.add_scalar('train/steps', steps, i)
            writer.add_scalar('train/epsilon', agent.epsilon, i)
            writer.add_scalar('train/replay_memory_size', len(agent.memory), i)

        reward_per_episode.append(rewards)
        # r_unbalance_per_episode.append(r_unbal)
        operation_cost_per_episode.append(op_cost)

        #steps_per_episode.append(steps)

        # if (i > 0) and (i % freq_episodes_evaluate_agent) == 0:
        # if (i + 1) % freq_episodes_evaluate_agent == 0:
        #     # evaluate agent
        #     eval_rewards, eval_steps = evaluate(agent, env,
        #                                         n_episodes=n_episodes_evaluate_agent,
        #                                         epsilon=0.01)

        #     # from src.utils import get_success_rate_from_n_steps
        #     # success_rate = get_success_rate_from_n_steps(env, eval_steps)
        #     print(f'Reward mean: {np.mean(eval_rewards):.5f}, std: {np.std(eval_rewards):.5f}')
        #     print(f'Num steps mean: {np.mean(eval_steps):.2f}, std: {np.std(eval_steps):.2f}')
        #     # print(f'Success rate: {success_rate:.2%}')
        #     if logging:
        #         writer.add_scalar('eval/avg_reward', np.mean(eval_rewards), i)
        #         writer.add_scalar('eval/avg_steps', np.mean(eval_steps), i)
            # writer.add_scalar('eval/success_rate', success_rate, i)

        # if global_step_counter > max_steps:
        #     break

    # return reward_per_episode, r_unbalance_per_episode, operation_cost_per_episode
    return reward_per_episode, operation_cost_per_episode
        


def evaluate(
    agent,
    env,
    day0, dayn,
    epsilon: Optional[float] = None,
    seed: Optional[int] = 0,
) -> Tuple[List, List]:

    from src.utils import set_seed
    set_seed(env, seed)

    # output metrics
    #reward_per_episode = []
    #r_unbalance_per_episode = []
    # steps_per_episode = []

    for day in tqdm(range(day0,dayn)):

        state = env.reset(day=day)
        rewards = 0
        # r_unbal = 0
        op_cos = 0
        #steps = 0
        done = False
        while not done:

            action = agent.act(state, epsilon=epsilon)
            next_state, reward, done, info = env.step(action)

            # r_ub, op_co = env.render(mode='Eval')
            op_co = env.render(mode='Eval')

            rewards += reward
            # r_unbal += r_ub
            op_cos += op_co
            #steps += 1
            state = next_state
        
        #print('R: ', rewards)
        #reward_per_episode.append(rewards)
        #r_unbalance_per_episode.append(r_unbal)

    # return rewards, r_unbal, op_cos
    return rewards, op_cos


def plot_reward_cost_curves(
    reward_per_episode: List[float],
    operation_cost_per_episode: List[float],
    moving_avg_window: int = 10,
    save_path: Optional[Union[str, Path]] = None,
    data_label: Optional[str] = None,
) -> None:
    """Plot per-episode reward and operating cost training curves, each with
    a moving-average overlay, and optionally save the figure to disk."""

    n_episodes = len(reward_per_episode)
    label_suffix = f' — {data_label}' if data_label else ''

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(reward_per_episode, alpha=0.3, color='tab:blue', label='Episode reward')
    if n_episodes >= moving_avg_window:
        reward_ma = np.convolve(reward_per_episode, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        axes[0].plot(np.arange(moving_avg_window - 1, n_episodes), reward_ma, color='tab:orange',
                     linewidth=2, label=f'{moving_avg_window}-ep moving average')
    axes[0].set_title(f'Training Reward{label_suffix}')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(operation_cost_per_episode, alpha=0.3, color='tab:red', label='Episode cost')
    if n_episodes >= moving_avg_window:
        cost_ma = np.convolve(operation_cost_per_episode, np.ones(moving_avg_window) / moving_avg_window, mode='valid')
        axes[1].plot(np.arange(moving_avg_window - 1, n_episodes), cost_ma, color='tab:green',
                     linewidth=2, label=f'{moving_avg_window}-ep moving average')
    axes[1].set_title(f'Training Operating Cost{label_suffix}')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Cost')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    plt.close(fig)