import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".2"

import time
import gymnasium as gym
import ml_collections
import numpy as np
import pandas as pd

from tqdm import trange
from models import SACAgent
from utils import ReplayBuffer, log_git, get_logger, make_env


###################
# Utils Functions #
###################
def eval_policy(agent: SACAgent,
                env: gym.Env,
                eval_episodes: int = 10):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        while True:
            avg_step += 1
            action = agent.sample_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, info = env.step(action)
            eval_reward += reward
            if terminated or truncated:
                eval_success += info["success"] 
                break

    eval_reward /= eval_episodes
    eval_success /= eval_episodes

    return eval_reward, eval_success, avg_step, time.time() - t1


def setup_logging(config):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 

    # logging
    exp_prefix = "sac"
    exp_name = f"{exp_prefix}/{config.env_name}/s{config.seed}_{timestamp}"
    os.makedirs(f"logs/{exp_prefix}/{config.env_name}", exist_ok=True)
    exp_info = f"# Running experiment for: {exp_name} #"
    print("#" * len(exp_info) + f"\n{exp_info}\n" + "#" * len(exp_info))
    logger = get_logger(f"logs/{exp_name}.log")

    # add git commit info
    log_git(config)
    logger.info(f"Config:\n{config}\n")

    # set random seed
    np.random.seed(config.seed)

    return exp_name, logger


def setup_exp(config):
    # initialize the environment
    env = make_env(config.env_name,
                   image_size=480,
                   seed=config.seed)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=480,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # SAC agent
    agent = SACAgent(obs_dim=obs_dim,
                     act_dim=act_dim,
                     max_action=max_action,
                     seed=config.seed,
                     tau=config.tau,
                     gamma=config.gamma,
                     lr=config.lr,
                     hidden_dims=config.hidden_dims)

    # Replay buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim)

    return env, eval_env, agent, replay_buffer


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging
    exp_name, logger = setup_logging(config)

    # experiment setup
    (env,
     eval_env,
     agent,
     replay_buffer) = setup_exp(config)

    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, _, _ = eval_policy(agent,    
                                                  eval_env,
                                                  eval_episodes=eval_episodes)
    logs = [{
        "step": 0,
        "eval_reward": eval_reward,
        "eval_success": eval_success
    }]

    # start training
    obs, _ = env.reset()
    success, cum_success, ep_step = 0, 0, 0
    ep_task_reward, ep_reward = 0, 0
    lst_ep_task_reward, lst_ep_reward = 0, 0
    for t in trange(1, config.max_timesteps + 1):
        ep_step += 1
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.sample_action(obs)
        next_obs, task_reward, terminated, truncated, info = env.step(action)
        cum_success += info["success"]

        replay_buffer.add(obs,
                          action,
                          next_obs,
                          info["success"]-1,
                          terminated)
        obs = next_obs
        ep_reward += info["success"]
        ep_task_reward += task_reward

        # start a new trajectory
        if terminated or truncated:
            obs, _ = env.reset()
            success = info["success"] 
            lst_ep_task_reward = ep_task_reward
            lst_ep_reward = ep_reward
            ep_task_reward = 0
            ep_reward = 0
            ep_step = 0

        # training
        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info = agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, _, _ = eval_policy(agent,
                                                          eval_env,
                                                          eval_episodes=eval_episodes)

        # logging
        if t % config.log_freq == 0:
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "success": success,
                    "reward": lst_ep_reward,
                    "task_reward": lst_ep_task_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "batch_reward": batch.rewards.mean(),
                    "batch_reward_max": batch.rewards.max(),
                    "batch_reward_min": batch.rewards.min(),
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[T {t//1000}K][{log_info['time']:.2f} min] "
                    f"task_R: {lst_ep_task_reward:.2f}, "
                    f"ep_R: {lst_ep_reward:.0f}\n"
                    f"\tq_loss: {log_info['critic_loss']:.3f}, "
                    f"a_loss: {log_info['alpha_loss']:.3f}, "
                    f"q: {log_info['q']:.2f}, "
                    f"q_max: {log_info['q_max']:.2f}\n"
                    f"\tR: {log_info['batch_reward']:.3f}, "
                    f"Rmax: {log_info['batch_reward_max']:.1f}, "
                    f"Rmin: {log_info['batch_reward_min']:.1f}, "
                    f"success: {success:.0f}, "
                    f"cum_success: {cum_success:.0f}\n")
                logs.append(log_info)
            else:
                logs.append({
                    "step": t,
                    "reward": lst_ep_reward,
                    "task_reward": lst_ep_task_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "time": (time.time() - start_time) / 60,
                })
                logger.info(
                    f"\n[T {t//1000}K][{logs[-1]['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"ep_reward: {lst_ep_reward:.2f}\n"
                )

    # save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{exp_name}.csv")

    # close env
    env.close()
    eval_env.close()
