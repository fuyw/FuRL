import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".4"

import cv2
import time
import clip
import git
import logging
import imageio
import gymnasium as gym
import ml_collections
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as T

from tqdm import trange
from models import VLMAgent
from utils import (TASKS, VLMBuffer, log_git, get_logger, make_env, load_liv)


###################
# Utils Functions #
###################
def crop_center(config, image):
    x1, x2, y1, y2 = 32, 224, 32, 224
    return image[x1:x2, y1:y2, :]


def eval_policy(agent: VLMAgent,
                env: gym.Env,              
                eval_episodes: int = 10,
                hidden_env: bool = True):
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
    exp_prefix = f"liv_rho{config.rho}"
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
    torch.manual_seed(config.seed)

    return exp_name, logger


def setup_exp(config):
    # liv
    transform = T.Compose([T.ToTensor()])
    liv = load_liv()
    liv.eval()

    # task description embedding
    with torch.no_grad():
        token = clip.tokenize([TASKS[config.env_name]])
        text_embedding = liv(input=token, modality="text")
    text_embedding = text_embedding.detach().cpu().numpy()

    # initialize the environment
    env = make_env(config.env_name, seed=config.seed)
    eval_seed = config.seed if "hidden" in config.env_name else config.seed+100
    eval_env = make_env(config.env_name,
                        seed=eval_seed,
                        image_size=480,
                        camera_id=config.camera_id)

    # environment parameter
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # fixed LIV representation projection
    vlm_agent = VLMAgent(obs_dim=obs_dim,
                         act_dim=act_dim,
                         max_action=max_action,
                         seed=config.seed,
                         tau=config.tau,
                         rho=config.rho,
                         gamma=config.gamma,
                         lr=config.lr,
                         text_embedding=text_embedding,
                         hidden_dims=config.hidden_dims)

    # Replay buffer
    replay_buffer = VLMBuffer(obs_dim=obs_dim, act_dim=act_dim)

    return transform, liv, env, eval_env, vlm_agent, replay_buffer


#################
# Main Function #
#################
def train_and_evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # logging setup
    exp_name, logger = setup_logging(config)

    # experiment setup
    (transform,
     liv,
     env,
     eval_env,
     vlm_agent,
     replay_buffer) = setup_exp(config)

    # reward for untrained agent
    eval_episodes = 1 if "hidden" in config.env_name else 10
    eval_reward, eval_success, _, _ = eval_policy(vlm_agent,
                                                  eval_env,
                                                  eval_episodes)
    logs = [{
        "step": 0,
        "eval_reward": eval_reward,
        "eval_success": eval_success
    }]

    # start training
    obs, _ = env.reset()
    success_cnt = 0
    ep_num, ep_step, success = 0, 0, 0
    ep_task_reward, ep_vlm_reward = 0, 0
    lst_ep_step, lst_ep_task_reward, lst_ep_vlm_reward = 0, 0, 0
    for t in trange(1, config.max_timesteps + 1):
        if t <= config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = vlm_agent.sample_action(obs)
        next_obs, task_reward, terminated, truncated, info = env.step(action)

        # vision language model reward
        image = env.mujoco_renderer.render(
            render_mode="rgb_array",
            camera_id=config.camera_id).copy()
        image = image[::-1]
        image = crop_center(config, image)
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        processed_image = transform(processed_image)
        with torch.no_grad():
            image_embedding = liv(input=processed_image.to("cuda")[None], modality="vision")
        image_embedding = image_embedding.detach().cpu().numpy()
        vlm_reward = vlm_agent.get_reward(image_embedding).item()  # (1, 1024)
        reward = info["success"]
        success_cnt += info["success"]

        # add to buffer
        replay_buffer.add(obs,
                          action,
                          next_obs,
                          vlm_reward,
                          reward-1,
                          terminated)
        obs = next_obs
        ep_vlm_reward += vlm_reward
        ep_task_reward += task_reward
        ep_step += 1

        # start a new trajectory
        if terminated or truncated:
            obs, _ = env.reset()
            success = info["success"] 
            lst_ep_step = ep_step
            lst_ep_task_reward = ep_task_reward
            lst_ep_vlm_reward = ep_vlm_reward
            ep_vlm_reward = 0
            ep_task_reward = 0
            ep_step = 0
            ep_num += 1

        # training
        if t > config.start_timesteps:
            batch = replay_buffer.sample(config.batch_size)
            log_info = vlm_agent.update(batch)

        # eval
        if t % config.eval_freq == 0:
            eval_reward, eval_success, _, _ = eval_policy(
                agent, eval_env, eval_episodes=eval_episodes)

        # logging
        if t % config.log_freq == 0:
            if t > config.start_timesteps:
                log_info.update({
                    "step": t,
                    "success": success,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "batch_reward": batch.rewards.mean(),
                    "batch_reward_max": batch.rewards.max(),
                    "batch_reward_min": batch.rewards.min(),
                    "time": (time.time() - start_time) / 60
                })
                logger.info(
                    f"\n[T {t//1000}K][{log_info['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"vlm_reward: {lst_ep_vlm_reward:.2f}\n"
                    f"\tq_loss: {log_info['critic_loss']:.3f}, "
                    f"a_loss: {log_info['alpha_loss']:.3f}, "
                    f"q: {log_info['q']:.3f}, q_max: {log_info['q_max']:.3f}\n"
                    f"\tR: {log_info['batch_reward']:.3f}, "
                    f"Rmax: {log_info['batch_reward_max']:.3f}, "
                    f"Rmin: {log_info['batch_reward_min']:.3f}\n"
                    f"\tep_num: {ep_num}, success_cnt: {success_cnt}, "
                    f"success: {success}\n"
                )
                logs.append(log_info)
            else:
                logs.append({
                    "step": t,
                    "task_reward": lst_ep_task_reward,
                    "vlm_reward": lst_ep_vlm_reward,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "time": (time.time() - start_time) / 60,
                })
                logger.info(
                    f"\n[T {t//1000}K][{logs[-1]['time']:.2f} min] "
                    f"task_reward: {lst_ep_task_reward:.2f}, "
                    f"vlm_reward: {lst_ep_vlm_reward:.2f}\n"
                )

    # save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"logs/{exp_name}.csv")

    # close env
    env.close()
    eval_env.close()
