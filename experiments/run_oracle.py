import os
import cv2
import time
import imageio
import numpy as np
import gymnasium as gym
import ml_collections
from utils import make_env, load_liv, TASKS


from metaworld.policies import (SawyerButtonPressTopdownV2Policy,
                                SawyerDrawerOpenV2Policy,
                                SawyerDrawerCloseV2Policy,
                                SawyerReachV2Policy,
                                SawyerDoorOpenV2Policy,
                                SawyerPushV2Policy,
                                SawyerWindowOpenV2Policy,
                                SawyerWindowCloseV2Policy,
                                SawyerBasketballV2Policy,
                                SawyerPegInsertionSideV2Policy,
                                SawyerPickPlaceV2Policy,
                                SawyerSweepV2Policy)


ORACLE_POLICY = {
    "button-press-topdown-v2-goal": SawyerButtonPressTopdownV2Policy,
    "drawer-open-v2-goal": SawyerDrawerOpenV2Policy,
    "drawer-close-v2-goal": SawyerDrawerCloseV2Policy,
    "reach-v2-goal": SawyerReachV2Policy,
    "door-open-v2-goal": SawyerDoorOpenV2Policy,
    "push-v2-goal": SawyerPushV2Policy,
    "window-open-v2-goal": SawyerWindowOpenV2Policy,
    "window-close-v2-goal": SawyerWindowCloseV2Policy,
    "basketball-v2-goal": SawyerBasketballV2Policy,
    "peg-insert-side-v2-goal": SawyerPegInsertionSideV2Policy,
    "pick-place-v2-goal": SawyerPickPlaceV2Policy,
    "sweep-v2-goal": SawyerSweepV2Policy,
}


def eval_policy(policy,
                env,
                camera_id: int = 2,
                eval_episodes: int = 1,
                video_dir: str = None,
                traj_dir: str = None):
    t1 = time.time()
    eval_reward, eval_success, avg_step = 0, 0, 0
    frames, success, states = [], [], []
    for i in range(1, eval_episodes + 1):
        obs, _ = env.reset()
        states = [obs]
        if video_dir and i == eval_episodes:
            frame = env.mujoco_renderer.render(render_mode="rgb_array",
                                               camera_id=camera_id)
            frames.append(frame[::-1])
            success.append(0.0)
        while True:
            avg_step += 1
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            states.append(obs)
            eval_success += info["success"] 
            success.append(info["success"])
            if video_dir and i == eval_episodes:
                frame = env.mujoco_renderer.render(render_mode="rgb_array",
                                                   camera_id=camera_id)
                frames.append(frame[::-1])
            eval_reward += reward
            if terminated or truncated:
                break
    eval_reward /= eval_episodes
    eval_success /= eval_episodes

    if video_dir:
        imageio.mimsave(f"{video_dir}_{eval_reward:.0f}.mp4", frames, fps=60)

    success = np.array(success)
    images = np.array(frames, dtype=np.uint8)
    np.savez(traj_dir, images=images, success=success)

    return eval_reward, eval_success, avg_step, time.time() - t1, images


def evaluate(config: ml_collections.ConfigDict):
    start_time = time.time()

    # initialize the dm_control environment
    traj_dir = f"data/oracle/{config.env_name}/s{config.seed}_c{config.camera_id}"
    video_dir = f"saved_videos/oracle/{config.env_name}"
    os.makedirs(f"saved_videos/oracle", exist_ok=True)
    os.makedirs(f"data/oracle/{config.env_name}", exist_ok=True)

    config.unlock()
    if config.env_name == "reach-v2-goal-hidden":
        config.env_name = "reach-v2-goal-observable"
    elif config.env_name == "push-v2-goal-hidden":
        config.env_name = "push-v2-goal-observable"
    elif config.env_name == "pick-place-v2-goal-hidden":
        config.env_name = "pick-place-v2-goal-observable" 
    elif config.env_name == "peg-insert-side-v2-goal-hidden":
        config.env_name == "peg-insert-side-v2-goal-observable"
    env = make_env(config.env_name,
                   seed=config.seed,
                   image_size=480,
                   camera_id=config.camera_id)

    policy = ORACLE_POLICY["-".join(config.env_name.split("-")[:-1])]()
    eval_reward, eval_success, _, _, images = eval_policy(policy,
                                                  env,
                                                  camera_id=config.camera_id,
                                                  video_dir=video_dir,
                                                  traj_dir=traj_dir)
    print(f"{config.env_name}: eval_reward={eval_reward:.2f}, eval_success={eval_success:.0f}")
