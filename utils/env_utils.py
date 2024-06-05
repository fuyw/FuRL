import os
import collections
import imageio
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from metaworld.envs import (
    ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
)


TASKS = {
    "window-open-v2-goal-hidden": "push and open a window",
    "window-close-v2-goal-hidden": "push and close a window", 
    "button-press-topdown-v2-goal-hidden": "press a button from the top",
    "door-open-v2-goal-hidden": "open a door with a revolving joint",
    "drawer-close-v2-goal-hidden": "push and close a drawer",
    "drawer-open-v2-goal-hidden": "open a drawer",
    "push-v2-goal-hidden": "push the puck to a goal",
    "reach-v2-goal-hidden": "reach a goal position",
    "window-open-v2-goal-observable": "push and open a window",
    "window-close-v2-goal-observable": "push and close a window",
    "button-press-topdown-v2-goal-observable": "press a button from the top",
    "door-open-v2-goal-observable": "open a door with a revolving joint",
    "drawer-close-v2-goal-observable": "push and close a drawer",
    "drawer-open-v2-goal-observable": "open a drawer",
    "push-v2-goal-observable": "push the puck to a goal",
    "reach-v2-goal-observable": "reach a goal position",
    "pick-place-v2-goal-observable": "pick and place a puck to a goal",
    "peg-insert-side-v2-goal-observable": "insert a peg sideways",
}


###################
# Utils Functions #
###################
def randomize_initial_state(env, num_step=50):
    dx = np.random.uniform(-0.5, 0.5)
    dy = np.random.uniform(-0.5, 0)
    dz = np.random.uniform(0., 0.05)
    actions = [np.array([dx, 0., 0., 0.]),
               np.array([0., dy, 0., 0.]),
               np.array([0., 0., dz, 0.])]
    for _ in range(num_step):
        action = actions[np.random.randint(3)]
        _ = env.step(action)
    env.curr_path_length = 0


################
# Env Wrappers #
################
class PixelObservationWrapper(gym.ObservationWrapper):
    def __init__(self,
                 env: gym.Env,
                 image_size: int,
                 camera_id: int=1):
        super().__init__(env)
        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(image_size, image_size, 3),
                                     dtype=np.uint8)
        self.viewer = self.env.unwrapped.mujoco_renderer._get_viewer(
            render_mode="rgb_array")
        self.camera_id = camera_id
        self.image_size = image_size

    def get_image(self):
        img = self.unwrapped.mujoco_renderer.render(
                render_mode="rgb_array", camera_id=self.camera_id)
        return img[::-1]

    def observation(self, observation):
        return self.get_image()

    def render_img(self, render_image_size: int = 256):
        self.viewer.viewport.width = render_image_size
        self.viewer.viewport.height = render_image_size
        frame = self.env.render()
        self.viewer.viewport.width = self.image_size
        self.viewer.viewport.height = self.image_size
        return frame[::-1]


class RepeatAction(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, combined_info


#################
# Main Function #
#################
def make_env(env_name: str = "drawer-open-v2-goal-hidden",
             seed: int = 42,
             camera_id: int = 1,
             render_image_size: int = 256,
             image_size: int = 256,  # 84
             use_pixel: bool = False,
             action_repeat: int = 1,
             render_mode: str = "rgb_array"):

    if "hidden" in env_name:
        env = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_name](seed=seed)
    else:
        env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed, render_mode=render_mode)
        env._freeze_rand_vec = False
    env.camera_id = camera_id
    viewer = env.unwrapped.mujoco_renderer._get_viewer(render_mode=render_mode)
    viewer.viewport.width = image_size
    viewer.viewport.height = image_size

    # sticky actions
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    # use pixel-based obs
    if use_pixel:
        env = PixelObservationWrapper(env, image_size, camera_id)

    # set random seed
    env.reset(seed=seed)
    env.action_space.seed(seed=seed)

    return env
