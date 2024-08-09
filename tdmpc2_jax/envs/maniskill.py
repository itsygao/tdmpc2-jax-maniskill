import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from gymnasium.vector import VectorEnvWrapper
from mani_skill.utils import common
from pathlib import Path
import hydra
import torch
import numpy as np
import os
from mani_skill.utils.visualization.misc import tile_images

def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path

class ManiSkillNumpyWrapper(gym.Wrapper): # VectorEnvWrapper
    """This wrapper wraps any maniskill env created via gym.make to ensure the outputs of
    env.render, env.reset, env.step are all numpy arrays. It also works when the input for
    env.step is a numpy array.
    This wrapper should generally be applied after
    wrappers as most wrappers for ManiSkill assume data returned is batched and is a torch tensor"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_envs = self.base_env.num_envs

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    @staticmethod
    def _convert_tensors_to_numpy(info):
        for key, value in info.items():
            if isinstance(value, dict):
                # Recursive call for nested dictionary
                ManiSkillNumpyWrapper._convert_tensors_to_numpy(value)
            elif isinstance(value, torch.Tensor):
                # Convert PyTorch tensor to numpy array
                info[key] = value.cpu().numpy()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        ManiSkillNumpyWrapper._convert_tensors_to_numpy(info)
        obs = obs.cpu().numpy()
        reward = reward.cpu().numpy()
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=dict()): # keep the default value the same as ManiskillVectorEnv if want to wrap out of it
        obs, info = super().reset(seed=seed, options=options)
        ManiSkillNumpyWrapper._convert_tensors_to_numpy(info)

        return obs.cpu().numpy(), info

    def render(self):
        return self.env.render()
    
    # def render(self):
    #     ret = self.env.render()
    #     if self.env.render_mode in ["rgb_array", "sensors"]:
    #         return common.to_numpy(ret)
    #     else:
    #         assert self.env.render_mode in ["rgb_array", "sensors"], f"{self.env.render_mode}"

class ManiskillVideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, cfg, wandb, fps=15):
        self.cfg = cfg
        self.maniskill_video_nrows = int(np.sqrt(cfg.env.num_envs))
        cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.env.env_id / str(cfg.seed) / cfg.exp_name
        self._save_dir = make_dir(cfg.work_dir / 'eval_video')
        self._wandb = wandb
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self._save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            self.frames.append(env.render())

    def save(self, step, key='videos/eval_video'):
        if self.enabled and len(self.frames) > 0:
            self.frames = [tile_images(rgbs, nrows=self.maniskill_video_nrows) for rgbs in self.frames]
            frames = np.stack(self.frames)
            return self._wandb.log(
                {key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
            )