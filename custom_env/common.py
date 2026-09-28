import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from env import CrossEmbodimentEnv

MAX_EPISODE_STEPS = 500

NORM_OBS_KEYS = ["target_obs", "base_obs", "j_obs", "ee_obs"]

class ActiveJointsAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n = len(env.unwrapped.active_env.actuated_jnt_ids)
        self.full_dim = env.unwrapped.action_space.shape[0]
        self._action_space = spaces.Box(-1.0, 1.0, shape=(self.n,), dtype=np.float32)

    @property
    def action_space(self):
        return self._action_space

    def action(self, act):
        full = np.zeros(self.full_dim, dtype=np.float32)
        full[: self.n] = act
        return full


def make_env(embodiment="biped", render_mode=None, target_update_interval=10, active_joints_only=False):
    def _thunk():
        env = CrossEmbodimentEnv(
            starting_embodiment=embodiment,
            target_update_interval=target_update_interval,
            render_mode=render_mode,
        )
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        if active_joints_only:
            env = ActiveJointsAction(env)
        return env
    return _thunk
