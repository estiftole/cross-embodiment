from gymnasium.wrappers import TimeLimit
from env import CrossEmbodimentEnv

MAX_EPISODE_STEPS = 500

NORM_OBS_KEYS = ["target_obs", "base_obs", "j_obs", "ee_obs"]

def make_env(embodiment="biped", render_mode=None, target_update_interval=10, use_padding=False):
    def _thunk():
        env = CrossEmbodimentEnv(
            starting_embodiment=embodiment,
            target_update_interval=target_update_interval,
            render_mode=render_mode,
            use_padding=use_padding
        )
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        return env
    return _thunk
