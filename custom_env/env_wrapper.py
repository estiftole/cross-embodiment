from gymnasium.wrappers import TimeLimit
from env import CrossEmbodimentEnv

MAX_EPISODE_STEPS = 500

NORM_OBS_KEYS = ["target_obs", "base_obs", "j_obs", "ee_obs"]

def make_env(starting_embodiment="biped", render_mode=None, target_update_interval=10, embodiment_update_interval=None, use_padding=False):
    def _thunk():
        env = CrossEmbodimentEnv(
            starting_embodiment=starting_embodiment,
            target_update_interval=target_update_interval,
            render_mode=render_mode,
            use_padding=use_padding,
            embodiment_update_interval=embodiment_update_interval
        )
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        return env
    return _thunk
