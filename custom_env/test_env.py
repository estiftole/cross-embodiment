from env import CrossEmbodimentEnv
from gymnasium.wrappers import TimeLimit
import numpy as np

if __name__ == "__main__":
    render_mode = "human"
    starting_embodiment="biped"
    max_episode_steps=250
    def make_env():
        env = CrossEmbodimentEnv(
            render_mode="human",
            starting_embodiment=starting_embodiment
        )
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    env = make_env()

    obs, info = env.reset()
    zero_action = np.zeros(env.action_space.shape)
    total_reward = 0.0

    rounds = 3
    while rounds > 0:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode finished with total reward: {total_reward}")
            obs, info = env.reset()
            rounds -= 1
    env.close()
