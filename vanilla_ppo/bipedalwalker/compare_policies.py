import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import CustomBipedalWalkerEnv
from train_univ_policy import PrivilegedObservationWrapper


def evaluate_model(model_type: str, num_episodes: int = 5, max_steps: int = 500, render_mode: str | None = None) -> list:
    print(f"\nEvaluating {model_type} Model...")

    def make_env():
        env = CustomBipedalWalkerEnv(render_mode=render_mode)

        if render_mode=='rgb_array':
            from gymnasium.wrappers import RecordVideo

            env = RecordVideo(
                env,
                video_folder=f"./videos/{model_type}",
                # name_prefix=model_type,
                episode_trigger=lambda episode_id: True,
            )
        env = PrivilegedObservationWrapper(env)
        return env

    vec_env = DummyVecEnv([make_env])

    if model_type == "UNIVERSAL":
        model = PPO.load("checkpoints/univ_bipedalwalker_ppo")
        vec_env = VecNormalize.load("checkpoints/univ_bipedal_vecnormalize.pkl", vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        model = PPO.load("checkpoints/naive_bipedalwalker_ppo")

    results = []

    for ep in range(num_episodes):
        obs = vec_env.reset()

        raw_env = vec_env.envs[0].unwrapped

        done = False
        steps = 0

        while not done:
            if model_type == "UNIVERSAL":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs[:, :24], deterministic=True)

            obs, rewards, dones, infos = vec_env.step(action)
            steps += 1

            done = dones[0] or (steps >= max_steps)

            if render_mode=='human':
                time.sleep(0.01)

        results.append({
            # "friction": raw_env.true_friction,
            # "density": raw_env.true_density,
            # "leg_scale": raw_env.true_leg_scale,
            "steps": steps
        })

    vec_env.close()
    return results

if __name__ == "__main__":
    max_steps = 500
    num_episodes = 5
    render_mode = "rgb_array"

    print(f"Running stress test comparison ({num_episodes} episodes each)...")

    naive_results = evaluate_model("NAIVE", num_episodes, max_steps, render_mode)
    naive_avg = np.mean([r['steps'] for r in naive_results])
    print("\n--- Results (Average Steps Survived) ---")
    print(f"Naive Policy:     {naive_avg:.2f} steps")
    naive_fails = [r for r in naive_results if r['steps'] < max_steps]
    print(f"Naive Policy failed on {len(naive_fails)} out of {num_episodes} episodes.")

    univ_results = evaluate_model("UNIVERSAL", num_episodes, max_steps, render_mode)
    univ_avg = np.mean([r['steps'] for r in univ_results])
    print("\n--- Results (Average Steps Survived) ---")
    print(f"Universal Policy: {univ_avg:.2f} steps")
    univ_fails = [r for r in univ_results if r['steps'] < max_steps]
    print(f"Universal Policy failed on {len(univ_fails)} out of {num_episodes} episodes.")
