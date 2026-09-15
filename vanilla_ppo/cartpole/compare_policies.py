import numpy as np
from stable_baselines3 import PPO
from env import CustomCartPoleEnv
from train_univ_policy import PrivilegedObservationWrapper

def evaluate_model(model_type: str, num_episodes: int =5, max_steps: int =200, render_mode: str | None = None) -> list:
    print(f"Evaluating {model_type} Model")
    env = CustomCartPoleEnv(render_mode=render_mode)
    if render_mode=='rgb_array':
        from gymnasium.wrappers import RecordVideo

        env = RecordVideo(
            env,
            video_folder=f"./videos/{model_type}",
            # name_prefix=model_type,
            episode_trigger=lambda episode_id: True,
        )
    env = PrivilegedObservationWrapper(env)

    if model_type == "UNIVERSAL":
        model = PPO.load("checkpoints/univ_cartpole_ppo")
    else:
        model = PPO.load("checkpoints/naive_cartpole_ppo")

    results = []

    for ep in range(num_episodes):
        obs, start_info = env.reset()
        print(f"Episode params: {start_info}")
        done = False
        steps = 0

        while not done:
            if model_type == "UNIVERSAL":
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs[:4], deterministic=True)

            obs, reward, done, _, info = env.step(action)
            steps += 1
            if steps >= max_steps: break

        results.append({
            "steps": steps
        })

    env.close()
    return results

if __name__ == "__main__":
    max_steps = 200
    # num_episodes = 200
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
