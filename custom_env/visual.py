from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env_wrapper import make_env
import argparse

def visualize(model):
    starting_embodiment = "biped"
    use_padding = False
    if model=="normal":
        use_padding = True
    env = DummyVecEnv([make_env(starting_embodiment, render_mode="human", target_update_interval=1, use_padding=use_padding)])
    env = VecNormalize.load(f"checkpoints/{model}_ppo_vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(f"checkpoints/{model}_ppo_model", env=env)
    print("Model loaded successfully!")

    obs = env.reset()
    total_reward, rounds = 0.0, 5
    while rounds > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done[0]:
            print(f"Episode finished with total reward: {total_reward:.2f}, "
                f"final distance to target: {info[0]['distance_from_target']:.2f}")
            total_reward = 0.0
            rounds -= 1
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model performance.")
    parser.add_argument("-m", "--model",
        help="Model to visualize",
        choices=['normal', 'urma', 'nervenet'],
        default="normal")
    args = parser.parse_args()
    visualize(args.model)
