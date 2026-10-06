from stable_baselines3 import PPO
from algorithms.nervenet.ppo import NerveNetPolicy

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env_wrapper import make_env, NORM_OBS_KEYS
import os
import argparse

def train(args):
    starting_embodiment = "biped"

    os.makedirs("checkpoints", exist_ok=True)

    env = make_vec_env(make_env(starting_embodiment, use_padding=True), n_envs=8)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, norm_obs_keys=NORM_OBS_KEYS, clip_obs=10.0)

    model = PPO(
        policy=NerveNetPolicy,
        env=env,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        seed=42,
        device="auto",
    )

    print("Starting NerveNet PPO Training Loop...")
    model.learn(total_timesteps=args.steps)

    model.save("checkpoints/nervenet_ppo_model")
    env.save("checkpoints/nervenet_ppo_vecnormalize.pkl")
    print("Model saved successfully.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NerveNet")
    parser.add_argument("--steps",
        help="Total training time steps",
        type=int,
        default=1_000_000)
    args = parser.parse_args()
    train(args)
