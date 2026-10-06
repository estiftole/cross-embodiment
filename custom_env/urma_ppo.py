from stable_baselines3 import PPO
from algorithms.urma.ppo import SB3URMAPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from env_wrapper import make_env, NORM_OBS_KEYS
import os
import argparse

def train(args):
    starting_embodiment = "biped"

    os.makedirs("checkpoints", exist_ok=True)

    env = make_vec_env(make_env(starting_embodiment, use_padding=False), n_envs=8)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, norm_obs_keys=NORM_OBS_KEYS, clip_obs=10.0)

    policy_kwargs = {
        "enc_hidden_dim": 64,
        "j_latent_dim": 64,
        "ee_latent_dim": 64,
        "embed_dim": 64,
        "attn_heads": 4,
        "hidden_dim": 128,
        "action_latent_dim": 64,
        "dec_hidden_dim": 64,
        "dec_out_dim": 32,
        "mu_hidden_dim": 32,
        "action_dim": 1,
        "ortho_init": False,
    }

    model = PPO(
        policy=SB3URMAPolicy,
        env=env,
        policy_kwargs=policy_kwargs,

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

    print("Starting URMA PPO Training Loop...")
    model.learn(total_timesteps=args.steps)

    model.save("checkpoints/urma_ppo_model")
    env.save("checkpoints/urma_ppo_vecnormalize.pkl")
    print("Model saved successfully.")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Universal Robot Morphology Architecture")
    parser.add_argument("--steps",
        help="Total training time steps",
        type=int,
        default=1_000_000)
    args = parser.parse_args()
    train(args)
