from stable_baselines3 import PPO
from algorithms.urma.ppo import SB3URMAPolicy
from stable_baselines3.common.env_util import make_vec_env
from common import make_env
import os

starting_embodiment = "biped"

os.makedirs("checkpoints", exist_ok=True)

env = make_vec_env(make_env(starting_embodiment, use_padding=True), n_envs=8)

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
    batch_size=128,
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
model.learn(total_timesteps=1_000_000)
env.close()

model.save("checkpoints/urma_ppo_model")
print("Model saved successfully.")
