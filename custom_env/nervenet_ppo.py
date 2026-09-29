from stable_baselines3 import PPO
from algorithms.nervenet.ppo import SB3NerveNetPolicy

from stable_baselines3.common.env_util import make_vec_env
from common import make_env
import os

starting_embodiment = "biped"

os.makedirs("checkpoints", exist_ok=True)

env = make_vec_env(make_env(starting_embodiment, use_padding=True), n_envs=8)

policy_kwargs = {
    "obs_enc_hidden_dim": 32,
    "hidden_state_dim": 64,
    "updater_hidden_dim": 64,
    "msg_hidden_dim": 32,
    "msg_dim": 16,
    "iterations": 2,
    "dec_hidden_dim": 32,
    "ortho_init": False
}

model = PPO(
    policy=SB3NerveNetPolicy,
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

print("Starting NerveNet PPO Training Loop...")
model.learn(total_timesteps=1_000_000)
env.close()

model.save("checkpoints/nervenet_ppo_model")
print("Model saved successfully.")
