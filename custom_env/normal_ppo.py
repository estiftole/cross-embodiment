from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from common import make_env
import os

starting_embodiment = "biped"

os.makedirs("checkpoints", exist_ok=True)

env = make_vec_env(make_env(starting_embodiment, use_padding=True), n_envs=8)

policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

model = PPO(
    "MultiInputPolicy",
    env,
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

print("Starting Normal PPO Training Loop...")
model.learn(total_timesteps=1_000_000)
env.close()

model.save("checkpoints/normal_ppo_model")
print("Model saved successfully.")
