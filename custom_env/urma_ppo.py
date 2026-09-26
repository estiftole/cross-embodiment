from env import CrossEmbodimentEnv
from stable_baselines3 import PPO
from algorithms.urma.ppo import SB3URMAPolicy
from stable_baselines3.common.env_util import make_vec_env

starting_embodiment="biped"
def make_env():
    wrapped_env = CrossEmbodimentEnv(
        # render_mode="human",
        starting_embodiment=starting_embodiment
    )
    return wrapped_env

env = make_vec_env(make_env, n_envs=1)

policy_kwargs = dict(
    enc_hidden_dim=64,
    j_latent_dim=64,
    ee_latent_dim=64,
    embed_dim=64,
    attn_heads=4,
    hidden_dim=128,
    action_latent_dim=64,
    dec_hidden_dim=64,
    dec_out_dim=32,
    mu_hidden_dim=32,
    action_dim=1,
    ortho_init=False,
)

model = PPO(
    policy=SB3URMAPolicy,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./urma_ppo_tb/"
)

print("Starting URMA PPO Training Loop...")
model.learn(total_timesteps=100_000)

model.save("checkpoints/urma_ppo_model")
print("Model saved successfully.")
