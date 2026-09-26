from env import CrossEmbodimentEnv
from stable_baselines3 import PPO
from algorithms.nervenet.ppo import SB3NerveNetPolicy
from stable_baselines3.common.env_util import make_vec_env

starting_embodiment="biped"
def make_env():
    wrapped_env = CrossEmbodimentEnv(
        # render_mode="human",
        starting_embodiment=starting_embodiment
    )
    return wrapped_env

env = make_vec_env(make_env, n_envs=4)

policy_kwargs = dict(
    obs_enc_hidden_dim=32,
    hidden_state_dim=64,
    updater_hidden_dim=64,
    msg_hidden_dim=32,
    msg_dim=16,
    iterations=2,
    dec_hidden_dim=32,
    ortho_init=False
)

model = PPO(
    policy=SB3NerveNetPolicy,
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
)

print("Starting NerveNet PPO Training Loop...")
model.learn(total_timesteps=200_000)
env.close()

model.save("checkpoints/nervenet_ppo_model")
print("Model saved successfully.")
