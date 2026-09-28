from env import CrossEmbodimentEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.wrappers import TimeLimit

starting_embodiment="biped"
max_episode_steps=250
def make_env():
    wrapped_env = CrossEmbodimentEnv(
        # render_mode="human",
        starting_embodiment=starting_embodiment,
        target_update_interval=1
    )
    wrapped_env = TimeLimit(wrapped_env, max_episode_steps=max_episode_steps)
    return wrapped_env

env = make_vec_env(make_env, n_envs=4)

policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],
        vf=[512, 512],
    )
)

model = PPO(
    "MultiInputPolicy",
    env,

    policy_kwargs=policy_kwargs,

    learning_rate=1.5e-4,

    n_steps=4096,
    batch_size=128,
    n_epochs=10,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,

    verbose=1,
    seed=42,

    device="auto",
)

print("Starting Normal PPO Training Loop...")
model.learn(total_timesteps=300_000)
env.close()

model.save("checkpoints/normal_ppo_model")
print("Model saved successfully.")
