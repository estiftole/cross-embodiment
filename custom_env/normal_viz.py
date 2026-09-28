from stable_baselines3 import PPO
from common import make_env

starting_embodiment = "biped"

env = make_env(
    starting_embodiment,
    render_mode="human",
    target_update_interval=1,
    active_joints_only=True
)()
env.training = False
env.norm_reward = False

model = PPO.load("checkpoints/normal_ppo_model", env=env)

print("Model loaded successfully!")
# model.learn(total_timesteps=500_000)
obs, info = env.reset()
total_reward = 0.0

rounds = 5
while rounds > 0:
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode finished with total reward: {total_reward}")
        obs, info = env.reset()
        rounds -= 1
env.close()
