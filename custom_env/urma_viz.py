from stable_baselines3 import PPO
from common import make_env

starting_embodiment = "biped"

env = make_env(
    starting_embodiment,
    render_mode="human",
    target_update_interval=1,
    use_padding=True
)()
env.training = False
env.norm_reward = False

model = PPO.load("checkpoints/urma_ppo_model", env=env)

print("Model loaded successfully!")
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
