from stable_baselines3 import PPO
from env import CrossEmbodimentEnv

starting_embodiment="biped"
env = CrossEmbodimentEnv(
    render_mode="human",
    starting_embodiment=starting_embodiment
)
model = PPO.load("checkpoints/urma_ppo_model", env=env)

print("Model loaded successfully!")
obs, info = env.reset()
total_reward = 0.0

for step in range(1000):
    action, _states = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode finished with total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0.0
