from stable_baselines3 import PPO
from env import CrossEmbodimentEnv
from gymnasium.wrappers import TimeLimit

starting_embodiment="biped"
def make_env():
    env = CrossEmbodimentEnv(
        render_mode="human",
        starting_embodiment=starting_embodiment
    )
    env = TimeLimit(env, max_episode_steps=200)
    return env

env = make_env()

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
