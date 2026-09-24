from env import BipedEnv, QuadpedEnv
import numpy as np

if __name__ == "__main__":
    render_mode = "rgb_array"
    # env = QuadpedEnv(render_mode=render_mode)
    env = BipedEnv(render_mode=render_mode)

    if render_mode=="rgb_array":
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(
            env,
            video_folder="./videos",
            episode_trigger=lambda episode_id: True,
        )

    obs, info = env.reset()
    zero_action = np.zeros(env.action_space.shape)

    max_steps = 100
    steps_left = max_steps
    trials_left = 1

    while (trials_left > 0):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        steps_left -= 1

        if terminated or truncated or steps_left == 0:
            steps_left = max_steps
            trials_left -= 1
            if trials_left > 0: obs, info = env.unwrapped.reset()

            # env.unwrapped.set_torso_dimensions(0.3, 0.3, 0.025)
            # env.unwrapped.set_leg_lengths(thigh_scale=0.8,shin_scale=0.3)
            # env.unwrapped.set_leg_thickness(thigh_radius=0.04,shin_radius=0.04)
            # env.unwrapped.set_wheel_dimensions(wheel_diameter=0.2)

    env.close()
