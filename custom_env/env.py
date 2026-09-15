import os
import tempfile
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np

class SwappableLocomotionEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    DEFAULT_CAMERA_CONFIG = {
        "distance": 5.5,
        "elevation": -20.0,
        "azimuth": 90.0,
        "lookat": [0.0, 0.0, 1.0],
    }

    def __init__(self, scene_xml_path="custom_models/flat_scene.xml", robot_xml_path="custom_models/model.xml", **kwargs):
        scene_xml_content = f"""
        <mujoco model="walking_scene">
          <include file="{os.path.abspath(scene_xml_path)}"/>
          <include file="{os.path.abspath(robot_xml_path)}"/>
        </mujoco>
        """

        self.tmp_model = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        self.tmp_model.write(scene_xml_content)
        self.tmp_model.close()

        super().__init__(
            model_path=self.tmp_model.name,
            frame_skip=5,
            observation_space=None,
            default_camera_config=self.DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        # camera setup
        self.setup_camera()


    def setup_camera(self):
        self.render()
        self.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.mujoco_renderer.viewer.cam.trackbodyid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso"
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        # Extract robot state for observations
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        obs = np.concatenate([qpos, qvel])

        # Calculate forward velocity reward along X-axis
        forward_reward = self.data.qvel[0]
        ctrl_cost = 0.001 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost

        torso_z_height = self.data.qpos[2]
        terminated = torso_z_height < 0.15
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        return np.concatenate([qpos, qvel]).astype(np.float32)

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)

        return self._get_obs()

    def close(self):
        super().close()
        if os.path.exists(self.tmp_model.name):
            os.remove(self.tmp_model.name)

if __name__ == "__main__":
    robot_xml_path="custom_models/quadped.xml"

    env = SwappableLocomotionEnv(robot_xml_path=robot_xml_path, render_mode="human")
    obs, info = env.reset()
    zero_action = np.zeros(env.action_space.shape)

    max_steps = 200
    steps_left = max_steps
    trials_left = 5

    while (trials_left > 0):
        obs, reward, terminated, truncated, info = env.step(zero_action)
        steps_left -= 1

        if terminated or truncated or steps_left == 0:
            obs, info = env.reset()
            steps_left = max_steps
            trials_left -= 1

        # time.sleep(env.dt)

    env.close()
