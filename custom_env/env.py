import os
import tempfile
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np

class CrossEmbodimentEnv(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    DEFAULT_CAMERA_CONFIG = {
        "distance": 5.5,
        "elevation": -35.26,
        "azimuth": 225.0,
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
        self.min_torso_height = 0.15

        super().__init__(
            model_path=self.tmp_model.name,
            frame_skip=5,
            observation_space=None,
            default_camera_config=self.DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

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

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        obs = np.concatenate([qpos, qvel])

        forward_reward = self.data.qvel[0]
        ctrl_cost = 0.001 * np.sum(np.square(action))
        reward = forward_reward - ctrl_cost

        torso_z_height = self.data.qpos[2]
        terminated = torso_z_height < self.min_torso_height
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

class BipedEnv(CrossEmbodimentEnv):
    def __init__(
        self,
        robot_xml_path="custom_models/quadruped.xml",
        render_mode="human"
    ):
        super().__init__(
            robot_xml_path=robot_xml_path,
            render_mode="human"
        )
        self._cache_nominal_geometry()

    def _cache_nominal_geometry(self):
        thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_thigh")
        self.nominal_thigh_len = self.model.geom_size[thigh_id][1] * 2.0 if thigh_id != -1 else 0.30

        shin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "left_shin_geom")
        self.nominal_shin_len = self.model.geom_size[shin_id][1] * 2.0 if shin_id != -1 else 0.60

        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_sphere")
        self.pelvis_radius = self.model.geom_size[pelvis_id][0] if pelvis_id != -1 else 0.04

    def set_torso_dimensions(
        self,
        half_length: float = 0.08,
        half_width: float = 0.18,
        half_height: float = 0.30
    ):
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_box")
        if torso_id != -1:
            self.model.geom_size[torso_id] = [half_length, half_width, half_height]
            self.model.geom_rbound[torso_id] = np.sqrt(
                half_length**2 + half_width**2 + half_height**2
            )

        pelvis_z = -(half_height + self.pelvis_radius)
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "pelvis_sphere")
        if pelvis_id != -1:
            self.model.geom_pos[pelvis_id][2] = pelvis_z

        leg_y_offset = max(0.05, half_width - 0.06)

        for prefix, sign in [("left", 1.0), ("right", -1.0)]:
            cyl_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"pelvis_cyl_{prefix}")
            if cyl_id != -1:
                r_cyl = self.model.geom_size[cyl_id][0]
                y_start = 0.05 * sign
                y_end = leg_y_offset * sign
                y_mid = (y_start + y_end) / 2.0
                h_cyl = abs(y_end - y_start) / 2.0

                self.model.geom_pos[cyl_id] = [0.0, y_mid, pelvis_z]
                self.model.geom_size[cyl_id][1] = h_cyl
                self.model.geom_rbound[cyl_id] = np.sqrt(r_cyl**2 + h_cyl**2)

        for prefix, sign in [("left", 1.0), ("right", -1.0)]:
            leg_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_leg")
            if leg_id != -1:
                self.model.body_pos[leg_id] = [0.0, sign * leg_y_offset, pelvis_z]

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_leg_lengths(self, thigh_scale: float = 1.0, shin_scale: float = 1.0):
        thigh_len = self.nominal_thigh_len * thigh_scale
        shin_len = self.nominal_shin_len * shin_scale

        for prefix in ["left", "right"]:
            thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_thigh")
            shin_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_shin")

            if thigh_id != -1:
                r_thigh = self.model.geom_size[thigh_id][0]
                h_thigh = thigh_len / 2.0
                self.model.geom_pos[thigh_id] = [0.0, 0.0, -h_thigh]
                self.model.geom_size[thigh_id][1] = h_thigh
                self.model.geom_rbound[thigh_id] = np.sqrt(r_thigh**2 + h_thigh**2)

            if shin_body_id != -1:
                self.model.body_pos[shin_body_id] = [0.0, 0.0, -thigh_len]

            shin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shin_geom")
            if shin_id != -1:
                r_shin = self.model.geom_size[shin_id][0]
                h_shin = shin_len / 2.0
                self.model.geom_pos[shin_id] = [0.0, 0.0, -h_shin]
                self.model.geom_size[shin_id][1] = h_shin
                self.model.geom_rbound[shin_id] = np.sqrt(r_shin**2 + h_shin**2)

            foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_foot")
            wheel_hub_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_wheel_hub")

            if foot_body_id != -1:
                self.model.body_pos[foot_body_id] = [0.0, 0.0, -shin_len]
            elif wheel_hub_id != -1:
                self.model.body_pos[wheel_hub_id] = [0.0, 0.0, -shin_len]

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_leg_thickness(self, thigh_radius: float = 0.03, shin_radius: float = 0.025):
        for prefix in ["left", "right"]:
            thigh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_thigh")
            if thigh_id != -1:
                h_thigh = self.model.geom_size[thigh_id][1]
                self.model.geom_size[thigh_id][0] = thigh_radius
                self.model.geom_rbound[thigh_id] = np.sqrt(thigh_radius**2 + h_thigh**2)

            shin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shin_geom")
            if shin_id != -1:
                h_shin = self.model.geom_size[shin_id][1]
                self.model.geom_size[shin_id][0] = shin_radius
                self.model.geom_rbound[shin_id] = np.sqrt(shin_radius**2 + h_shin**2)

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_wheel_dimensions(self, wheel_diameter: float = 0.24, wheel_thickness: float = 0.02):
        radius = wheel_diameter / 2.0
        half_width = wheel_thickness / 2.0

        updated = False
        for prefix in ["left", "right"]:
            wheel_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_wheel_disc"
            )
            if wheel_geom_id != -1:
                self.model.geom_size[wheel_geom_id][0] = radius
                self.model.geom_size[wheel_geom_id][1] = half_width
                self.model.geom_rbound[wheel_geom_id] = np.sqrt(radius**2 + half_width**2)
                updated = True

        if updated:
            mujoco.mj_setConst(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return self.reset()

class QuadpedEnv(CrossEmbodimentEnv):
    def __init__(
        self,
        robot_xml_path="custom_models/quadruped.xml",
        render_mode="human"
    ):
        super().__init__(
            robot_xml_path=robot_xml_path,
            render_mode="human"
        )

    def set_torso_dimensions(
            self,
            half_length: float,
            half_width: float,
            half_height: float
        ):
        torso_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso_box")
        self.model.geom_size[torso_geom_id] = [half_length, half_width, half_height]

        leg_x_offset = half_length - 0.08
        leg_y_offset = half_width + 0.02

        leg_positions = {
            "fl_leg": (leg_x_offset, leg_y_offset),
            "fr_leg": (leg_x_offset, -leg_y_offset),
            "bl_leg": (-leg_x_offset, leg_y_offset),
            "br_leg": (-leg_x_offset, -leg_y_offset),
        }

        for body_name, (x_pos, y_pos) in leg_positions.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            self.model.body_pos[body_id][0] = x_pos
            self.model.body_pos[body_id][1] = y_pos

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_leg_thickness(self, thigh_radius: float = 0.025, shin_radius: float = 0.02):
        for prefix in ["fl", "fr", "bl", "br"]:
            thigh_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_thigh")
            shin_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shin_geom")

            self.model.geom_size[thigh_geom_id][0] = thigh_radius
            self.model.geom_size[shin_geom_id][0] = shin_radius

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_wheel_dimensions(self, wheel_diameter: float = 0.16, wheel_thickness: float = 0.03):
        radius = wheel_diameter / 2.0
        half_width = wheel_thickness / 2.0

        for prefix in ["fl", "fr", "bl", "br"]:
            wheel_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_wheel_disc")
            if wheel_geom_id != -1:
                self.model.geom_size[wheel_geom_id][0] = radius
                self.model.geom_size[wheel_geom_id][1] = half_width

                self.model.geom_rbound[wheel_geom_id] = np.sqrt(radius**2 + half_width**2)

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.reset()

    def set_leg_lengths(self, thigh_scale: float = 1.0, shin_scale: float = 1.0):
        base_v_thigh = np.array([-0.08, 0.0, -0.3])
        base_v_shin = np.array([0.08, 0.0, -0.3])

        v_thigh = base_v_thigh * thigh_scale
        v_shin = base_v_shin * shin_scale

        for prefix in ["fl", "fr", "bl", "br"]:
            thigh_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_thigh")
            shin_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_shin")

            self.model.geom_pos[thigh_geom_id] = v_thigh / 2.0
            self.model.geom_size[thigh_geom_id][1] = np.linalg.norm(v_thigh) / 2.0
            self.model.body_pos[shin_body_id] = v_thigh

            shin_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shin_geom")
            self.model.geom_pos[shin_geom_id] = v_shin / 2.0
            self.model.geom_size[shin_geom_id][1] = np.linalg.norm(v_shin) / 2.0

            wheel_hub_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}_wheel_hub")
            foot_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_foot")

            if wheel_hub_id != -1:
                self.model.body_pos[wheel_hub_id] = v_shin
            elif foot_geom_id != -1:
                self.model.geom_pos[foot_geom_id] = v_shin

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        return self.reset()
