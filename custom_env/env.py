import os
import tempfile
import mujoco
from gymnasium.envs.mujoco import MujocoEnv
import numpy as np
import torch

import gymnasium as gym

class EnvTemplate(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}
    DEFAULT_CAMERA_CONFIG = {
        "distance": 5,
        "elevation": -35.26,
        "azimuth": 225.0,
        "lookat": [0.0, 0.0, 1.0],
    }

    def __init__(self, scene_xml_path="custom_models/flat_scene.xml", robot_xml_path="custom_models/model.xml", **kwargs):
        scene_xml_content = f"""
        <mujoco model="walking_scene">
          <include file="{os.path.abspath(scene_xml_path)}"/>
          <include file="{os.path.abspath(robot_xml_path)}"/>
          <worldbody>
              <site
                  name="target"
                  type="sphere"
                  pos="0 0 0.5"
                  size="0.12"
                  rgba="1 0 0 1"
              />
          </worldbody>
        </mujoco>
        """

        self.tmp_model = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        self.tmp_model.write(scene_xml_content)
        self.tmp_model.close()
        self.min_torso_height = 0.15

        self.target_pos = np.zeros(2)
        self.target_bounds = [-2.0, 2.0]
        self.target_reach_threshold = 0.5
        self.random_change_prob = 0.005

        super().__init__(
            model_path=self.tmp_model.name,
            frame_skip=5,
            observation_space=None,
            default_camera_config=self.DEFAULT_CAMERA_CONFIG,
            **kwargs
        )

        self.target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target"
        )

        if self.render_mode:
            self.setup_camera()
        self._init_embodiment_metadata()
        self.graph_topology = self._extract_graph_topology()

    def setup_camera(self):
        self.render()
        self.mujoco_renderer.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.mujoco_renderer.viewer.cam.trackbodyid = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso"
        )

    def _init_embodiment_metadata(self):
        self.actuated_jnt_ids = []
        self.is_wheel_joint = []
        joint_descriptors = []

        # cache joint descriptions
        for i in range(self.model.njnt):
            jnt_type = self.model.jnt_type[i]
            if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
                continue

            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""

            is_wheel = 1.0 if ("wheel" in jnt_name.lower() or self.model.jnt_limited[i] == 0) else 0.0
            limits = self.model.jnt_range[i]
            jnt_pos = self.model.jnt_pos[i]
            jnt_axis = self.model.jnt_axis[i]

            descriptor = np.concatenate([
                [is_wheel, limits[0], limits[1]],
                jnt_pos,
                jnt_axis
            ]).astype(np.float32)

            self.actuated_jnt_ids.append(i)
            self.is_wheel_joint.append(is_wheel)
            joint_descriptors.append(descriptor)

        self.cached_joint_descriptors = np.array(joint_descriptors, dtype=np.float32)

        # cache end-effector descriptions
        self.ee_site_ids = []
        ee_desc = []

        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
            if "foot" in site_name.lower() or "wheel" in site_name.lower():
                is_wheel = 1.0 if "wheel" in site_name.lower() else 0.0
                site_rest_pos = self.model.site_pos[i]
                ee_description = np.concatenate([[is_wheel], site_rest_pos]).astype(np.float32)

                self.ee_site_ids.append(i)
                ee_desc.append(ee_description)

        self.cached_ee_descriptors = (
            np.array(ee_desc, dtype=np.float32)
            if ee_desc
            else np.zeros((0, 4), dtype=np.float32)
        )

    def _extract_graph_topology(self):
        senders = []
        receivers = []

        # root torso is Node 0
        actuatable_nodes = list(range(1, len(self.actuated_jnt_ids) + 1))

        for node_idx, j_id in enumerate(self.actuated_jnt_ids, start=1):
            body_id = self.model.jnt_bodyid[j_id]
            parent_body_id = self.model.body_parentid[body_id]

            parent_node = 0 if parent_body_id == 1 else parent_body_id - 1

            senders.extend([parent_node, node_idx])
            receivers.extend([node_idx, parent_node])

        return {
            "senders": torch.tensor(senders, dtype=torch.long),
            "receivers": torch.tensor(receivers, dtype=torch.long),
            "actuatable_nodes": torch.tensor(actuatable_nodes, dtype=torch.long)
        }

    def _sample_target(self):
        self.target_pos = self.np_random.uniform(
            low=self.target_bounds[0],
            high=self.target_bounds[1],
            size=2
        )

        self.model.site_pos[self.target_site_id][:2] = self.target_pos
        self.model.site_pos[self.target_site_id][2] = 0.5

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        torso_xy = self.data.qpos[:2]
        torso_z = self.data.qpos[2]

        distance_to_target = np.linalg.norm(self.target_pos - torso_xy)
        dt = self.frame_skip * self.model.opt.timestep
        progress_reward = (self.prev_distance - distance_to_target) / dt
        self.prev_distance = distance_to_target

        healthy_reward = 1.0

        torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_z_orientation = self.data.xmat[torso_body_id][8]  # R22 element
        upright_reward = 0.5 * max(0.0, torso_z_orientation)

        ctrl_cost = 0.001 * np.sum(np.square(action))
        smoothness_cost = 0.01 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()

        reward = (
            progress_reward
            + healthy_reward
            + upright_reward
            - ctrl_cost
            - smoothness_cost
        )

        terminated = torso_z < self.min_torso_height

        if distance_to_target < self.target_reach_threshold:
            reward += 10.0
            terminated = True

        info = {
            "reward_progress": progress_reward,
            "reward_healthy": healthy_reward,
            "reward_upright": upright_reward,
            "cost_ctrl": ctrl_cost,
            "cost_smoothness": smoothness_cost,
            "distance_to_target": distance_to_target,
        }

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs()
        return obs, reward, terminated, False, info

    def _get_obs(self):
        torso_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")

        R_torso = self.data.xmat[torso_body_id].reshape(3, 3)
        yaw = np.arctan2(R_torso[1, 0], R_torso[0, 0])
        cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
        R_heading = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float32)

        torso_xy = self.data.qpos[:2]
        world_rel_target = np.array([self.target_pos[0] - torso_xy[0], self.target_pos[1] - torso_xy[1], 0.0])
        local_rel_target = (R_heading @ world_rel_target)[:2].astype(np.float32)

        world_lin_vel = self.data.qvel[:3]
        world_ang_vel = self.data.qvel[3:6]

        local_lin_vel = R_heading @ world_lin_vel
        local_ang_vel = R_heading @ world_ang_vel

        local_gravity = R_torso.T @ np.array([0.0, 0.0, -1.0])
        torso_z = self.data.qpos[2:3]

        base_obs = np.concatenate([
            torso_z,
            local_gravity,
            local_lin_vel,
            local_ang_vel
        ]).astype(np.float32)

        num_joints = len(self.actuated_jnt_ids)
        if num_joints > 0:
            q = self.data.qpos[self.jnt_qposadr]
            qvel = self.data.qvel[self.jnt_dofadr]

            joint_obs = np.zeros((num_joints, 3), dtype=np.float32)

            if np.any(self.wheel_mask):
                q_w = q[self.wheel_mask]
                joint_obs[self.wheel_mask, 0] = np.sin(q_w)
                joint_obs[self.wheel_mask, 1] = np.cos(q_w)
                joint_obs[self.wheel_mask, 2] = np.clip(qvel[self.wheel_mask], -10.0, 10.0)

            if np.any(self.hinge_mask):
                hinge_indices = self.actuated_jnt_ids[self.hinge_mask]
                q_h = q[self.hinge_mask]

                ranges = self.model.jnt_range[hinge_indices]
                q_min, q_max = ranges[:, 0], ranges[:, 1]
                q_mid = 0.5 * (q_max + q_min)
                q_half_range = 0.5 * (q_max - q_min)
                q_half_range[q_half_range == 0] = 1.0
                q_norm = (q_h - q_mid) / q_half_range

                joint_obs[self.hinge_mask, 0] = q_norm
                joint_obs[self.hinge_mask, 1] = np.clip(qvel[self.hinge_mask], -10.0, 10.0)
        else:
            joint_obs = np.zeros((0, 3), dtype=np.float32)

        if len(self.ee_site_ids) > 0:
            root_pos = self.data.qpos[:3]
            world_ee_pos = self.data.site_xpos[self.ee_site_ids] - root_pos
            local_ee_pos = (world_ee_pos @ R_heading.T).astype(np.float32)

            raw_contact_forces = self.data.cfrc_ext[self.ee_body_ids, 3:6]
            scaled_contact_forces = np.clip(raw_contact_forces / 100.0, -1.0, 1.0).astype(np.float32)

            ee_obs = np.hstack([local_ee_pos, scaled_contact_forces])
        else:
            ee_obs = np.zeros((0, 6), dtype=np.float32)

        return {
            "target_obs": local_rel_target,
            "base_obs": base_obs,
            "j_obs": joint_obs,
            "j_desc": self.cached_joint_descriptors,
            "ee_obs": ee_obs,
            "ee_desc": self.cached_ee_descriptors,
        }

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        qpos += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel += self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)

        self._sample_target()

        self.prev_distance = np.linalg.norm(self.target_pos - self.data.qpos[:2])
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        return self._get_obs()

    def close(self):
        super().close()
        if os.path.exists(self.tmp_model.name):
            os.remove(self.tmp_model.name)

class BipedEnv(EnvTemplate):
    def __init__(
        self,
        robot_xml_path="custom_models/biped.xml",
        render_mode=None
    ):
        if render_mode:
            super().__init__(
                robot_xml_path=robot_xml_path,
                render_mode=render_mode
            )
        else:
            super().__init__(
                robot_xml_path=robot_xml_path
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


class QuadpedEnv(EnvTemplate):
    def __init__(
        self,
        robot_xml_path="custom_models/quadped.xml",
        render_mode=None
    ):
        if render_mode:
            super().__init__(
                robot_xml_path=robot_xml_path,
                render_mode=render_mode
            )
        else:
            super().__init__(
                robot_xml_path=robot_xml_path
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


    def set_leg_thickness(self, thigh_radius: float = 0.025, shin_radius: float = 0.02):
        for prefix in ["fl", "fr", "bl", "br"]:
            thigh_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_thigh")
            shin_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}_shin_geom")

            self.model.geom_size[thigh_geom_id][0] = thigh_radius
            self.model.geom_size[shin_geom_id][0] = shin_radius

        mujoco.mj_setConst(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)


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

class CrossEmbodimentEnv(gym.Env):
    def __init__(self, starting_embodiment="quadped", render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.registry = {
            "biped": BipedEnv,
            "quadped": QuadpedEnv,
        }

        self.current_embodiment = None
        self.active_env = None
        self.switch_embodiment(starting_embodiment)

    def switch_embodiment(self, embodiment_name: str):
        if embodiment_name not in self.registry:
            raise ValueError(f"Unknown embodiment: {embodiment_name}")

        if self.current_embodiment == embodiment_name and self.active_env is not None:
            return

        if self.active_env is not None:
            self.active_env.close()

        self.current_embodiment = embodiment_name
        self.active_env = self.registry[embodiment_name](render_mode=self.render_mode)

        self.action_space = self.active_env.action_space

    def step(self, action):
        return self.active_env.step(action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.active_env.reset(seed=seed, options=options)

    def render(self):
        return self.active_env.render()

    def close(self):
        if self.active_env is not None:
            self.active_env.close()
            self.active_env = None
