"""Rule-based expert that follows the target humanoid in EVT-Bench, used to collect training episodes.

Per episode it writes, under ``<save_path>/<scene>/`` (success) or ``<save_path>_failed/<scene>/``:

* ``<id>.mp4``        forward RGB (``agent_1_articulated_agent_jaw_rgb``)
* ``<id>_info.json``  per-step robot / human pose, GT target box, base velocity
* ``<id>.json``       episode result and instruction

Control: A* path to the human, PID/PD tracking of a lookahead point at ``desired_dist``
behind the human; chase the next path point on large turns; back off when the human
walks towards the robot.
"""
from __future__ import annotations

import json
import math
import os
import os.path as osp
import warnings

import habitat
import habitat_sim
import imageio
import magnum as mn
import numpy as np
from habitat.config import read_write
from habitat_sim.gfx import LightInfo, LightPositionModel
from tqdm import trange

warnings.filterwarnings("ignore")


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def matrix_to_yaw(T: mn.Matrix4) -> float:
    x_axis_dir = T[0]
    return math.atan2(x_axis_dir.x, x_axis_dir.z)


def is_semi_circle_shape(path_points):
    """``arc / chord`` of the A* path; a ratio in (1.1, 1.8) is a large turn."""
    if len(path_points) < 4:
        return 0.0, False
    start = np.array(path_points[0])[[0, 2]]
    end = np.array(path_points[-1])[[0, 2]]
    chord_dist = np.linalg.norm(end - start)
    if chord_dist < 0.5:
        return 0.0, False
    arc_length = 0
    for i in range(len(path_points) - 1):
        arc_length += np.linalg.norm(np.array(path_points[i + 1])[[0, 2]] - np.array(path_points[i])[[0, 2]])
    ratio = arc_length / chord_dist
    return ratio, 1.1 < ratio < 1.8


def is_human_towards_robot(robot_zx, human_zx, human_yaw, cos_thresh=0.5, max_lateral=1.2, max_dist=5.0) -> bool:
    dir_h = np.array([np.cos(human_yaw), np.sin(human_yaw)], dtype=np.float32)
    vec_hr = np.array([robot_zx[0] - human_zx[0], robot_zx[1] - human_zx[1]], dtype=np.float32)
    dist = float(np.linalg.norm(vec_hr))
    if dist < 1e-6:
        return True
    if dist > max_dist:
        return False
    cos_theta = float(np.dot(dir_h, vec_hr / dist))
    lateral = float(abs(dir_h[0] * vec_hr[1] - dir_h[1] * vec_hr[0]))
    return (cos_theta > cos_thresh) and (lateral < max_lateral)


def densify_path(points, resolution=0.2):
    if len(points) < 2:
        return points
    dense_points = [np.array(points[0], dtype=np.float32)]
    for p0, p1 in zip(points[:-1], points[1:]):
        p0 = np.array(p0, dtype=np.float32)
        p1 = np.array(p1, dtype=np.float32)
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-6:
            continue
        direction = seg_vec / seg_len
        for i in range(1, int(np.floor(seg_len / resolution)) + 1):
            dense_points.append(p0 + direction * (i * resolution))
        dense_points.append(p1)
    return dense_points


def goal_behind(target_zx, robot_zx, distance):
    """``(z, x, yaw)`` column at ``distance`` from the target on the robot side, facing the target."""
    target_zx = np.array(target_zx).squeeze()
    robot_zx = np.array(robot_zx).squeeze()
    alpha = np.arctan2(target_zx[1] - robot_zx[1], target_zx[0] - robot_zx[0])
    return np.array([target_zx[0] - distance * np.cos(alpha), target_zx[1] - distance * np.sin(alpha), alpha])[:, np.newaxis]


class Expert:
    def __init__(self, result_path: str):
        self.result_path = result_path
        os.makedirs(self.result_path, exist_ok=True)
        self.rgb_list = []

        self.path_res = 0.1
        self.desired_dist = 1.2
        self.dt = 0.36

        # linear PID on distance, angular PD on heading + cross-track
        self.kp_v, self.ki_v, self.kd_v = 2.0, 0.0, 1.0
        self.kp_w, self.kd_w = 9.0, 0.25
        self.k_ct = 2.5
        self.v_max, self.v_min, self.w_max = 2.5, 0.0, 3.14
        self.lookahead = 0.8

        self.max_accel_v = 1.0
        self.max_accel_v_backward = 1.2
        self.max_accel_w = 3.0
        self.safety_dist_min = 0.8
        self.safety_dist_max = 2.5
        self.reset()

    def reset(self, episode=None, success: bool = False):
        if self.rgb_list and episode is not None:
            scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
            save_dir = os.path.join(self.result_path if success else self.result_path + "_failed", scene_key)
            os.makedirs(save_dir, exist_ok=True)
            imageio.mimsave(os.path.join(save_dir, f"{episode.episode_id}.mp4"), self.rgb_list)
            self.rgb_list = []
        self.path = None
        self.found = False
        self.goal_pose = None
        self.robot_zx = self.human_zx = self.robot_yaw = self.human_yaw = None
        self._int_v = 0.0
        self._prev_v = 0.0
        self._prev_w = 0.0
        self._prev_ev = 0.0
        self._prev_ew = 0.0

    def _select_lookahead_point(self, goal_traj, robot_zx):
        if goal_traj is None or len(goal_traj) == 0:
            return None
        traj = np.concatenate(goal_traj, axis=1).T
        nearest = int(np.argmin(np.linalg.norm(traj[:, :2] - robot_zx[None, :], axis=1)))
        idx = min(nearest + max(1, int(self.lookahead / max(self.path_res, 1e-6))), len(goal_traj) - 1)
        return np.array([float(traj[idx, 0]), float(traj[idx, 1]), float(traj[idx, 2])], dtype=np.float32)

    def _pid_pd_track(self, robot_zx, robot_yaw, target_zxyaw, direction=+1.0):
        dz = target_zxyaw[0] - robot_zx[0]
        dx = target_zxyaw[1] - robot_zx[1]
        dist = float(np.hypot(dz, dx))
        e_yaw = wrap_angle(target_zxyaw[2] - robot_yaw)
        ct = -np.sin(robot_yaw) * dz + np.cos(robot_yaw) * dx

        ev = dist
        self._int_v += ev * self.dt
        self._int_v = float(np.clip(self._int_v, -2.0, 2.0))
        dev = (ev - self._prev_ev) / max(self.dt, 1e-6)
        self._prev_ev = ev
        v = self.kp_v * ev + self.ki_v * self._int_v + self.kd_v * dev
        v_x = direction * float(np.clip(abs(v), self.v_min, self.v_max))

        dew = (e_yaw - self._prev_ew) / max(self.dt, 1e-6)
        self._prev_ew = e_yaw
        w = self.kp_w * e_yaw + self.kd_w * dew + self.k_ct * ct
        return v_x, float(np.clip(w, -self.w_max, self.w_max))

    def _slew_rate_limit(self, target_val, prev_val, max_accel):
        max_step = max_accel * self.dt
        return float(prev_val + np.clip(target_val - prev_val, -max_step, max_step))

    def _get_adaptive_v(self, raw_v, raw_w, dist_hr, state):
        """Scale speed by distance to the human, then rate-limit v and w."""
        dist_factor = np.clip(
            (dist_hr - self.safety_dist_min) / (self.safety_dist_max - self.safety_dist_min), 0.0, 1.0)
        if state == "FORWARD":
            v_target = raw_v * (0.2 + 0.8 * dist_factor)
        elif state == "CHASING":
            v_target = raw_v * (0.3 + 0.7 * dist_factor)
        else:  # BACKWARD
            v_target = raw_v * (0.2 + 0.8 * (1 - dist_factor))

        if state == "BACKWARD":
            v_smoothed = -self._slew_rate_limit(abs(v_target), abs(self._prev_v), self.max_accel_v_backward)
        else:
            v_smoothed = self._slew_rate_limit(abs(v_target), abs(self._prev_v), self.max_accel_v)
        w_smoothed = self._slew_rate_limit(raw_w, self._prev_w, self.max_accel_w)
        self._prev_v = v_smoothed
        self._prev_w = w_smoothed
        return v_smoothed, w_smoothed

    @staticmethod
    def _get_path_length(points):
        if len(points) < 2:
            return 0.0
        return sum(
            np.linalg.norm(np.array([points[i + 1][2], points[i + 1][0]]) - np.array([points[i][2], points[i][0]]))
            for i in range(len(points) - 1)
        )

    def act(self, sim, robot_pos, robot_rot, human_pos, human_rot, observations):
        """Returns body-frame ``[v_x, v_y, w]``."""
        self.rgb_list.append(observations["agent_1_articulated_agent_jaw_rgb"][:, :, :3])

        self.robot_zx = np.array([robot_pos[2], robot_pos[0]])
        self.human_zx = np.array([human_pos[2], human_pos[0]])
        self.robot_yaw = robot_rot
        self.human_yaw = human_rot
        dist_hr = np.linalg.norm(self.human_zx - self.robot_zx)

        goal_traj = []
        self.path = habitat_sim.ShortestPath()
        self.path.requested_start = robot_pos
        self.path.requested_end = human_pos
        self.found = sim.pathfinder.find_path(self.path)
        _, is_large_turn = is_semi_circle_shape(self.path.points)
        is_towards = is_human_towards_robot(
            self.robot_zx, self.human_zx, self.human_yaw, cos_thresh=0.7, max_lateral=1.1, max_dist=5.0)
        path_length = self._get_path_length(self.path.points)
        should_backward = False

        if self.found and is_large_turn and path_length > self.desired_dist:
            # Large turn: chase the next A* waypoint.
            state = "CHASING"
            goal_zx = np.array([self.path.points[2][2], self.path.points[2][0]])
            next_goal_zx = np.array([self.path.points[3][2], self.path.points[3][0]])
            next_goal_yaw = np.arctan2(next_goal_zx[1] - goal_zx[1], next_goal_zx[0] - goal_zx[0])
            robot_goal_yaw = np.arctan2(goal_zx[1] - self.robot_zx[1], goal_zx[0] - self.robot_zx[0])
            target = np.array([goal_zx[0], goal_zx[1], wrap_angle((robot_goal_yaw + next_goal_yaw) / 2.0)])
            self.goal_pose = target[:, np.newaxis]
            v_raw, w_raw = self._pid_pd_track(self.robot_zx, self.robot_yaw, target, direction=+1.0)
            v_x, w = self._get_adaptive_v(v_raw, w_raw, dist_hr, state)
            v_y = 0.0

        elif not is_towards:
            state = "FORWARD"
            closed_to_goal = False
            target = None
            if not self.found:
                closed_to_goal = True
            else:
                dense_path = densify_path(self.path.points, resolution=self.path_res)
                desired_index = 0
                for index, candidate_p in enumerate(dense_path[::-1]):
                    dist_to_goal = np.linalg.norm(np.array(candidate_p)[[2, 0]] - self.human_zx)
                    if self.desired_dist < dist_to_goal < self.desired_dist + 1.0:
                        desired_index = len(dense_path) - index
                        break
                for p in dense_path[:desired_index]:
                    goal_p = np.array([p[2], p[0]])
                    goal_yaw = np.arctan2(self.human_zx[1] - goal_p[1], self.human_zx[0] - goal_p[0])
                    goal_traj.append(np.array([goal_p[0], goal_p[1], goal_yaw])[:, np.newaxis])
                target = self._select_lookahead_point(goal_traj, self.robot_zx)
            if target is None:
                closed_to_goal = True
                goal_yaw = np.arctan2(self.human_zx[1] - self.robot_zx[1], self.human_zx[0] - self.robot_zx[0])
                target = np.array([self.robot_zx[0], self.robot_zx[1], goal_yaw])
            self.goal_pose = target[:, np.newaxis]
            if closed_to_goal and dist_hr < 1.0:
                should_backward = True
            else:
                v_raw, w_raw = self._pid_pd_track(self.robot_zx, self.robot_yaw, target, direction=+1.0)
                v_x, w = self._get_adaptive_v(v_raw, w_raw, dist_hr, state)
                v_y = 0.0

        if is_towards or should_backward:
            state = "BACKWARD"
            self.path = habitat_sim.ShortestPath()
            self.path.requested_start = robot_pos
            end_goal = goal_behind(self.human_zx, self.robot_zx, self.desired_dist)
            self.path.requested_end = np.array([end_goal[1, 0], robot_pos[1], end_goal[0, 0]])
            self.found = sim.pathfinder.find_path(self.path)
            if not self.found or dist_hr > self.desired_dist + 1.0:
                goal_yaw = np.arctan2(self.human_zx[1] - self.robot_zx[1], self.human_zx[0] - self.robot_zx[0])
                target = np.array([self.robot_zx[0], self.robot_zx[1], goal_yaw])
            else:
                target = np.array([end_goal[0, 0], end_goal[1, 0], end_goal[2, 0]])
            self.goal_pose = target[:, np.newaxis]
            v_raw, w_raw = self._pid_pd_track(self.robot_zx, self.robot_yaw, target, direction=-1.0)
            v_x, w = self._get_adaptive_v(v_raw, w_raw, dist_hr, state)
            v_y = 0.0

        return [v_x, v_y, w]


_LIGHTS = [
    LightInfo(vector=v, color=[1.0, 1.0, 1.0], model=LightPositionModel.Global)
    for v in ([10.0, -2.0, 0.0, 0.0], [-10.0, -2.0, 0.0, 0.0], [0.0, -2.0, 10.0, 0.0], [0.0, -2.0, -10.0, 0.0])
]


def _episode_paths(save_path: str, episode):
    scene_key = osp.splitext(osp.basename(episode.scene_id))[0].split(".")[0]
    return scene_key, [os.path.join(d, scene_key) for d in (save_path, save_path + "_failed")]


def collect_episodes(config, dataset_split, save_path: str) -> None:
    expert = Expert(save_path)
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        sim = env.sim
        for _ in trange(len(env.episodes)):
            env.reset()
            sim.set_light_setup(_LIGHTS)
            ep = env.current_episode
            _, dirs = _episode_paths(save_path, ep)
            if any(os.path.exists(os.path.join(d, f"{ep.episode_id}_info.json")) for d in dirs):
                continue
            instruction = ep.info.get("instruction", None)

            human = sim.agents_mgr[0].articulated_agent
            robot = sim.agents_mgr[1].articulated_agent
            iter_step = followed_step = too_far_count = 0
            status = "Normal"
            finished = False
            records = []
            while not env.episode_over:
                human_pos = human.base_pos
                robot_pos = robot.base_pos
                robot_rot = matrix_to_yaw(robot.base_transformation)
                human_rot = matrix_to_yaw(human.base_transformation)
                obs = sim.get_sensor_observations()
                detector = env.task._get_observations(ep)
                action = expert.act(sim, robot_pos, robot_rot, human_pos, human_rot, obs)
                env.step({
                    "action": (
                        "agent_0_humanoid_navigate_action", "agent_1_base_velocity",
                        "agent_2_oracle_nav_randcoord_action_obstacle", "agent_3_oracle_nav_randcoord_action_obstacle",
                        "agent_4_oracle_nav_randcoord_action_obstacle", "agent_5_oracle_nav_randcoord_action_obstacle",
                    ),
                    "action_args": {"agent_1_base_vel": action},
                })
                iter_step += 1

                info = env.get_metrics()
                if info["human_following"] == 1.0:
                    followed_step += 1
                    too_far_count = 0
                if np.linalg.norm(robot.base_pos - human.base_pos) > 4.0:
                    too_far_count += 1
                    if too_far_count > 20:
                        status = "Lost"
                        break
                records.append({
                    "step": iter_step,
                    "human_pos": list(human_pos),
                    "human_yaw": human_rot,
                    "robot_pos": list(robot_pos),
                    "robot_yaw": robot_rot,
                    "human_bbox": detector["agent_1_main_humanoid_detector_sensor"]["box"].tolist(),
                    "dis_to_human": float(np.linalg.norm(robot.base_pos - human.base_pos)),
                    "facing": info["human_following"],
                    "base_velocity": action,
                })
                if info["human_collision"] == 1.0:
                    status = "Collision"
                    break

            info = env.get_metrics()
            if env.episode_over:
                finished = True
            result = {
                "finish": finished,
                "status": status,
                "success": (info["human_following_success"] and info["human_following"]) if iter_step < 300
                           else info["human_following"],
                "following_rate": followed_step / iter_step,
                "following_step": followed_step,
                "total_step": iter_step,
                "collision": info["human_collision"],
            }
            if instruction is not None:
                result["instruction"] = instruction

            save_dir = dirs[0] if result["success"] else dirs[1]
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"{ep.episode_id}_info.json"), "w") as f:
                json.dump(records, f, indent=2)
            with open(os.path.join(save_dir, f"{ep.episode_id}.json"), "w") as f:
                json.dump(result, f, indent=2)
            expert.reset(ep, success=result["success"])


def chunk_instruction(config, dataset_split) -> str:
    """Instruction of the first episode ``TrackEnv`` visits for this chunk (no rendering).

    The released checkpoint was trained with this instruction written to every episode of
    the chunk; ``collect_expert.py`` reproduces that labeling.
    """
    with read_write(config):
        config.habitat.simulator.create_renderer = False
        config.habitat.gym.obs_keys = []
        for agent_name in config.habitat.simulator.agents:
            config.habitat.simulator.agents[agent_name].sim_sensors = {}
        config.habitat.task.lab_sensors = {}
        config.habitat.task.measurements = {}
    with habitat.TrackEnv(config=config, dataset=dataset_split) as env:
        env.reset()
        return env.current_episode.info.get("instruction", None)


def write_chunk_instruction(dataset_split, save_path: str, instruction: str) -> int:
    n = 0
    for ep in dataset_split.episodes:
        _, dirs = _episode_paths(save_path, ep)
        for d in dirs:
            path = os.path.join(d, f"{ep.episode_id}.json")
            if os.path.exists(path):
                with open(path) as f:
                    result = json.load(f)
                result["instruction"] = instruction
                with open(path, "w") as f:
                    json.dump(result, f, indent=2)
                n += 1
                break
    return n
