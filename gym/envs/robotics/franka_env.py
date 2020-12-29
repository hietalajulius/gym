# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#

import os
import copy
import numpy as np

import gym
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import utils
from gym import utils as gym_utils
import mujoco_py

import math

DEFAULT_SIZE = 500


class FrankaEnv(gym.GoalEnv, gym_utils.EzPickle):
    def __init__(self,  mocap_control=True):
        self.mocap_control = mocap_control
        if self.mocap_control:
            model = mujoco_py.load_model_from_path(
                "../../franka_sim/franka_cloth_mocap.xml")
        else:
            model = mujoco_py.load_model_from_path(
                "../../franka_sim/franka_cloth_ctrl.xml")
        self.sim = mujoco_py.MjSim(model, nsubsteps=1)
        self.n_substeps = 40

        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.velocities = [np.zeros(7)]
        self.accelerations = [np.zeros(7)]
        self.jerks = [np.zeros(7)]

        self.ee_velocities = [np.zeros(3)]
        self.ee_accelerations = [np.zeros(3)]
        self.ee_jerks = [np.zeros(3)]

        self.joint_names = np.array(
            ["panda0_joint" + str(name_idx) for name_idx in range(1, 8)])
        self.vel_sensor_names = np.array(
            [joint_name + "_vel_sensor" for joint_name in self.joint_names])
        self.pos_sensor_names = np.array(
            [joint_name + "_pos_sensor" for joint_name in self.joint_names])
        self.site_names = ["S0_0", "S4_0", "S8_0", "S0_4",
                           "S0_8", "S4_8", "S8_8", "S8_4", 'robot']

        self.end_effector_vel_limit = 1.7
        self.end_effector_acc_limit = 13
        self.end_effector_jerk_limit = 6500

        self.velocity_limits = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
        self.acceleration_limits = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
        self.jerk_limits = np.array(
            [7500, 3750, 5000, 6250, 7500, 10000, 10000])
        self.initial_joint_values = np.array(
            [-0.102, -0.116, -0.364, -2.68, -0.08, 2.58, -0.396])

        self.vel_violations = 0
        self.acc_violations = 0
        self.jerk_violations = 0
        self.substeps_taken = 0

        self.seed()
        utils.disable_mocap_weld(self.sim, "end_effector_body", "B8_0")
        self.initial_EE_position = np.array([0, 0, 0])  # Remove this
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()

        self.initial_EE_position = self.sim.data.get_site_xpos(
            "end_effector_site").copy()

        self.key_callback_function = None
        obs = self._get_obs()

        if self.mocap_control:
            self.action_space = spaces.Box(-1.,
                                           1., shape=(4,), dtype='float32')
        else:
            self.action_space = spaces.Box(-1.,
                                           1., shape=(9,), dtype='float32')

        if 'image' in obs.keys():
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf,
                                        shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf,
                                         shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf,
                                       shape=obs['observation'].shape, dtype='float32'),
                robot_observation=spaces.Box(-np.inf, np.inf,
                                             shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=spaces.Box(-np.inf, np.inf,
                                        shape=obs['model_params'].shape, dtype='float32'),
                image=spaces.Box(-np.inf, np.inf,
                                 shape=obs['image'].shape, dtype='float32')
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf,
                                        shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf,
                                         shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf,
                                       shape=obs['observation'].shape, dtype='float32'),
                robot_observation=spaces.Box(-np.inf, np.inf,
                                             shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=spaces.Box(-np.inf, np.inf,
                                        shape=obs['model_params'].shape, dtype='float32')
            ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.n_substeps

    def set_key_callback_function(self, key_callback_function):
        self.key_callback_function = key_callback_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_action(self, action):
        control = action.copy()[:-1]
        grasp = action.copy()[-1]
        if grasp < 0:
            utils.disable_mocap_weld(self.sim, "end_effector_body", "B8_0")

        if self.mocap_control:
            utils.increase_mocap_position(self.sim, control)
        else:
            utils.franka_ctrl_set_action(self.sim, control)

    def _take_substeps(self, action=None):
        self.substeps_taken += 1
        #print("\nTaking subs ", self.substeps_taken, "Viols", self.vel_violations, self.acc_violations, self.jerk_violations)

        for step in range(1, self.n_substeps+1):
            if not action is None and self.mocap_control:
                linear_action = (action / self.n_substeps)*step
                self._set_action(linear_action)
            elif not action is None:
                self._set_action(action)

            self.sim.step()

            ee_moves = self.initial_EE_position - \
                self.sim.data.get_site_xpos("end_effector_site").copy()

            #print(ee_moves[0] ,ee_moves[1] , ee_moves[2], )

            positions = np.array([self.sim.data.get_sensor(
                pos_sensor_name).copy() for pos_sensor_name in self.pos_sensor_names])

            velocities = np.array([self.sim.data.get_sensor(
                vel_sensor_name) for vel_sensor_name in self.vel_sensor_names])
            accelerations = (velocities - self.velocities[-1]) / 0.001
            jerks = (accelerations - self.accelerations[-1]) / 0.001

            ee_vel = self.sim.data.get_sensor("end_effector_vel_sensor").copy()
            ee_acc = self.sim.data.get_sensor("end_effector_acc_sensor").copy()
            #print("senso", self.sim.data.sensordata)

            #print("sensss", ee_vel, ee_acc, abs(ee_vel-self.ee_velocities[-1])/0.001)

            self.ee_velocities.append(ee_vel)
            self.ee_accelerations.append(ee_acc)

            if np.any(self.velocity_limits < abs(velocities)):
                #print("vels", velocities, self.velocity_limits < abs(velocities), "\n")
                #print("Vel viol at substep ", step, self.joint_names[self.velocity_limits < abs(velocities)], abs(velocities-self.velocity_limits)[self.velocity_limits < abs(velocities)], velocities[self.velocity_limits < abs(velocities)])
                self.vel_violations += 1

            if np.any(self.acceleration_limits < abs(accelerations)):
                #print("accs", accelerations, self.acceleration_limits < abs(accelerations), "\n")
                #print("acc viol at substep ", step, self.joint_names[self.acceleration_limits < abs(accelerations)], abs(accelerations-self.acceleration_limits)[self.acceleration_limits < abs(accelerations)], accelerations[self.acceleration_limits < abs(accelerations)])
                self.acc_violations += 1

            if np.any(self.jerk_limits < abs(jerks)):
                #print("jerks", jerks, self.jerk_limits < abs(jerks), "\n")
                #print("jerk viol at substep ", step, self.joint_names[self.jerk_limits < abs(jerks)], abs(jerks-self.jerk_limits)[self.jerk_limits < abs(jerks)], jerks[self.jerk_limits < abs(jerks)])
                self.jerk_violations += 1

            self.velocities.append(velocities)
            self.accelerations.append(accelerations)
            self.jerks.append(jerks)

        if not action is None and not self.mocap_control:
            diff_vec = action[:-1]-positions
            #print("Substep", step, "diff", np.linalg.norm(diff_vec))

        elif not action is None:
            pass
            #print([pos for pos in positions], ",")

    def step(self, action):
        #action = np.clip(action, self.action_space.low, self.action_space.high)
        #action = np.array(action)
        # self._set_action(action)
        self._take_substeps(action)
        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        done = False
        if info['is_success']:
            done = True
        return obs, reward, done, info

    def reset(self):
        self.velocities = [np.zeros(7)]
        self.accelerations = [np.zeros(7)]
        self.jerks = [np.zeros(7)]

        self.ee_velocities = [np.zeros(7)]
        self.ee_accelerations = [np.zeros(7)]
        self.ee_jerks = [np.zeros(7)]

        self.vel_violations = 0
        self.acc_violations = 0
        self.jerk_violations = 0
        self.substeps_taken = 0

        self._reset_sim()
        self._reset_view()
        obs = self._get_obs()
        return obs

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(
                    self.sim, self.key_callback_function)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _env_setup(self):
        for i, value in enumerate(self.initial_joint_values):
            self.sim.data.set_joint_qpos(self.joint_names[i], value)

        self.sim.forward()
        if self.mocap_control:
            utils.reset_mocap2body_xpos(self.sim)
        else:
            self.sim.data.ctrl[:] = self.initial_joint_values

        utils.enable_mocap_weld(self.sim, "end_effector_body", "B8_0")
        for _ in range(10):
            self._take_substeps()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()

    def _reset_view(self):
        self.sim.forward()

    def _get_obs(self):

        body_0_8 = self.sim.data.get_site_xpos("S0_8").copy()
        body_0_0 = self.sim.data.get_site_xpos("S0_0").copy()
        body_8_8 = self.sim.data.get_site_xpos("S8_8").copy()
        body_8_0 = self.sim.data.get_site_xpos("S8_0").copy()
        achieved_goal = np.concatenate(
            [body_8_8, body_8_0, body_0_8, body_0_0]).flatten()

        pos = np.array([self.sim.data.get_site_xpos(site).copy()
                        for site in self.site_names]).flatten()
        dt = self.n_substeps * self.sim.model.opt.timestep
        vel = np.array([self.sim.data.get_site_xvelp(site).copy()
                        for site in self.site_names]).flatten() * dt

        obs = np.concatenate([pos, vel])

        robot_pos = self.sim.data.get_site_xpos('robot').copy()
        robot_vel = self.sim.data.get_site_xvelp('robot').copy() * dt
        robot_obs = np.concatenate([robot_pos, robot_vel])
        model_params = np.array([0])

        observation = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation': robot_obs
        }

        image_obs = copy.deepcopy(self.render(
            width=84, height=84, mode='rgb_array'))
        observation['image'] = (image_obs / 255).flatten()
        if False:
            print("Image obs", image_obs)
            cv2.imshow('env', image_obs)
            cv2.imshow('env', cv2.cvtColor(image_obs, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        return observation

    def _is_success(self, achieved_goal, desired_goal):
        return False

    def _sample_goal(self):
        return np.zeros(12)

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def compute_reward(self, achieved_goal, goal, info):
        return 1
