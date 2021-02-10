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
import cv2
import time

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500


class ClothRobotEnv(gym.GoalEnv):
    def __init__(self, action_space, model, sparse_dense, sparse_dense_max_steps, n_substeps, randomize_params, uniform_jnt_tend, randomize_geoms, pixels, max_advance, random_seed, debug_render_success):

        self.action_space = action_space
        self.previous_action = np.zeros(action_space.shape[0])

        self.sim = mujoco_py.MjSim(model, nsubsteps=1)
        self.pixels = pixels
        self.max_advance = max_advance
        self.n_substeps = n_substeps

        self.randomize_params = randomize_params
        self.randomize_geoms = randomize_geoms
        self.uniform_jnt_tend = uniform_jnt_tend

        self.viewer = None
        self.key_callback_function = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.origin = np.array([0.12, 0.12, 0])
        self.maxdist = 0.13

        # TODO: enable dynamic size and orientation config
        self.maximum = self.origin[0] + self.maxdist
        self.minimum = self.origin[0] - self.maxdist
        self.min_damping = 0.00001  # TODO: pass ranges in from outside
        self.max_damping = 0.02

        self.min_stiffness = 0.00001  # TODO: pass ranges in from outside
        self.max_stiffness = 0.02

        self.min_geom_size = 0.005  # TODO: pass ranges in from outside
        self.max_geom_size = 0.011
        self.current_geom_size = self.min_geom_size

        self.current_joint_stiffness = self.min_stiffness
        self.current_joint_damping = self.min_damping

        self.current_tendon_stiffness = self.min_stiffness
        self.current_tendon_damping = self.min_damping

        self.sparse_dense = sparse_dense

        self.sparse_dense_max_steps = sparse_dense_max_steps
        self.sparse_dense_steps = 0

        self.debug_render_success = debug_render_success

        if self.randomize_params:
            self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                         self.current_tendon_stiffness, self.current_tendon_damping)
        if self.randomize_geoms:
            self.set_geom_params()

        # Adjust params before this
        self.seed(random_seed)
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()
        self.mocap_beginning = self.sim.data.get_site_xpos('S8_0').copy()

        obs = self._get_obs()

        if self.pixels:
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

    def set_aux_positions(self, corner1, corner2, corner3, corner4):
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner1, label="corner1")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner2, label="corner2")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner3, label="corner3")
        self.viewer.add_marker(size=np.array(
            [.001, .001, .001]), pos=corner4, label="corner4")

    def clear_aux_positions(self):
        del self.viewer._markers[:]

    def set_key_callback_function(self, key_callback_function):
        self.key_callback_function = key_callback_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_action(self, action):
        raise NotImplementedError()

    def _take_substeps(self):
        for _ in range(self.n_substeps):
            self.sim.step()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.array(action)

        self._set_action(action)
        self._take_substeps()
        self._step_callback()
        obs = self._get_obs()
        # Set previous action only after current obs returned
        self.previous_action = action
        reward = self.compute_reward(np.reshape(
            obs['achieved_goal'], (1, -1)), np.reshape(self.goal, (1, -1)), dict(real_sim=True))[0]


        info = {"task_reward": reward,
                "velocity_penalty": 0.0,
                "position_penalty": 0.0,
                "acceleration_penalty": 0.0,
                "velocity_over_limit": 0.0,
                "position_over_limit": 0.0,
                "acceleration_over_limit": 0,
                "control_penalty": 0.0,
                'is_success': not reward < 0}

        done = False

        if info['is_success']:
            if self.sparse_dense:
                self.sparse_dense_steps += 1

            if not self.sparse_dense or self.sparse_dense_steps >= self.sparse_dense_max_steps:
                done = True

            if self.sparse_dense_steps == 1 or done:
                print("Real sim ep success", reward,
                      info, self.sparse_dense_steps, self.current_joint_damping, self.current_joint_stiffness)

            if self.debug_render_success:
                self.render(mode='rgb_array', height=500, width=500, image_capture=True,
                            filename='success_images/' + str(time.time()).replace(".", "-") + '.png')

        return obs, reward, done, info

    def set_geom_params(self):
        for geom_name in self.sim.model.geom_names:
            if "G" in geom_name:
                geom_id = self.sim.model.geom_name2id(geom_name)
                self.sim.model.geom_size[geom_id] = self.current_geom_size * \
                    (1 + np.random.randn()*0.01)  # TODO: Figure out if this makes sense

        self.sim.forward()

    def set_joint_tendon_params(self, joint_stiffness, joint_damping, tendon_stiffness, tendon_damping):
        for _, joint_name in enumerate(self.sim.model.joint_names):
            joint_id = self.sim.model.joint_name2id(joint_name)
            self.sim.model.jnt_stiffness[joint_id] = joint_stiffness
            self.sim.model.dof_damping[joint_id] = joint_damping

        for _, tendon_name in enumerate(self.sim.model.tendon_names):
            tendon_id = self.sim.model.tendon_name2id(tendon_name)
            self.sim.model.tendon_stiffness[tendon_id] = tendon_stiffness
            self.sim.model.tendon_damping[tendon_id] = tendon_damping

        self.sim.forward()

    def reset(self):
        self.sparse_dense_steps = 0
        self._reset_sim()
        self.goal = self._sample_goal().copy()  # Sample goal only after reset
        self._reset_view()  # Set goal sites based on sampled goal

        if self.randomize_params:
            self.current_joint_stiffness = self.np_random.uniform(
                self.min_stiffness, self.max_stiffness)
            self.current_joint_damping = self.np_random.uniform(
                self.min_damping, self.max_damping)

            if self.uniform_jnt_tend:
                self.current_tendon_stiffness = self.current_joint_stiffness
                self.current_tendon_damping = self.current_joint_damping
            else:
                # Own damping/stiffness for tendons
                self.current_tendon_stiffness = self.np_random.uniform(
                    self.min_stiffness, self.max_stiffness)
                self.current_tendon_damping = self.np_random.uniform(
                    self.min_damping, self.max_damping)

            self.set_joint_tendon_params(self.current_joint_stiffness, self.current_joint_damping,
                                         self.current_tendon_stiffness, self.current_tendon_damping)

        if self.randomize_geoms:
            self.current_geom_size = self.np_random.uniform(
                self.min_geom_size, self.max_geom_size)
            self.set_geom_params()

        self.previous_action = np.zeros(self.action_space.shape[0])
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, image_capture=False, filename=None):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # original image is upside-down, so flip it
            image_obs = data[::-1, :, :]
            if image_capture and not filename is None:
                image_obs_save = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(filename, image_obs_save)
            return image_obs
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
        raise NotImplementedError()

    def _reset_sim(self):
        raise NotImplementedError()

    def _reset_view(self):
        raise NotImplementedError()

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
