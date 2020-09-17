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

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class ClothRobotEnv(gym.GoalEnv):
    def __init__(self, model_path, n_actions, n_substeps):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        utils.remove_mocap_welds(self.sim)
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()
        self.key_callback_function = None

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')

        if 'image' in obs.keys():
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                image=spaces.Box(-np.inf, np.inf, shape=obs['image'].shape, dtype='float32')
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32')
            ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def set_aux_positions(self, corner1, corner2, corner3, corner4):
        self.viewer.add_marker(size=np.array([.001, .001, .001]),pos=corner1, label="corner1")
        self.viewer.add_marker(size=np.array([.001, .001, .001]),pos=corner2, label="corner2")
        self.viewer.add_marker(size=np.array([.001, .001, .001]),pos=corner3, label="corner3")
        self.viewer.add_marker(size=np.array([.001, .001, .001]),pos=corner4, label="corner4")

    def clear_aux_positions(self):
        del self.viewer._markers[:]

    def set_key_callback_function(self, key_callback_function):
        self.key_callback_function = key_callback_function

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_action(self, action):
        action = action.copy()  
        pos_ctrl = action[:3]
        pos_ctrl *= 0.05

        grasp = action[3]
        if action[3] > 0:
            if not self.grasp_is_active:
                body_name, dist = utils.get_closest_body_to_mocap(self.sim)
                print("Body and dist", body_name, dist)
                if dist < 0.023:
                    utils.grasp(self.sim, body_name)
                    self.grasp_is_active = True
        else:
            utils.remove_mocap_welds(self.sim)
            self.grasp_is_active = False

        utils.mocap_set_action_cloth(self.sim, pos_ctrl, grasp, self.minimum, self.maximum)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = np.array(action)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        done = False
        if info['is_success']:
            print("Real sim success", reward, info)
            done = True
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(ClothRobotEnv, self).reset()
        self._reset_sim()
        self.goal = self._sample_goal().copy() #Sample goal only after reset
        self._reset_sim() #Reset again to have goals in correct place
        #utils.enable_mocap_welds(self.sim)
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim, self.key_callback_function)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer


    def _env_setup(self):
        utils.reset_mocap_welds(self.sim)
        gripper_target = self.sim.data.get_site_xpos('S8_0')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
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
