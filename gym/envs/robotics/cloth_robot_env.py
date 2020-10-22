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
    def __init__(self, model_path, n_actions, n_substeps, learn_grasp, randomize_params, uniform_jnt_tend, pixels, max_advance):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))
        
        assert (n_actions == 3 and not learn_grasp) or (n_actions == 4 and learn_grasp)

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=1)
        self.pixels = pixels
        self.max_advance = max_advance
        self.n_substeps = n_substeps
        self.learn_grasp = learn_grasp
        self.randomize_params = randomize_params
        self.uniform_jnt_tend = uniform_jnt_tend
        if self.learn_grasp:
            utils.remove_mocap_welds(self.sim)
            self.grasp_is_active = False

        self.viewer = None
        self.key_callback_function = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.origin = np.array([0.12,0.12,0])
        self.maxdist = 0.15
        self.maximum = self.origin[0] + self.maxdist #What is this
        self.minimum = self.origin[0] - self.maxdist #What is this

        self.min_damping = 0.0001
        self.max_damping = 0.2
        self.min_stiffness = 0.0001
        self.max_stiffness = 1
        self.current_joint_stiffness = 0.1
        self.current_joint_damping = 0.1
        self.current_tendon_stiffness = 0.1
        self.current_tendon_damping = 0.1
        self.set_joint_tendon_params()
        
        #Adjust params before this
        self.seed()
        self._env_setup()
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()
        self.mocap_beginning = self.sim.data.get_site_xpos('S8_0').copy()
        

        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')

        if self.pixels:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                robot_observation=spaces.Box(-np.inf, np.inf, shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=spaces.Box(-np.inf, np.inf, shape=obs['model_params'].shape, dtype='float32'),
                image=spaces.Box(-np.inf, np.inf, shape=obs['image'].shape, dtype='float32')
            ))
        else:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
                observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
                robot_observation=spaces.Box(-np.inf, np.inf, shape=obs['robot_observation'].shape, dtype='float32'),
                model_params=spaces.Box(-np.inf, np.inf, shape=obs['model_params'].shape, dtype='float32')
            ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.n_substeps

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
        pos_ctrl *= self.max_advance

        if self.learn_grasp:
            grasp = action[3]
            if grasp > 0:
                if not self.grasp_is_active:
                    body_name, dist = utils.get_closest_body_to_mocap(self.sim)
                    if dist < 0.023:
                        utils.grasp(self.sim, body_name)
                        self.grasp_is_active = True
            else:
                utils.remove_mocap_welds(self.sim)
                self.grasp_is_active = False

        utils.mocap_set_action_cloth(self.sim, pos_ctrl, self.minimum, self.maximum)

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

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)

        done = False
        if info['is_success']:
            print("Real sim success", reward, info)
            done = True
        return obs, reward, done, info

    def set_joint_tendon_params(self):
        for _, joint_name in enumerate(self.sim.model.joint_names):
            joint_id = self.sim.model.joint_name2id(joint_name)
            self.sim.model.jnt_stiffness[joint_id] = self.current_joint_stiffness
            self.sim.model.dof_damping[joint_id] = self.current_joint_damping

        for _, tendon_name in enumerate(self.sim.model.tendon_names):
            tendon_id = self.sim.model.tendon_name2id(tendon_name)
            self.sim.model.tendon_stiffness[tendon_id] = self.current_tendon_stiffness
            self.sim.model.tendon_damping[tendon_id] = self.current_tendon_damping


    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        #print("dofs", self.sim.model.dof_jntid)
        #print("self sim", dir(self.sim.model))
        #print("geoms", self.sim.model.geom_names)
        '''
        for geom_name in self.sim.model.geom_names:
            if "G" in geom_name:
                geom_id = self.sim.model.geom_name2id(geom_name)
                self.sim.model.geom_size[geom_id] += 0.001
        
        '''
        '''
        for i, joint_name in enumerate(self.sim.model.joint_names):
            if i == 1:
                break
            joint_id = self.sim.model.joint_name2id(joint_name)
            print("joint vals: ", self.sim.model.jnt_stiffness[joint_id], self.sim.model.dof_damping[joint_id])
        for i, tendon_name in enumerate(self.sim.model.tendon_names):
            if i == 1:
                break
            tendon_id = self.sim.model.tendon_name2id(tendon_name)
            print("tendon vals: ", self.sim.model.tendon_stiffness[tendon_id], self.sim.model.tendon_damping[tendon_id])
        '''
        

        self._reset_sim()
        self.goal = self._sample_goal().copy() #Sample goal only after reset
        self._reset_view() #Set goal sites based on sampled goal

        if self.randomize_params:
            self.current_joint_stiffness = np.random.uniform(self.min_stiffness, self.max_stiffness)
            self.current_joint_damping = np.random.uniform(self.min_damping, self.max_damping)
            if self.uniform_jnt_tend:
                self.current_tendon_stiffness = self.current_joint_stiffness 
                self.current_tendon_damping = self.current_joint_damping
            else:
                self.current_tendon_stiffness = np.random.uniform(self.min_stiffness, self.max_stiffness)
                self.current_tendon_damping = np.random.uniform(self.min_damping, self.max_damping)

            self.set_joint_tendon_params()
            self.sim.forward()


        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
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
        utils.reset_mocap2body_xpos(self.sim)
        lim1_id = self.sim.model.site_name2id('limit0')
        lim2_id = self.sim.model.site_name2id('limit1')
        lim3_id = self.sim.model.site_name2id('limit2')
        lim4_id = self.sim.model.site_name2id('limit3')

        self.sim.model.site_pos[lim1_id] = self.origin + np.array([-self.maxdist,-self.maxdist,0])
        self.sim.model.site_pos[lim2_id] = self.origin + np.array([self.maxdist,-self.maxdist,0])
        self.sim.model.site_pos[lim3_id] = self.origin + np.array([-self.maxdist,self.maxdist,0])
        self.sim.model.site_pos[lim4_id] = self.origin + np.array([self.maxdist,self.maxdist,0])
        for _ in range(10):
            self._take_substeps()

    def _reset_sim(self):
        if self.learn_grasp:
            self.grasp_is_active = False
            utils.remove_mocap_welds(self.sim)

        self.sim.set_state(self.initial_state)

        if self.learn_grasp:
            mocap_beginning = self.mocap_beginning + np.random.randint(-10,10,3)/300
        else:
            mocap_beginning = self.mocap_beginning

        utils.set_mocap_position(self.sim, mocap_beginning)
        self.sim.forward()

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
