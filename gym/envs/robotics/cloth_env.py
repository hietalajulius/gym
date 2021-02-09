# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#

import numpy as np

from gym.envs.robotics import rotations, cloth_robot_env, utils, reward_calculation
import cv2
import copy
from PIL import Image
import mujoco_py
import os
from gym import error, spaces


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ClothEnv(cloth_robot_env.ClothRobotEnv):
    """Cloth envs
    """

    def __init__(
        self,
        model_path,
        constraints,
        n_substeps,
        template_kwargs={},
        debug_render_success=False,
        velocity_in_obs=True,
        pixels=False,
        random_seed=1,
        goal_noise_range=(0, 0.05),
        sparse_dense=False,
        sparse_dense_max_steps=10,
        max_advance=0.05,
        randomize_params=False,
        randomize_geoms=False,
        image_size=84,
        uniform_jnt_tend=True
    ):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)

        self.goal_noise_range = goal_noise_range
        self.velocity_in_obs = velocity_in_obs
        self.image_size = image_size
        self.site_names = ["S0_0", "S4_0", "S8_0", "S0_4",
                           "S0_8", "S4_8", "S8_8", "S8_4", 'robot']
        self.constraints = constraints
        self.single_goal_dim = 3  # 3D Positions only
        self.reward_function = reward_calculation.get_reward_function(
            self.constraints, self.single_goal_dim, sparse_dense)

        action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        super(ClothEnv, self).__init__(
            action_space=action_space,
            debug_render_success=debug_render_success,
            sparse_dense=sparse_dense,
            sparse_dense_max_steps=sparse_dense_max_steps,
            model=model,
            n_substeps=n_substeps,
            randomize_params=randomize_params,
            randomize_geoms=randomize_geoms,
            uniform_jnt_tend=uniform_jnt_tend,
            pixels=pixels,
            max_advance=max_advance,
            random_seed=random_seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_function(achieved_goal, desired_goal, info)

    def _set_action(self, action):
        action = action.copy()
        pos_ctrl = action[:3]
        pos_ctrl *= self.max_advance
        utils.mocap_set_action_cloth(
            self.sim, pos_ctrl, self.minimum, self.maximum)

    def _get_obs(self):
        achieved_goal = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy()

        pos = np.array([self.sim.data.get_site_xpos(site).copy()
                        for site in self.site_names]).flatten()
        robot_pos = self.sim.data.get_site_xpos('robot').copy()

        if self.velocity_in_obs:
            dt = self.n_substeps * self.sim.model.opt.timestep

            vel = np.array([self.sim.data.get_site_xvelp(site).copy()
                            for site in self.site_names]).flatten() * dt
            robot_vel = self.sim.data.get_site_xvelp('robot').copy() * dt
            obs = np.concatenate([pos, vel])
            robot_obs = np.concatenate([robot_pos, robot_vel])
        else:
            obs = pos
            robot_obs = robot_pos

        if self.randomize_geoms and self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        elif self.randomize_params:
            model_params = np.array([self.current_joint_stiffness, self.current_joint_damping,
                                     self.current_tendon_stiffness, self.current_tendon_damping])
        else:
            model_params = np.array([0])

        observation = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation': robot_obs
        }

        if self.pixels:
            image_obs = copy.deepcopy(self.render(
                width=self.image_size, height=self.image_size, mode='rgb_array'))

            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)

            observation['image'] = (image_obs / 255).flatten()
        return observation

    def _viewer_setup(self):
        lookat = self.origin

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 0.37
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -90.

    def _env_setup(self):
        utils.reset_mocap_welds(self.sim)
        utils.reset_mocap2body_xpos(self.sim)
        lim1_id = self.sim.model.site_name2id('limit0')
        lim2_id = self.sim.model.site_name2id('limit1')
        lim3_id = self.sim.model.site_name2id('limit2')
        lim4_id = self.sim.model.site_name2id('limit3')

        self.sim.model.site_pos[lim1_id] = self.origin + \
            np.array([-self.maxdist, -self.maxdist, 0])
        self.sim.model.site_pos[lim2_id] = self.origin + \
            np.array([self.maxdist, -self.maxdist, 0])
        self.sim.model.site_pos[lim3_id] = self.origin + \
            np.array([-self.maxdist, self.maxdist, 0])
        self.sim.model.site_pos[lim4_id] = self.origin + \
            np.array([self.maxdist, self.maxdist, 0])

        for _ in range(10):
            self._take_substeps()

    def _reset_view(self):
        targets = ['target0', 'target1']
        next_target = 0
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            origin = constraint['origin']
            if not target == origin:
                site_id = self.sim.model.site_name2id(targets[next_target])
                self.sim.model.site_pos[site_id] = self.goal[i *
                                                             self.single_goal_dim:(i+1)*self.single_goal_dim]
                next_target += 1

        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        utils.set_mocap_position(self.sim, self.mocap_beginning)
        self.sim.forward()

    def _sample_goal(self):
        goal = np.zeros(self.single_goal_dim*len(self.constraints))
        noise = np.random.uniform(self.goal_noise_range[0],
                                  self.goal_noise_range[1])
        for i, constraint in enumerate(self.constraints):
            target = constraint['target']
            target_pos = self.sim.data.get_site_xpos(
                target).copy()

            offset = np.zeros(self.single_goal_dim)
            if 'noise_directions' in constraint.keys():
                for idx, offset_dir in enumerate(constraint['noise_directions']):
                    offset[idx] = offset_dir*noise

            goal[i*self.single_goal_dim: (i+1) *
                 self.single_goal_dim] = target_pos + offset

        return goal.copy()

    def render(self, mode='human', width=500, height=500, image_capture=False, filename=None):
        return super(ClothEnv, self).render(mode, width, height, image_capture, filename)
