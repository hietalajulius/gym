# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#

import numpy as np

from gym.envs.robotics import rotations, cloth_robot_env, utils, reward_calculation
import cv2
import copy
from PIL import Image
import mujoco_py
from panda_gym.template_renderer import TemplateRenderer
from gym import error, spaces


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def construct_model(template_file, **kwargs):
    renderer = TemplateRenderer()
    xml_data = renderer.render_template(template_file, **kwargs)
    model = mujoco_py.load_model_from_xml(xml_data)
    return model


class FrankaEnv(cloth_robot_env.ClothRobotEnv):
    """Cloth envs
    """

    def __init__(
        self,
        model_path,
        constraints,
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
        n_substeps=40,
        uniform_jnt_tend=True
    ):
        self.goal_noise_range = goal_noise_range
        self.velocity_in_obs = velocity_in_obs
        self.image_size = image_size
        self.site_names = ["S0_0", "S4_0", "S8_0", "S0_4",
                           "S0_8", "S4_8", "S8_8", "S8_4", 'panda0_end_effector']
        self.initial_joint_values = np.array(
            [0.148, 0.027, 0, -2.58, -0.0551, 0.996, -0.742])
        #self.initial_joint_values = np.zeros(7)
        self.joint_names = np.array(
            ["panda0_joint" + str(name_idx) for name_idx in range(1, 8)])
        self.constraints = constraints
        self.single_goal_dim = 3  # 3D Positions only
        self.reward_function = reward_calculation.get_reward_function(
            self.constraints, self.single_goal_dim, sparse_dense)

        model = construct_model(model_path, **template_kwargs)
        action_space = spaces.Box(-1., 1., shape=(7,), dtype='float32')

        super(FrankaEnv, self).__init__(
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
        action *= self.max_advance
        utils.ctrl_set_action(self.sim, action)

    def _get_obs(self):
        achieved_goal = np.zeros(self.single_goal_dim*len(self.constraints))
        for i, constraint in enumerate(self.constraints):
            origin = constraint['origin']
            achieved_goal[i*self.single_goal_dim:(i+1)*self.single_goal_dim] = self.sim.data.get_site_xpos(
                origin).copy()

        pos = np.array([self.sim.data.get_site_xpos(site).copy()
                        for site in self.site_names]).flatten()

        robot_pos = np.array([self.sim.data.get_joint_qpos(joint).copy()
                              for joint in self.joint_names]).flatten()

        if self.velocity_in_obs:
            dt = self.n_substeps * self.sim.model.opt.timestep

            vel = np.array([self.sim.data.get_site_xvelp(site).copy()
                            for site in self.site_names]).flatten() * dt
            robot_vel = np.array([self.sim.data.get_joint_qvel(joint).copy()
                                  for joint in self.joint_names]).flatten() * dt
            obs = np.concatenate([pos, vel])
            robot_obs = np.concatenate([robot_pos, robot_vel])
        else:
            obs = pos
            robot_obs = robot_pos

        robot_obs = np.concatenate([robot_obs, self.previous_action])

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

        lookat = np.array([0.5, 0, .9])

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = .7
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -70.

    def _env_setup(self):
        #utils.disable_weld(self.sim, "panda0_end_effector_vb", "B0_8")
        for i, value in enumerate(self.initial_joint_values):
            self.sim.data.set_joint_qpos(self.joint_names[i], value)
        utils.franka_ctrl_set_action(self.sim, self.initial_joint_values)
        #utils.enable_weld(self.sim, "panda0_end_effector_vb", "B0_8")

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

    def render(self, mode='human', width=1000, height=1000, image_capture=False, filename=None):
        return super(FrankaEnv, self).render(mode, width, height, image_capture, filename)
