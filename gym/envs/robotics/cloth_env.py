# Copyright 2020 (c) Aalto University - All Rights Reserved
# Author: Julius Hietala <julius.hietala@aalto.fi>
#

import numpy as np

from gym.envs.robotics import rotations, cloth_robot_env, utils
import cv2
import copy
from PIL import Image


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ClothEnv(cloth_robot_env.ClothRobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, task, n_actions, learn_grasp, distance_threshold, strict, pixels, max_advance, start_grasped, randomize_params,
        n_substeps=40, noise_range=0.03, uniform_jnt_tend=True
    ):

        self.noise_range = noise_range
        self.task = task
        if self.task == "sideways":
            limit_workspace = True
        else:
            limit_workspace = False
        self.strict = strict
        self.distance_threshold = distance_threshold
        self.site_names =  ["S0_0", "S4_0", "S8_0", "S0_4", "S0_8", "S4_8", "S8_8", "S8_4", 'robot']

        super(ClothEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions, learn_grasp=learn_grasp, randomize_params=randomize_params, uniform_jnt_tend=uniform_jnt_tend, pixels=pixels, max_advance=max_advance, limit_workspace=limit_workspace, start_grasped=start_grasped)


    def compute_reward(self, achieved_goal, goal, info):
        if self.task == "sideways":
            if self.strict:
                achieved_goal = np.reshape(achieved_goal, (-1,12))
                goal = np.reshape(goal, (-1,12))

                dist1 = goal_distance(achieved_goal[:, :3], goal[:, :3])
                d1 =  dist1 > self.distance_threshold
                
                dist2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
                d2 = dist2  > self.distance_threshold

                dist3 = goal_distance(achieved_goal[:, 6:9], goal[:, 6:9])
                d3 =  dist3 > self.distance_threshold

                dist4 = goal_distance(achieved_goal[:, 9:12], goal[:, 9:12])
                d4 =  dist4 > self.distance_threshold

                res = -(np.any(np.array([d1, d2, d3, d4]), axis=0)).astype(np.float32).flatten()

            else:
                achieved_goal = np.reshape(achieved_goal, (-1,6))
                goal = np.reshape(goal, (-1,6))
                dist1 = goal_distance(achieved_goal[:, :3], goal[:, :3])
                d1 =  dist1 > self.distance_threshold

                dist2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
                d2 =  dist2 > self.distance_threshold
                res = -(np.any(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()

            if len(res) == 1:
                res = res[0]
        else:
            if self.strict:
                achieved_goal = np.reshape(achieved_goal, (-1,6))
                goal = np.reshape(goal, (-1,6))

                dist1 = goal_distance(achieved_goal[:, :3], goal[:, :3])
                d1 =  dist1 > self.distance_threshold

                dist2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
                d2 =  dist2 > self.distance_threshold
                res = -(np.any(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()

                if len(res) == 1:
                    res = res[0]
            else:
                d = goal_distance(achieved_goal, goal)
                res = -(d > self.distance_threshold).astype(np.float32)
        
        return res


    def _get_obs(self):
        if self.task == "sideways":
            if self.strict:
                body_0_8 = self.sim.data.get_site_xpos("S0_8").copy()
                body_0_0= self.sim.data.get_site_xpos("S0_0").copy()
                body_8_8 = self.sim.data.get_site_xpos("S8_8").copy()
                body_8_0 = self.sim.data.get_site_xpos("S8_0").copy()
                achieved_goal = np.concatenate([body_8_8, body_8_0, body_0_8, body_0_0]).flatten()
            else:
                manip1 = self.sim.data.get_site_xpos("S8_8").copy()
                manip2 = self.sim.data.get_site_xpos("S8_0").copy()
                achieved_goal = np.concatenate([manip1, manip2]).flatten()

        else:
            if self.strict:
                body_8_0 = self.sim.data.get_site_xpos("S8_0").copy()
                body_0_8 = self.sim.data.get_site_xpos("S0_8").copy()
                achieved_goal = np.concatenate([body_8_0, body_0_8])
            
            else:
                achieved_goal = self.sim.data.get_site_xpos("S8_0").copy() #Modded


        pos = np.array([self.sim.data.get_site_xpos(site).copy() for site in self.site_names]).flatten()
        dt = self.n_substeps * self.sim.model.opt.timestep
        vel = np.array([self.sim.data.get_site_xvelp(site).copy() for site in self.site_names]).flatten() * dt

        obs = np.concatenate([pos, vel])

        robot_pos = self.sim.data.get_site_xpos('robot').copy()
        robot_vel = self.sim.data.get_site_xvelp('robot').copy() * dt
        robot_obs = np.concatenate([robot_pos, robot_vel])

        model_params = np.array([self.current_joint_stiffness, self.current_joint_damping, self.current_tendon_stiffness, self.current_tendon_damping, self.current_geom_size])
        observation = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
            'model_params': model_params.copy(),
            'robot_observation' : robot_obs
        }
        
        if self.pixels:
            image_obs = self.render(width=140, height=140, mode='rgb_array').copy()
            image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('env', image_obs)
            #cv2.imshow('env', cv2.cvtColor(image_obs, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
            observation['image'] = (image_obs / 255).flatten()
        return observation

    def _viewer_setup_diagonal(self):
        lookat = self.origin

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 0.7
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -90.

    def _viewer_setup(self):
        if self.task == 'sideways':
            self._viewer_setup_sideways()
        else:
            self._viewer_setup_diagonal()
    
    def _viewer_setup_sideways(self):
        lookat = self.origin 

        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 0.45
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -90.
    
    def _reset_view(self):
        if self.task == "sideways":
            site1_id = self.sim.model.site_name2id('target0')
            site2_id = self.sim.model.site_name2id('target1')

            self.sim.model.site_pos[site1_id] = self.goal[:3]
            self.sim.model.site_pos[site2_id] = self.goal[3:6]
        else:
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal[:3]

        self.sim.forward()


    def _sample_goal(self):
        offset = self.np_random.uniform(low=0, high=self.noise_range)
        if self.task == "sideways":
            if self.strict:
                goal1 = self.sim.data.get_site_xpos("S0_8").copy()
                goal2 = self.sim.data.get_site_xpos("S0_0").copy()
                goal = np.concatenate([goal1 + np.array([abs(offset),0,0]),goal2 + np.array([abs(offset),0,0]), goal1, goal2]).flatten()
            else:
                goal1 = self.sim.data.get_site_xpos("S0_8").copy()
                goal2 = self.sim.data.get_site_xpos("S0_0").copy()
                goal = np.concatenate([goal1 + np.array([abs(offset),0,0]),goal2 + np.array([abs(offset),0,0])]).flatten()
        else:
            if self.strict:
                goal1 = self.sim.data.get_site_xpos("S0_8").copy() + np.array([offset,-offset,0])
                goal2 = self.sim.data.get_site_xpos("S0_8").copy()
                goal = np.concatenate([goal1, goal2])
                
            else:
                goal = self.sim.data.get_site_xpos("S0_8").copy() + np.array([offset,-offset,0]) #Modified

        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        if self.task == "sideways":
            if self.strict:
                achieved_goal = np.reshape(achieved_goal, (-1,12))
                goal = np.reshape(desired_goal, (-1,12))
                d1 = goal_distance(achieved_goal[:, :3], goal[:, :3]) <= self.distance_threshold
                d2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6]) <= self.distance_threshold
                d3 = goal_distance(achieved_goal[:, 6:9], goal[:, 6:9]) <= self.distance_threshold/2
                d4 = goal_distance(achieved_goal[:, 9:12], goal[:, 9:12]) <= self.distance_threshold/2
                res = (np.all(np.array([d1, d2, d3, d4]), axis=0)).astype(np.float32).flatten()
            else:
                achieved_goal = np.reshape(achieved_goal, (-1,6))
                goal = np.reshape(desired_goal, (-1,6))
                d1 = goal_distance(achieved_goal[:, :3], goal[:, :3]) <= self.distance_threshold
                d2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6]) <= self.distance_threshold
                res = (np.all(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()

            if len(res) == 1:
                res = res[0]
        else:
            if self.strict:
                achieved_goal = np.reshape(achieved_goal, (-1,6))
                goal = np.reshape(desired_goal, (-1,6))
                d1 = goal_distance(achieved_goal[:, :3], goal[:, :3]) <= self.distance_threshold
                d2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6]) <= self.distance_threshold
                res = (np.all(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()
                if len(res) == 1:
                    res = res[0]
            else:
                d = goal_distance(achieved_goal, desired_goal)
                res = (d < self.distance_threshold).astype(np.float32)
        return res

    def render(self, mode='human', width=500, height=500):
        return super(ClothEnv, self).render(mode, width, height)
