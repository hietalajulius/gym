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
        self, model_path, n_substeps, 
        n_actions, distance_threshold=0.02, noise_range=0.05, task="diagonal", strict=False, pixels=False
    ):

        self.noise_range = noise_range
        self.task = task
        self.pixels = pixels
        self.strict = strict
        self.distance_threshold = distance_threshold
        self.site_names =  ["S0_0", "S4_0", "S8_0", "S0_4", "S0_8", "S4_8", "S8_8", "S8_4", 'robot']

        self.origin = np.array([0.12,0.12,0])
        self.maxdist = 0.15
        self.maximum = self.origin[0] + self.maxdist
        self.minimum = self.origin[0] - self.maxdist

        super(ClothEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions)


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
                d3 =  dist3 > self.distance_threshold/2

                dist4 = goal_distance(achieved_goal[:, 9:12], goal[:, 9:12])
                d4 =  dist4 > self.distance_threshold/2

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
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        vel = np.array([self.sim.data.get_site_xvelp(site).copy() for site in self.site_names]).flatten() * dt
        obs = np.concatenate([pos, vel])
        observation = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        
        if self.pixels:
            image_obs = copy.deepcopy(self.render(width=84, height=84, mode='rgb_array'))
            #image_obs = cv2.cvtColor(image_obs, cv2.COLOR_BGR2GRAY)
            #cv2.imshow('env', cv2.cvtColor(image_obs, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
            observation['image'] = (image_obs / 255)
            #print("added pixel observations", observation['image'].shape)
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

        self.viewer.cam.distance = 0.5
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -90.


    def _reset_sim(self):
        mocap_beginning = self.sim.data.get_site_xpos('S8_0').copy() + np.random.randint(-10,10,3)/300
        utils.reset_mocap_position(self.sim, mocap_beginning)
        self.sim.set_state(self.initial_state)

        lim1_id = self.sim.model.site_name2id('limit0')
        lim2_id = self.sim.model.site_name2id('limit1')
        lim3_id = self.sim.model.site_name2id('limit2')
        lim4_id = self.sim.model.site_name2id('limit3')

        self.sim.model.site_pos[lim1_id] = self.origin + np.array([-self.maxdist,-self.maxdist,0])
        self.sim.model.site_pos[lim2_id] = self.origin + np.array([self.maxdist,-self.maxdist,0])
        self.sim.model.site_pos[lim3_id] = self.origin + np.array([-self.maxdist,self.maxdist,0])
        self.sim.model.site_pos[lim4_id] = self.origin + np.array([self.maxdist,self.maxdist,0])

        if self.task == "sideways":
            site1_id = self.sim.model.site_name2id('target0')
            site2_id = self.sim.model.site_name2id('target1')

            self.sim.model.site_pos[site1_id] = self.goal[:3]
            self.sim.model.site_pos[site2_id] = self.goal[3:6]
        else:
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal[:3]
            
        self.sim.forward()
        return True

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
