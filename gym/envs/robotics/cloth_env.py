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
        self, model_path, n_substeps, noise_range,
        distance_threshold, n_actions, task="diagonal", strict=False, sparse_dense=False, pixels=False, baselines=False
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            target_offset (float or array with 3 elements): offset of the target
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.noise_range = noise_range
        self.task = task
        self.pixels = pixels
        self.strict = strict
        self.sparse_dense = sparse_dense
        self.distance_threshold = distance_threshold
        self.site_names =  ["S0_0", "S4_0", "S8_0", "S0_4", "S0_8", "S4_8", "S8_8", "S8_4"]

        super(ClothEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions, baselines=baselines)

    # GoalEnv methods
    # ----------------------------

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

                if self.sparse_dense:
                    mask = res == 0
                    res[mask] = (4*np.ones(mask.shape[0]) - (dist1[mask] + dist2[mask] + dist3[mask] + dist4[mask])/self.distance_threshold)/4
            else:
                achieved_goal = np.reshape(achieved_goal, (-1,6))
                goal = np.reshape(goal, (-1,6))
                dist1 = goal_distance(achieved_goal[:, :3], goal[:, :3])
                d1 =  dist1 > self.distance_threshold

                dist2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6])
                d2 =  dist2 > self.distance_threshold
                res = -(np.any(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()

                if self.sparse_dense:
                    mask = res == 0
                    res[mask] = (2*np.ones(mask.shape[0]) - (dist1[mask] + dist2[mask])/self.distance_threshold)/2


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

                if self.sparse_dense:
                    mask = res == 0
                    res[mask] = (2*np.ones(mask.shape[0]) - (dist1[mask] + dist2[mask])/self.distance_threshold)/2

                if len(res) == 1:
                    res = res[0]
            else:
                d = goal_distance(achieved_goal, goal)
                res = -(d > self.distance_threshold).astype(np.float32)

                if self.sparse_dense:
                    mask = res == 0
                    res[mask] = np.ones(mask.shape[0]) - d/self.distance_threshold
        
        return res

        


    # RobotEnv methods
    # ----------------------------


    def _set_action(self, action):
        action = action.copy()  # ensure that we don't change the action outside of this scope
        pos_ctrl = action[:3]
        pos_ctrl *= 0.05
        if self.task == "sideways":
            assert action.shape == (3,)
            #grip = action[3]
            utils.mocap_set_action_cloth(self.sim, pos_ctrl)
        else:
            assert action.shape == (3,)
            utils.mocap_set_action_cloth(self.sim, pos_ctrl)

    def _get_obs(self):
        if self.task == "sideways":
            if self.strict:
                body_0_8 = self.sim.data.get_site_xpos("S0_8").copy()
                body_0_0= self.sim.data.get_site_xpos("S0_0").copy()
                body_8_8 = self.sim.data.get_site_xpos("S8_8").copy()
                body_8_0 = self.sim.data.get_site_xpos("S8_0").copy()
                achieved_goal = np.concatenate([body_0_8, body_0_0, body_8_8, body_8_0]).flatten()
            else:
                manip1 = self.sim.data.get_site_xpos("S0_8").copy()
                manip2 = self.sim.data.get_site_xpos("S0_0").copy()
                achieved_goal = np.concatenate([manip1, manip2]).flatten()

        else:
            if self.strict:
                body_0_0 = self.sim.data.get_site_xpos("S0_0").copy()
                body_8_8 = self.sim.data.get_site_xpos("S8_8").copy()
                achieved_goal = np.concatenate([body_0_0, body_8_8])
            
            else:
                achieved_goal = self.sim.data.get_site_xpos("S0_0").copy()


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
            #cv2.imshow('env', cv2.cvtColor(image_obs, cv2.COLOR_RGB2BGR))
            #cv2.waitKey(1)
            observation['image'] = image_obs / 255
            #print("added pixel observations")

        return observation

    def _viewer_setup_original(self):
        body_id = self.sim.model.body_name2id('B0_0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('B4_4')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value

        self.viewer.cam.distance = 0.32
        self.viewer.cam.azimuth = 0.
        self.viewer.cam.elevation = -90.

    def _render_callback(self):
        #Resetting sites was here
        #self.sim.forward()
        pass

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        if self.task == "sideways":
            site1_id = self.sim.model.site_name2id('target0')
            site2_id = self.sim.model.site_name2id('target1')
            site3_id = self.sim.model.site_name2id('target2')
            site4_id = self.sim.model.site_name2id('target3')
            self.sim.model.site_pos[site1_id] = self.goal[:3]
            self.sim.model.site_pos[site2_id] = self.goal[3:6]
            if self.strict:
                self.sim.model.site_pos[site3_id] = self.goal[6:9]
                self.sim.model.site_pos[site4_id] = self.goal[9:12]
            else:
                self.sim.model.site_pos[site3_id] = self.goal[:3]
                self.sim.model.site_pos[site4_id] = self.goal[3:6]

        else:
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal[:3]
            
        self.sim.forward()
        return True

    def _sample_goal(self):
        offset = self.np_random.uniform(low=-self.noise_range, high=0)
        if self.task == "sideways":
            if self.strict:
                goal1 = self.sim.data.get_site_xpos("S8_8").copy()
                goal2 = self.sim.data.get_site_xpos("S8_0").copy()
                goal = np.concatenate([goal1 + np.array([offset,0,0]),goal2 + np.array([offset,0,0]), goal1, goal2]).flatten()
            else:
                goal1 = self.sim.data.get_site_xpos("S8_8").copy()
                goal2 = self.sim.data.get_site_xpos("S8_0").copy()
                goal = np.concatenate([goal1 + np.array([offset,0,0]),goal2 + np.array([offset,0,0])]).flatten()
        else:
            if self.strict:
                goal1 = self.sim.data.get_site_xpos("S8_8").copy() + np.array([offset,offset,0])
                goal2 = self.sim.data.get_site_xpos("S8_8").copy()
                goal = np.concatenate([goal1, goal2])
                
            else:
                goal = self.sim.data.get_site_xpos("S8_8").copy() + np.array([offset,offset,0])

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

    def _env_setup(self):
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        gripper_target = np.array([0, 0, 0.01])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(ClothEnv, self).render(mode, width, height)
