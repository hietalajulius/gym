import numpy as np

from gym.envs.robotics import rotations, cloth_robot_env, utils


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class ClothEnv(cloth_robot_env.ClothRobotEnv):
    """Superclass for all Fetch environments.
    """

    def __init__(
        self, model_path, n_substeps, noise_range,
        distance_threshold, n_actions, task="diagonal"
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
        self.distance_threshold = distance_threshold
        self.site_names =  ["S0_0", "S4_0", "S8_0", "S0_4", "S0_8", "S4_8", "S8_8", "S8_4"]

        super(ClothEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=n_actions)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        if self.task == "sideways":
            achieved_goal = np.reshape(achieved_goal, (-1,6))
            goal = np.reshape(goal, (-1,6))
            d1 = goal_distance(achieved_goal[:, :3], goal[:, :3]) > self.distance_threshold
            d2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6]) > self.distance_threshold
            res = -(np.any(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()
            if len(res) == 1:
                res = res[0]
            return res

        else:
            d = goal_distance(achieved_goal, goal)
            res = -(d > self.distance_threshold).astype(np.float32)
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
            manip1 = self.sim.data.get_site_xpos("S0_8").copy()
            manip2 = self.sim.data.get_site_xpos("S0_0").copy()
            achieved_goal = np.concatenate([manip1, manip2]).flatten()

        else:
            achieved_goal = self.sim.data.get_site_xpos("S0_0").copy()
            #targ_pos = self.sim.data.get_site_xpos('target0').copy()


        pos = np.array([self.sim.data.get_site_xpos(site).copy() for site in self.site_names]).flatten()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        vel = np.array([self.sim.data.get_site_xvelp(site).copy() for site in self.site_names]).flatten() * dt
        obs = np.concatenate([pos, vel])
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('B0_0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        if self.task == "sideways":
            site1_id = self.sim.model.site_name2id('target0')
            site2_id = self.sim.model.site_name2id('target1')
            self.sim.model.site_pos[site1_id] = self.goal[:3]
            self.sim.model.site_pos[site2_id] = self.goal[3:6]
        else:
            site_id = self.sim.model.site_name2id('target0')
            self.sim.model.site_pos[site_id] = self.goal
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        offset = self.np_random.uniform(low=-self.noise_range, high=self.noise_range)
        if self.task == "sideways":
            goal1 = self.sim.data.get_site_xpos("S8_8")
            goal2 = self.sim.data.get_site_xpos("S8_0")
            goal = np.concatenate([goal1 + np.array([-abs(offset),0,0]),goal2 + np.array([-abs(offset),0,0])]).flatten()
        else:
            goal = self.sim.data.get_site_xpos("S8_8") + np.array([offset,offset,0])
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        if self.task == "sideways":
            achieved_goal = np.reshape(achieved_goal, (-1,6))
            goal = np.reshape(desired_goal, (-1,6))
            d1 = goal_distance(achieved_goal[:, :3], goal[:, :3]) <= self.distance_threshold
            d2 = goal_distance(achieved_goal[:, 3:6], goal[:, 3:6]) <= self.distance_threshold
            res = (np.all(np.array([d1, d2]), axis=0)).astype(np.float32).flatten()
            if len(res) == 1:
                res = res[0]
            return res
        else:
            d = goal_distance(achieved_goal, desired_goal)
            res = (d < self.distance_threshold).astype(np.float32)
            return res

    def _env_setup(self):
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        gripper_target = np.array([0, 0, 0])
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

    def render(self, mode='human', width=500, height=500):
        return super(ClothEnv, self).render(mode, width, height)
