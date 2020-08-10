import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'reach.xml')


class ClothReachEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self,reward_type=None):
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, n_actions=3, noise_range=0, distance_threshold=0.05, strict=False, sparse_dense=False, pixels=True)
        utils.EzPickle.__init__(self)