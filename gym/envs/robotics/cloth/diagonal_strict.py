import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'diagonal.xml')


class ClothDiagonalStrictEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self,reward_type=None):
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=40, n_actions=3, distance_threshold=0.05, strict=True, sparse_dense=False, pixels=False)
        utils.EzPickle.__init__(self)
