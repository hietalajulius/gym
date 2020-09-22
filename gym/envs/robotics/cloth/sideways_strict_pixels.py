import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'cloth.xml')


class ClothSidewaysStrictPixelsEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self,reward_type=None):
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=40, noise_range=0.03, distance_threshold=0.03, n_actions=3, task="sideways", strict=True, pixels=True)
        utils.EzPickle.__init__(self)
