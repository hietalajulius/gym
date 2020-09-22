import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'cloth.xml')


class ClothSidewaysStrictEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self,reward_type=None):
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=40, noise_range=0.02, distance_threshold=0.03, n_actions=4, task="sideways", strict=True)
        utils.EzPickle.__init__(self)
