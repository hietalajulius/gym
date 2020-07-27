import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'sideways.xml')


class ClothSidewaysEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self,reward_type=None):
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, n_substeps=20, n_actions=3, task="sideways", noise_range=0.01, distance_threshold=0.05)
        utils.EzPickle.__init__(self)
