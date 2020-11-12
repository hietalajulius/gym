import os
from gym import utils
from gym.envs.robotics import cloth_env

MODEL_XML_PATH = os.path.join('cloth', 'cloth.xml')


class ClothEnv(cloth_env.ClothEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        print("Creating env")
        cloth_env.ClothEnv.__init__(
            self, MODEL_XML_PATH, **kwargs)
        utils.EzPickle.__init__(self)
