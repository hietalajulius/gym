import os
from gym import utils
from gym.envs.robotics import franka_env

MODEL_XML_PATH = 'franka_cloth.xml'


class FrankaEnv(franka_env.FrankaEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        print("Creating env")
        franka_env.FrankaEnv.__init__(
            self, MODEL_XML_PATH, **kwargs)
        utils.EzPickle.__init__(self)
