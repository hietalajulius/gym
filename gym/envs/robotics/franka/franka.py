import os
from gym import utils
from gym.envs.robotics import franka_env


os.environ["FRANKA_TEMPLATE_PATH"] = '/Users/juliushietala/robotics/panda-gym/panda_gym/franka_sim/templates'
os.environ["FRANKA_MESH_PATH"] = '/Users/juliushietala/robotics/panda-gym/panda_gym/franka_sim/meshes'
MODEL_XML_PATH = 'franka_cloth.xml'


class FrankaEnv(franka_env.FrankaEnv, utils.EzPickle):
    def __init__(self, **kwargs):
        print("Creating env")
        franka_env.FrankaEnv.__init__(
            self, MODEL_XML_PATH, **kwargs)
        utils.EzPickle.__init__(self)
