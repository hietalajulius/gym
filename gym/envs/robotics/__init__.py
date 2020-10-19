from gym.envs.robotics.fetch_env import FetchEnv
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv

from gym.envs.robotics.hand.reach import HandReachEnv
from gym.envs.robotics.hand.manipulate import HandBlockEnv
from gym.envs.robotics.hand.manipulate import HandEggEnv
from gym.envs.robotics.hand.manipulate import HandPenEnv

from gym.envs.robotics.hand.manipulate_touch_sensors import HandBlockTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandEggTouchSensorsEnv
from gym.envs.robotics.hand.manipulate_touch_sensors import HandPenTouchSensorsEnv

from gym.envs.robotics.cloth.diagonal import ClothDiagonalEnv
from gym.envs.robotics.cloth.diagonal_strict_pixels import ClothDiagonalStrictPixelsEnv
from gym.envs.robotics.cloth.diagonal_pixels import ClothDiagonalPixelsEnv
from gym.envs.robotics.cloth.diagonal_strict import ClothDiagonalStrictEnv
from gym.envs.robotics.cloth.diagonal_strict_learn_grasp import ClothDiagonalStrictLearnGraspEnv

from gym.envs.robotics.cloth.sideways import ClothSidewaysEnv
from gym.envs.robotics.cloth.sideways_strict import ClothSidewaysStrictEnv
from gym.envs.robotics.cloth.sideways_strict_pixels import ClothSidewaysStrictPixelsEnv

from gym.envs.robotics.franka_env import FrankaEnv

