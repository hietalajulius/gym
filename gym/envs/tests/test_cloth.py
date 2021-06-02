import pytest
import gym
import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


c_d = 0.03  # constraint distance
s_o = 0.0000000000001  # small offset
c1 = dict(origin="S0_0", target="S8_0", distance=c_d)
c2 = dict(origin="S4_4", target="S8_4", distance=c_d)
c3 = dict(origin="S8_8", target="S0_8", distance=c_d)
c4 = dict(origin="S8_8", target="S0_8", distance=c_d)

constraints = [c1, c2, c3, c4]

# Base values
x1 = np.array([1, 1, 0])
x2 = 2*x1
x3 = 3*x1
x4 = 4*x1


def test_cloth_reward_calculation_no_sparse_dense():
    env = gym.make('Cloth-v1', pixels=False,
                   randomize_params=False, randomize_geoms=False, output_max=0.05, random_seed=1, sparse_dense=False, constraints=constraints)

    achieved_1 = np.concatenate((x1, x2, x3, x4))
    desired_1 = np.concatenate((x1, x2, x3, x4))

    achieved_2 = np.concatenate((x1, x2, x3, x4*2))  # Way off
    desired_2 = np.concatenate((x1, x2, x3, x4))

    achieved = np.array([achieved_1, achieved_2])
    desired = np.array([desired_1, desired_2])
    r = env.compute_reward(achieved, desired, {})
    assert r[0] == 0
    assert r[1] == -1


def test_cloth_reward_calculation_no_sparse_dense_very_close():
    env = gym.make('Cloth-v1', pixels=False,
                   randomize_params=False, randomize_geoms=False, output_max=0.05, random_seed=1, sparse_dense=False, constraints=constraints)

    achieved_1 = np.concatenate((x1, x2, x3, x4))
    desired_1 = np.concatenate((x1, x2, x3, x4))

    offset_2 = np.array([1, 1, 0]) * np.sqrt(((c_d - s_o)**2)/2)
    achieved_2 = np.concatenate(
        (x1, x2, x3, x4+offset_2))  # Just close enough
    desired_2 = np.concatenate((x1, x2, x3, x4))

    offset_3 = np.array([1, 1, 0]) * np.sqrt(((c_d + s_o)**2)/2)
    achieved_3 = np.concatenate(
        (x1, x2, x3, x4+offset_3))  # Slightly off
    desired_3 = np.concatenate((x1, x2, x3, x4))

    achieved = np.array([achieved_1, achieved_2, achieved_3])
    desired = np.array([desired_1, desired_2, desired_3])
    r = env.compute_reward(achieved, desired, {})
    assert r[0] == 0
    assert r[1] == 0
    assert r[2] == -1


spec_list = [dict(c_d=0.01, s_o=0.0000123),
             dict(c_d=0.02, s_o=0.00000012345),
             dict(c_d=0.04, s_o=0.00010030090001),
             dict(c_d=0.05, s_o=0.0000011),
             dict(c_d=0.05, s_o=0.00000000019999),
             dict(c_d=0.01, s_o=0.00000000001),
             dict(c_d=0.02, s_o=0.00001),
             dict(c_d=0.03, s_o=0.00010030090001),
             dict(c_d=0.04, s_o=0.0000011),
             dict(c_d=0.70, s_o=0.00000000019898989999),
             dict(c_d=22.01, s_o=1.0000123),
             dict(c_d=0.02, s_o=0.00000012345),
             dict(c_d=0.14, s_o=0.00010030090001),
             dict(c_d=0.05, s_o=0.0000011),
             dict(c_d=0.05, s_o=0.00000000019999),
             dict(c_d=0.01, s_o=0.00000000001),
             dict(c_d=9.02, s_o=0.00001),
             dict(c_d=99.03, s_o=9.00010030090001),
             dict(c_d=0.04, s_o=0.0000011),
             dict(c_d=1.70, s_o=0.008880000000019898989999)]


@pytest.mark.parametrize("spec", spec_list)
def test_cloth_reward_calculation_sparse_dense(spec):
    c_d = spec['c_d']
    s_o = spec['s_o']
    c1 = dict(origin="S0_0", target="S8_0", distance=c_d)
    c2 = dict(origin="S4_4", target="S8_4", distance=c_d)
    c3 = dict(origin="S8_8", target="S0_8", distance=c_d)
    c4 = dict(origin="S8_8", target="S0_8", distance=c_d)

    constraints = [c1, c2, c3, c4]

    env = gym.make('Cloth-v1', pixels=False,
                   randomize_params=False, randomize_geoms=False, output_max=0.05, random_seed=1, sparse_dense=True, constraints=constraints)

    achieved_1 = np.concatenate((x1, x2, x3, x4))
    desired_1 = np.concatenate((x1, x2, x3, x4))

    offset_2 = np.array([1, 1, 0]) * np.sqrt(((c_d - s_o)**2)/2)
    achieved_2 = np.concatenate(
        (x1, x2, x3, x4+offset_2))  # Just close enough one corner
    desired_2 = np.concatenate((x1, x2, x3, x4))

    offset_3 = np.array([1, 1, 0]) * np.sqrt(((c_d + s_o)**2)/2)
    achieved_3 = np.concatenate(
        (x1, x2, x3, x4+offset_3))  # Slightly off
    desired_3 = np.concatenate((x1, x2, x3, x4))

    offset_4 = np.array([1, 1, 0]) * np.sqrt(((c_d - s_o)**2)/2)
    achieved_4 = np.concatenate(
        (x1+offset_4, x2+offset_4, x3+offset_4, x4+offset_4))  # Just close enough all corners
    desired_4 = np.concatenate((x1, x2, x3, x4))

    achieved = np.array([achieved_1, achieved_2, achieved_3, achieved_4])
    desired = np.array([desired_1, desired_2, desired_3, desired_4])
    r = env.compute_reward(achieved, desired, {'real_sim': True})

    one_off_desired = (
        len(constraints)-(c_d-s_o)/c_d)/len(constraints)

    all_off_desired = 1-(c_d-s_o)/c_d

    print("One", one_off_desired, r[1], goal_distance(achieved_2, desired_2))
    print("All", all_off_desired, r[3], goal_distance(achieved_4, desired_4))
    assert r[0] == 1
    assert np.isclose(one_off_desired, r[1])
    assert r[2] == -1
    assert np.isclose(all_off_desired, r[3])
    #assert 1 == 2
