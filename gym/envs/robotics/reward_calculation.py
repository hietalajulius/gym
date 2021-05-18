import numpy as np


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_task_reward_function(constraints, single_goal_dim, sparse_dense, success_reward, fail_reward, extra_reward):
    def task_reward_function(achieved_goal, desired_goal, info):
        achieved_oks = np.zeros(
            (achieved_goal.shape[0], len(constraints)))
        achieved_distances = np.zeros(
            (achieved_goal.shape[0], len(constraints)))

        constraint_distances = [constraint['distance']
                                for constraint in constraints]

        for i, constraint_distance in enumerate(constraint_distances):
            achieved = achieved_goal[:, i *
                                     single_goal_dim:(i+1)*single_goal_dim]
            desired = desired_goal[:, i *
                                   single_goal_dim:(i+1)*single_goal_dim]

            achieved_distances_per_constraint = goal_distance(
                achieved, desired)

            constraint_ok = achieved_distances_per_constraint < constraint_distance

            achieved_distances[:, i] = achieved_distances_per_constraint
            achieved_oks[:, i] = constraint_ok

        successes = np.all(achieved_oks, axis=1)
        fails = np.invert(successes)

        task_rewards = successes.astype(np.float32).flatten()*success_reward

        if sparse_dense:
            dist_rewards = np.sum((1 - achieved_distances/np.array(constraint_distances)),
                                  axis=1) / len(constraints)

            task_rewards = task_rewards + dist_rewards*(extra_reward-success_reward)
            
            if "num_future_goals" in info.keys():
                num_future_goals = info['num_future_goals']
                task_rewards[-num_future_goals:] = success_reward

        task_rewards[fails] = fail_reward

        return task_rewards

    return task_reward_function


'''
if __name__ == '__main__':
    import gym
    constraint_distance = 0.0001
    c1 = dict(origin="S0_0", target="S0_0", distance=constraint_distance)
    c2 = dict(origin="S4_4", target="S4_4", distance=constraint_distance)
    c3 = dict(origin="S8_8", target="S8_8", distance=constraint_distance)
    c4 = dict(origin="S0_8", target="S0_8", distance=constraint_distance)

    constraints = [c1, c2, c3, c4]
    env = gym.make('Cloth-v1', pixels=False,
                   randomize_params=False, randomize_geoms=False, max_advance=0.05, random_seed=1, sparse_dense=True, constraints=constraints)

    print("Env has reward func", env.reward_function)

    rew = env.compute_reward(np.reshape(
        env.goal, (1, -1)), np.reshape(
        env.goal, (1, -1)), {'real_sim': True})
    print("Sparse dense", rew[0])
    obs, reward, done, info = env.step(np.array([0, 0, 0]))

    env = gym.make('Cloth-v1', pixels=False,
                   randomize_params=False, randomize_geoms=False, max_advance=0.05, random_seed=1, sparse_dense=False, constraints=constraints)

    rew = env.compute_reward(np.reshape(
        env.goal, (1, -1)), np.reshape(
        env.goal, (1, -1)), {'real_sim': True})
    print("Binary", rew[0])
    obs, reward, done, info = env.step(np.array([0, 0, 0]))
'''
