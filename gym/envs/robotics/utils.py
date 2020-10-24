import numpy as np

from gym import error
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

def distance(body_a, body_b):
    return np.linalg.norm(body_a - body_b, axis=-1)

def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith('robot')]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7, ))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]

def franka_ctrl_set_action(sim, action):
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            sim.data.ctrl[i] = action[i]

def mocap_set_action_cloth(sim, pos_ctrl, minimum, maximum, origin, limit_workspace):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        reset_mocap2body_xpos(sim) #Mocap reset
        action = sim.data.mocap_pos + pos_ctrl[:3]
        action = action.flatten()
        action = np.clip(action, minimum, maximum)
        if limit_workspace:
            y = np.clip(action[1], minimum, 0.18)
            action = np.array([action[0], y, action[2]])
        if action[2] < 0: #Limit mocap from going under the floor
            action[2] = 0
        sim.data.mocap_pos[:] = action

def increase_mocap_position(sim, position_increase):
    action = sim.data.mocap_pos.copy() + position_increase.copy()
    sim.data.mocap_pos[:] = action
    #sim.forward()

def set_mocap_position(sim, position):
    sim.data.mocap_pos[:] = position.copy()
    sim.forward()

def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
    sim.forward()

def remove_mocap_welds(sim):
    """Removes the mocap welds that we use for actuation.
    """
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_active[i] = False
    sim.forward()

def enable_mocap_welds(sim):
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_active[i] = True
    sim.forward()

def grasp(sim, body_name):
    eq_obj_id = sim.model.body_name2id(body_name)
    for i in range(sim.model.eq_data.shape[0]):
        if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD and (sim.model.eq_obj1id[i] == eq_obj_id or sim.model.eq_obj2id[i] == eq_obj_id):
            sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
            sim.model.eq_active[i] = True
    sim.forward()

def get_closest_body_to_mocap(sim):
    mocap_pos = sim.data.mocap_pos.copy()
    body_names = ["B" + str(i) + "_" + str(j) for i in range(9) for j in range(9)]
    dists = []
    for body_name in body_names:
        body_pos = sim.data.get_body_xpos(body_name).copy()
        dist = distance(body_pos, mocap_pos)
        dists.append(dist)
    dists = np.array(dists)
    min_idx = np.argmin(dists)
    return body_names[min_idx], dists[min_idx]

    

def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (sim.model.eq_type is None or
        sim.model.eq_obj1id is None or
        sim.model.eq_obj2id is None):
        return

    for eq_type, obj1_id, obj2_id, eq_active, eq_data in zip(sim.model.eq_type,
                                         sim.model.eq_obj1id,
                                         sim.model.eq_obj2id,
                                         sim.model.eq_active,
                                         sim.model.eq_data):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue


        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        #assert (mocap_id != -1)
        if not mocap_id == -1:
            sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        #sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]
        
    #sim.forward()

def disable_mocap_weld(sim, body1, body2):
    body1_id = sim.model.body_name2id(body1)
    body2_id = sim.model.body_name2id(body2)

    for i, (eq_type, obj1_id, obj2_id) in enumerate(zip(sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id)):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        if (obj1_id == body1_id and obj2_id == body2_id) or (obj2_id == body1_id and obj1_id == body2_id):
            sim.model.eq_active[i] = False

def enable_mocap_weld(sim, body1, body2):
    body1_id = sim.model.body_name2id(body1)
    body2_id = sim.model.body_name2id(body2)

    for i, (eq_type, obj1_id, obj2_id) in enumerate(zip(sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id)):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        if (obj1_id == body1_id and obj2_id == body2_id) or (obj2_id == body1_id and obj1_id == body2_id):
            sim.model.eq_data[i, :] = np.array(
                    [0., 0., 0., 1., 0., 0., 0.])
            sim.model.eq_active[i] = True
            