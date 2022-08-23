import carla
import math
import numpy as np

def carla_loc_to_arr(loc):
    return [loc.x, loc.y, loc.z]

def to_body_frame(ego, w_locs):
    if w_locs is None or len(w_locs) == 0:
        return []

    loc = ego.get_location()
    yaw = ego.get_transform().rotation.yaw
    radian = math.radians(yaw)

    rotate_mat = np.array([[np.cos(-radian), np.sin(-radian), 0],
                            [-np.sin(-radian), np.cos(-radian), 0], [0, 0, 1]])

    w_locs_mat = []
    for w_loc in w_locs:
        w_locs_mat.append(carla_loc_to_arr(w_loc))

    shift_w_locs = np.array(w_locs_mat) - carla_loc_to_arr(loc)
    r_locs = np.dot(shift_w_locs, rotate_mat)

    return r_locs

def to_global_frame(ego, r_locs):
    if r_locs is None or len(r_locs) == 0:
        return []

    loc = ego.get_location()
    yaw = ego.get_transform().rotation.yaw
    radian = math.radians(yaw)

    rotate_mat = np.array([[np.cos(radian), np.sin(radian), 0],
                            [-np.sin(radian), np.cos(radian), 0]])
    M = np.dot(r_locs, rotate_mat) + carla_loc_to_arr(loc)

    w_locs = []
    for row in M:
        w_locs.append(carla.Location(x=row[0], y=row[1], z=0))

    return w_locs

def r_loc_2_vec_3d(r_loc):
    return carla.Vector3D(float(r_loc[0]), float(r_loc[1]), 0)

def get_vector_degree(vec1, vec2):
    length_product = vec1.length() * vec2.length()
    if length_product == 0:
        return 0
    return np.rad2deg(np.arccos(vec1.dot(vec2) / length_product))