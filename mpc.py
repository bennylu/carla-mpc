import math
import numpy as np
import utils
from scipy.optimize import minimize

class MPC:

    HORIZON = 6
    L = 2.5

    def __init__(self, drawer, ego):
        self.drawer = drawer
        self.ego = ego

    def run_step(self, path, speed_m_s, dt):
        # convert path to body frame
        path = utils.to_body_frame(self.ego, path)

        # fit the path with a polynomial
        poly = np.polyfit(path[:, 0], path[:, 1], 3)

        # generate lane center locations
        self.speed_dt = speed_m_s * dt

        x = 0
        x_arr = []
        for i in range(0, self.HORIZON):
            x += self.speed_dt
            x_arr.append(x)
        self.locs = np.vstack((x_arr, np.polyval(poly, x_arr))).T
        self.drawer.draw_camera_lines((255, 255, 255), utils.to_global_frame(self.ego, self.locs), 1)

        self.psi_arr = []
        vec_0 = utils.r_loc_2_vec_3d([1, 0, 0])
        for i in range(self.HORIZON - 1):
            vec_1 = utils.r_loc_2_vec_3d(self.locs[i + 1]) - utils.r_loc_2_vec_3d(self.locs[i])
            psi = np.deg2rad(utils.get_vector_degree(vec_0, vec_1))
            if vec_1.y < 0:
                psi = -psi
            self.psi_arr.append(psi)

        # mpc
        bounds = np.full((self.HORIZON, 2), (-0.3, 0.3))
        init_steer_arr = np.full(self.HORIZON, 0)
        solution = minimize(self.objective, init_steer_arr, (), method='SLSQP', bounds=bounds, tol=1e-4)
        eval_states = np.array(self.evaluate_states(solution.x))[:, 0:2]
        self.drawer.draw_camera_lines((0, 0, 255), utils.to_global_frame(self.ego, eval_states), 1)

        return solution.x[0]
    
    def mpc_model(self, state, steering):
        x_t = state[0]
        y_t = state[1]
        psi_t = state[2]

        x_t_1 = x_t + self.speed_dt * np.cos(psi_t)
        y_t_1 = y_t + self.speed_dt * np.sin(psi_t)
        psi_t_1 = psi_t + self.speed_dt * np.tan(steering) / self.L

        return [x_t_1, y_t_1, psi_t_1]
    
    def evaluate_states(self, steer_arr):
        states = []
        state = [0, 0, 0]
        for steer in steer_arr:
            state = self.mpc_model(state, steer)
            states.append(state)
        return states

    def objective(self, steer_arr, *args):
        states = self.evaluate_states(steer_arr)

        cost_cte = 0
        for i in range(self.HORIZON):
            state = states[i]
            loc = self.locs[i]

            if i < self.HORIZON - 1:
                psi = self.psi_arr[i]
            else:
                psi = 0

            cost_cte += math.sqrt((state[0] - loc[0]) ** 2 + (state[1] - loc[1]) ** 2 + (state[2] - psi) ** 2)

        return cost_cte