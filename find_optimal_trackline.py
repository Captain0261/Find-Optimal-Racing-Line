import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import gurobipy as gp
from gurobipy import GRB


def draw_track(cone_data, boundary_data):
    left_bound = []
    for cone in boundary_data['left']:
        x, y = cone_data[cone]
        left_bound.append([x, y])
    left_bound = np.array(left_bound)

    right_bound = []
    for cone in boundary_data['right']:
        x, y = cone_data[cone]
        right_bound.append([x, y])
    right_bound = np.array(right_bound)
    return left_bound, right_bound


def fit_track(left_bound, right_bound, N=1000):
    tck_left, u_left = splprep(
        [left_bound[:, 0], left_bound[:, 1]], s=10.0, per=1, k=3)
    tck_right, u_right = splprep(
        [right_bound[:, 0], right_bound[:, 1]], s=10.0, per=1, k=3)
    u_new = np.linspace(0, 1, N+1)[:-1]
    x_left, y_left = splev(u_new, tck_left, der=0)
    x_right, y_right = splev(u_new, tck_right, der=0)
    return x_left, y_left, x_right, y_right


def get_optimal_line(x_left, y_left, x_right, y_right, N=1000):
    dx_bound = x_right - x_left
    dy_bound = y_right - y_left

    model = gp.Model("get_optimal_line")
    model.Params.LogToConsole = 1
    alpha = model.addVars(N, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="alpha")
    obj = gp.QuadExpr()
    for i in range(N):
        im1 = (i - 1) % N
        ip1 = (i + 1) % N

        x_im1 = x_left[im1] + alpha[im1] * dx_bound[im1]
        x_i = x_left[i] + alpha[i] * dx_bound[i]
        x_ip1 = x_left[ip1] + alpha[ip1] * dx_bound[ip1]

        y_im1 = y_left[im1] + alpha[im1] * dy_bound[im1]
        y_i = y_left[i] + alpha[i] * dy_bound[i]
        y_ip1 = y_left[ip1] + alpha[ip1] * dy_bound[ip1]

        obj += (x_im1 - 2*x_i + x_ip1) * (x_im1 - 2*x_i + x_ip1)
        obj += (y_im1 - 2*y_i + y_ip1) * (y_im1 - 2*y_i + y_ip1)

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        alpha_opt = np.array([alpha[i].X for i in range(N)])
        opt_x = x_left + alpha_opt * dx_bound
        opt_y = y_left + alpha_opt * dy_bound
        return opt_x, opt_y, alpha_opt
    else:
        raise RuntimeError("Cannot find a solution")


def generate_velocity_profile(opt_x, opt_y, a_lat_max=10, a_acc=5, a_dec=8, v_max=25):
    N = len(opt_x)
    tck_opt, u_opt = splprep([opt_x, opt_y], s=0.0, per=1, k=3)

    dx, dy = splev(u_opt, tck_opt, der=1)
    ddx, ddy = splev(u_opt, tck_opt, der=2)

    kappa = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**(1.5) + 1e-8)

    v_profile = np.zeros(N)
    for i in range(N):
        if kappa[i] < 1e-5:
            v_profile[i] = v_max
        else:
            v_profile[i] = min(v_max, np.sqrt(a_lat_max / kappa[i]))

    ds = np.zeros(N)
    for i in range(N):
        next_i = (i + 1) % N
        ds[i] = np.hypot(opt_x[next_i] - opt_x[i], opt_y[next_i] - opt_y[i])

    for i in range(N - 2, -1, -1):
        v_limit_from_braking = np.sqrt(v_profile[i+1]**2 + 2 * a_dec * ds[i])
        v_profile[i] = min(v_profile[i], v_limit_from_braking)

    v_limit_from_braking = np.sqrt(v_profile[0]**2 + 2 * a_dec * ds[-1])
    v_profile[-1] = min(v_profile[-1], v_limit_from_braking)

    for i in range(N - 2, -1, -1):
        v_limit_from_braking = np.sqrt(v_profile[i+1]**2 + 2 * a_dec * ds[i])
        v_profile[i] = min(v_profile[i], v_limit_from_braking)

    for i in range(N - 1):
        v_limit_from_accel = np.sqrt(v_profile[i]**2 + 2 * a_acc * ds[i])
        v_profile[i+1] = min(v_profile[i+1], v_limit_from_accel)

    return v_profile, kappa


if __name__ == "__main__":
    with open('track_dataset/cone_map_1.yaml', 'r') as cone:
        cone_data = yaml.safe_load(cone)

    with open('track_dataset/boundaries_1.yaml', 'r') as boundary:
        boundary_data = yaml.safe_load(boundary)

    left_bound, right_bound = draw_track(cone_data, boundary_data)
    x_left, y_left, x_right, y_right = fit_track(
        left_bound, right_bound)

    opt_x, opt_y, alpha_opt = get_optimal_line(
        x_left, y_left, x_right, y_right)

    v_target, curvatures = generate_velocity_profile(opt_x, opt_y)

# draw the track
    plt.plot(left_bound[:, 0], left_bound[:, 1], 'y-o', label='Left Boundary')
    plt.plot(right_bound[:, 0], right_bound[:, 1],
             'b-o', label='Right Boundary')
    plt.plot(x_left, y_left, 'g-', linewidth=2, label='Fitted Track')
    plt.plot(x_right, y_right, 'g-', linewidth=2)
    plt.plot(opt_x, opt_y, 'r-', linewidth=2, label='Optimal Line')
    plt.axis('equal')
    plt.legend()
    plt.title("Track Boundaries")
    plt.show()
