from collections import OrderedDict as ODict
from collections import Iterable

import math
import numpy as np
import matplotlib.pyplot as plt

from car_dynamics import sample_input_front_steered, FrontSteered, sample_input_rover, RoverPartialDynEst
from estimators import kinematic_state_observer, PointBasedFilter, fit_data_rover


def plot_stuff(dynamic_obj, est_states, num_row=1):
    """
    Plot 2 figures: 1 -> estimated parameters vs gt, 2-> main dynamic states such as trajectory, heading, etc
    """
    # first figure for the parameters
    num_est_params = len(dynamic_obj.est_params)
    num_main_states = dynamic_obj.num_states - num_est_params
    num_col = int(math.ceil(num_est_params/float(num_row)))
    plt.figure(1)
    for i, j in enumerate(range(num_main_states, dynamic_obj.num_states)):
        plt.subplot(num_row, num_col, i+1)
        plt.plot(dynamic_obj.T, dynamic_obj.gt_states[j, :],
                 label='gt', linestyle='--')
        plt.plot(dynamic_obj.T, est_states[j, :], label='est')
        plt.legend()
        plt.grid(True, "both")
        plt.xlabel('Time (seconds)')
        plt.ylabel(dynamic_obj.est_params[i])

    # second figure is to examine the main states
    plt.figure(2)
    num_col = int(math.ceil((num_main_states-1)/2.0))
    plt.subplot(2, num_col, 1)
    plt.plot(est_states[dynamic_obj.state_dict['x'], :],
             est_states[dynamic_obj.state_dict['y'], :], label='est')
    plt.plot(dynamic_obj.gt_states[dynamic_obj.state_dict['x'], :],
             dynamic_obj.gt_states[dynamic_obj.state_dict['y'], :], label='gt')
    if dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict['y'] in dynamic_obj.state_indices:
        plt.plot(dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['x']), :],
                 dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['y']), :], label='output')

    plt.grid(True, "both")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()

    other_states_dict = {x: dynamic_obj.state_dict[x]
                         for x in dynamic_obj.state_dict if x not in ['x', 'y'] and x not in dynamic_obj.est_params}
    for i, key in enumerate(other_states_dict):
        state_ind = other_states_dict[key]
        plt.subplot(2, num_col, i+2)
        plt.plot(dynamic_obj.T, est_states[state_ind, :], label='est')
        plt.plot(dynamic_obj.T,
                 dynamic_obj.gt_states[state_ind, :], label='gt')
        if state_ind in dynamic_obj.state_indices:
            plt.plot(dynamic_obj.T, dynamic_obj.outputs[dynamic_obj.state_indices.index(
                state_ind), :], label='output')
        plt.grid(True, "both")
        plt.xlabel('Time (s)')
        plt.ylabel(key)
        plt.legend()
    plt.show()


def simulate_rover_data(param_dict, U, T, seed=0, std_x=0.0, std_y=0.0, std_theta=0.0, std_vx=0.0, std_x_out=1e-1, std_y_out=1e-1, std_theta_out=1.0, std_vx_out=0.1, std_x_dot_out=1e-1, std_y_dot_out=1e-1, std_theta_dot_out=1.0, std_vx_dot_out=1e-1, est_params=['c8', 'c9'], output_keys=['x', 'y', 'theta', 'vx'], output_dot_keys=[], init_state_cov=0.0, init_param_cov=0.0):
    # check if expected inputs are iterable, if not convert to as such
    if not isinstance(est_params, Iterable):
        est_params = [est_params]

    if not isinstance(init_state_cov, Iterable):
        init_state_cov = [init_state_cov]*4

    if not isinstance(init_param_cov, Iterable):
        init_param_cov = [init_param_cov]*len(est_params)

    assert len(est_params) == len(init_param_cov), "Expected parameters to be estimated to be of the same length as initial covariance but instead got {} and {} respectively".format(
        len(est_params), len(init_param_cov))
    assert len(init_state_cov) == 4, "Expected initial covariance of the states to be of size 4 but instead got {}".format(
        len(init_state_cov))

    # control random seed generator
    np.random.seed(seed)

    # covariance matrix of additive GWN in stochastic model
    std_theta *= math.pi/180.0
    temp = [std_x**2, std_y**2, std_theta ** 2, std_vx**2]
    temp.extend([0.0]*len(est_params))
    Q = np.diag(temp)

    # covariance matrix of additive GWN in observation model
    std_theta_out *= math.pi/180.0
    std_theta_dot_out *= math.pi/180.0
    vars_out = []
    for output_key in output_keys:
        if output_key == 'x':
            vars_out.append(std_x_out**2)
        elif output_key == 'y':
            vars_out.append(std_y_out**2)
        elif output_key == 'theta':
            vars_out.append(std_theta_out**2)
        elif output_key == 'vx':
            vars_out.append(std_vx_out**2)
    for output_dot_key in output_dot_keys:
        if output_dot_key == 'x':
            vars_out.append(std_x_dot_out**2)
        elif output_dot_key == 'y':
            vars_out.append(std_y_dot_out**2)
        elif output_dot_key == 'theta':
            vars_out.append(std_theta_dot_out**2)
        elif output_dot_key == 'vx':
            vars_out.append(std_vx_dot_out**2)

    R = np.diag(vars_out)

    # ground truth initial condition
    z0 = [0.0]*4
    for est_param in est_params:
        z0.append(param_dict[est_param])
    z0 = np.array([z0]).T

    # initial belief, uncertainty of initial condition
    temp = list(init_state_cov)
    temp.extend(init_param_cov)
    P0 = np.diag(temp)

    # create the ground truth and noisy states
    dynamic_obj = RoverPartialDynEst(
        param_dict, est_params, state_keys=output_keys, state_dot_keys=output_dot_keys)
    dynamic_obj.sample_nlds(z0, U, T, Q=Q, P0=P0, R=R, store_variables=True)

    return dynamic_obj


def simulate_rover_data_wrapper(est_params, output_keys, output_dot_keys, seed=0):
    """
    Just calls simulate_rover_data with parameters highlighted here.
    To see the modification to tests, parameters can just be modified in this file.
    """
    # ground truth parameters
    param_dict = ODict([('c1', 1.5), ('c2', 0.2), ('c3', 2.35), ('c4', 0.1),
                        ('c5', -0.0811), ('c6', -1.4736), ('c7', 0.1257), ('c8', 0.0765), ('c9', -0.0140)])

    # stds for stochastic model
    std_x = std_y = std_theta = std_vx = 0.0

    # stds for output
    std_x_out = std_y_out = std_x_dot_out = std_y_dot_out = 0.10
    std_theta_out = std_theta_dot_out = 1.0
    std_vx_out = std_vx_dot_out = 0.10

    # timing variables
    dt = 0.05
    tf = 20.0
    T = np.arange(0.0, tf, dt)

    # create input vector for rover model
    U = sample_input_rover(T)

    # initial covariance
    init_state_cov = 0.0
    init_param_cov = 1.0

    # get dynamic object
    dynamic_obj = simulate_rover_data(param_dict, U, T, seed=seed, std_x=std_x, std_y=std_y, std_theta=std_theta, std_vx=std_vx, std_x_out=std_x_out, std_y_out=std_y_out, std_theta_out=std_theta_out, std_vx_out=std_vx_out, std_x_dot_out=std_x_dot_out,
                                      std_y_dot_out=std_y_dot_out, std_theta_dot_out=std_theta_dot_out, std_vx_dot_out=std_vx_dot_out, est_params=est_params, output_keys=output_keys, output_dot_keys=output_dot_keys, init_state_cov=init_state_cov, init_param_cov=init_param_cov)

    return dynamic_obj


def create_filtered_estimates(dynamic_obj, method='CKF', order=2):
    """
    Generate mean and covariance of filtered distribution at various times for the problem defined by:
    dynamic_obj: source of information for the filter
    """

    # create instance of the filter
    pbgf = PointBasedFilter(method, order)

    # filtering loop
    num_sol = len(dynamic_obj.T)
    est_states = np.zeros((dynamic_obj.num_states, num_sol))
    est_states[:, 0:1] = dynamic_obj.initial_cond
    cov_states = np.zeros(
        (dynamic_obj.num_states, dynamic_obj.num_states, num_sol))
    cov_states[:, :, 0] = dynamic_obj.P0.copy()
    for i in range(1, num_sol):
        est_states[:, i:i+1], cov_states[:, :, i] = pbgf.predict_and_update(est_states[:, i-1:i], cov_states[:, :, i-1], dynamic_obj.process_model,
                                                                            dynamic_obj.observation_model, dynamic_obj.Q, dynamic_obj.R, dynamic_obj.U[:, i-1], dynamic_obj.outputs[:, i:i+1], additional_args_pm=dynamic_obj.additional_args_pm_array[:, i-1], additional_args_om=dynamic_obj.additional_args_om_array[:, i])

    return est_states, cov_states


def test_fit_data_rover(num_mc=100, back_rotate=False):
    # get data
    est_params = []
    output_keys = ['x', 'y', 'theta', 'vx']
    output_dot_keys = ['theta']
    square_errors = np.array([])
    interested_mc = int(math.floor(num_mc/2.0))
    for i in range(num_mc):
        dynamic_obj = simulate_rover_data_wrapper(
            est_params, output_keys, output_dot_keys, seed=i)

        if square_errors.shape[0] == 0:
            square_errors = np.zeros((4, len(dynamic_obj.T), num_mc))

        # store the ground truth data
        gt_states = dynamic_obj.gt_states.copy()

        # prepare input for fitting function
        dt = np.diff(dynamic_obj.T).mean()
        vxdot = dynamic_obj.gt_states_dot[dynamic_obj.state_dict['vx'], :]
        yawrate = dynamic_obj.gt_states_dot[dynamic_obj.state_dict['theta'], :]
        vy = yawrate*(dynamic_obj.param_dict['c8'] + dynamic_obj.param_dict['c9']
                      * dynamic_obj.gt_states[dynamic_obj.state_dict['vx'], :]**2)

        if back_rotate and dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict['y'] in dynamic_obj.state_indices:
            vxy = dynamic_obj.cal_vxvy_from_coord(
                dynamic_obj.outputs[:, 1:], dynamic_obj.outputs[:, :-1], dt, output=True)
            vxind = dynamic_obj.state_indices.index(
                dynamic_obj.state_dict['vx'])
            dynamic_obj.outputs[vxind, :-1] = vxy[0, :]
            vy[:-1] = vxy[1, :]

            # LS + NLS
            parameters = fit_data_rover(
                dynamic_obj.outputs, dynamic_obj.U, dt, vy=vy)
        else:
            # LS + NLS
            parameters = fit_data_rover(
                dynamic_obj.outputs, dynamic_obj.U, dt)

        # forward integrate the model with new parameters
        for key, parameter in zip(dynamic_obj.param_dict, parameters):
            dynamic_obj.param_dict[key] = parameter

        dynamic_obj.sample_nlds(
            dynamic_obj.initial_cond, dynamic_obj.U, dynamic_obj.T)

        square_errors[:, :, i] = np.square(gt_states - dynamic_obj.gt_states)

        if i == interested_mc:
            i_gt_states = gt_states
            i_approx_states = dynamic_obj.gt_states

    MSE = np.mean(square_errors, axis=2)

    # Visually compare the trajectory at interested mc
    plt.subplot(1, 2, 1)
    plt.plot(i_gt_states[dynamic_obj.state_dict['x'], :],
             i_gt_states[dynamic_obj.state_dict['y'], :], label='gt')
    plt.plot(i_approx_states[dynamic_obj.state_dict['x'], :],
             i_approx_states[dynamic_obj.state_dict['y'], :], label='nls_fit')
    plt.grid(True, "both")
    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title("Trajectory comparison at {}th experiment".format(interested_mc))

    plt.subplot(1, 2, 2)
    for key in dynamic_obj.state_dict:
        state_ind = dynamic_obj.state_dict[key]
        plt.plot(dynamic_obj.T, MSE[state_ind, :], label=key)
    plt.grid(True, "both")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.title('MSE at different time instant from {} mc simulations'.format(num_mc))
    plt.show()


def test_pbgf():
    # get data
    est_params = ['c8', 'c9']
    output_keys = ['x', 'y']
    output_dot_keys = ['theta']
    dynamic_obj = simulate_rover_data_wrapper(
        est_params, output_keys, output_dot_keys)

    # get filtered estimates
    est_states = create_filtered_estimates(dynamic_obj, order=2)[0]

    # plot the convergence of the parameters
    plot_stuff(dynamic_obj, est_states, num_row=2)


if __name__ == '__main__':
    test_fit_data_rover(back_rotate=False)
    test_pbgf()
else:
    # control random seed generator
    np.random.seed(0)

    # stds of GWN on ground truth states
    std_x = 0.05
    std_y = 0.05
    std_theta = 2.0*math.pi/180.0
    std_v = 0.1
    std_w = 1.0*math.pi/180.0
    Q = np.diag([std_x**2, std_y**2, std_theta **
                 2, std_v**2, std_v**2, std_w**2])

    # stds of GWN on output
    std_ax = 1e-6
    std_ay = 1e-6
    std_vx = 0.
    std_omega = 0.
    R = 0.0*np.diag([std_ax**2, std_ay**2, std_vx**2, std_omega**2])

    # parameters for the model
    param_dict = dict()
    param_dict['mass'] = 1301
    param_dict['lr'] = 1.45
    param_dict['lf'] = 1.0
    param_dict['e_wr'] = 0.33
    param_dict['cxf'] = 0.75
    param_dict['cxr'] = 0.8
    param_dict['cyf'] = 0.5
    param_dict['cyr'] = 0.6
    param_dict['iz'] = 1627

    # initial condition of the car states
    z0 = np.zeros((6, 1))  # [x, y, theta, vx, vy, omega]
    z0[3] = 1.0

    # timing information
    dt = 0.05
    t_f = 20.0
    T = np.arange(0, t_f, dt)

    # create input vector specific to dynamic model of interest
    U = sample_input_front_steered(T)

    # create the ground truth and noisy states
    dynamic_obj = FrontSteered(
        param_dict, output_type='inertial_acc', state_keys=['vx', 'omega'])
    gt_states, _, initial_cond, outputs = dynamic_obj.sample_nlds(
        z0, U, T, Q=Q, R=R)

    # use kinematic state observer
    alpha = 1.0
    sub_states = kinematic_state_observer(
        initial_cond, outputs[3, :], outputs[0:2, :], outputs[2, :], T, alpha)

    temp = PointBasedFilter('UKF', 2)
    X = 2*np.random.rand(3, 1)
    P = 5*np.random.rand(3, 3)
    P = np.matmul(P, np.transpose(P))
    x, L, W, WeightMat = temp.sigmas2(X, P)
    #temp.predict_and_update(X, P, None, None, Q, R, np.zeros((R.shape[0], 1)))
    print(temp.verifySigma(x, WeightMat, X, P))

    # plot the theoretical vs noisy trajectory
    plt.subplot(2, 1, 1)
    plt.plot(gt_states[0, :], gt_states[1, :], label='gt')
    plt.title('Driven Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # plot the estimated lateral & longitudinal velocity
    plt.subplot(2, 2, 3)
    plt.plot(T, gt_states[3, :], label='gt')
    plt.plot(T, sub_states[0, :], label='est')
    plt.title('Comparison of Estimated long. velocity to GT')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('V (m/s)')

    plt.subplot(2, 2, 4)
    plt.plot(T, gt_states[4, :], label='gt')
    plt.plot(T, sub_states[1, :], label='est')
    plt.title('Comparison of Estimated lat. velocity to GT')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('V (m/s)')

    plt.show()
