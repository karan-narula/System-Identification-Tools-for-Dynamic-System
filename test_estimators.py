from collections import OrderedDict as ODict

import math
import numpy as np
import matplotlib.pyplot as plt

from car_dynamics import sample_linear, FrontSteered, RoverPartialDynEst, FrontDriveFrontSteerEst, OneWheelFrictionEst
from estimators import kinematic_state_observer, fit_data_rover, fit_data_rover_dynobj
from utilities import create_dyn_obj, create_filtered_estimates, create_smoothed_estimates, plot_stuff, solve_ivp_dyn_obj


def test_fit_data_rover(param_dict, num_mc=100, back_rotate=False, **kwargs):
    """
    Test the function fit_data_rover for a specific configuration for a number of times. Generates various plots to
    see MSE at various times and how well trajectory fits with the estimated parameters.

    Args:
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        num_mc (int): number of monte carlo experiment to perform
        back_rotate (bool): produce linear and lateral velocities from rotating state coordinates?
        kwargs: dictionary of variable length for additional parameters for the create_dyn_obj function;
            see create_dyn_obj for more details

    """
    # timing variables
    dt = 0.05
    tf = 20.0
    T = np.arange(0.0, tf, dt)

    # create input vector for rover model
    cruise_time = 2.0
    U = sample_linear(T, cruise_time, *[math.pi/6.0, 5.0])

    kwargs['U'] = U
    kwargs['T'] = T

    square_errors = np.array([])
    interested_mc = int(math.floor(num_mc/2.0))
    for i in range(num_mc):
        # get data
        # control random seed generator
        np.random.seed(i)
        dynamic_obj = create_dyn_obj(
            RoverPartialDynEst, param_dict, simulate_gt=True, real_output=False, **kwargs)

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
            vxy = dynamic_obj.cal_vxvy_from_coord(output=True)
            vxind = dynamic_obj.state_indices.index(
                dynamic_obj.state_dict['vx'])
            dynamic_obj.outputs[vxind, :-1] = vxy[0, :]
            vy[:-1] = vxy[1, :]

            # LS + NLS
            parameters = fit_data_rover_dynobj(dynamic_obj)
            #parameters = fit_data_rover(
            #    dynamic_obj.outputs, dynamic_obj.U, dt, vy=vy)
        else:
            # LS + NLS
            #parameters = fit_data_rover(
            #    dynamic_obj.outputs, dynamic_obj.U, dt)
            parameters = fit_data_rover_dynobj(dynamic_obj)

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


def test_pbgf(dyn_class, param_dict, timing_vars={}, input_vars={}, ode_vars={}, **kwargs):
    """
    Test the PBGF in estimating the parameters and states.

    Args:
        dyn_class (class of type inherited from AbstractDyn): type to create dynamic object
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        timing_vars (dict): dictionary of parameters related to time such as duration of the simulation and dt
        input_vars (dict): dictionary of parameters related to input
        ode_vars (dict): dictionary of parameters required by the ode solver for the vehicle dynamics
        kwargs: dictionary of variable length for additional parameters for the create_dyn_obj function;
            see create_dyn_obj for more details

    """
    # timing variables
    dt = timing_vars.get('dt', 0.05)
    tf = timing_vars.get('tf', 20.0)
    T = np.arange(0.0, tf, dt)

    # create input vector from input variables
    sample_linear_flag = input_vars.get('sample_linear_flag', False)
    if sample_linear_flag:
        cruise_time = input_vars.get('cruise_time', 5.0)
        max_inputs_list = input_vars.get('max_inputs_list', [])
        U = sample_linear(T, cruise_time, *max_inputs_list)
    else:
        U = input_vars.get('U', np.array([]))

    kwargs['U'] = U
    kwargs['T'] = T

    # control random seed generator
    seed = kwargs.get('seed', 0)
    np.random.seed(seed)

    # get data
    dynamic_obj = create_dyn_obj(
        dyn_class, param_dict, simulate_gt=True, real_output=False, re_initialise=len(ode_vars) == 0, **kwargs)

    # optionally, use ode solver for the dynamic system and compare the results
    if len(ode_vars):
        if 'T' in ode_vars and sample_linear_flag:
            U = sample_linear(ode_vars['T'], cruise_time, *max_inputs_list)
            ode_vars['U'] = U
        solve_ivp_dyn_obj(dynamic_obj, **ode_vars)

    # get filtered or smoothed estimates
    operation = kwargs.get('operation', 'filter')
    assert operation in [
        'filter', 'smoother'], "Invalid estimation operation requested"
    if operation == 'filter':
        est_states = create_filtered_estimates(dynamic_obj, order=2)[0]
    else:
        lag_interval = kwargs.get('lag_interval', 5)
        est_states = create_smoothed_estimates(
            dynamic_obj, order=2, lag_interval=lag_interval)[0]

    # plot the convergence of the parameters
    plot_stuff(dynamic_obj, est_states, angle_states=configuration.get(
        'angle_states', []), encapsulated_gt=True, num_rows=[2, 2])


if __name__ == '__main__':
    # parameters under test for the rover model
    param_dict = ODict([('c1', 1.5), ('c2', 0.2), ('c3', 2.35), ('c4', 0.1),
                        ('c5', -0.0811), ('c6', -1.4736), ('c7', 0.1257), ('c8', 0.0765), ('c9', -0.0140)])
    configuration = {'output_keys': ['x', 'y', 'theta', 'vx'],
                     'output_dot_keys': ['theta'],
                     'init_param_cov': 1.0,
                     'std_x_out': 0.10,
                     'std_y_out': 0.10,
                     'std_theta_out': math.pi/180.0,
                     'std_theta_dot_out': math.pi/180.0}
    max_steering = 30.0*math.pi/180.0
    max_speed = 5.0
    max_inputs_list = [max_steering, max_speed]

    # test LS & NLS of rover model
    test_fit_data_rover(param_dict, back_rotate=False, **configuration)

    ## test pbgf for rover model
    configuration = {'seed': 0,
                     'output_keys': ['x', 'y'],
                     'est_params': ['c8', 'c9'],
                     'init_param_cov': 1.0,
                     'std_x_out': 0.10,
                     'std_y_out': 0.10}
    input_vars = {'sample_linear_flag': True,
                  'max_inputs_list': max_inputs_list}

    test_pbgf(RoverPartialDynEst, param_dict,
              input_vars=input_vars, **configuration)

    # test pbgf for FrontDriveFrontSteer
    param_dict = ODict([('fx', 15), ('cf', 0.75), ('cr', 0.75), ('lf', 0.8),
                        ('lr', 1.75), ('m', 1000.0), ('iz', 100.0), ('rc', 0.0), ('fr', 0.0), ('g', 9.8)])
    configuration = {'seed': 0,
                     'output_keys': ['x', 'y'],
                     'est_params': ['m', 'iz'],
                     'init_param_cov': 100,
                     'std_x_out': 0.10,
                     'std_y_out': 0.10,
                     'std_theta_out': math.pi/180.0,
                     'std_theta_dot_out': math.pi/180.0}
    max_a = 20.0
    max_steering = 30.0*math.pi/180.0
    input_vars = {'sample_linear_flag': True,
                  'max_inputs_list': [max_a, max_steering]}

    test_pbgf(FrontDriveFrontSteerEst, param_dict,
              input_vars=input_vars, **configuration)

    # test pbgf for FrontDriveFrontSteer with sudden change in fr
    param_dict = ODict([('fx', 15), ('cf', 0.75), ('cr', [0.75, 1.0]), ('lf', 0.8),
                        ('lr', 1.75), ('m', 1000.0), ('iz', 100.0), ('rc', 0.0), ('fr', [1.0, 2.0]), ('g', 9.8)])
    configuration = {}
    configuration = {'seed': 0,
                     'output_keys': ['theta', 'x'],
                     'output_dot_keys': ['theta'],
                     'est_params': ['fr', 'cr'],
                     'init_param_cov': 10,
                     'std_x_out': 0.1,
                     'std_y_out': 0.1,
                     'std_theta_out': math.pi/180.0,
                     'std_theta_dot_out': math.pi/180.0,
                     'time_varying_q': 1e-4,
                     'angle_states': ['theta', 'w']}
    max_a = 20.0
    max_steering = 30.0*math.pi/180.0
    input_vars = {'sample_linear_flag': True,
                  'max_inputs_list': [max_a, max_steering]}

    test_pbgf(FrontDriveFrontSteerEst, param_dict,
              input_vars=input_vars, **configuration)

    # same as previous one but get results using smoother
    configuration['operation'] = 'smoother'
    configuration['lag_interval'] = int(math.ceil(20*2))
    test_pbgf(FrontDriveFrontSteerEst, param_dict,
              input_vars=input_vars, **configuration)

    # test pbgf for the one wheel friction model with sudden change in road condition coefficient (parameters from EKF paper)
    param_dict = ODict([('sigma_0', 40.0), ('sigma_1', 4.9487), ('sigma_2', 0.0018), ('sigma_w', 0.0), ('L', 0.25), ('mu_c', 0.5),
                        ('mu_s', 0.9), ('vs', 12.5), ('r', 0.25), ('m', 5.0), ('J', 0.2344), ('Fn', 14.0), ('k', 1.1), ('theta', [1.0, 2.0])])
    configuration = {}
    configuration = {'seed': 0,
                     'output_keys': ['w'],
                     'est_params': ['theta'],
                     'init_param_cov': 1e-2,
                     'std_w': 1e-4,
                     'std_v': 1e-4,
                     'std_z': 1e-4,
                     'std_w_out': 3.0*math.pi/180.0,
                     'time_varying_q': 1e-4,
                     'angle_states': []}
    max_torque = 5.0
    timing_vars = {'dt': 0.0005, 'tf': 30}
    input_vars = {'sample_linear_flag': True, 'max_inputs_list': [max_torque]}
    ode_vars = {'T': np.arange(
        0.0, timing_vars['tf'], 0.1), 'plot_result': True, 'plot_euler_result': True, 'num_rows': 1}

    test_pbgf(OneWheelFrictionEst, param_dict, timing_vars=timing_vars,
              input_vars=input_vars, ode_vars=ode_vars, **configuration)
"""
Testing kinematic observer (Not working yet!)

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
    #temp.predict_and_or_update(X, P, None, None, Q, R, np.zeros((R.shape[0], 1)))
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
"""
