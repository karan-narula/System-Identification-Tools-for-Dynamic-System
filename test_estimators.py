from collections import OrderedDict as ODict
from collections import Iterable

import math
import numpy as np
import matplotlib.pyplot as plt

from car_dynamics import sample_linear, FrontSteered, RoverPartialDynEst, FrontDriveFrontSteerEst
from estimators import kinematic_state_observer, PointBasedFilter, fit_data_rover, fit_data_rover_dynobj


def bind_npi_pi(angles):
    angles = np.fmod(angles + math.pi, 2*math.pi)
    angles[angles < 0] += 2*math.pi
    angles -= math.pi

    return angles


def plot_stuff(dynamic_obj, est_states, num_row=1):
    """
    Useful function for plotting stuff. It plots 2 figures: 1 -> estimated parameters vs gt,
    2-> main dynamic states such as trajectory, heading, etc

    Args:
        dynamic_obj (RoverPartialDynEst obj): dynamic object
        est_states (numpy array [4+len(dynamic_obj.est_params) x nt]): estimated states of the dynamic model,
            same shape as dynamic_obj.gt_states
        num_row (int): number of rows of subplots in figure 1

    """
    # first figure for the parameters
    num_est_params = len(dynamic_obj.est_params)
    num_main_states = dynamic_obj.num_states - num_est_params
    if num_est_params < num_row:
        num_row = num_est_params
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


def simulate_data(dyn_class, param_dict, U, T, **kwargs):
    """
    Simulate ground truth, initial condition and output data for a specified dynamics class.

    Args:
        dyn_class (class of type inherited from AbstractDyn): type to create dynamic object
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        U (numpy array [2 x nt]): input consisting of steering angle and velocity at different time instances
        T (numpy array [nt]): time instances corresponding to the input U
        kwargs: dictionary of variable length for additional parameters

    kwargs:
        angle_states (list): list of heading state keys to be collated with binding function and stored in dynamic object
        seed (int): seed for the random generator for repeatability of the tests; defaults to 0
        std_* (float): standard deviation for additive noise for evolution model of state *; * must be a key in the
            state_dict of dyn_class; defaults to 0.0 when not given for a state in state_dict of dyn_class
        std_*_out (float): standard deviation for additive noise for observation of state *; * must be a key in the
            state_dict of dyn_class; defaults to 1e-1 when not given for a state in output_keys
        std_*_dot_out (float): standard deviation for additive noise for observation of rate of change of state *;
            * must be a key in the state_dict of dyn_class; defaults to 1e-1 when not given for a state in output_dot_keys
        est_params (list): list of parameter strings to be estimated; defaults to empty
        output_keys (list): list of state strings that are observed; must be from state_dict of dyn_class; defaults to
            keys of state_dict of dyn_class
        output_dot_keys (list): list of state derivative strings that are observed; must be from state_dict of dyn_class;
            defaults to empty
        init_state_cov (list or float): if list must be of the same length as state_dict of dyn_class; covariance of initial
            condition for the dynamic states; defaults to 0.0
        init_param_cov (list or float): if list must be of size as len(est_params); covariance of initial condition for the
            parameters; defaults to 0.0

    Returns:
        dynamic_obj (dyn_class obj): dynamic object

    """
    # retrieve state dictionary from dynamic class
    state_dict = dyn_class.global_state_dict

    # check if expected inputs are iterable, if not convert to as such
    est_params = kwargs.get('est_params', [])
    if not isinstance(est_params, Iterable):
        est_params = [est_params]

    init_state_cov = kwargs.get('init_state_cov', 0.0)
    if not isinstance(init_state_cov, Iterable):
        init_state_cov = [init_state_cov]*len(state_dict)

    init_param_cov = kwargs.get('init_param_cov', 0.0)
    if not isinstance(init_param_cov, Iterable):
        init_param_cov = [init_param_cov]*len(est_params)

    assert len(est_params) == len(init_param_cov), "Expected parameters to be estimated to be of the same length as initial covariance but instead got {} and {} respectively".format(
        len(est_params), len(init_param_cov))
    assert len(init_state_cov) == len(state_dict), "Expected initial covariance of the states to be of size {} but instead got {}".format(
        len(state_dict), len(init_state_cov))

    # control random seed generator
    seed = kwargs.get('seed', 0)
    np.random.seed(seed)

    # covariance matrix of additive GWN in stochastic model
    temp = [0.0]*len(state_dict)
    for key in state_dict:
        temp[state_dict[key]] = kwargs.get('std_' + key, 0.0)**2
    time_varying_q = kwargs.get('time_varying_q', 1e-4)
    for est_param in est_params:
        if isinstance(param_dict[est_param], Iterable):
            temp.append(time_varying_q)
        else:
            temp.append(0.0)
    Q = np.diag(temp)

    # covariance matrix of additive GWN in observation model
    output_keys = kwargs.get('output_keys', state_dict.keys())
    output_dot_keys = kwargs.get('output_dot_keys', [])

    vars_out = []
    for output_key in output_keys:
        if output_key in state_dict:
            vars_out.append(kwargs.get(
                'std_' + output_key + '_out', 0.10)**2)

    for output_dot_key in output_dot_keys:
        if output_dot_key in state_dict:
            vars_out.append(kwargs.get(
                'std_' + output_dot_key + '_dot_out', 0.10)**2)

    R = np.diag(vars_out)

    # adjust param dict when full array is not specified
    overwrite_keys = []
    overwrite_vals = []
    for key in param_dict:
        if isinstance(param_dict[key], Iterable):
            if len(param_dict[key]) != len(T):
                temp = param_dict[key].copy()
                param_dict[key] = [temp[-1]]*len(T)

                num_each = math.floor(len(T)/len(temp))
                ind = 0
                for item in temp:
                    param_dict[key][ind:ind+num_each] = [item]*num_each
                    ind += num_each
            if key in est_params:
                overwrite_keys.append(key)
                overwrite_vals.append(param_dict[key].copy())

    # ground truth initial condition
    z0 = [0.0]*len(state_dict)
    for est_param in est_params:
        if isinstance(param_dict[est_param], Iterable):
            z0.append(param_dict[est_param][0])
        else:
            z0.append(param_dict[est_param])
    z0 = np.array([z0]).T

    # initial belief, uncertainty of initial condition
    temp = list(init_state_cov)
    temp.extend(init_param_cov)
    P0 = np.diag(temp)

    # create the ground truth and noisy states
    dynamic_obj = dyn_class(
        param_dict, est_params, state_keys=output_keys, state_dot_keys=output_dot_keys, simulate_gt=True)
    dynamic_obj.sample_nlds(z0, U, T, Q=Q, P0=P0, R=R, store_variables=True,
                            overwrite_keys=overwrite_keys, overwrite_vals=overwrite_vals)
    dynamic_obj.re_initialise()

    # create innovation bound function mapping for heading states
    innovation_bound_func = {}
    angle_states = kwargs.get('angle_states', [])
    if not isinstance(angle_states, Iterable):
        angle_states = [angle_states]
    for angle_state in angle_states:
        assert angle_state in state_dict, "Specified angle state not in state dictionary"
        innovation_bound_func[dynamic_obj.state_indices.index(
            state_dict[angle_state])] = bind_npi_pi
        innovation_bound_func[len(dynamic_obj.state_indices) +
                              dynamic_obj.state_dot_indices.index(state_dict[angle_state])] = bind_npi_pi

    dynamic_obj.innovation_bound_func = innovation_bound_func

    return dynamic_obj


def create_filtered_estimates(dynamic_obj, method='CKF', order=2):
    """
    Generate mean and covariance of filtered distribution at various times for the problem defined by the dynamic object.

    Args:
        dynamic_obj (obj derived from AbstractDyn): dynamic object encapsulating source of information for the filter
        method (str): The method for filtering algorithm, see estimators.py for the methods currently implemented; defaults to 'CKF'
        order (int): Order of accuracy for integration rule, see estimators.py for orders currently implemented; defaults to 2

    Returns:
        est_states (numpy array [dynamic_obj.num_states x nt]): filtered mean estimates of the states at different time instances
        cov_states (numpy array [dynamic_obj.num_states x dynamic_obj.num_states x nt]): filtered covariance of the states
            at different time instances

    """

    # create instance of the filter
    pbgf = PointBasedFilter(method, order)

    if hasattr(dynamic_obj, 'innovation_bound_func'):
        innovation_bound_func = dynamic_obj.innovation_bound_func
    else:
        innovation_bound_func = {}

    # filtering loop
    num_sol = len(dynamic_obj.T)
    est_states = np.zeros((dynamic_obj.num_states, num_sol))
    est_states[:, 0:1] = dynamic_obj.initial_cond
    cov_states = np.zeros(
        (dynamic_obj.num_states, dynamic_obj.num_states, num_sol))
    cov_states[:, :, 0] = dynamic_obj.P0.copy()
    for i in range(1, num_sol):
        est_states[:, i:i+1], cov_states[:, :, i] = pbgf.predict_and_or_update(est_states[:, i-1:i], cov_states[:, :, i-1], dynamic_obj.process_model, dynamic_obj.observation_model, dynamic_obj.Q, dynamic_obj.R, dynamic_obj.U[:, i-1], dynamic_obj.outputs[:, i:i+1], additional_args_pm=[
                                                                               sub[i-1] for sub in dynamic_obj.additional_args_pm_list], additional_args_om=[sub[i] for sub in dynamic_obj.additional_args_om_list], innovation_bound_func=innovation_bound_func)

    return est_states, cov_states


def test_fit_data_rover(param_dict, num_mc=100, back_rotate=False, **kwargs):
    """
    Test the function fit_data_rover for a specific configuration for a number of times. Generates various plots to
    see MSE at various times and how well trajectory fits with the estimated parameters.

    Args:
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        num_mc (int): number of monte carlo experiment to perform
        back_rotate (bool): produce linear and lateral velocities from rotating state coordinates?
        kwargs: dictionary of variable length for additional parameters for the simulate_data function;
            see simulate_data for more details

    """
    # timing variables
    dt = 0.05
    tf = 20.0
    T = np.arange(0.0, tf, dt)

    # create input vector for rover model
    cruise_time = 2.0
    U = sample_linear(T, cruise_time, *[math.pi/6.0, 5.0])

    square_errors = np.array([])
    interested_mc = int(math.floor(num_mc/2.0))
    for i in range(num_mc):
        # get data
        kwargs['seed'] = i
        dynamic_obj = simulate_data(
            RoverPartialDynEst, param_dict, U, T, **kwargs)

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


def test_pbgf(dyn_class, param_dict, max_inputs_list, **kwargs):
    """
    Test the PBGF in estimating the parameters and states.

    Args:
        dyn_class (class of type inherited from AbstractDyn): type to create dynamic object
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        max_inputs_list (list): list of maximum inputs for generating inputs for the dynamic model
        kwargs: dictionary of variable length for additional parameters for the simulate_data function;
            see simulate_data for more details

    """
    # timing variables
    dt = 0.05
    tf = 20.0
    T = np.arange(0.0, tf, dt)

    # create input vector for rover model
    cruise_time = 5.0
    U = sample_linear(T, cruise_time, *max_inputs_list)

    # get data
    dynamic_obj = simulate_data(
        dyn_class, param_dict, U, T, **kwargs)

    # get filtered estimates
    est_states = create_filtered_estimates(dynamic_obj, order=2)[0]

    # plot the convergence of the parameters
    plot_stuff(dynamic_obj, est_states, num_row=2)


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

    test_pbgf(RoverPartialDynEst, param_dict, max_inputs_list, **configuration)

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
    max_inputs_list = [max_a, max_steering]
    test_pbgf(FrontDriveFrontSteerEst, param_dict,
              max_inputs_list, **configuration)

    # test pbgf for FrontDriveFrontSteer with sudden change in fr
    param_dict = ODict([('fx', 15), ('cf', 0.75), ('cr', 0.75), ('lf', 0.8),
                        ('lr', 1.75), ('m', 1000.0), ('iz', 100.0), ('rc', 0.0), ('fr', [1.0, 2.0]), ('g', 9.8)])
    configuration = {}
    configuration = {'seed': 0,
                     'output_keys': ['x', 'y', 'theta'],
                     'est_params': ['fr'],
                     'init_param_cov': 10,
                     'std_x_out': 0.10,
                     'std_y_out': 0.10,
                     'std_theta_out': math.pi/180.0,
                     'std_theta_dot_out': math.pi/180.0,
                     'time_varying_q': 1e-4}
    max_a = 20.0
    max_steering = 30.0*math.pi/180.0
    max_inputs_list = [max_a, max_steering]
    test_pbgf(FrontDriveFrontSteerEst, param_dict,
              max_inputs_list, **configuration)

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
