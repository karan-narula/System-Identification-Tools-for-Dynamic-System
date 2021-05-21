from collections import Iterable

import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from estimators import PointBasedFilter, PointBasedFixedLagSmoother


def bind_npi_pi(angles):
    angles = np.fmod(angles + math.pi, 2*math.pi)
    angles[angles < 0] += 2*math.pi
    angles -= math.pi

    return angles


def create_dyn_obj(dyn_class, param_dict, simulate_gt=False, real_output=True, re_initialise=False, **kwargs):
    """
    Create dynamic object for specified class encapsulating some information useful during the estimation process

    Args:
        dyn_class (class of type inherited from AbstractDyn): type to create dynamic object
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        simulate_gt (bool): whether to simulate ground truth data for dynamic object; defaults to false
        real_output (bool): whether dynamic object will later be used to encapsulate real data; defaults to true
        re_initialise (bool): whether to reinitialise the dynamic object after simulating ground truth data; defaults to false
        kwargs: dictionary of variable length for additional parameters

    kwargs:
        U (numpy array [num_in x nt]): simulated input for the model to be used for generating ground truth data
        T (numpy array [nt]): time instances corresponding to the input U
        angle_states (list): list of heading state keys to be collated with binding function and stored in dynamic object
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
        init_params (list or float): if list must be of size as len(est_params); initial value of parameters; defaults to empty

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

    init_params = kwargs.get('init_params', [0.0]*len(est_params))
    if not isinstance(init_params, Iterable):
        init_params = [init_params]

    init_param_cov = kwargs.get('init_param_cov', 0.0)
    if not isinstance(init_param_cov, Iterable):
        init_param_cov = [init_param_cov]*len(est_params)

    assert len(est_params) == len(init_param_cov), "Expected parameters to be estimated to be of the same length as initial covariance but instead got {} and {} respectively".format(
        len(est_params), len(init_param_cov))
    assert len(init_state_cov) == len(state_dict), "Expected initial covariance of the states to be of size {} but instead got {}".format(
        len(state_dict), len(init_state_cov))
    assert len(init_params) == len(est_params), "Initial value of parameters must be of the same length as the list of parameters to be instead but instead got {} and {} respectively".format(
        len(init_params), len(est_params))

    # covariance matrix of additive GWN in stochastic model
    temp = [0.0]*len(state_dict)
    for key in state_dict:
        temp[state_dict[key]] = kwargs.get('std_' + key, 0.0)**2
    time_varying_q = kwargs.get('time_varying_q', 0.0)
    temp.extend([time_varying_q]*len(est_params))
    Q = np.diag(temp)

    # covariance matrix of additive GWN in observation model
    output_keys = kwargs.get('output_keys', state_dict.keys())
    output_dot_keys = kwargs.get('output_dot_keys', [])
    output_additional_keys = kwargs.get('output_additional_keys', [])

    if real_output:
        output_data_keys = kwargs.get('output_data_keys', state_dict.keys())
        output_data_dot_keys = kwargs.get('output_data_dot_keys', [])
        output_data_additional_keys = kwargs.get(
            'output_data_additional_keys', [])
        assert len(output_data_keys) == len(output_keys), "Expected output keys to be of the same length as its mapping in data file but instead got {} and {} respectively".format(
            len(output_data_keys), len(output_keys))
        assert len(output_dot_keys) == len(output_data_dot_keys), "Expected derivative of output keys to be of the same length as its mapping in data file but instead got {} and {} respectively".format(
            len(output_dot_keys), len(output_data_dot_keys))
        assert len(output_additional_keys) == len(output_data_additional_keys), "Expected additional output keys to be of the same length as its mapping in data file but instead got {} and {} respectively".format(
            len(output_additional_keys), len(output_data_additional_keys))

    vars_out = []
    for output_key in output_keys:
        if output_key in state_dict:
            vars_out.append(kwargs.get(
                'std_' + output_key + '_out', 0.10)**2)

    for output_dot_key in output_dot_keys:
        if output_dot_key in state_dict:
            vars_out.append(kwargs.get(
                'std_' + output_dot_key + '_dot_out', 0.10)**2)

    for output_additional_key in output_additional_keys:
        vars_out.append(kwargs.get(
            'std_' + output_additional_key + '_out', 0.10)**2)

    R = np.diag(vars_out)

    # adjust param dict when full array is not specified
    overwrite_keys = []
    overwrite_vals = []
    T = kwargs.get('T', np.array([]))
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
    if real_output:
        z0.extend(init_params)
    else:
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

    # create dynamic object, optionall ground truth data and noisy states
    dynamic_obj = dyn_class(param_dict, est_params, state_keys=output_keys, state_dot_keys=output_dot_keys,
                            additional_output_keys=output_additional_keys, simulate_gt=simulate_gt)
    if simulate_gt and not real_output:
        U = kwargs.get('U', np.array([]))
        dynamic_obj.sample_nlds(z0, U, T, Q=Q, P0=P0, R=R, store_variables=True,
                                overwrite_keys=overwrite_keys, overwrite_vals=overwrite_vals)
        if re_initialise:
            dynamic_obj.re_initialise()
    else:
        dynamic_obj.P0 = P0
        dynamic_obj.Q = Q
        dynamic_obj.R = R
        dynamic_obj.initial_cond = z0

    # create innovation bound function mapping for heading states
    innovation_bound_func = {}
    angle_states = kwargs.get('angle_states', [])
    if not isinstance(angle_states, Iterable):
        angle_states = [angle_states]
    for angle_state in angle_states:
        assert angle_state in state_dict, "Specified angle state not in state dictionary"
        if state_dict[angle_state] in dynamic_obj.state_indices:
            innovation_bound_func[dynamic_obj.state_indices.index(
                state_dict[angle_state])] = bind_npi_pi
        if state_dict[angle_state] in dynamic_obj.state_dot_indices:
            innovation_bound_func[len(
                dynamic_obj.state_indices) + dynamic_obj.state_dot_indices.index(state_dict[angle_state])] = bind_npi_pi

    dynamic_obj.innovation_bound_func = innovation_bound_func

    # randomly add or substract even number of pi to angle outputs
    if len(dynamic_obj.innovation_bound_func.keys()) and simulate_gt and not real_output:
        pis_factor_list = [i for i in range(10) if i % 2 == 0]
        original_outputs = dynamic_obj.outputs.copy()
        for i in range(dynamic_obj.outputs.shape[1]):
            index_to_mod = np.random.choice(
                list(dynamic_obj.innovation_bound_func.keys()))
            add_or_subtract = np.random.choice([0, 1])
            number_of_pis = np.random.choice(pis_factor_list)
            original_value = dynamic_obj.outputs[index_to_mod, i]
            dynamic_obj.outputs[index_to_mod,
                                i] += ((-1)**add_or_subtract)*number_of_pis*math.pi

    return dynamic_obj


def solve_ivp_dyn_obj(dynamic_obj, T=None, U=None, plot_result=False, plot_euler_result=False, num_rows=3,  method='RK45'):
    """
    Useful function to instead solve the first-order differential equation using generic solver and compare against first order Euler

    Args:
        dynamic_obj (any inherited AbstractDyn obj): dynamic object
        T (numpy array [nt x 1 or 1 x nt]): time array to evaluate the states at; defaults to None (read from dynamic object)
        U (numpy array [nu x nt]): input array required for the first-order differential equation; defaults to None (read
            from dynamic object)
        plot_result (bool): whether to plot the result or not; defaults to False
        plot_euler_result (bool): whether to plot the euluer result encapsulated in the dynamic object or not
        num_rows (int): number of rows of subplots in the figure
        method (string): method of solver to pick and use for the differential equation

    """
    # retrieve time and input if user specified otherwise get from dynamic object
    taken_T_from_dyn_obj = False
    if isinstance(T, Iterable):
        if len(T) == 0:
            T = dynamic_obj.T
        taken_T_from_dyn_obj = True
    if isinstance(U, Iterable):
        if len(U) == 0:
            U = dynamic_obj.U
    if T is None:
        T = dynamic_obj.T
        taken_T_from_dyn_obj = True
    if U is None:
        U = dynamic_obj.U
    assert np.array(T.shape).max() == np.array(U.shape).max(
    ), "Time and input matrices should be of the same length"

    # create time span based on stored time instances
    t_span = [T[0], T[-1]]

    def dxdt(t, z):
        # interpolate inputs
        u = np.zeros(dynamic_obj.num_in)
        for i in range(dynamic_obj.num_in):
            u[i] = np.interp(t, T, U[i, :])

        # find param dict at this time instant
        index = np.abs(t - dynamic_obj.T).argmin()
        param_dict = dynamic_obj.param_list[index]

        return dynamic_obj.dxdt(z, u, param_dict).flatten()

    ivp_result = solve_ivp(
        dxdt, t_span, dynamic_obj.initial_cond.flatten(), method=method, t_eval=T)

    # re-initialise dynamic object if still in simulated gt
    if dynamic_obj.simulate_gt:
        dynamic_obj.re_initialise()

    # plot the results if user requested it
    if plot_result or plot_euler_result:
        # calculate number of columns for subplots
        num_subplots = len(dynamic_obj.global_state_dict.keys())
        if num_subplots < num_rows:
            num_rows = num_subplots
        num_cols = int(math.ceil(num_subplots/float(num_rows)))

        # plot the states
        plt.figure(1)
        for i, key in enumerate(dynamic_obj.global_state_dict):
            # subplot
            plt.subplot(num_rows, num_cols, i+1)

            index = dynamic_obj.global_state_dict[key]
            # plot the gt states from first-order Euler
            if plot_euler_result:
                plt.plot(
                    dynamic_obj.T, dynamic_obj.gt_states[index, :], label='first order Euler', linestyle='--')

            # plot the states from solver
            if plot_result:
                plt.plot(T, ivp_result.y[index], label=method)

            # legend and labels
            plt.legend(loc='upper right')
            plt.xlabel("Time (seconds")
            plt.ylabel(key)
            plt.grid(True, "both")

        plt.show()

    return ivp_result


def create_filtered_estimates(dynamic_obj, method='CKF', order=2, obs_freq=float('inf')):
    """
    Generate mean and covariance of filtered distribution at various times for the problem defined by the dynamic object.

    Args:
        dynamic_obj (obj derived from AbstractDyn): dynamic object encapsulating source of information for the filter
        method (str): The method for filtering algorithm, see estimators.py for the methods currently implemented; defaults to 'CKF'
        order (int): Order of accuracy for integration rule, see estimators.py for orders currently implemented; defaults to 2
        obs_freq (float): Frequency of using observations stored in dynamic object for update step

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

    # keep track of time since last update
    time_since_last_update = 0.0

    # filtering loop
    num_sol = len(dynamic_obj.T)
    est_states = np.zeros((dynamic_obj.num_states, num_sol))
    est_states[:, 0:1] = dynamic_obj.initial_cond
    cov_states = np.zeros(
        (dynamic_obj.num_states, dynamic_obj.num_states, num_sol))
    cov_states[:, :, 0] = dynamic_obj.P0.copy()
    for i in range(1, num_sol):
        # get output for desired frquency
        if time_since_last_update >= 1/obs_freq:
            time_since_last_update = 0.0
            output = dynamic_obj.outputs[:, i:i+1]
        else:
            time_since_last_update += dynamic_obj.T[i] - dynamic_obj.T[i-1]
            output = []

        # perform prediction and update
        est_states[:, i:i+1], cov_states[:, :, i] = pbgf.predict_and_or_update(est_states[:, i-1:i], cov_states[:, :, i-1], dynamic_obj.process_model, dynamic_obj.observation_model, dynamic_obj.Q, dynamic_obj.R, dynamic_obj.U[:, i-1], output, dynamic_obj.U[:, i], additional_args_pm=[
                                                                               sub[i-1] for sub in dynamic_obj.additional_args_pm_list], additional_args_om=[sub[i] for sub in dynamic_obj.additional_args_om_list], innovation_bound_func=innovation_bound_func)

    return est_states, cov_states


def create_smoothed_estimates(dynamic_obj, method='CKF', order=2, lag_interval=5):
    """
    Generate mean and covariance of smoothing distribution at various times for the problem defined by the dynamic object.

    Args:
        dynamic_obj (obj derived from AbstractDyn): dynamic object encapsulating source of information for the filter
        method (str): The method for filtering algorithm, see estimators.py for the methods currently implemented; defaults to 'CKF'
        order (int): Order of accuracy for integration rule, see estimators.py for orders currently implemented; defaults to 2
        lag_interval (int): lag interval for producing smoothed estimate

    Returns:
        est_states (numpy array [dynamic_obj.num_states x nt]): filtered mean estimates of the states at different time instances
        cov_states (numpy array [dynamic_obj.num_states x dynamic_obj.num_states x nt]): filtered covariance of the states
            at different time instances

    """
    # create instance of the smoother
    pbgf = PointBasedFixedLagSmoother(method, order, lag_interval)

    if hasattr(dynamic_obj, 'innovation_bound_func'):
        innovation_bound_func = dynamic_obj.innovation_bound_func
    else:
        innovation_bound_func = {}

    # set initial condition for smoothers
    pbgf.set_initial_cond(dynamic_obj.initial_cond, dynamic_obj.P0)

    # smoothing loop
    num_sol = len(dynamic_obj.T)
    est_states = np.zeros((dynamic_obj.num_states, num_sol))
    cov_states = np.zeros(
        (dynamic_obj.num_states, dynamic_obj.num_states, num_sol))
    for i in range(1, num_sol):
        X_smooth_fi, P_smooth_fi, smooth_flag = pbgf.predict_and_or_update(dynamic_obj.process_model, dynamic_obj.observation_model, dynamic_obj.Q, dynamic_obj.R, dynamic_obj.U[:, i-1], dynamic_obj.outputs[:, i:i+1], dynamic_obj.U[:, i], additional_args_pm=[
                                                                           sub[i-1] for sub in dynamic_obj.additional_args_pm_list], additional_args_om=[sub[i] for sub in dynamic_obj.additional_args_om_list], innovation_bound_func=innovation_bound_func)
        if smooth_flag and i - lag_interval >= 0:
            est_states[:, i-lag_interval:i - lag_interval+1] = X_smooth_fi[0]
            cov_states[:, :, i-lag_interval] = P_smooth_fi[0]
            if i == num_sol-1:
                for k in range(1, len(X_smooth_fi)):
                    est_states[:, i-lag_interval+k:i -
                               lag_interval+k+1] = X_smooth_fi[k]
                    cov_states[:, :, i-lag_interval+k] = P_smooth_fi[k]

    return est_states, cov_states


def plot_stuff(dynamic_obj, est_states, angle_states=[], encapsulated_gt=False, ref_params=None, data=None, data_state_mapping={}, data_indices=None, num_rows=[1, 2]):
    """
    Useful function for plotting stuff. It plots 2 figures: 1 -> estimated parameters vs gt,
    2-> main dynamic states such as trajectory, heading, etc

    Args:
        dynamic_obj (any inherited AbstractDyn obj): dynamic object
        est_states (numpy array [len(dynamic_obj.state_dict)+len(dynamic_obj.est_params) x nt]): estimated states
            of the dynamic model, same shape as dynamic_obj.gt_states
        angle_states (list): list of heading state keys to be bound between -pi and pi; defaults to empty
        encapsulated_gt (bool): whether ground truth data is encapsulated in dynamic_obj (either through simulation or externally)
        ref_params (list): list of parameters reference to compare against that produced by estimator
        data (dict): dictionary of external data (gt or otherwise) not encapsulated in dynamic_obj
        data_state_mapping (dict): dictionary containing mapping of state keys of dynamic_obj to data keys
        data_indices (list of 2 items): start and end indices of data that is of interest for plotting
        num_rows (list of 2 items): number of rows of subplots in figures 1 & 2

    """
    # check if ref_params is of the right size
    if ref_params is not None:
        if not isinstance(ref_params, Iterable):
            ref_params = [ref_params]
        assert len(ref_params) == len(dynamic_obj.est_params), "Expected parameters to be estimated to be of the same length as the provided reference parameters but instead got {} and {} respectively".format(
            len(ref_params), len(dynamic_obj.est_params))
    # check if data_indices when provided is of the right size
    if data_indices is not None:
        assert len(
            data_indices) == 2, "Expect only two items in data indices list"
    # check num_rows when provided is of the right size
    if not isinstance(num_rows, Iterable):
        num_rows = [num_rows]*2
    assert len(num_rows) == 2, "Expect only two items in num rows for each figure"

    # first figure for the parameters
    num_est_params = len(dynamic_obj.est_params)
    num_main_states = dynamic_obj.num_states - num_est_params
    if num_est_params < num_rows[0]:
        num_rows[0] = num_est_params
    num_col = int(math.ceil(num_est_params/float(num_rows[0])))
    plt.figure(1)
    for i, j in enumerate(range(num_main_states, dynamic_obj.num_states)):
        plt.subplot(num_rows[0], num_col, i+1)

        plt.plot(dynamic_obj.T, est_states[j, :], label='est')
        if encapsulated_gt:
            plt.plot(dynamic_obj.T,
                     dynamic_obj.gt_states[j, :], label='gt', linestyle='--')
        elif ref_params is not None:
            if isinstance(ref_params[i], Iterable):
                assert len(ref_params[i]) == len(dynamic_obj.T), "Reference parameter size is inconsistent with time instances ({} vs {})".format(
                    len(ref_params[i]), len(dynamic_obj.T))
                ref_param = ref_params[i]
            else:
                ref_param = np.ones(dynamic_obj.T.shape)*ref_params[i]

            plt.plot(dynamic_obj.T, ref_param, label='ref', linestyle='--')

        plt.legend()
        plt.grid(True, "both")
        plt.xlabel('Time (seconds)')
        plt.ylabel(dynamic_obj.est_params[i])

    # second figure is to examine the main states
    plt.figure(2)
    start_ind = 1
    if 'x' in dynamic_obj.state_dict and 'y' in dynamic_obj.state_dict:
        num_col = int(math.ceil((num_main_states-1)/num_rows[1]))

        plt.subplot(num_rows[1], num_col, 1)

        plt.plot(est_states[dynamic_obj.state_dict['x'], :],
                 est_states[dynamic_obj.state_dict['y'], :], label='est')
        if dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict['y'] in dynamic_obj.state_indices:
            plt.plot(dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['x']), :],
                     dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['y']), :], label='output')
        if data is not None:
            if 'x' in data_state_mapping and 'y' in data_state_mapping:
                if data_state_mapping['x'] in data and data_state_mapping['y'] in data:
                    if data_indices is None:
                        temp_indices = [
                            0, len(data[data_state_mapping['x']])-1]
                    else:
                        temp_indices = data_indices
                    plt.plot(data[data_state_mapping['x']][temp_indices[0]:temp_indices[1]+1],
                             data[data_state_mapping['y']][temp_indices[0]:temp_indices[1]+1], label='data')
        if encapsulated_gt:
            plt.plot(dynamic_obj.gt_states[dynamic_obj.state_dict['x'], :],
                     dynamic_obj.gt_states[dynamic_obj.state_dict['y'], :], label='gt')

        plt.grid(True, "both")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()

        start_ind += 1
    else:
        num_col = int(math.ceil(num_main_states/num_rows[1]))

    other_states_dict = {x: dynamic_obj.state_dict[x] for x in dynamic_obj.state_dict if x not in [
        'x', 'y'] and x not in dynamic_obj.est_params}
    for i, key in enumerate(other_states_dict):
        state_ind = other_states_dict[key]
        plt.subplot(num_rows[1], num_col, start_ind+i)

        if key in angle_states:
            plt.plot(dynamic_obj.T, bind_npi_pi(
                est_states[state_ind, :]), label='est')
        else:
            plt.plot(dynamic_obj.T, est_states[state_ind, :], label='est')
        if state_ind in dynamic_obj.state_indices:
            if key in angle_states:
                plt.plot(dynamic_obj.T, bind_npi_pi(
                    dynamic_obj.outputs[dynamic_obj.state_indices.index(state_ind), :]), label='output')
            else:
                plt.plot(dynamic_obj.T, dynamic_obj.outputs[dynamic_obj.state_indices.index(
                    state_ind), :], label='output')
        if data is not None:
            if key in data_state_mapping:
                if data_state_mapping[key] in data:
                    if data_indices is None:
                        temp_indices = [
                            0, len(data[data_state_mapping[key]])-1]
                    else:
                        temp_indices = data_indices

                    assert len(dynamic_obj.T) == temp_indices[1]-temp_indices[0]+1, "Expected number of time instances to be the same as data instances for key {} but instead got {} & {}".format(
                        key, len(dynamic_obj.T), temp_indices[1]-temp_indices[0]+1)

                    if key in angle_states:
                        plt.plot(dynamic_obj.T, bind_npi_pi(
                            data[data_state_mapping[key]][temp_indices[0]:temp_indices[1]+1]), label='data')
                    else:
                        plt.plot(dynamic_obj.T, data[data_state_mapping[key]]
                                 [temp_indices[0]:temp_indices[1]+1], label='data')
        if encapsulated_gt:
            if key in angle_states:
                plt.plot(dynamic_obj.T, bind_npi_pi(
                    dynamic_obj.gt_states[state_ind, :]), label='gt')
            else:
                plt.plot(dynamic_obj.T,
                         dynamic_obj.gt_states[state_ind, :], label='gt')

        plt.grid(True, "both")
        plt.xlabel('Time (s)')
        plt.ylabel(key)
        plt.legend()
    plt.show()
