from collections import OrderedDict as ODict
from collections import Iterable

import glob

import math
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from car_dynamics import RearDriveFrontSteerEst
from test_estimators import create_filtered_estimates


def fit_and_plot(A1, A2, B, all_params, model_tag, axs, t_vec, ax, ay, est_params=['sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr']):
    A = np.concatenate((A1, A2), axis=0)
    # perform least square
    parameters, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)
    assert len(parameters) == len(est_params), "Key for parameters should be of the same length as matrices"

    # calculate mse from this set of parameters
    mse = ((np.matmul(A, parameters) - B)**2).mean()

    # save parameters to the dictionary
    param_dict = {}
    for key, parameter in zip(est_params, parameters):
        param_dict[key] = parameter[0]
    all_params['param_list'].append(param_dict)
    all_params['mse'].append(mse)
    all_params['model'].append(model_tag)

    # save to best parameter if mse is lower
    if mse < all_params['best']['mse']:
        all_params['best']['param'] = param_dict
        all_params['best']['model'] = model_tag
        all_params['best']['mse'] = mse

    # plot the comparison
    axs[0].plot(t_vec, ax, label='true')
    axs[0].plot(t_vec, np.matmul(A1, parameters), label='fit')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('vx dot')
    axs[0].legend()
    axs[0].grid(True, "both")

    axs[1].plot(t_vec, ay, label='true')
    axs[1].plot(t_vec, np.matmul(A2, parameters), label='fit')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('vy dot')
    axs[1].legend()
    axs[1].grid(True, "both")


def least_square_test(param_dict, data, threshold_ws=20.0):
    # get the constants from the dictionary
    m = param_dict["m"]
    iz = param_dict["iz"]
    lf = param_dict["lf"]
    lr = param_dict["lr"]
    ref = param_dict["ref"]
    rer = param_dict["rer"]
    g = param_dict["g"]

    # get friction data
    friction = np.array(data['friction']).T
    friction = friction[0, :].flatten()

    # best parameter dictionary to be filled and returned
    all_params = {}

    # iterate through the unique friction data
    unique_frictions = np.unique(friction)
    for ref_friction in unique_frictions:
        all_params[ref_friction] = {}
        all_params[ref_friction]['param_list'] = []
        all_params[ref_friction]['mse'] = []
        all_params[ref_friction]['model'] = []
        all_params[ref_friction]['best'] = {}
        all_params[ref_friction]['best']['mse'] = float('inf')

        # get front wheel speed, rear wheel speed and steering angle
        wheelspeed = np.array(data['wheelspeed'])
        wheelangle = np.array(data['wheelangle'])
        wf = 0.5*(wheelspeed[:, 0:1] + wheelspeed[:, 1:2])
        wr = 0.5*(wheelspeed[:, 2:3] + wheelspeed[:, 3:4])
        steering_angle = 0.5*(wheelangle[:, 0:1] + wheelangle[:, 1:2])

        # use wheel speed to threshold data when the vehicle is moving
        filter_condition = (np.abs(wr) > threshold_ws).flatten()
        filter_condition = np.logical_and(
            filter_condition, friction == ref_friction)
        wf = wf[filter_condition, :]
        wr = wr[filter_condition, :]
        steering_angle = steering_angle[filter_condition, :]

        # time vector
        t_vec = np.array(data['tvec'])[filter_condition, :]

        # get lateral and longitudinal velocity
        vx = np.array(data['vx'])[filter_condition, :]
        vy = np.array(data['vy'])[filter_condition, :]
        dts = np.diff(t_vec, axis=0)
        vxdot = np.diff(vx, axis=0)/dts
        vydot = np.diff(vy, axis=0)/dts

        # get yawrate
        w = np.array(data['yawrate'])[filter_condition, :]

        # get lateral and longitudinal acceleration
        ax = np.array(data['ax'])[filter_condition, :]
        ay = np.array(data['ay'])[filter_condition, :]

        # compose a least square problem in cr, cf, dr, df and fr
        sigma_xf = ref*wf - vx
        sigma_xf[sigma_xf < 0.0] /= vx[sigma_xf < 0.0]
        sigma_xf[sigma_xf > 0.0] /= ref*wf[sigma_xf > 0.0]
        """
        sigma_xf[(sigma_xf < 0.0) & (np.logical_not(np.isclose(vx, 0.0)))
                 ] /= vx[(sigma_xf < 0.0) & (np.logical_not(np.isclose(vx, 0.0)))]
        sigma_xf[(sigma_xf < 0.0) & (np.isclose(vx, 0.0))] = 0.0
        sigma_xf[(sigma_xf > 0.0) & (np.logical_not(np.isclose(wf, 0.0)))
                 ] /= (ref * wf[(sigma_xf > 0.0) & (np.logical_not(np.isclose(wf, 0.0)))])
        sigma_xf[(sigma_xf > 0.0) & (np.isclose(wf, 0.0))] = 0.0
        """

        sigma_xr = rer*wr - vx
        sigma_xr[sigma_xr < 0.0] /= vx[sigma_xr < 0.0]
        sigma_xr[sigma_xr > 0.0] /= rer*wr[sigma_xr > 0.0]
        """
        sigma_xr[(sigma_xr < 0.0) & (np.logical_not(np.isclose(vx, 0.0)))
                 ] /= vx[(sigma_xr < 0.0) & (np.logical_not(np.isclose(vx, 0.0)))]
        sigma_xr[(sigma_xr < 0.0) & (np.isclose(vx, 0.0))] = 0.0
        sigma_xr[(sigma_xr > 0.0) & (np.logical_not(np.isclose(wr, 0.0)))
                 ] /= (rer * wr[(sigma_xr > 0.0) & (np.logical_not(np.isclose(wr, 0.0)))])
        sigma_xr[(sigma_xr > 0.0) & (np.isclose(wr, 0.0))] = 0.0
        """
        rx = m*g*np.ones(sigma_xr.shape)

        theta_vf = (vy + lf*w)/vx
        theta_vr = (vy - lr*w)/vx
        """
        theta_vf = np.zeros(rx.shape)
        theta_vr = np.zeros(rx.shape)
        theta_vf[np.logical_not(np.isclose(vx, 0.0))] = (vy[np.logical_not(np.isclose(
            vx, 0.0))] + lf*w[np.logical_not(np.isclose(vx, 0.0))])/vx[np.logical_not(np.isclose(vx, 0.0))]
        theta_vr[np.logical_not(np.isclose(vx, 0.0))] = (vy[np.logical_not(np.isclose(
            vx, 0.0))] - lr*w[np.logical_not(np.isclose(vx, 0.0))])/vx[np.logical_not(np.isclose(vx, 0.0))]
        theta_vf[(np.isclose(vx, 0.0)) & (np.logical_not(
            np.isclose(vy + lf*w, 0.0)))] = 0.5*math.pi
        theta_vr[(np.isclose(vx, 0.0)) & (np.logical_not(
            np.isclose(vy + lf*w, 0.0)))] = 0.5*math.pi
        """
        alpha_f = steering_angle - theta_vf
        alpha_r = -theta_vr

        ### The first subset of figures not considering aerodynamic drag
        fig, axs = plt.subplots(4, 2, constrained_layout=True, num=1)
        fig.suptitle(
            'Friction coefficient of {} + no aerodynamic drag'.format(ref_friction))
        ## first least square: separate front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        B = np.concatenate((ax, ay))
        A1 = np.concatenate((sigma_xr/m, sigma_xf*np.cos(steering_angle)/m,
                             np.zeros(alpha_f.shape), -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape), sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        model_tag = "separate front & back coeffs + don't neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[0, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr'])

        ## second least square: separate front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m, np.zeros(alpha_f.shape),
                             alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        model_tag = "separate front & back coeffs + neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[1, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sc_r', 'sc_f', 'fr'])

        ## third least square: same front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m + sigma_xf*np.cos(steering_angle)/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        model_tag = "same front & back coeffs + don't neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag,
                     axs[2, :], t_vec, ax, ay, est_params=['sl_r', 'sc_r', 'fr'])

        ## fourth least square: same front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        model_tag = "same front & back coeffs + neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag,
                     axs[3, :], t_vec, ax, ay, est_params=['sl_r', 'sc_r', 'fr'])

        ### The second subset of figures considering aerodynamic drag
        fig, axs = plt.subplots(4, 2, constrained_layout=True, num=2)
        fig.suptitle(
            'Friction coefficient of {} + aerodynamic drag'.format(ref_friction))
        ## first least square: separate front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m, sigma_xf*np.cos(steering_angle)/m,
                             np.zeros(alpha_f.shape), -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape), sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "separate front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[0, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr', 'da'])

        ## second least square: separate front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             np.zeros(alpha_f.shape), alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "separate front & back coeffs + neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[1, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sc_r', 'sc_f', 'fr', 'da'])

        ## third least square: same front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m + sigma_xf*np.cos(steering_angle)/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "same front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[2, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sc_r', 'fr', 'da'])

        ## fourth least square: same front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "same front & back coeffs + neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[3, :], t_vec, ax, ay, est_params=[
                     'sl_r', 'sc_r', 'fr', 'da'])

        """
        ### The third subset of figure do consider aerodynamic drag but instead use numerically differenced long & lat acceleration
        fig, axs = plt.subplots(4, 2, constrained_layout=True, num=3)
        fig.suptitle(
            'Friction coefficient of {} + aerodynamic drag + numerically differenced ax & ay'.format(ref_friction))
        # use everything until the last index
        vx = vx[:-1, :]
        vy = vy[:-1, :]
        w = w[:-1, :]
        t_vec = t_vec[:-1, :]
        sigma_xr = sigma_xr[:-1, :]
        sigma_xf = sigma_xf[:-1, :]
        steering_angle = steering_angle[:-1, :]
        alpha_f = alpha_f[:-1, :]
        alpha_r = alpha_r[:-1, :]
        rx = rx[:-1, :]

        ## first least square: separate front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        B = np.concatenate((vxdot, vydot))
        B -= np.concatenate((vy*w, -vx*w))
        A1 = np.concatenate((sigma_xr/m, sigma_xf*np.cos(steering_angle)/m,
                             np.zeros(alpha_f.shape), -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape), sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "separate front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag + use numerically differenced vx & vy as ax & ay"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[0, :], t_vec, vxdot, vydot, est_params=[
                     'sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr', 'da'])

        ## second least square: separate front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             np.zeros(alpha_f.shape), alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "separate front & back coeffs + neglect driven wheel long. force + aerodynamic drag + use numerically differenced vx & vy as ax & ay"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[1, :],
                     t_vec, vxdot, vydot, est_params=['sl_r', 'sc_r', 'sc_f', 'fr', 'da'])

        ## third least square: same front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m + sigma_xf*np.cos(steering_angle)/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "same front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag + use numerically differenced vx & vy as ax & ay"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[2, :],
                     t_vec, vxdot, vydot, est_params=['sl_r', 'sc_r', 'fr', 'da'])

        ## fourth least square: same front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        model_tag = "same front & back coeffs + neglect driven wheel long. force + aerodynamic drag + use numerically differenced vx & vy as ax & ay"
        fit_and_plot(A1, A2, B, all_params[ref_friction], model_tag, axs[3, :],
                     t_vec, vxdot, vydot, est_params=['sl_r', 'sc_r', 'fr', 'da'])
        """

        plt.show()

    print(all_params)

    return all_params


def plot_stuff(dynamic_obj, data, est_states, configuration, num_row=1):
    """
    Useful function for plotting stuff. It plots 2 figures: 1 -> estimated parameters vs gt,
    2-> main dynamic states such as trajectory, heading, etc

    Args:
        dynamic_obj (RoverPartialDynEst obj): dynamic object
        est_states (numpy array [4+len(dynamic_obj.est_params) x nt]): estimated states of the dynamic model,
            same shape as dynamic_obj.gt_states
        num_row (int): number of rows of subplots in figure 1
        plot_gt (bool): whether to plot the ground truth states (may not be present), defaults to True

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
    if dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict['y'] in dynamic_obj.state_indices:
        plt.plot(dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['x']), :],
                 dynamic_obj.outputs[dynamic_obj.state_indices.index(dynamic_obj.state_dict['y']), :], label='output')
    else:
        plt.plot(data[configuration['data_state_mapping']['x']],
                 data[configuration['data_state_mapping']['y']], label='data')

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
        if state_ind in dynamic_obj.state_indices:
            plt.plot(dynamic_obj.T, dynamic_obj.outputs[dynamic_obj.state_indices.index(
                state_ind), :], label='output')
        else:
            plt.plot(
                dynamic_obj.T, data[configuration['data_state_mapping'][key]], label='data')

        plt.grid(True, "both")
        plt.xlabel('Time (s)')
        plt.ylabel(key)
        plt.legend()
    plt.show()


def create_dyn_obj(dyn_class, param_dict, **kwargs):
    """
    Create dynamic object for specified class encapsulating some information useful during the estimation process

    Args:
        dyn_class (class of type inherited from AbstractDyn): type to create dynamic object
        param_dict (dict): dictionary of parameters needed for defining the dynamics
        kwargs: dictionary of variable length for additional parameters

    kwargs:
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

    init_params = kwargs.get('init_params', [])
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
    time_varying_q = kwargs.get('time_varying_q', 1e-4)
    temp.extend([time_varying_q]*len(est_params))
    Q = np.diag(temp)

    # covariance matrix of additive GWN in observation model
    output_keys = kwargs.get('output_keys', state_dict.keys())
    output_dot_keys = kwargs.get('output_dot_keys', [])
    output_data_keys = kwargs.get('output_data_keys', state_dict.keys())
    output_data_dot_keys = kwargs.get('output_data_dot_keys', [])
    assert len(output_data_keys) == len(output_keys), "Expected output keys to be of the same length as its mapping in data file but instead got {} and {} respectively".format(
        len(output_data_keys), len(output_keys))
    assert len(output_dot_keys) == len(output_data_dot_keys), "Expected derivative of output keys to be of the same length as its mapping in data file but instead got {} and {} respectively".format(
        len(output_dot_keys), len(output_data_dot_keys))

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

    # initial belief, uncertainty of initial condition
    temp = list(init_state_cov)
    temp.extend(init_param_cov)
    P0 = np.diag(temp)

    # create ground truth data
    dynamic_obj = dyn_class(
        param_dict, est_params, state_keys=output_keys, state_dot_keys=output_dot_keys, simulate_gt=False)

    # store the covariance matrices
    dynamic_obj.P0 = P0
    dynamic_obj.Q = Q
    dynamic_obj.R = R

    # initial condition (zero for states as they will be overwritten)
    z0 = [0.0]*len(state_dict)
    z0.extend(init_params)
    dynamic_obj.initial_cond = np.array([z0]).T

    return dynamic_obj


def extract_model(data, dynamic_obj):
    # timing matrices
    T = np.array(data['tvec']).flatten()
    nt = len(T)
    dts = np.diff(T)
    dts = np.append(dts, dts[-1])

    # models
    def process_model(x, u, noise, dt, param_dict): return dynamic_obj.forward_prop(
        x, dynamic_obj.dxdt(x, u, param_dict), dt) + noise

    def observation_model(
        x, u, noise, param_dict): return dynamic_obj.output_model(x, u, param_dict) + noise

    # additional arguments
    additional_args_pm_list = np.zeros((2, nt)).tolist()
    additional_args_om_list = np.zeros((1, nt)).tolist()
    additional_args_pm_list[0] = dts
    additional_args_pm_list[1] = [dynamic_obj.param_dict]*nt
    additional_args_om_list[0] = [dynamic_obj.param_dict]*nt

    dynamic_obj.T = T
    dynamic_obj.process_model = process_model
    dynamic_obj.observation_model = observation_model
    dynamic_obj.additional_args_pm_list = additional_args_pm_list
    dynamic_obj.additional_args_om_list = additional_args_om_list


def extract_input(data, dynamic_obj):
    nt = len(data['tvec'])
    U = np.zeros((dynamic_obj.num_in, nt))

    wheelspeed = np.array(data['wheelspeed']).T
    wheelangle = np.array(data['wheelangle']).T
    U[0:1, :] = 0.5*(wheelspeed[0, :] + wheelspeed[1, :])
    U[1:2, :] = 0.5*(wheelspeed[2, :] + wheelspeed[3, :])
    U[2:3, :] = 0.5*(wheelangle[0, :] + wheelangle[1, :])

    dynamic_obj.U = U


def extract_output(data, data_filename, configuration, dynamic_obj):
    # check if mapping exists in the data file
    for key in configuration['output_data_keys']:
        assert key in data.keys(), "Key {} not present in data file {}".format(key, data_filename)
    for key in configuration['output_data_dot_keys']:
        assert key in data.keys(), "Key {} not present in data file {}".format(key, data_filename)

    # extract data
    nt = len(data['tvec'])
    if nt < 0:
        return False
    num_out = dynamic_obj.num_out
    outputs = np.zeros((num_out, nt))
    index = 0
    for key in configuration['output_data_keys']:
        outputs[index:index+1, :] = np.array(data[key]).T
        index += 1
    for key in configuration['output_data_dot_keys']:
        outputs[index:index+1, :] = np.array(data[key]).T
        index += 1
    dynamic_obj.outputs = outputs

    return True


def extract_initial_cond(data, configuration, dynamic_obj, first_file, continue_estimation):
    # put in the dynamic states
    for key, index in dynamic_obj.global_state_dict.items():
        dynamic_obj.initial_cond[index] = data[configuration['data_state_mapping'][key]][0][0]

    # reset the parameters if requested
    if not continue_estimation:
        est_params = configuration.get('est_params', [])
        if not isinstance(est_params, Iterable):
            est_params = [est_params]

        init_params = configuration.get('init_params', [])
        if not isinstance(init_params, Iterable):
            init_params = [init_params]

        for est_param, init_param in zip(est_params, init_params):
            assert est_param in dynamic_obj.state_dict.keys(
            ), "Parameter {} to be estimated is not registered in state dictionary".format(est_param)
            dynamic_obj.initial_cond[dynamic_obj.state_dict[est_param]] = init_param
    elif not first_file:
        dynamic_obj.P0 = np.copy(dynamic_obj.P_end)
        # assume absolute confidence in initial condition
        for index in dynamic_obj.global_state_dict.values():
            dynamic_obj.P0[index, :] = 0.0
            dynamic_obj.P0[:, index] = 0.0

    print(dynamic_obj.initial_cond)
    print(dynamic_obj.P0)


if __name__ == '__main__':
    # create instance of filter
    method = 'CKF'
    order = 2

    # get a list of matlab files in the folder
    folder_name = './matlab_sim_data/'
    mat_files = glob.glob(folder_name + '*.mat')

    # parameter dictionary based on single matlab file
    data = loadmat(mat_files[0])
    print(data.keys())
    radii = data['wheelradii'][0]
    param_dict = ODict([('m', float(data['m'])), ('iz', float(data['I'])), ('lf', float(data['lf'])), ('lr', float(
        data['lr'])), ('ref', 0.5*(float(radii[0])+float(radii[1]))), ('rer', 0.5*(float(radii[2])+float(radii[3]))), ('g', 9.81)])

    # create dynamic object and get initial condition
    configuration = {'output_keys': ['x', 'y'],
                     'output_data_keys': ['x', 'y'],
                     'output_dot_keys': ['vx', 'vy'],
                     'output_data_dot_keys': ['ax', 'ay'],
                     'est_params': ['fr', 'sl_r', 'sc_f', 'sc_r', 'cd'],
                     'init_params': [0.0, 1e3, 1e3, 1e3, 3.0],
                     'init_param_cov': 1000.0,
                     'std_x_out': 0.1,
                     'std_y_out': 0.1,
                     'std_theta_out': math.pi/180.0,
                     'std_vx_out': 0.1,
                     'std_vy_out': 0.1,
                     'std_vx_dot_out': 0.15,
                     'std_vy_dot_out': 0.15,
                     'std_theta_dot_out': math.pi/180.0,
                     'time_varying_q': 0.0,
                     'threshold_ws': 20.0}
    configuration['data_state_mapping'] = {
        'x': 'x', 'y': 'y', 'theta': 'heading', 'vx': 'vx', 'vy': 'vy', 'w': 'yawrate'}
    dynamic_obj = create_dyn_obj(
        RearDriveFrontSteerEst, param_dict, **configuration)

    first_file = True
    # expect to restart from previous cycle (previous mat file?)
    continue_estimation = True

    # load the data from matlab file
    for mat_file in mat_files:
        # read data
        data = loadmat(mat_file)

        # perform lls fitting
        lls_params = least_square_test(
            param_dict, data, threshold_ws=configuration['threshold_ws'])


        # create process and observation models
        extract_model(data, dynamic_obj)

        # extract input from data
        extract_input(data, dynamic_obj)

        # extract output from data
        ok = extract_output(data, mat_file, configuration, dynamic_obj)
        if not ok:
            continue

        # get initial condition from data
        extract_initial_cond(data, configuration, dynamic_obj,
                             first_file, continue_estimation)

        # perform filtering
        est_states, cov_states = create_filtered_estimates(
            dynamic_obj, method, order)
        dynamic_obj.P_end = cov_states[:, :, -1]
        dynamic_obj.initial_cond = est_states[:, -1:]

        # plot the evolution of states and parameters
        plot_stuff(dynamic_obj, data, est_states, configuration, num_row=2)

        if first_file:
            first_file = False
