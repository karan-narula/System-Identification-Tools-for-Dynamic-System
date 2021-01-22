from collections import OrderedDict as ODict
from collections import Iterable

import glob

import math
import numpy as np
from scipy.io import loadmat
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from car_dynamics import RearDriveFrontSteerEst, RearDriveFrontSteerSubStateVelEst, cal_vxvy_from_coord
from utilities import create_filtered_estimates, create_dyn_obj, plot_stuff


def fit_and_plot(A_list, B, all_params, model_tag, axs, t_vec, ys, est_params=['sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr'], ylabels=['vx dot', 'vy dot', 'yaw rate dot']):
    assert len(ylabels) >= len(
        ys), "Labels should be at least as long as the fitted data but instead got {} and {}".format(len(ylabels), len(ys))

    A = np.concatenate(A_list, axis=0)
    # perform least square
    parameters, residuals, _, _ = np.linalg.lstsq(A, B, rcond=None)
    assert len(parameters) == len(
        est_params), "Key for parameters should be of the same length as matrices"

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
    for i, (y, ylabel) in enumerate(zip(ys, ylabels[:len(ys)])):
        axs[i].plot(t_vec, y, label='true')
        axs[i].plot(t_vec, np.matmul(A_list[i], parameters), label='fit')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel(ylabel)
        axs[i].legend()
        axs[i].grid(True, "both")


def fit_and_plot_vxvy(t_vec, x, y, yaw, vx, vy, fig_num=1):
    # get LS fit for vx & vy
    dts = np.diff(t_vec, axis=0).T
    state = np.concatenate((x[:-1, :], y[:-1, :], yaw[:-1, :]), axis=1).T
    state_prev = np.concatenate((x[1:, :], y[1:, :], yaw[1:, :]), axis=1).T
    vxy = cal_vxvy_from_coord(state, state_prev, dts)
    vxy = -vxy  # shouldn't need to adjust!

    # plot the results comparing read values against fitted values
    fig, axs = plt.subplots(1, 2, constrained_layout=True, num=fig_num)

    axs[0].plot(t_vec, vx, label='data')
    axs[0].plot(t_vec[:-1, :], vxy[0, :], label='fit')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Vx (m/s)')
    axs[0].legend()
    axs[0].grid(True, "both")

    axs[1].plot(t_vec, vy, label='data')
    axs[1].plot(t_vec[:-1, :], vxy[1, :], label='fit')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Vy (m/s)')
    axs[1].legend()
    axs[1].grid(True, "both")

    return vxy


def bind_npi_pi(angles):
    angles = np.fmod(angles + math.pi, 2*math.pi)
    angles[angles < 0] += 2*math.pi
    angles -= math.pi

    return angles


def forward_integrate_ode(init_state, axs, ays, ws, t_vec):
    t_vec = t_vec[:, 0]
    ws = ws[:, 0]
    ws[1:] = ws[:-1]
    axs = axs[:, 0]
    axs[1:] = axs[:-1]
    ays = ays[:, 0]
    ays[1:] = ays[:-1]

    def dyn_model(t, z):
        # get current states
        heading = z[2]
        vx = z[3]
        vy = z[4]

        # get inputs via linear interpolation
        w = np.interp(t, t_vec, ws)
        ax = np.interp(t, t_vec, axs)
        ay = np.interp(t, t_vec, ays)

        # calculate derivative
        x_dot = vx*math.cos(heading) - vy*math.sin(heading)
        y_dot = vx*math.sin(heading) + vy*math.cos(heading)
        theta_dot = w
        vx_dot = ax + vy*w
        vy_dot = ay - vx*w

        dzdt = [x_dot, y_dot, theta_dot, vx_dot, vy_dot]

        return dzdt

    # use ODE solver to integrate the model
    solver_states = solve_ivp(dyn_model, [
                              t_vec[0], t_vec[-1]], init_state, method='RK45', t_eval=t_vec)
    solver_states.y[2, :] = bind_npi_pi(solver_states.y[2, :])

    return solver_states


def forward_integrate_kinematic(init_state, axs, ays, ws, t_vec):
    states = np.zeros((5, len(t_vec)))
    states[:, 0] = init_state

    dts = np.diff(t_vec, axis=0)

    # first order Euler forward integration
    for j in range(1, len(t_vec)):
        # get previous states
        heading = states[2, j-1]
        vx = states[3, j-1]
        vy = states[4, j-1]

        # get previous inputs
        w = ws[j-1, 0]
        ax = axs[j-1, 0]
        ay = ays[j-1, 0]
        dt = dts[j-1, 0]

        x_dot = vx*math.cos(heading) - vy*math.sin(heading)
        y_dot = vx*math.sin(heading) + vy*math.cos(heading)
        theta_dot = w
        vx_dot = ax + vy*w
        vy_dot = ay - vx*w

        states[:, j] = states[:, j-1] + dt * \
            np.array([x_dot, y_dot, theta_dot, vx_dot, vy_dot])

    states[2, :] = bind_npi_pi(states[2, :])

    return states


def plot_states_evol(states, solver_states, x, y, yaw, vx, vy, t_vec):
    fig, axs = plt.subplots(2, 2, constrained_layout=True, num=0)
    # plot the states
    xaxis_gts = [x, t_vec, t_vec, t_vec]
    xaxis_integrates = [states[0, :], t_vec, t_vec, t_vec]
    xaxis_solver_integrates = [solver_states[0, :], t_vec, t_vec, t_vec]
    xlabels = ['X (m)', 'Time (s)', 'Time (s)', 'Time (s)']
    yaxis_gts = [y, yaw, vx, vy]
    yaxis_integrates = [states[1, :], states[2, :], states[3, :], states[4, :]]
    yaxis_solver_integrates = [
        solver_states[1, :], solver_states[2, :], solver_states[3, :], solver_states[4, :]]
    ylabels = ['Y (m)', 'Heading (rad)', 'Vx (m/s)', 'Vy (m/s)']
    i = 0
    j = 0

    for xaxis_gt, xaxis_integrate, xaxis_solver_integrate, xlabel, yaxis_gt, yaxis_integrate, yaxis_solver_integrate, ylabel in zip(xaxis_gts, xaxis_integrates, xaxis_solver_integrates, xlabels, yaxis_gts, yaxis_integrates, yaxis_solver_integrates, ylabels):
        axs[i, j].plot(xaxis_gt, yaxis_gt, label='gt')
        axs[i, j].plot(xaxis_integrate, yaxis_integrate, label='integrate')
        axs[i, j].plot(xaxis_solver_integrate,
                       yaxis_solver_integrate, label='RK45 integrate')
        axs[i, j].set_xlabel(xlabel)
        axs[i, j].set_ylabel(ylabel)
        axs[i, j].grid(True, "both")
        axs[i, j].legend()

        j += 1
        if j >= 2:
            i += 1
            j = 0


def least_square_test(param_dict, data, threshold_ws=20.0, yaw_rate_derivative=False, estimate_vx_vy=False, use_fitted_vx_vy=False):
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
        wdot = np.diff(w, axis=0)/dts

        # get lateral and longitudinal acceleration
        ax = np.array(data['ax'])[filter_condition, :]
        ay = np.array(data['ay'])[filter_condition, :]

        # get kinematic states
        x = np.array(data['x'])[filter_condition, :]
        y = np.array(data['y'])[filter_condition, :]
        yaw = np.array(data['heading'])[filter_condition, :]

        # fit lateral & longitudinal velocity using position derivative and heading
        fig_num = 1
        if estimate_vx_vy:
            vxy = fit_and_plot_vxvy(t_vec, x, y, yaw, vx, vy, fig_num=fig_num)
            fig_num += 1

        # adjust data indices if derivatives are to be used
        if yaw_rate_derivative or (estimate_vx_vy and use_fitted_vx_vy):
            if estimate_vx_vy and use_fitted_vx_vy:
                vx = vxy[0:1, :].T
                vy = vxy[1:2, :].T
            else:
                vx = vx[:-1, :]
                vy = vy[:-1, :]
            ax = ax[:-1, :]
            ay = ay[:-1, :]
            t_vec = t_vec[:-1, :]
            x = x[:-1, :]
            y = y[:-1, :]
            yaw = yaw[:-1, :]
            w = w[:-1, :]
            wf = wf[:-1, :]
            wr = wr[:-1, :]
            steering_angle = steering_angle[:-1, :]

        # forward integrate the model with yaw rate and inertial acceleration as inputs
        init_state = [x[0, 0], y[0, 0], yaw[0, 0], vx[0, 0], vy[0, 0]]
        states = forward_integrate_kinematic(init_state, ax, ay, w, t_vec)
        solver_states = forward_integrate_ode(
            init_state, ax.copy(), ay.copy(), w.copy(), t_vec.copy())
        plot_states_evol(states, solver_states.y, x, y, yaw, vx, vy, t_vec)

        #compose a least square problem in cr, cf, dr, df and fr
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
        if yaw_rate_derivative:
            num_cols = 3
        else:
            num_cols = 2
        fig, axs = plt.subplots(
            4, num_cols, constrained_layout=True, num=fig_num)
        fig_num += 1
        fig.suptitle(
            'Friction coefficient of {} + no aerodynamic drag'.format(ref_friction))
        ## first least square: separate front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        B = np.concatenate((ax, ay))
        y_list = [ax, ay]

        A1 = np.concatenate((sigma_xr/m, sigma_xf*np.cos(steering_angle)/m,
                             np.zeros(alpha_f.shape), -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape), sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), lf*sigma_xf*np.sin(steering_angle)/iz, -
                             lr*alpha_r/iz, lf*alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape)), axis=1)
        A_list = [A1, A2]

        if yaw_rate_derivative:
            B = np.concatenate((B, wdot))
            y_list.append(wdot)
            A_list.append(A3)

        model_tag = "separate front & back coeffs + don't neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[0, :], t_vec, y_list, est_params=[
                     'sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr'])

        ## second least square: separate front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m, np.zeros(alpha_f.shape),
                             alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), -lr*alpha_r/iz, lf *
                             alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "separate front & back coeffs + neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[1, :], t_vec, y_list, est_params=[
                     'sl_r', 'sc_r', 'sc_f', 'fr'])

        ## third least square: same front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m + sigma_xf*np.cos(steering_angle)/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        A3 = np.concatenate((lf*sigma_xf*np.sin(steering_angle)/iz, -lr*alpha_r /
                             iz + lf*alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "same front & back coeffs + don't neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag,
                     axs[2, :], t_vec, y_list, est_params=['sl_r', 'sc_r', 'fr'])

        ## fourth least square: same front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), -lr*alpha_r/iz + lf *
                             alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "same front & back coeffs + neglect driven wheel long. force + no aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag,
                     axs[3, :], t_vec, y_list, est_params=['sl_r', 'sc_r', 'fr'])

        ### The second subset of figures considering aerodynamic drag
        fig, axs = plt.subplots(
            4, num_cols, constrained_layout=True, num=fig_num)
        fig_num += 1
        fig.suptitle(
            'Friction coefficient of {} + aerodynamic drag'.format(ref_friction))
        ## first least square: separate front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m, sigma_xf*np.cos(steering_angle)/m,
                             np.zeros(alpha_f.shape), -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape), sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), lf*sigma_xf*np.sin(steering_angle)/iz, -
                             lr*alpha_r/iz, lf*alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "separate front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[0, :], t_vec, y_list, est_params=[
                     'sl_r', 'sl_f', 'sc_r', 'sc_f', 'fr', 'da'])

        ## second least square: separate front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             np.zeros(alpha_f.shape), alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m, alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), -lr*alpha_r/iz, lf *
                             alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "separate front & back coeffs + neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[1, :], t_vec, y_list, est_params=[
                     'sl_r', 'sc_r', 'sc_f', 'fr', 'da'])

        ## third least square: same front and back coeffs + don't neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m + sigma_xf*np.cos(steering_angle)/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((sigma_xf*np.sin(steering_angle)/m,
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A3 = np.concatenate((lf*sigma_xf*np.sin(steering_angle)/iz, -lr*alpha_r /
                             iz + lf*alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "same front & back coeffs + don't neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[2, :], t_vec, y_list, est_params=[
                     'sl_r', 'sc_r', 'fr', 'da'])

        ## fourth least square: same front and back coeffs + neglect driven wheel long. force
        # construct matrices
        A1 = np.concatenate((sigma_xr/m,
                             -alpha_f*np.sin(steering_angle)/m, -rx/m, -(vx**2)/m), axis=1)
        A2 = np.concatenate((np.zeros(alpha_f.shape),
                             alpha_r/m + alpha_f*np.cos(steering_angle)/m, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A3 = np.concatenate((np.zeros(alpha_f.shape), -lr*alpha_r/iz + lf *
                             alpha_f*np.cos(steering_angle)/iz, np.zeros(alpha_f.shape), np.zeros(vx.shape)), axis=1)
        A_list = [A1, A2]
        if yaw_rate_derivative:
            A_list.append(A3)
        model_tag = "same front & back coeffs + neglect driven wheel long. force + aerodynamic drag"
        fit_and_plot(A_list, B, all_params[ref_friction], model_tag, axs[3, :], t_vec, y_list, est_params=[
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


def get_data_indices(data, threshold_ws):
    # extract wheel speed
    wheelspeed = np.array(data['wheelspeed'])
    wr = 0.5*(wheelspeed[:, 2:3] + wheelspeed[:, 3:4])

    first_index = 0
    for i in range(len(wr)):
        if wr[i, :] > threshold_ws:
            first_index = i
            break

    for i in range(len(wr)-1, -1, -1):
        if wr[i, :] > threshold_ws:
            last_index = i
            break

    return (first_index, last_index)


def extract_model(data, data_indices, dynamic_obj):
    # timing matrices
    T = np.array(data['tvec'])[data_indices[0]:data_indices[1]+1, :].flatten()
    nt = len(T)
    dts = np.diff(T)
    dts = np.append(dts, dts[-1])

    # models
    def process_model(x, u, noise, input_noise, dt, param_dict): return dynamic_obj.forward_prop(
        x, dynamic_obj.dxdt(x, u + input_noise, param_dict), dt) + noise

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


def extract_input(data, data_indices, dynamic_obj):
    nt = data_indices[1] - data_indices[0] + 1
    U = np.zeros((dynamic_obj.num_in, nt))

    wheelspeed = np.array(data['wheelspeed']).T
    wheelangle = np.array(data['wheelangle']).T
    w = np.array(data['yawrate']).T
    U[0:1, :] = 0.5*(wheelspeed[0, data_indices[0]:data_indices[1]+1] +
                     wheelspeed[1, data_indices[0]:data_indices[1]+1])
    U[1:2, :] = 0.5*(wheelspeed[2, data_indices[0]:data_indices[1]+1] +
                     wheelspeed[3, data_indices[0]:data_indices[1]+1])
    U[2:3, :] = 0.5*(wheelangle[0, data_indices[0]:data_indices[1]+1] +
                     wheelangle[1, data_indices[0]:data_indices[1]+1])
    U[3:4, :] = w[0, data_indices[0]:data_indices[1]+1]

    dynamic_obj.U = U


def extract_output(data, data_indices, data_filename, configuration, dynamic_obj):
    # check if mapping exists in the data file
    for key in configuration['output_data_keys']:
        assert key in data.keys(), "Key {} not present in data file {}".format(key, data_filename)
    for key in configuration['output_data_dot_keys']:
        assert key in data.keys(), "Key {} not present in data file {}".format(key, data_filename)

    # extract data
    nt = data_indices[1] - data_indices[0] + 1
    num_out = dynamic_obj.num_out
    outputs = np.zeros((num_out, nt))
    index = 0
    for key in configuration['output_data_keys']:
        if key == 'ax' or key == 'ay':
            outputs[index:index+1,
                    :] = np.array(data[key])[data_indices[0]-1:data_indices[1], :].T
        else:
            outputs[index:index+1,
                    :] = np.array(data[key])[data_indices[0]:data_indices[1]+1, :].T
        index += 1
    for key in configuration['output_data_dot_keys']:
        outputs[index:index+1,
                :] = np.array(data[key])[data_indices[0]:data_indices[1]+1, :].T
        index += 1
    dynamic_obj.outputs = outputs

    return True


def extract_initial_cond(data, data_indices, configuration, dynamic_obj, first_file, continue_estimation):
    # put in the dynamic states
    for key, index in dynamic_obj.global_state_dict.items():
        dynamic_obj.initial_cond[index] = data[configuration['data_state_mapping']
                                               [key]][data_indices[0]][0]

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
        data['lr'])), ('ref', 0.5*(float(radii[0])+float(radii[1]))), ('rer', 0.5*(float(radii[2])+float(radii[3]))), ('g', 9.81), ('rho', 1.225), ('af', 1.6 + 0.00056*(float(data['m']) - 765.0)), ('sl_f', 0.0)])

    # configuration for testing the estimator + create dynamic object
    configuration = {'output_keys': ['vx', 'vy', 'ax', 'ay'],
                     'output_data_keys': ['vx', 'vy', 'ax', 'ay'],
                     'output_dot_keys': [],
                     'output_data_dot_keys': [],
                     'est_params': ['fr', 'sl_r', 'sc_f', 'sc_r', 'da'],
                     'init_params': [0.0, 0.0, 0.0, 0.0, 0.0],
                     'init_param_cov': [1e1, 1e1, 1e1, 1e1, 1e1],
                     'std_x': 0.05,
                     'std_y': 0.05,
                     'std_theta': 0.5*math.pi/180.0,
                     'std_vx': 1e-2,
                     'std_vy': 1e-2,
                     'std_ax': 1e-2,
                     'std_ay': 1e-2,
                     'std_w': 0.25*math.pi/180.0,
                     'std_x_out': 0.1,
                     'std_y_out': 0.1,
                     'std_theta_out': math.pi/180.0,
                     'std_vx_out': 0.05,
                     'std_vy_out': 0.05,
                     'std_ax_out': 0.05,
                     'std_ay_out': 0.05,
                     'std_vx_dot_out': 0.05,
                     'std_vy_dot_out': 0.05,
                     'std_theta_dot_out': math.pi/180.0,
                     'time_varying_q': 0.0,
                     'threshold_ws': 20.0}
    configuration['data_state_mapping'] = {
        'x': 'x', 'y': 'y', 'theta': 'heading', 'vx': 'vx', 'vy': 'vy', 'w': 'yawrate', 'ax': 'ax', 'ay': 'ay'}
    dynamic_obj = create_dyn_obj(
        RearDriveFrontSteerSubStateVelEst, param_dict, **configuration)

    first_file = True
    # expect to restart from previous cycle (previous mat file?)
    continue_estimation = False

    # load the data from matlab file
    for mat_file in mat_files:
        # read data
        data = loadmat(mat_file)

        # perform lls fitting
        lls_params = least_square_test(
            param_dict, data, threshold_ws=configuration['threshold_ws'], estimate_vx_vy=True, use_fitted_vx_vy=False, yaw_rate_derivative=False)

        # Use initial condition of parameters based on LLS (for debugging purposes) to check dynamic model
        first_friction = data['friction'][0][0]
        for i, key in enumerate(configuration['est_params']):
            if key != 'cd':
                configuration['init_params'][i] = lls_params[first_friction]['best']['param'][key]
            else:
                configuration['init_params'][i] = 2*lls_params[first_friction]['best']['param']['da']/(
                    param_dict['rho']*param_dict['af'])
            param_dict[key] = configuration['init_params'][i]

        # get data filter based on wheel speed threshold
        data_indices = get_data_indices(data, configuration['threshold_ws'])
        if data_indices[1] - data_indices[0] + 1 < 0:
            continue

        # create process and observation models
        extract_model(data, data_indices, dynamic_obj)

        # extract input from data
        extract_input(data, data_indices, dynamic_obj)

        # extract output from data
        extract_output(data, data_indices, mat_file,
                       configuration, dynamic_obj)

        # get initial condition from data
        extract_initial_cond(data, data_indices, configuration,
                             dynamic_obj, first_file, continue_estimation)

        # perform filtering
        est_states, cov_states = create_filtered_estimates(
            dynamic_obj, method, order)
        dynamic_obj.P_end = cov_states[:, :, -1]
        dynamic_obj.initial_cond = est_states[:, -1:]

        # plot the evolution of states and parameters
        plot_stuff(dynamic_obj, est_states, angle_states=configuration.get('angle_states', []), data_state_mapping=configuration.get(
            'data_state_mapping', {}), data=data, data_indices=data_indices, num_rows=[2, 2])

        if first_file:
            first_file = False
