import math
import numpy as np

from estimators import sample_gaussian


class AbstractDyn(object):
    """
    Abstract dynamics class to be expanded upon
    """

    def __init__(self, param_dict, expected_keys=[], state_keys=[], state_dot_keys=[], state_dict={}):
        # store the parameter dictionary for the vehicle dynamics
        self.param_dict = param_dict

        # store expected keys of param dictionary
        self.expected_keys = expected_keys

        # store the dictionary linking state to index
        self.state_dict = state_dict

        # check if param dictionary is valid
        assert self.check_param_dict(), "Parameter dictionary does not contain all requried keys"

        # store state its derivative indices for generating output
        self.state_indices = []
        self.state_dot_indices = []
        for key in state_keys:
            if key in self.state_dict:
                self.state_indices.append(self.state_dict[key])
        for key_dot in state_dot_keys:
            if key_dot in self.state_dict:
                self.state_dot_indices.append(self.state_dict[key_dot])
        # dimensionality of states, inputs and outputs
        self.num_states = len(self.state_dict.keys())
        self.num_out = len(self.state_indices) + len(self.state_dot_indices)
        self.num_in = 0

    def check_param_dict(self):
        """
        Can be overriden by inherited class 
        """
        for key in self.expected_keys:
            if key not in self.param_dict.keys():
                return False

        return True

    def sample_nlds(self, z0, U, T, Q=None, P0=None, R=None, store_variables=True):
        """
        get ground truth and output data (SNLDS: Stochastic non-linear dynamic system)
        """
        if Q is None:
            Q = np.zeros((len(z0), len(z0)))
        if P0 is None:
            P0 = np.zeros((len(z0), len(z0)))
        if R is None:
            R = np.zeros((self.num_out, self.num_out))

        # check sizes of received matrices
        num_sol = len(T)
        assert len(
            z0) == self.num_states, "True initial condition is of incorrect size"
        assert U.shape == (
            self.num_in, num_sol), "Incorrect size of input matrix"
        assert Q.shape == (
            self.num_states, self.num_states), "Inconsistent size of process noise matrix"
        assert P0.shape == (
            self.num_states, self.num_states), "Inconsistent size of initial covariance matrix"
        assert R.shape == (
            self.num_out, self.num_out), "Inconsistent size of observation noise matrix"
        # generate noise samples for stochastic model and observations
        state_noise_samples = sample_gaussian(np.zeros(z0.shape), Q, num_sol)
        obs_noise_samples = sample_gaussian(
            np.zeros((self.num_out, 1)), R, num_sol)

        # initialise matrices to return
        gt_states = np.zeros((z0.shape[0], num_sol))
        gt_states[:, 0:1] = z0
        gt_states_dot = np.zeros(gt_states.shape)
        initial_cond = sample_gaussian(z0, P0, 1)
        outputs = np.zeros((self.num_out, num_sol))
        outputs[:, 0] = self.output_model(
            gt_states[:, 0], U[:, 0]) + obs_noise_samples[:, 0]

        for i in range(1, num_sol):
            gt_states_dot[:, i-1] = self.dxdt(gt_states[:, i-1], U[:, i-1])
            # first order euler approximation
            dt = T[i] - T[i-1]
            gt_states[:, i] = self.forward_prop(
                gt_states[:, i-1], gt_states_dot[:, i-1], dt) + state_noise_samples[:, i-1]

            # output
            outputs[:, i] = self.output_model(
                gt_states[:, i], U[:, i]) + obs_noise_samples[:, i]

        gt_states_dot[:, num_sol -
                      1] = self.dxdt(gt_states[:, num_sol-1], U[:, num_sol-1])

        # store varibles to avoid parsing things around?
        if store_variables:
            self.Q = Q
            self.R = R
            self.P0 = P0

            self.U = U
            self.T = T

            self.gt_states = gt_states
            self.gt_states_dot = gt_states_dot
            self.initial_cond = initial_cond
            self.outputs = outputs

        return gt_states, gt_states_dot, initial_cond, outputs

    def forward_prop(self, state, state_dot, dt):
        return state + dt*state_dot

    def output_model(self, state, u):
        return np.concatenate((state[self.state_indices], self.dxdt(state, u)[0, self.state_dot_indices]))


class FrontSteered(AbstractDyn):
    """
    Assumptions:
    (i) steering angles of the front left and right wheels are the same. front steered only.
    (ii) steering command is small enough such that sin(steering) = steering
    (iii) vehicle operates in the linear region of the tire-force curve with negligible inclination and bang angles
    (iv) left and right wheels on each axle have the same stiffness
    (v) dominant forces are the tire-road contact forces. influences due to wind and air resistance are ignored.
    
    model from: "Tire-Stiffness and Vehicle-State Estimation Based on Noise-Adaptive Particle Filtering"
    inputs = [steering_angle, wf, wr]; wf -> front wheel rotation rate, wr -> rear wheel rotation rate
    states = [x, y, theta, vx, vy, omega]
    """

    def __init__(self, param_dict, state_keys, state_dot_keys=[], acc_output=False):
        # state dictionary for this model
        state_dict = {'x': 0, 'y': 1, 'theta': 2, 'vx': 3, 'vy': 4, 'omega': 5}

        # expected parameter keys
        expected_keys = ["mass", "lr", "lf",
                         "e_wr", "cxf", "cxr", "cyf", "cyr", "iz"]

        super(FrontSteered, self).__init__(
            param_dict, expected_keys=expected_keys, state_keys=state_keys, state_dot_keys=state_dot_keys, state_dict=state_dict)

        # dimensionality of input
        self.num_in = 3

        # inertial acceleration as an output
        if acc_output:
            self.num_out += 2
        self.acc_output = acc_output

    def dxdt(self, state, u):
        # get the inputs
        steering_angle = u[0]
        wf = u[1]
        wr = u[2]

        # get the states
        heading = state[2]
        vx = state[3]
        vy = state[4]
        omega = state[5]

        # calculate lateral tire forces
        if vx != 0.0:
            slip_ang_f = steering_angle - (vy + self.param_dict['lf']*omega)/vx
            slip_ang_r = (self.param_dict['lr']*omega - vy)/vx
        else:
            slip_ang_f = 0.0
            slip_ang_r = 0.0
        fy_f = self.param_dict['cyf']*slip_ang_f
        fy_r = self.param_dict['cyr']*slip_ang_r

        # calculate longitudinal force
        wheel_slip_f = (vx - self.param_dict['e_wr']*wf) / \
            max(vx, self.param_dict['e_wr']*wf)
        wheel_slip_r = (vx - self.param_dict['e_wr']*wr) / \
            max(vx, self.param_dict['e_wr']*wr)
        fx_f = self.param_dict['cxf']*wheel_slip_f
        fx_r = self.param_dict['cxr']*wheel_slip_r

        # calculate lateral and longitudinal acceleration
        vx_dot = (fx_f*math.cos(steering_angle) + fx_r + fy_f *
                  math.sin(steering_angle) + self.param_dict['mass']*vy*omega)/self.param_dict['mass']
        ax_inertial = vx_dot - vy*omega
        vy_dot = (fy_f*math.cos(steering_angle) + fy_r + fx_f *
                  math.sin(steering_angle) - self.param_dict['mass']*vx*omega)/self.param_dict['mass']
        ay_inertial = vy_dot + vx*omega

        # calculate angular acceleration
        omega_dot = (self.param_dict['lf']*(fy_f*math.cos(steering_angle) + fx_f *
                                            math.sin(steering_angle)) - self.param_dict['lr']*fy_r)/self.param_dict['iz']

        # kinematic model based on derived dynamic quantities
        x_dot = vx*math.cos(heading) - vy*math.sin(heading)
        y_dot = vx*math.sin(heading) + vy*math.cos(heading)
        heading_dot = omega

        return np.array([[x_dot, y_dot, heading_dot, vx_dot, vy_dot, omega_dot]])

    def get_acc(self, state, u):
        # get the inputs
        steering_angle = u[0]
        wf = u[1]
        wr = u[2]

        # get the states
        heading = state[2]
        vx = state[3]
        vy = state[4]
        omega = state[5]

        # calculate lateral tire forces
        if vx != 0.0:
            slip_ang_f = steering_angle - (vy + self.param_dict['lf']*omega)/vx
            slip_ang_r = (self.param_dict['lr']*omega - vy)/vx
        else:
            slip_ang_f = 0.0
            slip_ang_r = 0.0
        fy_f = self.param_dict['cyf']*slip_ang_f
        fy_r = self.param_dict['cyr']*slip_ang_r

        # calculate longitudinal force
        wheel_slip_f = (vx - self.param_dict['e_wr']*wf) / \
            max(vx, self.param_dict['e_wr']*wf)
        wheel_slip_r = (vx - self.param_dict['e_wr']*wr) / \
            max(vx, self.param_dict['e_wr']*wr)
        fx_f = self.param_dict['cxf']*wheel_slip_f
        fx_r = self.param_dict['cxr']*wheel_slip_r

        ax_inertial = (fx_f*math.cos(steering_angle) + fx_r + fy_f *
                       math.sin(steering_angle))/self.param_dict['mass']
        ay_inertial = (fy_f*math.cos(steering_angle) + fy_r + fx_f *
                       math.sin(steering_angle))/self.param_dict['mass']

        return np.array([[ax_inertial, ay_inertial]])

    def output_model(self, state, u):
        if self.acc_output:
            return np.concatenate((super(FrontSteered, self).output_model(state, u), self.get_acc(state, u)[0]))
        else:
            return super(FrontSteered, self).output_model(state, u)


class RoverDyn(AbstractDyn):
    """
    model from: Michigan guys' RTD python repo
    inputs = [steering_angle, commanded velocity]
    states = [x, y, theta, vx]
    """

    def __init__(self, param_dict, state_keys, state_dot_keys=[], expected_keys=None):
        # state dictionary for this model
        state_dict = {'x': 0, 'y': 1, 'theta': 2, 'vx': 3}

        # expected parameter keys
        if expected_keys is None:
            expected_keys = ["c1", "c2", "c3",
                             "c4", "c5", "c6", "c7", "c8", "c9"]
        super(RoverDyn, self).__init__(param_dict, expected_keys=expected_keys,
                                       state_keys=state_keys, state_dot_keys=state_dot_keys, state_dict=state_dict)

        # specify expected dimensionality of input
        self.num_in = 2

    def dxdt(self, state, u):
        # get the inputs
        steering_angle = u[0]
        vx_cmd = u[1]

        # get the states
        theta = state[2]
        vx = state[3]

        ang_rate = math.tan(self.param_dict['c1']*steering_angle + self.param_dict['c2'])*vx/(
            self.param_dict['c3'] + self.param_dict['c4']*vx**2)
        vy = ang_rate*(self.param_dict['c8'] + self.param_dict['c9']*vx**2)

        # calculate the derivatives
        x_dot = vx*math.cos(theta) - vy*math.sin(theta)
        y_dot = vx*math.sin(theta) + vy*math.cos(theta)
        heading_dot = ang_rate
        vx_dot = self.param_dict['c5'] + self.param_dict['c6'] * \
            (vx - vx_cmd) + self.param_dict['c7']*(vx - vx_cmd)**2

        return np.array([[x_dot, y_dot, heading_dot, vx_dot]])

    def cal_vxvy_from_coord(self, state, state_prev, dt, output=False):
        """
        Calculate longitudinal and lateral velocity by rotating current position into the frame of previous position
        """
        if len(state.shape) == 1:
            state = state[:, np.newaxis]
        if len(state_prev.shape) == 1:
            state_prev = state_prev[:, np.newaxis]

        if output:
            x_ind = self.state_indices.index(self.state_dict['x'])
            y_ind = self.state_indices.index(self.state_dict['y'])
            theta_ind = self.state_indices.index(self.state_dict['theta'])
        else:
            x_ind = self.state_dict['x']
            y_ind = self.state_dict['y']
            theta_ind = self.state_dict['theta']

        prev_ang = state_prev[theta_ind, :]
        R = np.array([[np.cos(prev_ang), np.sin(prev_ang)],
                      [-np.sin(prev_ang), np.cos(prev_ang)]])
        xy_dot = (state[[x_ind, y_ind], :] - state_prev[[x_ind, y_ind], :])/dt
        vxy = np.zeros(xy_dot.shape)
        for i in range(xy_dot.shape[1]):
            vxy[:, i:i+1] = np.matmul(R[:, :, i], xy_dot[:, i:i+1])

        return vxy


class RoverPartialDynEst(RoverDyn):
    """
    model from: same as RoverDyn but some parameters are the states themselves to be estimated online
    inputs = [steering_angle, commanded velocity]
    states = [x, y, theta, vx, ...]
    """

    def __init__(self, param_dict, est_params, state_keys, state_dot_keys=[]):
        expected_keys = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
        # check if est_params are in expected keys
        for est_param in est_params:
            assert est_param in expected_keys, "Parameter {} to be estimated is not in expected keys".format(
                est_param)
            expected_keys.remove(est_param)

        # remove parameters to be estimated from dictionary
        pruned_param_dict = param_dict.copy()
        for est_param in est_params:
            if est_param in pruned_param_dict:
                del pruned_param_dict[est_param]

        super(RoverPartialDynEst, self).__init__(pruned_param_dict,
                                                 state_keys, state_dot_keys=state_dot_keys, expected_keys=expected_keys)

        # append state dictionary
        for i, est_param in enumerate(est_params):
            self.state_dict[est_param] = self.num_states + i

        # dimensionalities of different things
        self.est_params = list(est_params)
        self.num_states += len(self.est_params)

    def dxdt(self, state, u):
        # put the parameters into the dictionary
        for i, est_param in enumerate(self.est_params):
            self.param_dict[est_param] = state[4+i]

        state_dot = super(RoverPartialDynEst, self).dxdt(state, u)
        return np.concatenate((state_dot, np.zeros((1, len(self.est_params)))), axis=1)


def sample_input_rover(T, max_steering=30*math.pi/180.0, max_speed=5.0, cruise_time=2.0):
    """
    Create an example input vector for the rover dynamic model in which:
    (i) steering angle linearly increase to max and decrease back to zero 
    (ii) velocity linearly increase to max and decrease back to zero
    """
    # create input vector
    U = np.zeros((2, len(T)))
    t_after_accel = (T[-1] - cruise_time)/2.0
    t_before_deaccel = t_after_accel + cruise_time

    slope_speed = max_speed/t_after_accel
    slope_steer = max_steering/t_after_accel

    # acceleration phase
    U[0, T <= t_after_accel] = slope_steer*T[T <= t_after_accel]
    U[1, T <= t_after_accel] = slope_speed*T[T <= t_after_accel]

    # cruise phase
    U[0, (T > t_after_accel) & (T < t_before_deaccel)] = max_steering
    U[1, (T > t_after_accel) & (T < t_before_deaccel)] = max_speed

    # deacceleration phase
    U[0, T >= t_before_deaccel] = -slope_steer * \
        (T[T >= t_before_deaccel] - t_before_deaccel) + max_steering
    U[1, T >= t_before_deaccel] = -slope_speed * \
        (T[T >= t_before_deaccel] - t_before_deaccel) + max_speed

    return U


def sample_input_front_steered(T, max_w=45*math.pi/180.0, max_steering=20*math.pi/180.0, cruise_time=2.0):
    """
    Create an example input vector for the front_steered dynamic model in which:
    (i) wheel rate linearly increase to max and decrease back to zero
    (ii) steering angle linearly increase to max and decrease back to zero 
    """
    # create input vector
    U = np.zeros((3, len(T)))
    t_after_accel = (T[-1] - cruise_time)/2.0
    t_before_deaccel = t_after_accel + cruise_time

    slope_w = max_w/t_after_accel
    slope_steer = max_steering/t_after_accel

    # acceleration phase
    U[0, T <= t_after_accel] = slope_steer*T[T <= t_after_accel]
    U[1, T <= t_after_accel] = slope_w*T[T <= t_after_accel]
    U[2, T <= t_after_accel] = slope_w*T[T <= t_after_accel]

    # cruise phase
    U[0, (T > t_after_accel) & (T < t_before_deaccel)] = max_steering
    U[1, (T > t_after_accel) & (T < t_before_deaccel)] = max_w
    U[2, (T > t_after_accel) & (T < t_before_deaccel)] = max_w

    # deacceleration phase
    U[0, T >= t_before_deaccel] = -slope_steer * \
        (T[T >= t_before_deaccel] - t_before_deaccel) + max_steering
    U[1, T >= t_before_deaccel] = -slope_w * \
        (T[T >= t_before_deaccel] - t_before_deaccel) + max_w
    U[2, T >= t_before_deaccel] = -slope_w * \
        (T[T >= t_before_deaccel] - t_before_deaccel) + max_w

    return U


def test_rover_dyn():
    # assume parameter values in accordance to Michigan's rover
    param_dict = {'c1': 1.6615e-5, 'c2': -1.9555e-07, 'c3': 3.6190e-06, 'c4': 4.3820e-07,
                  'c5': -0.0811, 'c6': -1.4736, 'c7': 0.1257, 'c8': 0.0765, 'c9': -0.0140}

    # timing information
    dt = 0.05
    t_f = 20.0
    T = np.arange(0, t_f, dt)

    # create input vector for rover dynamics
    U = sample_input_rover(T)

    # get the ground truth states for propagating rover model
    z0 = np.zeros((4, 1))
    state_keys = ['x', 'y', 'theta', 'vx']
    dynamic_obj = RoverDyn(param_dict, state_keys=state_keys)
    gt_states, _, initial_cond, outputs = dynamic_obj.sample_nlds(z0, U, T)

    assert np.allclose(
        gt_states, outputs), "Gt not the same as output when no noises are present"
    assert np.allclose(
        z0, initial_cond), "Initial condition is not the same as first ground truth despite confident initial condition"

    # check result from the same model with unknown parameters
    z0 = np.zeros((13, 1))
    for i in range(1, 10):
        z0[3+i] = param_dict['c'+str(i)]
    est_params = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
    est_param_dynamic_obj = RoverPartialDynEst({}, est_params, state_keys)
    gt_states1, _, initial_cond1, outputs1 = est_param_dynamic_obj.sample_nlds(
        z0, U, T)

    assert np.allclose(np.matlib.repmat(z0[4:, :], 1, len(
        T)), gt_states1[4:, :]), "States should remain the same, stationary"
    assert np.allclose(
        gt_states, gt_states1[:4, :]), "Gt should be the same given no uncertainty in parameters"
    assert np.allclose(
        outputs, outputs1), "Outputs should also be the same given no uncertainty in parameters"

    # do some plots of the resultant trajectory (sanity check)
    import matplotlib.pyplot as plt
    plt.plot(gt_states[0, :], gt_states[1, :], marker='x')
    plt.show()


if __name__ == '__main__':
    test_rover_dyn()
