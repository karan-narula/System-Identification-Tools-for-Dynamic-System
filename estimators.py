import math
import numpy.matlib
import numpy as np
from scipy.linalg import block_diag
from scipy.special import comb
from scipy.optimize import least_squares
import itertools


def sample_gaussian(mu, Sigma, N=1):
    """
    Draw N random row vectors from a Gaussian distribution
    """

    N = int(N)
    n = len(mu)
    U, s, V = np.linalg.svd(Sigma)
    S = np.zeros(Sigma.shape)
    for i in range(min(Sigma.shape)):
        S[i, i] = math.sqrt(s[i])

    M = np.random.normal(size=(n, N))
    M = np.dot(np.dot(U, S), M) + mu

    return M


def kinematic_state_observer(initial_cond, yaw_rates, inertial_accs, long_vs, T, alpha):
    num_sol = len(T)
    states = np.zeros((2, num_sol))
    states[:, 0] = np.squeeze(initial_cond[3:5])

    # fixed matrices
    C = np.array([1, 0])
    B = np.identity(2)
    A = np.zeros((2, 2))

    for i in range(1, num_sol):
        # current yaw rate
        yaw_rate = yaw_rates[i-1]

        # put yaw_rate in A matrix
        A[0, 1] = yaw_rate
        A[1, 0] = -yaw_rate

        # gain matrix based on yaw rate
        K = 1.0*np.array([2*alpha*math.fabs(yaw_rate),
                          (alpha**2 - 1)*yaw_rate])

        # state observer equation
        states_dot = np.matmul(
            (A - np.matmul(K, C)), states[:, i-1]) + np.matmul(B, inertial_accs[:, i-1]) + K*long_vs[i-1]

        dt = T[i] - T[i-1]
        states[:, i] = states[:, i-1] + dt*states_dot

    return states


class PointBasedFilter(object):
    """
    Class for performing UKF/CKF prediction or update
    method-> The method for filtering algorithm, there are two choices: 'UKF' for unscented Filter and 'CKF' for Cubature Filter
    order-> Order of accuracy for integration rule. Currently, there are two choices: 2 and 4
    """

    def __init__(self, method, order):
        methods = ['UKF', 'CKF']
        orders = [2, 4]

        assert method in methods, "Given method not implemented or doesn't exist. Current methods available are 'UKF' and 'CKF'"
        assert order in orders, "Given order not implemented. Current available orders are 2 and 4"

        self.method = method
        self.order = order

    def predict_and_update(self, X, P, f, h, Q, R, u, y, iq=None):
        """
        X: expected value of the states (n x 1) array
        P: covariance of the states (n x n) array
        f: function handle for the process model
        h: function handle for the observation model
        Q: process model noise covariance in the prediction step (nq x nq) array
        R: observation model noise covariance in the update step (nu x nu) array
        u: current input required for function f & possibly function h
        y: current measurement/output of the system (nu x 1) array
        iq: index of the states that have additive noise q in the process model (nq x 1) array, i.e. X(ia) = f(X(ia)) + q 
        """
        if iq is None:
            iq = np.arange(X.shape[0])
        # create augmented system of the states and the noises
        n = len(X)
        nq = len(iq)
        nu = len(y)
        X1 = np.concatenate((X, np.zeros((nq, 1)), np.zeros((nu, 1))), axis=0)
        P1 = block_diag(P, Q, R)

        # generate cubature/sigma points and the weights based on the method
        if self.method == 'UKF':
            if self.order == 2:
                x, L, W, WeightMat = self.sigmas2(X1, P1)
            elif self.order == 4:
                x, L, W, WeightMat = self.sigmas4(X1, P1)
        elif self.method == 'CKF':
            if self.order == 2:
                x, L, W, WeightMat = self.cubature2(X1, P1)
            elif self.order == 4:
                x, L, W, WeightMat = self.cubature4(X1, P1)

        # prediction step
        ia = np.arange(n)
        ib = np.arange(n, n+nq)
        X, x, P, x1 = self.unscented_transformF(
            x, W, WeightMat, L, f, u, iq, ia, ib)

        # update step
        if nu > 0:
            ip = np.arange(n+nq, n+nq+nu)
            Z, _, Pz, z2 = self.unscented_transformH(
                x, W, WeightMat, L, h, u, ia, ip)
            # transformed cross-covariance
            Pxy = np.matmul(np.matmul(x1, WeightMat), z2.T)
            # Kalman gain
            K = np.matmul(Pxy, np.linalg.inv(Pz))
            # state update
            X += np.matmul(K, y - Z)
            # covariance update
            P -= np.matmul(K, Pxy.T)

        return X, P

    def unscented_transformH(self, x, W, WeightMat, L, f, u, ia, ip):
        n = len(ip)
        Y = np.zeros((n, 1))
        y = np.zeros((n, L))
        for k in range(L):
            y[:, k] = f(x[ia, k], u) + x[ip, k]
            Y += W.flat[k]*y[:, k:k+1]

        y1 = y - Y
        P = np.matmul(np.matmul(y1, WeightMat), y1.T)

        return Y, y, P, y1

    def unscented_transformF(self, x, W, WeightMat, L, f, u, iq, ia, ib):
        """
        Function to propagate sigma/cubature points during the prediction step
        """
        order = len(ia)
        Y = np.zeros((order, 1))
        y = x
        for k in range(L):
            # prediction using function handlea
            y[ia, k] = f(y[ia, k], u)
            y[iq, k] += y[ib, k]

            # iterative computation of expected value
            Y += W.flat[k]*y[np.arange(order), k:k+1]

        y1 = y[np.arange(order), :] - Y
        P = np.matmul(np.matmul(y1, WeightMat), y1.T)

        return Y, y, P, y1

    def sigmas2(self, X, P):
        """
        function to generate second order sigma points
        """
        n = X.shape[0]
        # some constants based on augmented dimentionality
        Params = [1 - n/3.0, 1.0/6.0, math.sqrt(3.0)]
        L = 2*n + 1
        W = np.concatenate(
            (np.array([[Params[0]]]), np.matlib.repmat(Params[1], 1, 2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set
        temp = np.zeros((n, L))
        loc = np.arange(n)
        l_index = loc*L + loc + 1
        temp.flat[l_index] = Params[2]
        l_index += n
        temp.flat[l_index] = -Params[2]

        Y = np.matlib.repmat(X, 1, L)
        x = Y + np.matmul(sqP, temp)

        # for debugging
        # self.verifySigma(temp, W, 3)

        return x, L, W, WeightMat

    def sigmas4(self, X, P):
        """
        function to generate fourth order sigma points
        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n**2 + 1
        W = np.concatenate((np.array([[1 + (n**2-7.0*n)/18.0]]), np.matlib.repmat(
            (4-n)/18.0, 1, 2*n), np.matlib.repmat(1.0/36.0, 1, 2*n**2-2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create first type of sigma point set
        s = math.sqrt(3.0)
        temp = np.zeros((n, 2*n+1))
        loc = np.arange(n)
        l_index = loc*(2*n+1) + loc + 1
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        Y = np.matlib.repmat(X, 1, 2*n+1)
        x = Y + np.matmul(sqP, temp)

        # create second type of sigma point: 2n**2 - 2n points based on (s2,s2) structure
        temp1 = np.zeros((n, 2*n**2 - 2*n))
        count = comb(n, 2, exact=True)
        loc = np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(range(n), 2)), int, count=count*2).reshape(-1, 2)
        l_index = loc*(2*n**2 - 2*n) + \
            np.matlib.repmat(np.arange(count)[:, np.newaxis], 1, 2)
        for i in itertools.product([1, 2], repeat=2):
            temp1.flat[l_index[:, 0]] = (-1)**i[0]*s
            temp1.flat[l_index[:, 1]] = (-1)**i[1]*s
            l_index += count

        Y = np.matlib.repmat(X, 1, 2*n**2 - 2*n)
        x = np.concatenate((x, Y + np.matmul(sqP, temp1)), axis=1)

        # for debugging
        # temp = np.concatenate((temp, temp1), axis=1)
        # self.verifySigma(temp, W, 5)

        return x, L, W, WeightMat

    def cubature2(self, X, P):
        """
        function to generate second order cubature points
        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n
        W = np.matlib.repmat(1.0/L, 1, L)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set
        s = math.sqrt(n)
        temp = np.zeros((n, L))
        loc = np.arange(n)
        l_index = loc*L + loc
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        Y = np.matlib.repmat(X, 1, L)
        x = Y + np.matmul(sqP, temp)

        # for debugging
        # self.verifySigma(temp, W, 2)

        return x, L, W, WeightMat

    def cubature4(self, X, P):
        """
        function to generate fourth order cubature points
        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n**2 + 1
        W = np.concatenate((np.array([[2.0/(n+2.0)]]), np.matlib.repmat((4-n)/(2.0*(
            n+2)**2), 1, 2*n), np.matlib.repmat(1.0/((n+2.0)**2), 1, 2*n**2-2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set
        s = math.sqrt(n+2.0)
        temp = np.zeros((n, 2*n+1))
        loc = np.arange(n)
        l_index = loc*(2*n+1) + loc + 1
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        Y = np.matlib.repmat(X, 1, 2*n+1)
        x = Y + np.matmul(sqP, temp)

        # create second type of sigma point: 2n**2 - 2n points based on (s2,s2) structure
        s = math.sqrt(n+2.0)/math.sqrt(2.0)
        temp1 = np.zeros((n, 2*n**2 - 2*n))
        count = comb(n, 2, exact=True)
        loc = np.fromiter(itertools.chain.from_iterable(
            itertools.combinations(range(n), 2)), int, count=count*2).reshape(-1, 2)
        l_index = loc*(2*n**2 - 2*n) + \
            np.matlib.repmat(np.arange(count)[:, np.newaxis], 1, 2)
        for i in itertools.product([1, 2], repeat=2):
            temp1.flat[l_index[:, 0]] = (-1)**i[0]*s
            temp1.flat[l_index[:, 1]] = (-1)**i[1]*s
            l_index += count

        Y = np.matlib.repmat(X, 1, 2*n**2 - 2*n)
        x = np.concatenate((x, Y + np.matmul(sqP, temp1)), axis=1)

        # for debugging
        # temp = np.concatenate((temp, temp1), axis=1)
        # self.verifySigma(temp, W, 5)

        return x, L, W, WeightMat

    def verifyTransformedSigma(self, x, WeightMat, X, P):
        sigma_mean = np.zeros(X.shape)
        W = np.diag(WeightMat)
        for i in range(x.shape[1]):
            sigma_mean += W[i]*x[:, i:i+1]

        sigma_cov = np.matmul(
            np.matmul((x - sigma_mean), WeightMat), np.transpose(x - sigma_mean))

        mean_close = np.allclose(X, sigma_mean)
        cov_close = np.allclose(P, sigma_cov)

        return mean_close, cov_close

    def verifySigma(self, x, W, order=2):
        n, L = x.shape

        # check moment and cross moment of each order
        for i in range(1, order+1):
            # find all possible combinations for adding up to order i
            arr = [0]*n
            outputs = []
            findCombinationsUtil(arr, 0, i, i, outputs)
            for output in outputs:
                theoretical_moment = 1.0
                for power in output:
                    theoretical_moment *= self.stdGaussMoment(power)
                    if theoretical_moment == 0:
                        break

                elem_combinations = itertools.permutations(
                    range(n), len(output))
                for elem_combination in elem_combinations:
                    moment = (
                        W*np.prod(x[elem_combination, :]**np.matlib.repmat(output, L, 1).T, axis=0)).sum()
                    assert np.isclose(moment, theoretical_moment), "The {}th moment with element {} and power {} yielded value of {} instead of {}".format(
                        i, elem_combination, output, moment, theoretical_moment)

    def stdGaussMoment(self, order):
        """
        Calculate order-th moment of univariate std Gaussian distribution (zero mean, 1 std)
        """
        if order % 2:
            return 0.0
        else:
            prod = 1.0
            for i in range(1, order, 2):
                prod *= i

            return prod


def findCombinationsUtil(arr, index, num, reducedNum, output):
    # Base condition
    if reducedNum < 0:
        return

    # If combination is found, store it
    if reducedNum == 0:
        output.append(arr[:index])
        return

    # find pervious number stored
    prev = 1 if (index == 0) else arr[index-1]

    # start loop from previous number
    for k in range(prev, num+1):
        # next element of array
        arr[index] = k

        # recursively try this combination with reduced number
        findCombinationsUtil(arr, index+1, num, reducedNum-k, output)


def fit_data_rover(states, U, dt, vxdot=np.array([]), yawrate=np.array([]), vy=np.array([])):
    """
    Perform LS and NLS fitting parameters estimation for the rover dynamics (c1-c9)
    states: x, y, theta and vx at different time instances (4 x nt array)
    U: input to the model at different time instances consisting of steering angle and commanded velocity (2 x nt array)
    vxdot: optionally, linear longitudinal acceleration at different time instances (nt x 1 array)
    yawrate: optionally, yaw rate at different time instances (nt x 1 array)
    vy: optionally, lateral velocity if observed (nt x 1 array)
    """

    parameters = [0]*9

    # first fit the longitudinal acceleration
    if vxdot.shape[0] == 0:
        vxdot = np.diff(states[3, :])/dt
    else:
        vxdot = vxdot[:-1]
    diff = np.reshape(states[3, :-1] - U[1, :-1], [-1, 1])
    A_long_accel = np.concatenate(
        (np.ones((len(vxdot), 1)), diff, np.square(diff)), axis=1)
    parameters[4:7] = np.linalg.lstsq(
        A_long_accel, vxdot[:, np.newaxis], rcond=None)[0][:, 0].tolist()

    # fitting for yaw rate
    if yawrate.shape[0] == 0:
        yawrate = np.diff(states[2, :])/dt
    else:
        yawrate = yawrate[:-1]

    def nls_yawrate(x, yaw_rate, steering_cmd, vx):
        return yaw_rate - np.tan(x[0]*steering_cmd + x[1])*vx/(x[2] + x[3]*vx**2)

    x0 = np.array([1, 0, 1.775, 0])
    res_l = least_squares(nls_yawrate, x0, args=(
        yawrate, U[0, :-1], states[3, :-1]))
    parameters[:4] = res_l.x

    if vy.shape[0] == 0:
        xdot = np.diff(states[0, :])/dt
        ydot = np.diff(states[1, :])/dt

        # fitting for xdot and ydot coordinates
        def nls_xy(params, xdot, ydot, vx, yaw, yaw_rate):
            vy = yaw_rate*(params[0] + params[1]*vx**2)
            res_x = xdot - (vx*np.cos(yaw) - vy*np.sin(yaw))
            res_y = ydot - (vx*np.sin(yaw) + vy*np.cos(yaw))
            return np.concatenate((res_x, res_y)).flatten()

        x0 = np.array([0.1, 0.1])
        res_l = least_squares(nls_xy, x0, args=(
            xdot, ydot, states[3, :-1], states[2, :-1], yawrate))
        parameters[7:9] = res_l.x
    else:
        # fitting for lateral velocity
        prod = yawrate*(states[3, :-1]**2)
        A_lat_vel = np.concatenate(
            (yawrate[:, np.newaxis], prod[:, np.newaxis]), axis=1)
        parameters[7:9] = np.linalg.lstsq(A_lat_vel, vy[:-1, np.newaxis], rcond=None)[
            0][:, 0].tolist()

    return parameters


def sample_nlds(z0, U, nt, f, h, num_out, Q=None, P0=None, R=None):
    """
    Retrieve ground truth, initial and output data (SNLDS: Stochastic non-linear dynamic system)
    z0: initial ground truth condition (n x 1 array)
    U: inputs for the process and observation model (nu x nt array)
    nt: number of simulation steps (scalar)
    f: function handle for one-time step forward propagating the state
    h: function handle for retrieving the outputs of the system as a function of system states
    num_out: number of outputs from function h (scalar)
    Q: noise covariance matrix for additive noise in the stochastic model f (n x n array)
    P0: initial covariance for the initial estimate around the ground truth (n x n array)
    R: covariance matrix of additive noise in h function (num_out x num_out array)
    """
    if not U:
        U = np.zeros((0, nt))
    if Q is None:
        Q = np.zeros((len(z0), len(z0)))
    if P0 is None:
        P0 = np.zeros((len(z0), len(z0)))
    if R is None:
        R = np.zeros((num_out, num_out))

    # check sizes of received matrices
    assert U.shape[1] == nt, "Expected input for all {} time instances but only received {}".format(
        nt, U.shape[1])
    assert Q.shape == (len(z0), len(
        z0)), "Inconsistent size of process noise matrix"
    assert P0.shape == (len(z0), len(
        z0)), "Inconsistent size of initial covariance matrix"
    assert R.shape == (
        num_out, num_out), "Inconsistent size of observation noise matrix"

    # generate noise samples for stochastic model and observations
    state_noise_samples = sample_gaussian(np.zeros(z0.shape), Q, nt)
    obs_noise_samples = sample_gaussian(
        np.zeros((num_out, 1)), R, nt)

    # initialise matrices to return
    gt_states = np.zeros((z0.shape[0], nt))
    gt_states[:, 0:1] = z0
    initial_cond = sample_gaussian(z0, P0, 1)
    outputs = np.zeros((num_out, nt))
    outputs[:, 0] = h(gt_states[:, 0], U[:, 0]) + obs_noise_samples[:, 0]

    for i in range(1, nt):
        gt_states[:, i] = f(gt_states[:, i-1], U[:, i-1]) + \
            state_noise_samples[:, i-1]
        outputs[:, i] = h(gt_states[:, i], U[:, i]) + obs_noise_samples[:, i]

    return gt_states, initial_cond, outputs


def test_pbgf_linear(n=10, m=5, nt=10):
    """
    Test the PointBasedFilter against KF when problem is linear
    """
    # control random seed generator
    np.random.seed(0)

    # set up the true initial condition
    X = 5.0*np.random.randn(n, 1)
    P = 10.0*np.random.randn(n, n)
    P = np.matmul(P, P.T)

    # process and measurement models (linear)
    dt = 0.05
    J = np.eye(n) + dt*(-2.0*np.eye(n) +
                        np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1))

    def process_model(x, u=[]): return np.matmul(J, x)
    Q = 5.0*np.eye(n)
    out_loc = np.random.permutation(n)[:m]
    R = 1.0*np.eye(m)
    H = np.zeros((m, n))
    l_ind = out_loc + np.arange(m)*n
    H.flat[l_ind] = 1.0
    def observation_model(x, u=[]): return np.matmul(H, x)

    ## generate the output of the real time system
    x_gt, x0, outputs = sample_nlds(
        X, [], nt, process_model, observation_model, m, Q, P, R)

    ## loop through and compare result from KF and a pbgf
    pbgf = PointBasedFilter('CKF', 2)
    X1 = x0.copy()
    X2 = x0.copy()
    P1 = P.copy()
    P2 = P.copy()
    mse = np.zeros((nt, 1))
    mse[0] = np.mean((X1-x_gt[:, 0])**2)
    trace = np.zeros(mse.shape)
    trace[0] = np.trace(P1)
    for i in range(1, nt):
        # KF code
        # prediction step
        X1 = process_model(X1)
        P1 = np.matmul(np.matmul(J, P1), J.T) + Q

        # update step
        z = outputs[:, i:i+1] - observation_model(X1)
        S = np.matmul(np.matmul(H, P1), H.T) + R
        K = np.matmul(np.matmul(P1, H.T), np.linalg.inv(S))
        X1 += np.matmul(K, z)
        P1 -= np.matmul(np.matmul(K, H), P1)

        ## PBGF code
        X2, P2 = pbgf.predict_and_update(
            X2, P2, process_model, observation_model, Q, R, [], outputs[:, i:i+1])

        assert np.allclose(
            P1, P2), "Covariance from KF and PBGF should be the same as problem is linear"
        assert np.allclose(
            X1, X2), "Expected Value from KF and PBGF should be the same as problem is linear"

        # calculate mse and put in array
        mse[i] = np.mean((X1-x_gt[:, i])**2)
        trace[i] = np.trace(P1)

    import matplotlib.pyplot as plt
    plt.plot(mse, marker='x', label='mse')
    plt.plot(trace, marker='o', label='trace')
    plt.grid(True, "both")
    plt.legend()
    plt.show()


def test_pbgf_1d_linear(gt_const=10.0, initial_cov=10.0, q_cov=1e-2, r_cov=1, nt=50):
    """
    estimate a random constant
    """
    # control random seed generator
    np.random.seed(0)

    # set up the true initial condition
    X = np.array([[gt_const]])
    P = initial_cov*np.ones((1, 1))

    # process and observation model
    def process_model(x, u=[]): return x
    def observation_model(x, u=[]): return x

    # process and observation noises
    R = np.array([[r_cov]])
    Q = np.array([[q_cov]])

    # generate the initial condition
    x_gt, x0, outputs = sample_nlds(
        X, [], nt, process_model, observation_model, 1, Q, P, R)

    ## loop through and compare result from KF and a pbgf
    pbgf = PointBasedFilter('CKF', 2)
    X1 = x0.copy()
    X2 = x0.copy()
    P1 = P.copy()
    P2 = P.copy()
    est_history = np.zeros((nt, 1))
    est_history[0] = x0.copy()
    mse = np.zeros((nt, 1))
    mse[0] = np.mean((X1-x_gt[:, 0])**2)
    trace = np.zeros(mse.shape)
    trace[0] = np.trace(P1)
    for i in range(1, nt):
        # KF code
        # prediction step
        P1 = P1 + Q

        # update step
        z = outputs[:, i:i+1] - X1
        S = P1 + R
        K = np.matmul(P1, np.linalg.inv(S))
        X1 += np.matmul(K, z)
        P1 -= np.matmul(K, P1)

        ## PBGF code
        X2, P2 = pbgf.predict_and_update(
            X2, P2, process_model, observation_model, Q, R, [], outputs[:, i:i+1])

        assert np.allclose(
            P1, P2), "Covariance from KF and PBGF should be the same as problem is linear"
        assert np.allclose(
            X1, X2), "Expected Value from KF and PBGF should be the same as problem is linear"

        # calculate mse and put in array
        mse[i] = np.mean((X1-x_gt[:, i])**2)
        trace[i] = np.trace(P1)
        est_history[i] = X1[:].copy()

    import matplotlib.pyplot as plt
    plt.plot(est_history, label='est_voltage')
    plt.plot(x_gt[0,:], linestyle='--', label='real_voltage')
    plt.plot(mse, marker='x', label='mse')
    plt.plot(trace, marker='o', label='trace')
    plt.grid(True, "both")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_pbgf_linear()
    test_pbgf_1d_linear(q_cov=1e-2)
