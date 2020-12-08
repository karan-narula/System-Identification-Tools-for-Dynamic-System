import math
import numpy.matlib
import numpy as np
from scipy.linalg import block_diag
from scipy.special import comb
from scipy.optimize import least_squares
import itertools
from collections import Iterable


def sample_gaussian(mu, Sigma, N=1):
    """
    Draw N random row vectors from a Gaussian distribution

    Args:
        mu (numpy array [n x 1]): expected value vector
        Sigma (numpy array [n x n]): covariance matrix
        N (int): scalar number of samples

    Returns:
        M (numpy array [n x N]): samples from Gaussian distribtion

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
    """
    Not working yet!
    """
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

    Args:
        method (str): The method for filtering algorithm, there are two choices: 'UKF' for unscented Filter 
            and 'CKF' for Cubature Filter
        order (int): Order of accuracy for integration rule. Currently, there are two choices: 2 and 4

    """

    def __init__(self, method, order):
        methods = ['UKF', 'CKF']
        orders = [2, 4]

        assert method in methods, "Given method not implemented or doesn't exist. Current methods available are 'UKF' and 'CKF'"
        assert order in orders, "Given order not implemented. Current available orders are 2 and 4"

        self.method = method
        self.order = order

    def predict_and_or_update(self, X, P, f, h, Q, R, u, y, Qu=None, additional_args_pm=[], additional_args_om=[], innovation_bound_func={}, predict_flag=True):
        """
        Perform one iteration of prediction and/or update.
        algorithm reference: Algorithm 5.1, page 104 of "Compressed Estimation in Coupled High-dimensional Processes"

        Args:
            X (numpy array [n x 1]): expected value of the states
            P (numpy array [n x n]): covariance of the states
            f (function): function handle for the process model; expected signature f(state, input, model noise, input noise, ...)
            h (function): function handle for the observation model; expected signature h(state, input, noise, ...)
            Q (numpy array [nq x nq]): process model noise covariance in the prediction step
            R (numpy array [nr x nr]): observation model noise covariance in the update step
            u (*): current input required for function f & possibly function h
            y (numpy array [nu x 1]): current measurement/output of the system
            Qu (numpy array [nqu x nqu]): input noise covariance in the prediction step
            additional_args_pm (list): list of additional arguments to be passed to the process model during the prediction step
            additional_args_om (list): list of additional arguments to be passed to the observation model during the update step
            innovation_bound_func (dict): dictionary with innovation index as keys and callable function as value to bound
                innovation when needed
            predict_flag (bool): perform prediction? defaults to true

        Returns:
            X (numpy array [n x 1]): expected value of the states after prediction and update
            P (numpy array [n x n]): covariance of the states after prediction and update

        """
        # create augmented system of the states and the noises (step 1 of algorithm 5.1, equation 5.42)
        n = len(X)
        nq = Q.shape[0]
        if Qu is not None:
            nqu = Qu.shape[0]
        else:
            nqu = 0
            Qu = np.zeros((nqu, nqu))
        nr = R.shape[0]
        X1 = np.concatenate((X, np.zeros((nq+nqu+nr, 1))), axis=0)
        P1 = block_diag(P, Q, Qu, R)

        # generate cubature/sigma points and the weights based on the method (steps 2-4 of algorithm 5.1)
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

        if predict_flag:
            # prediction step (step 5 of algorithm 5.1) by implementing equations 5.25, 5.34 and 5.35 (pages 105-106)
            ia = np.arange(n)
            ib = np.arange(n, n+nq)
            ic = np.arange(n+nq, n+nq+nqu)
            X, x, P, x1 = self.unscented_transformF(
                x, W, WeightMat, L, f, u, ia, ib, ic, additional_args_pm)

        # update step (step 6 of algorithm 5.1) by implementing equations 5.36-5.41 (page 106)
        if len(y):
            # check if innovation keys is valid
            for key in innovation_bound_func:
                assert key in range(len(y)), "Key of innovation bound function dictionary should be within the length of the output"
                assert callable(innovation_bound_func[key]), "Innovation bound function is not callable"

            ip = np.arange(n+nq, n+nq+nr)
            Z, _, Pz, z2 = self.unscented_transformH(
                x, W, WeightMat, L, h, u, ia, ip, len(y), additional_args_om)
            # transformed cross-covariance (equation 5.38)
            Pxy = np.matmul(np.matmul(x1, WeightMat), z2.T)
            # Kalman gain
            K = np.matmul(Pxy, np.linalg.inv(Pz))
            # state update (equation 5.40)
            innovation = y - Z
            for key in innovation_bound_func:
                innovation[key,:] = innovation_bound_func[key](innovation[key,:])
            X += np.matmul(K, innovation)
            # covariance update (equation 5.41)
            P -= np.matmul(K, Pxy.T)

        return X, P

    def unscented_transformH(self, x, W, WeightMat, L, f, u, ia, iq, n, additional_args):
        """
        Function to propagate sigma/cubature points through observation function.

        Args:
            x (numpy array [n_a x L]): sigma/cubature points
            W (numpy array [L x 1 or 1 x L]: 1D Weight array
            WeightMat (numpy array [L x L]): weight matrix with weights of the points on the diagonal
            L (int): number of points
            f (function): function handle for the observation model; expected signature f(state, input, noise, ...)
            u (?): current input required for function f
            ia (numpy array [n_s x 1]): row indices of the states in sima/cubature points
            iq (numpy array [n_q x 1]): row indices of the observation noise in sigma/cubature points
            n (int): dimensionality of output or return from function f
            additional_args (list): list of additional arguments to be passed to the observation model

        Returns:
            Y (numpy array [n x 1]): Expected value vector of the result from transformation function f
            y (numpy array [n x L]): Transformed sigma/cubature points
            P (numpy array [n x n]): Covariance matrix of the result from transformation function f
            y1 (numpy array [n x L]): zero-mean Transformed sigma/cubature points

        """
        Y = np.zeros((n, 1))
        y = np.zeros((n, L))
        # Propagating sigma/cubature points through function (equation 5.36)
        for k in range(L):
            y[:, k] = f(x[ia, k], u, x[iq, k], *additional_args)
            # Calculating mean (equation 5.37)
            Y += W.flat[k]*y[:, k:k+1]

        # Calculating covariance (equation 5.39)
        y1 = y - Y
        P = np.matmul(np.matmul(y1, WeightMat), y1.T)

        return Y, y, P, y1

    def unscented_transformF(self, x, W, WeightMat, L, f, u, ia, iq, iqu, additional_args):
        """
        Function to propagate sigma/cubature points through process model function.

        Args:
            x (numpy array [n_a x L]): sigma/cubature points
            W (numpy array [L x 1 or 1 x L]: 1D Weight array of the sigma/cubature points
            WeightMat (numpy array [L x L]): weight matrix with weights in W of the points on the diagonal
            L (int): number of points
            f (function): function handle for the process model; expected signature f(state, input, noise, ...)
            u (?): current input required for function f
            ia (numpy array [n_s x 1]): row indices of the states in sima/cubature points
            iq (numpy array [n_q x 1]): row indices of the process noise in sigma/cubature points
            iqu (numpy array [n_qu x 1]): row indices of the input noise in sigma/cubature points
            additional_args (list): list of additional arguments to be passed to the process model

        Returns:
            Y (numpy array [n_s x 1]): Expected value vector of the result from transformation function f
            y (numpy array [n_a x L]): Transformed sigma/cubature points
            P (numpy array [n_s x n_s]): Covariance matrix of the result from transformation function f
            y1 (numpy array [n_s x L]): zero-mean Transformed sigma/cubature points

        """
        order = len(ia)
        Y = np.zeros((order, 1))
        y = x
        # Propagating sigma/cubature points through function (equation 5.25)
        for k in range(L):
            if len(iqu):
                y[ia, k] = f(x[ia, k], u, x[iq, k],
                             x[iqu, k], *additional_args)
            else:
                y[ia, k] = f(x[ia, k], u, x[iq, k],
                             np.zeros(u.shape), *additional_args)
            # Calculating mean (equation 5.34)
            Y += W.flat[k]*y[np.arange(order), k:k+1]

        # Calculating covariance (equation 5.35)
        y1 = y[np.arange(order), :] - Y
        P = np.matmul(np.matmul(y1, WeightMat), y1.T)

        return Y, y, P, y1

    def sigmas2(self, X, P):
        """
        function to generate second order sigma points
        reference: Appendix G.1 of "Compressed Estimation in Coupled High-dimensional Processes"

        Args:
            X (numpy array [n x 1]): mean of Gaussian distribution
            P (numpy array [n x n]): covariance matrix of Gaussian distribution

        Returns:
            x (numpy array [n x L]): second order sigma point
            L (int): number of sigma points
            W (numpy array [1 x L]): 1D Weight array of sigma points
            WeightMat (numpy array [L x L]): weight matrix with weights in W of the points on the diagonal

        """
        n = X.shape[0]
        # some constants based on augmented dimentionality
        Params = [1 - n/3.0, 1.0/6.0, math.sqrt(3.0)]
        L = 2*n + 1
        W = np.concatenate(
            (np.array([[Params[0]]]), np.matlib.repmat(Params[1], 1, 2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix (step 3 of algorithm 5.1, equation 5.22)
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set (step 2 of algorithm 5.1)
        temp = np.zeros((n, L))
        loc = np.arange(n)
        l_index = loc*L + loc + 1
        temp.flat[l_index] = Params[2]
        l_index += n
        temp.flat[l_index] = -Params[2]

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, L)
        x = Y + np.matmul(sqP, temp)

        # for debugging
        # self.verifySigma(temp, W, 3)

        return x, L, W, WeightMat

    def sigmas4(self, X, P):
        """
        function to generate fourth order sigma points
        Note: No analytical results exist for generating 4th order sigma points as it requires performing
        non-linear least square (see Appendix G.2 of "Compressed Estimation in Coupled High-dimensional Processes".

        A separate scheme is used here, see equation 5.20 instead.

        Args:
            X (numpy array [n x 1]): mean of Gaussian distribution
            P (numpy array [n x n]): covariance matrix of Gaussian distribution

        Returns:
            x (numpy array [n x L]): fourth order sigma point
            L (int): number of sigma points
            W (numpy array [1 x L]): 1D Weight array of sigma points
            WeightMat (numpy array [L x L]): weight matrix with weights in W of the points on the diagonal

        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n**2 + 1
        W = np.concatenate((np.array([[1 + (n**2-7.0*n)/18.0]]), np.matlib.repmat(
            (4-n)/18.0, 1, 2*n), np.matlib.repmat(1.0/36.0, 1, 2*n**2-2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix (step 3 of algorithm 5.1, equation 5.22)
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set (step 2 of algorithm 5.1)
        s = math.sqrt(3.0)
        temp = np.zeros((n, 2*n+1))
        loc = np.arange(n)
        l_index = loc*(2*n+1) + loc + 1
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, 2*n+1)
        x = Y + np.matmul(sqP, temp)

        # create second type of sigma point: 2n**2 - 2n points based on (s2,s2) structure (step 2 of algorithm 5.1)
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

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, 2*n**2 - 2*n)
        x = np.concatenate((x, Y + np.matmul(sqP, temp1)), axis=1)

        # for debugging
        # temp = np.concatenate((temp, temp1), axis=1)
        # self.verifySigma(temp, W, 5)

        return x, L, W, WeightMat

    def cubature2(self, X, P):
        """
        function to generate second order cubature points
        reference: paper "Cubature Kalman Fitlers"

        Args:
            X (numpy array [n x 1]): mean of Gaussian distribution
            P (numpy array [n x n]): covariance matrix of Gaussian distribution

        Returns:
            x (numpy array [n x L]): second order cubature point
            L (int): number of cubature points
            W (numpy array [1 x L]): 1D Weight array of cubature points
            WeightMat (numpy array [L x L]): weight matrix with weights in W of the points on the diagonal

        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n
        W = np.matlib.repmat(1.0/L, 1, L)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix (step 3 of algorithm 5.1, equation 5.22)
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set (step 2 of algorithm 5.1)
        s = math.sqrt(n)
        temp = np.zeros((n, L))
        loc = np.arange(n)
        l_index = loc*L + loc
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, L)
        x = Y + np.matmul(sqP, temp)

        # for debugging
        # self.verifySigma(temp, W, 2)

        return x, L, W, WeightMat

    def cubature4(self, X, P):
        """
        function to generate fourth order cubature points
        reference: paper "High-degree cubature kalman filter"

        Args:
            X (numpy array [n x 1]): mean of Gaussian distribution
            P (numpy array [n x n]): covariance matrix of Gaussian distribution

        Returns:
            x (numpy array [n x L]): fourth order cubature point
            L (int): number of cubature points
            W (numpy array [1 x L]): 1D Weight array of cubature points
            WeightMat (numpy array [L x L]): weight matrix with weights in W of the points on the diagonal

        """
        n = X.shape[0]
        # some constants based on augmented dimensionality
        L = 2*n**2 + 1
        W = np.concatenate((np.array([[2.0/(n+2.0)]]), np.matlib.repmat((4-n)/(2.0*(
            n+2)**2), 1, 2*n), np.matlib.repmat(1.0/((n+2.0)**2), 1, 2*n**2-2*n)), axis=1)
        WeightMat = np.diag(np.squeeze(W))

        # first perform SVD to get the square root matrix (step 3 of algorithm 5.1, equation 5.22)
        U, D, _ = np.linalg.svd(P)
        sqP = np.matmul(U, np.diag(D**0.5))

        # create sigma point set (step 2 of algorithm 5.1)
        s = math.sqrt(n+2.0)
        temp = np.zeros((n, 2*n+1))
        loc = np.arange(n)
        l_index = loc*(2*n+1) + loc + 1
        temp.flat[l_index] = s
        l_index += n
        temp.flat[l_index] = -s

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, 2*n+1)
        x = Y + np.matmul(sqP, temp)

        # create second type of sigma point: 2n**2 - 2n points based on (s2,s2) structure (step 2 of algorithm 5.1)
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

        # step 4 of algorithm 5.1 equation 5.24
        Y = np.matlib.repmat(X, 1, 2*n**2 - 2*n)
        x = np.concatenate((x, Y + np.matmul(sqP, temp1)), axis=1)

        # for debugging
        # temp = np.concatenate((temp, temp1), axis=1)
        # self.verifySigma(temp, W, 5)

        return x, L, W, WeightMat

    def verifyTransformedSigma(self, x, WeightMat, X, P):
        """
        Verify if the transformed sigma/cubature point captures the mean and covariance of the 
        target Gaussian distribution

        Args:
            x (numpy array [n x L]): sigma/cubature points
            WeightMat (numpy array [L x L]): weight matrix with weights of the points on the diagonal
            X (numpy array [n x 1]): mean of Gaussian distribution
            P (numpy array [n x n]): covariance matrix of Gaussian distribution

        Returns:
            mean_close (bool): whether mean of the distibution is captured by the sigma/cubature points
            cov_close (bool): whether covariance of the distibution is captured by the sigma/cubature points

        """
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
        """
        Since originally the points of PBGF are generated from standard Gaussian distribution,
        check if moments up to specified order are being captured. Raises error when mismatch is found.

        Args:
            x (numpy array [n x L]): sigma/cubature points
            W (numpy array [1 x L or L x 1]): 1D Weight array of sigma/cubature points
            order (int): moment order in which the sampled points are generated from

        """
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
        Calculate order-th moment of univariate standard Gaussian distribution (zero mean, 1 std)

        Args:
            order (int): scalar moment order

        Returns:
            prod (int): requested order-th moment of standard Gaussian distribution

        """
        if order % 2:
            return 0.0
        else:
            prod = 1.0
            for i in range(1, order, 2):
                prod *= i

            return prod


def findCombinationsUtil(arr, index, num, reducedNum, output):
    """
    Find all combinations of < n numbers from 1 to num with repetition that add up to reducedNum 

    Args:
        arr (list size n): current items that add up to <= reducedNum (in the 0th recursion)
        index (int): index of the next slot of arr list
        num (int): limit of what numbers to be chosen from -> [1, num]
        reducedNum (int): remaining number to add up to required sum
        output (list): for appending the results to

    """

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


def fit_data_rover_dynobj(dynamic_obj, vy=np.array([]), back_rotate=False):
    """
    Perform LS and NLS fitting parameters estimation for the rover dynamics (c1-c9) using dynamic object.

    Args:
        dynamic_obj (RoverPartialDynEst or RoverDyn obj): dynamic object
        vy (numpy array [nt]): optionally, lateral velocity if observed; defaults to empty
        back_rotate (bool): produce linear and lateral velocities from rotating state coordinates? defaults to False

    Returns:
        parameters (list): consists of parameters c1-c9 in that order
    """

    parameters = [0]*9

    # check if we have access to longitudinal velocity
    if dynamic_obj.state_dict['vx'] not in dynamic_obj.state_indices or back_rotate:
        assert dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict[
            'y'] in dynamic_obj.state_indices, "No source for vehicle coordinates from output data"
        vx = dynamic_obj.cal_vxvy_from_coord(output=True)[0, :]
    else:
        vx = dynamic_obj.outputs[dynamic_obj.state_indices.index(
            dynamic_obj.state_dict['vx']), :]

    # first fit the longitudinal acceleration
    if dynamic_obj.state_dict['vx'] in dynamic_obj.state_dot_indices:
        vx_dot_ind = len(dynamic_obj.state_indices) + \
            dynamic_obj.state_dot_indices.index(dynamic_obj.state_dict['vx'])
        vxdot = dynamic_obj.outputs[vx_dot_ind, :]
    else:
        dts = np.diff(dynamic_obj.T)
        if len(vx) < len(dynamic_obj.T):
            dts = dts[:len(vx)-1]
        vxdot = np.diff(vx)/dts
    last_ind = min(len(vx), len(vxdot))

    diff = np.reshape(vx[:last_ind] -
                      dynamic_obj.U[1, :last_ind], [-1, 1])
    A_long_accel = np.concatenate(
        (np.ones((len(vxdot[:last_ind]), 1)), diff, np.square(diff)), axis=1)
    parameters[4:7] = np.linalg.lstsq(
        A_long_accel, vxdot[:last_ind, np.newaxis], rcond=None)[0][:, 0].tolist()

    # fitting for yaw rate
    if dynamic_obj.state_dict['theta'] in dynamic_obj.state_dot_indices:
        theta_dot_ind = len(dynamic_obj.state_indices) + \
            dynamic_obj.state_dot_indices.index(
                dynamic_obj.state_dict['theta'])
        thetadot = dynamic_obj.outputs[theta_dot_ind, :]
    else:
        theta_ind = dynamic_obj.state_indices.index(
            dynamic_obj.state_dict['theta'])
        thetadot = np.diff(
            dynamic_obj.outputs[theta_ind, :])/np.diff(dynamic_obj.T)
    last_ind = min(len(thetadot), len(vx))

    def nls_yawrate(x, yaw_rate, steering_cmd, vx):
        return yaw_rate - np.tan(x[0]*steering_cmd + x[1])*vx/(x[2] + x[3]*vx**2)

    x0 = np.array([1, 0, 1.775, 0])
    res_l = least_squares(nls_yawrate, x0, args=(
        thetadot[:last_ind], dynamic_obj.U[0, :last_ind], vx[:last_ind]))
    parameters[:4] = res_l.x

    # calculate lateral velocity
    if vy.shape[0] == 0 and back_rotate:
        assert dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict[
            'y'] in dynamic_obj.state_indices, "No source for vehicle coordinates from output data"
        vy = dynamic_obj.cal_vxvy_from_coord_wrapper(output=True)[1, :]

    # depending on availability of vy, perform NLS or LS
    if vy.shape[0] == 0:
        assert dynamic_obj.state_dict['x'] in dynamic_obj.state_indices and dynamic_obj.state_dict[
            'y'] in dynamic_obj.state_indices, "No source for vehicle coordinates from output data"

        x_ind = dynamic_obj.state_indices.index(dynamic_obj.state_dict['x'])
        y_ind = dynamic_obj.state_indices.index(dynamic_obj.state_dict['y'])
        theta_ind = dynamic_obj.state_indices.index(
            dynamic_obj.state_dict['theta'])

        xdot = np.diff(dynamic_obj.outputs[x_ind, :])/np.diff(dynamic_obj.T)
        ydot = np.diff(dynamic_obj.outputs[y_ind, :])/np.diff(dynamic_obj.T)
        theta = dynamic_obj.outputs[theta_ind, :]

        last_ind = min(len(xdot), len(vx), len(theta), len(thetadot))

        def nls_xy(params, xdot, ydot, vx, yaw, yaw_rate):
            vy = yaw_rate*(params[0] + params[1]*vx**2)
            res_x = xdot - (vx*np.cos(yaw) - vy*np.sin(yaw))
            res_y = ydot - (vx*np.sin(yaw) + vy*np.cos(yaw))
            return np.concatenate((res_x, res_y)).flatten()

        x0 = np.array([0.1, 0.1])
        res_l = least_squares(nls_xy, x0, args=(
            xdot[:last_ind], ydot[:last_ind], vx[:last_ind], theta[:last_ind], thetadot[:last_ind]))
        parameters[7:9] = res_l.x
    else:
        # LS fitting based on lateral velocity
        last_ind = min(len(thetadot), len(vx), len(vy))
        prod = thetadot[:last_ind]*(vx[:last_ind]**2)
        A_lat_vel = np.concatenate(
            (thetadot[:last_ind, np.newaxis], prod[:, np.newaxis]), axis=1)
        parameters[7:9] = np.linalg.lstsq(
            A_lat_vel, vy[:last_ind, np.newaxis], rcond=None)[0][:, 0].tolist()

    return parameters


def fit_data_rover(states, U, dt, vxdot=np.array([]), yawrate=np.array([]), vy=np.array([])):
    """
    Perform LS and NLS fitting parameters estimation for the rover dynamics (c1-c9).

    Args:
        states (numpy array [4 x nt]): rover states consisting of x, y, theta and vx at different time instances
        U (numpy array [2 x nt]): input to the model at different time instances consisting of steering angle and commanded velocity
        vxdot (numpy array [nt]): optionally, linear longitudinal acceleration at different time instances
        yawrate (numpy array [nt]): optionally, yaw rate at different time instances
        vy (numpy array [nt]): optionally, lateral velocity if observed

    Returns:
        parameters (list): consists of parameters c1-c9 in that order

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


def sample_nlds(z0, U, nt, f, h, num_out, Q=None, P0=None, R=None, Qu=None, additional_args_pm=[], additional_args_om=[], overwrite_inds=[], overwrite_vals=[]):
    """
    Retrieve ground truth, initial and output data (SNLDS: Stochastic non-linear dynamic system)

    Args:
        z0 (numpy array [n x 1]): initial ground truth condition
        U (numpy array [nu x nt]): inputs for the process and observation model
        nt (int): number of simulation steps
        f (function): function handle for one-time step forward propagating the state; expected signature f(state, input, noise, ...)
        h (function): function handle for retrieving the outputs of the system as a function of system states;
            expected signature h(state, input, noise, ...)
        num_out (int): number of outputs from function h
        Q (numpy array [nq x nq]): noise covariance matrix involved in the stochastic model f
        P0 (numpy array [n x n]): initial covariance for the initial estimate around the ground truth
        R (numpy array [nr x nr]): covariance matrix of the noise involved in h function
        Qu (numpy array [nqu x nqu]): noise covariance matrix involved in the input to the stochastic model f
        additional_args_pm (list): list of additional arguments to be passed to function f
        additional_args_om (list): list of additional arguments to be passed to function h
        overwrite_inds (list): list of state indices to be overwritten
        overwrite_vals (list): list of ground truth values to overwrite state propagation

    Returns:
        gt_states (numpy array [n x nt]): ground truth states at different time instances
        initial_cond (numpy array [n x 1]): initial condition from Gaussian distribution with mean z0 and covariance P0
        outputs (numpy array [num_out x nt]): simulated outputs of the system
        additional_args_pm_list (2d list [len(additional_args_pm) x nt]): additional arguments to be passed to 
            function f at each time instant
        additional_args_om_list (2d list [len(additional_args_om) x nt]): additional arguments to be passed to
            function h at each time instant

    """
    if not len(U):
        U = np.zeros((0, nt))
    if Q is None:
        Q = np.zeros((len(z0), len(z0)))
    if Qu is None:
        Qu = np.zeros((U.shape[0], U.shape[0]))
    if P0 is None:
        P0 = np.zeros((len(z0), len(z0)))
    if R is None:
        R = np.zeros((num_out, num_out))

    # check sizes of received matrices
    assert U.shape[1] == nt, "Expected input for all {} time instances but only received {}".format(
        nt, U.shape[1])
    assert Q.shape == (len(z0), len(
        z0)), "Inconsistent size of process noise matrix"
    assert Qu.shape == (U.shape[0], U.shape[0]
                        ), "Inconsistent size of input noise matrix"
    assert P0.shape == (len(z0), len(
        z0)), "Inconsistent size of initial covariance matrix"
    assert R.shape == (
        num_out, num_out), "Inconsistent size of observation noise matrix"

    # check the additional arguments
    additional_args_pm_list = np.zeros((len(additional_args_pm), nt)).tolist()
    additional_args_om_list = np.zeros((len(additional_args_om), nt)).tolist()
    for i, argument in enumerate(additional_args_pm):
        if not isinstance(argument, Iterable):
            additional_args_pm_list[i] = [argument] * nt
        else:
            assert len(
                argument) == nt, "If iterable argument for pm is provided, it should have the length of nt"
            additional_args_pm_list[i] = argument
    for i, argument in enumerate(additional_args_om):
        if not isinstance(argument, Iterable):
            additional_args_om_list[i] = [argument] * nt
        else:
            assert len(
                argument) == nt, "If iterable argument for om is provided, it should have the length of nt"
            additional_args_om_list[i] = argument

    # check the information to be overwritten
    assert len(overwrite_inds) == len(
        overwrite_vals), "Inconsitent sizes of information to be overwritten"
    for ind in overwrite_inds:
        assert ind >= 0 and ind < len(
            z0), "Overwrite index not within range [{},{})".format(0, len(z0))

    overwrite_vals_array = np.zeros((len(overwrite_inds), nt))
    for i, val in enumerate(overwrite_vals):
        if isinstance(val, Iterable):
            assert len(
                val) == nt, "Iterable information should have the length of nt"
        overwrite_vals_array[i] = val

    # generate noise samples for stochastic model and observations
    state_noise_samples = sample_gaussian(np.zeros(z0.shape), Q, nt)
    input_noise_samples = sample_gaussian(np.zeros((Qu.shape[0], 1)), Qu, nt)
    obs_noise_samples = sample_gaussian(
        np.zeros((num_out, 1)), R, nt)

    # initialise matrices to return
    gt_states = np.zeros((z0.shape[0], nt))
    gt_states[:, 0:1] = z0
    initial_cond = sample_gaussian(z0, P0, 1)
    outputs = np.zeros((num_out, nt))
    outputs[:, 0] = h(gt_states[:, 0], U[:, 0],
                      obs_noise_samples[:, 0], *[sub[0] for sub in additional_args_om_list])

    for i in range(1, nt):
        gt_states[:, i] = f(gt_states[:, i-1], U[:, i-1], state_noise_samples[:, i-1],
                            input_noise_samples[:, i-1], *[sub[i-1] for sub in additional_args_pm_list])

        # overwrite information as per user requirement
        gt_states[overwrite_inds, i] = overwrite_vals_array[:, i]

        outputs[:, i] = h(gt_states[:, i], U[:, i],
                          obs_noise_samples[:, i], *[sub[i] for sub in additional_args_om_list])

    return gt_states, initial_cond, outputs, additional_args_pm_list, additional_args_om_list


def test_pbgf_linear(n=10, m=5, nt=10):
    """
    Test the PointBasedFilter against KF when problem is linear. Raises error when mean and covariance from
    PBGF differs from that of KF.

    Args:
        n (int): dimensionality of problem; defaults to 10
        m (int): number of outputs which are randomly selected from the states; defaults to 5
        nt (int): number of filtering iterations; defaults to 10

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

    def process_model(x, u, noise, input_noise): return np.matmul(J, x) + noise
    Q = 5.0*np.eye(n)
    out_loc = np.random.permutation(n)[:m]
    R = 1.0*np.eye(m)
    H = np.zeros((m, n))
    l_ind = out_loc + np.arange(m)*n
    H.flat[l_ind] = 1.0
    def observation_model(x, u, noise): return np.matmul(H, x) + noise

    ## generate the output of the real time system
    x_gt, x0, outputs = sample_nlds(
        X, [], nt, process_model, observation_model, m, Q, P, R)[0:3]

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
        X1 = process_model(X1, [], 0.0, 0.0)
        P1 = np.matmul(np.matmul(J, P1), J.T) + Q

        # update step
        z = outputs[:, i:i+1] - observation_model(X1, [], 0.0)
        S = np.matmul(np.matmul(H, P1), H.T) + R
        K = np.matmul(np.matmul(P1, H.T), np.linalg.inv(S))
        X1 += np.matmul(K, z)
        P1 -= np.matmul(np.matmul(K, H), P1)

        ## PBGF code
        X2, P2 = pbgf.predict_and_or_update(
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


def test_pbgf_1d_linear(gt_const=10.0, initial_cov=10.0, q_cov=1e-2, r_cov=1.0, nt=50):
    """
    Test the PBGF against KF when problem is linear. This problem is one-dimensional estimate of a random constant.

    Args:
        gt_const (float): parameter to be estimated; defaults to 10.0
        initial_cov (float): initial uncertainty of gt_const; defaults to 10.0
        q_cov (float): stochastic noise for evolution of the parameter; defaults to 1e-2, value of 0 implies parameter is constant
        r_cov (float): observation noise of the parameter; defaults to 1.0
        nt (int): number of filtering iterations; defaults to 50

    """
    # control random seed generator
    np.random.seed(0)

    # set up the true initial condition
    X = np.array([[gt_const]])
    P = initial_cov*np.ones((1, 1))

    # process and observation model
    def process_model(x, u=[], noise=0.0, input_noise=0.0): return x + noise
    def observation_model(x, u=[], noise=0.0): return x + noise

    # process and observation noises
    R = np.array([[r_cov]])
    Q = np.array([[q_cov]])

    # generate the initial condition
    x_gt, x0, outputs = sample_nlds(
        X, [], nt, process_model, observation_model, 1, Q, P, R)[0:3]

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
        X2, P2 = pbgf.predict_and_or_update(
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
    plt.plot(x_gt[0, :], linestyle='--', label='real_voltage')
    plt.plot(mse, marker='x', label='mse')
    plt.plot(trace, marker='o', label='trace')
    plt.grid(True, "both")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_pbgf_linear()
    test_pbgf_1d_linear(q_cov=1e-2)
