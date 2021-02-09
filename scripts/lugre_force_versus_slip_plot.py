import math

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    """
    This is to mimick Figure 3 in the paper titled 'Dynamic Tire Friction Models for Vehicle Traction Control' and 
    therefore see that the parameters are correct
    """
    # parameters used in the model
    sigma_0 = 40.0
    sigma_1 = 4.9487
    sigma_2 = 0.0018
    mu_c = 0.5
    mu_s = 0.9
    vs = 12.5
    L = 0.25

    # slip range, constant velocity and road condition coefficient
    v = 20.0
    s = np.linspace(0.0, 1.0, 1000)
    thetas = [1.0, 1.0/0.8, 1.0/0.6, 1/0.4] 
    norm_force = np.zeros((len(thetas), len(s)))

    for i in range(len(s)):
        for j, theta in enumerate(thetas):
            # intermediate calculations (for during braking only)
            vr = s[i]*v
            gs = (1.0/theta)*(mu_c + (mu_s - mu_c)*math.exp(-math.sqrt(math.fabs(vr/vs))))
            gamma = 1.0 - sigma_1*math.fabs(vr)/gs
            if s[i] != 0.0:
                p = gs/(sigma_0*L*math.fabs(s[i]))

                # calculate current normal force
                norm_force[j, i] = math.copysign(1.0, vr)*gs*L*(1.0 + gamma*p*(math.exp(-1.0/p) - 1.0)) + sigma_2*vr

    # plot the curves
    for j, theta in enumerate(thetas):
        plt.plot(s, norm_force[j, :], label="1/theta = {}".format(1.0/theta))

    plt.xlabel('s [slip rate]')
    plt.ylabel('normalised friction')
    plt.title('Static view of the distributed Lugre friction model')
    plt.legend()
    
    plt.show()
