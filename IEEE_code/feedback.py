import numpy as np

"""
This script predicts the number of sessions a user in week t and with sessions x will have given that they are
recommended app a.
"""
params = {'alpha': np.array([-53.92, 10.7, -23.18, -147.26, -76.28, 130.16, -92.76, -125.61, -47.87, -131.76,
                             223.46, -104.4]),
          'beta': np.array([0.35, -0.81, 0.16, 0.72, 0.32, -0.54, 0.49, 0.59, 0.24, 0.7, -1.24, 0.5]),
          'gamma': np.array(
              [119.97, 332.21, 146.85, 210.29, 216.65, 835.54, 225.47, 212.72, 209.58, 222.50, 1353.36, 119.98])}


def feedback_mean(t, x, a):
    """
    predict Poisson mean of the feedback at time t, state x and taking action a

    :param t: time(range 1-16)
    :param x: total state of 1000 users(nonnegative integer)
    :param a: action(range 0-11)
    :return the Poisson mean of the feedback of one user
    """
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']

    new_x = x * 1000
    f_mean = alpha[a] * t + beta[a] * new_x + gamma[a] if alpha[a] * t + beta[a] * new_x + gamma[a] > 0 else 0.001
    f_mean = f_mean / 1000
    return f_mean


def feedback(t, x, a):
    f_mean = feedback_mean(t, x, a)
    f = np.random.poisson(f_mean)
    return f
