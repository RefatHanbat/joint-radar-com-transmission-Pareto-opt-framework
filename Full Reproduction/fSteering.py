import numpy as np

'''
fSteering.py

'''
def myf_steering_vector(sys_param, theta_deg, N):

    d = sys_param["d"]

    theta_rad = np.deg2rad(theta_deg)

    a_theta = np.exp(
        1j * 2 * np.pi * d * np.arange(N) * np.sin(theta_rad)
    ).reshape(-1, 1)

    return a_theta


def myf_outer_a(sys_param, theta_deg, N):

    a_theta = myf_steering_vector(sys_param, theta_deg, N)

    A_theta = a_theta @ a_theta.conj().T

    return A_theta