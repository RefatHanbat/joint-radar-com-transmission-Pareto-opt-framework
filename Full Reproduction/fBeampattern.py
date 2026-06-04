import numpy as np

from fSteering import *

from fParam import *

from fConstraints import *


def myf_compute_beampattern(sys_param, S_total, N):


    theta_grid_plot = sys_param["theta_grid_plot"]

    P_theta = np.zeros(len(theta_grid_plot))

    total_power = np.real(np.trace(S_total))

    for idx, theta in enumerate(theta_grid_plot):

        a_theta = myf_steering_vector(sys_param, theta, N)

        P_theta[idx] = np.real(

            a_theta.conj().T @ S_total @ a_theta

        ).item()


    P_theta_gain = P_theta / total_power

    P_theta_dB = myf_lin2db(P_theta_gain)


    return theta_grid_plot, P_theta_dB


def myf_compute_min_DPSL_numeric(sys_param, S_total, N):

    theta_sidelobe = sys_param["theta_sidelobe"]

    eta_list = []

    for theta_m in theta_sidelobe:

        _, _, C_m = myf_C_matrices(sys_param, theta_m, N)

        eta_m = np.real(np.trace(C_m @ S_total))

        eta_list.append(eta_m)

    eta_min = np.min(np.array(eta_list))

    return eta_min