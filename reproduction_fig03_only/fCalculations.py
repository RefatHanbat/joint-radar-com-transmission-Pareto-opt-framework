import numpy as np
from fParam import *

'''
fCalculations.py
'''


def myf_steering_vector(sys_param, theta_deg, N):

    ### Parameters ###

    d = sys_param["d"]


    ### Functions ###

    theta_rad = np.deg2rad(theta_deg)
    a_vec = np.exp(1j*2*np.pi*d*np.arange(N)*np.sin(theta_rad)).reshape(-1, 1)

    return a_vec


def myf_beampattern(sys_param, R_mat, theta_grid):

    ### Parameters ###

    N = np.size(R_mat, axis=0)


    ### Functions ###

    P_vec = np.zeros(np.size(theta_grid), dtype=float)
    for ind in range(0, np.size(theta_grid)):
        a_vec = myf_steering_vector(sys_param, theta_grid[ind], N)
        P_vec[ind] = np.real(a_vec.conj().T@R_mat@a_vec).item()

    P_vec = np.maximum(P_vec, 10**(-12))

    return P_vec


def myf_beampattern_dBi(sys_param, R_mat, theta_grid):

    ### Parameters ###

    P0 = sys_param["P0"]


    ### Functions ###

    P_vec = myf_beampattern(sys_param, R_mat, theta_grid)
    P_dBi = myf_lin2db(P_vec/P0)

    return P_dBi


def myf_DPSL(sys_param, R_mat):

    ### Parameters ###

    theta0 = sys_param["theta0"]
    theta_sidelobe = sys_param["theta_sidelobe"]
    N = np.size(R_mat, axis=0)


    ### Functions ###

    a_0 = myf_steering_vector(sys_param, theta0, N)
    P_peak = np.real(a_0.conj().T@R_mat@a_0).item()

    DPSL = 10**12
    for ind in range(0, np.size(theta_sidelobe)):
        a_m = myf_steering_vector(sys_param, theta_sidelobe[ind], N)
        P_side = np.real(a_m.conj().T@R_mat@a_m).item()
        DPSL = min(DPSL, P_peak - P_side)

    DPSL = max(DPSL, 10**(-12))

    return DPSL


def myf_DPSL_dB(sys_param, R_mat):

    ### Functions ###

    DPSL_dB = myf_lin2db(myf_DPSL(sys_param, R_mat))

    return DPSL_dB


def myf_SINR_dB(sys_param, channel, solutions):

    ### Parameters ###

    H = channel["H"]
    S = solutions["S"]
    sigma2 = sys_param["sigma2"]
    K = np.size(H, axis=0)


    ### Functions ###

    SINR_dB = np.zeros(K)
    for ind1 in range(0, K):
        h_vec = H[ind1, :].reshape(-1, 1)
        signal = np.real(h_vec.conj().T@S[ind1]@h_vec).item()
        interference = 0
        for ind2 in range(0, K):
            if (ind2 != ind1):
                interference = interference + np.real(h_vec.conj().T@S[ind2]@h_vec).item()
        SINR_dB[ind1] = myf_lin2db(signal/(interference + sigma2))

    return SINR_dB
