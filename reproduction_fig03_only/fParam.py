import numpy as np

"""
fParam.py

Chen et al.,
"Joint Radar-Communication Transmission:
A Generalized Pareto Optimization Framework"
"""


# ============================================================
# Unit conversion functions
# ============================================================

def myf_dbm2watt(x_dBm):
    x_watt = 10 ** ((x_dBm - 30) / 10)

    return x_watt


def myf_watt2dbm(x_watt):

    x_dBm = 10 * np.log10(np.maximum(x_watt, 1e-18)) + 30

    return x_dBm


def myf_db2lin(x_dB):

    x = 10 ** (x_dB / 10)

    return x


def myf_lin2db(x):

    x_dB = 10 * np.log10(np.maximum(x, 1e-18))

    return x_dB


### system parameters 

def myf_sys_param():

    sys_param = {}


    sys_param["P0_dBm"] = 20

    sys_param["P0"] = myf_dbm2watt(sys_param["P0_dBm"])

    sys_param["sigma2_dBm"] = 0

    sys_param["sigma2"] = myf_dbm2watt(sys_param["sigma2_dBm"])

    sys_param["P0_over_sigma2_dB"] = (
        sys_param["P0_dBm"] - sys_param["sigma2_dBm"]
    )

    ###### Radar / array parameters
   

    sys_param["d"] = 0.5

    sys_param["theta0"] = 0

    sys_param["beamwidth_3dB"] = 10

    sys_param["theta_a"] = (
        sys_param["theta0"] - sys_param["beamwidth_3dB"] / 2
    )

    sys_param["theta_b"] = (
        sys_param["theta0"] + sys_param["beamwidth_3dB"] / 2
    )

    sys_param["theta_grid"] = np.linspace(-90, 90, 181)

    sys_param["theta_grid_plot"] = np.linspace(-90, 90, 721)

    sys_param["theta_sidelobe_exclusion"] = 10

    sys_param["theta_sidelobe"] = sys_param["theta_grid"][

        np.abs(sys_param["theta_grid"] - sys_param["theta0"])

        > sys_param["theta_sidelobe_exclusion"]
    ]

    
    ####### Solver parameters #######
   

    sys_param["solver_name"] = "CLARABEL"

    sys_param["solver_max_iter"] = 3000

    sys_param["solver_verbose"] = False

    sys_param["solver_accept_status"] = [
        "optimal",
        "optimal_inaccurate",
    ]

    ######### Figure 03 parameters ##########
   

    sys_param["N_fig3_cand"] = [12, 16]

    sys_param["K_cand"] = [1, 2, 3]

    sys_param["Gamma_fig3_dB"] = 20

    sys_param["Gamma_fig3"] = myf_db2lin(sys_param["Gamma_fig3_dB"])

    return sys_param


def myf_fixed_channel_fig3():

    fixed_channel = {}

    h1_12 = np.array([

        -0.21 - 0.75j,

        0.58 - 0.021j,

        -0.15 - 0.06j,

        -0.73 + 0.029j,

        0.096 + 0.16j,

        -0.67 - 0.17j,

        -0.38 + 1.6j,

        -0.34 - 1.2j,

        -0.14 + 0.3j,

        -0.18 - 0.86j,

        0.87 + 0.23j,

        -0.31 - 0.46j,

    ], dtype=complex)

    h2_12 = np.array([

        0.016 + 0.66j,

        1.1 + 0.13j,

        0.44 + 1.1j,

        0.67 - 0.52j,

        0.36 + 0.3j,

        -0.11 + 1.4j,

        1.2 + 0.24j,

        -0.5 - 0.42j,

        -0.19 - 1.2j,

        -0.75 + 0.047j,

        -0.16 + 0.77j,

        -0.11 + 0.18j,

    ], dtype=complex)

    h3_12 = np.array([

        0.036 + 0.25j,

        0.33 - 1.1j,

        0.13 + 0.07j,

        0.22 - 0.022j,

        0.18 - 0.26j,

        -0.1 - 1.6j,

        -0.62 + 0.71j,

        -0.83 - 0.2j,

        1.1 + 0.33j,

        1.1 + 0.46j,

        -1.1 + 0.71j,

        0.2 - 0.67j,

    ], dtype=complex)


    h1_16 = np.concatenate((
        h1_12,
        np.array([
            -0.18 - 0.93j,
            -0.88 - 0.039j,
            -0.36 + 0.25j,
            -2.1 + 0.17j,
        ], dtype=complex)
    ))

    h2_16 = np.concatenate((
        h2_12,
        np.array([
            0.31 + 0.65j,
            -0.67 + 0.64j,
            -0.23 + 0.88j,
            -0.32 - 0.49j,
        ], dtype=complex)
    ))

    h3_16 = np.concatenate((
        h3_12,
        np.array([
            0.28 + 1.0j,
            -0.52 + 0.42j,
            0.0 + 0.66j,
            0.88 - 0.46j,
        ], dtype=complex)
    ))

    fixed_channel[12] = np.array([
        h1_12,
        h2_12,
        h3_12,
    ], dtype=complex)

    fixed_channel[16] = np.array([
        h1_16,
        h2_16,
        h3_16,
    ], dtype=complex)

    return fixed_channel
