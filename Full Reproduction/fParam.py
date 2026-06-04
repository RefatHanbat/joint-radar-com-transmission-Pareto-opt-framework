import numpy as np

'''
fParam.py

'''

def myf_dbm2watt(x_dBm):

    return 10 ** ((x_dBm - 30) / 10)


def myf_db2lin(x_dB):

    return 10 ** (x_dB / 10)


def myf_lin2db(x):

    return 10 * np.log10(np.maximum(x, 1e-18))


def myf_sys_param():

    sys_param = {}

    ###### Power ########

    sys_param["P0_dBm"] = 20

    sys_param["P0"] = myf_dbm2watt(sys_param["P0_dBm"])

    sys_param["sigma2_dBm"] = 0

    sys_param["sigma2"] = myf_dbm2watt(sys_param["sigma2_dBm"])

    ####### Radar / array ###############

    sys_param["d"] = 0.5

    sys_param["theta0"] = 0

    sys_param["beam_width_3dB"] = 10

    sys_param["theta_a"] = (

        sys_param["theta0"] - sys_param["beam_width_3dB"] / 2
    )

    sys_param["theta_b"] = (

        sys_param["theta0"] + sys_param["beam_width_3dB"] / 2
    )

    ##### Fig. 3 candidate   ####

    sys_param["N_fig3_cand"] = [12, 16]

    sys_param["K_cand"] = [1, 2, 3]

    sys_param["Gamma_dB"] = 20
    
    sys_param["Gamma"] = myf_db2lin(sys_param["Gamma_dB"])

    ########## Angle grids #############

    sys_param["theta_grid"] = np.linspace(-90, 90, 181)

    sys_param["theta_grid_plot"] = np.linspace(-90, 90, 721)

    ######## Sidelobe region #############

    sys_param["theta_sidelobe_exclusion"] = 10

    sys_param["theta_sidelobe"] = sys_param["theta_grid"][

        np.abs(sys_param["theta_grid"] - sys_param["theta0"])

        > sys_param["theta_sidelobe_exclusion"]
    ]

    ########### Solver #####################

    sys_param["solver_name"] = "MOSEK"

    # sys_param["solver_max_iter"] = 3000

    sys_param["solver_verbose"] = False

    sys_param["solver_accept_status"] = ["optimal", "optimal_inaccurate"]



    #### Figure 04 candidate ####

    sys_param["N_fig4_cand"] = np.arange(10, 21, 2)

    sys_param["Num_MC_fig4"] = 30

    sys_param["Gamma_fig4_dB"] = 20

    sys_param["Gamma_fig4"] = myf_db2lin(sys_param["Gamma_fig4_dB"])


    #### Paper [22] waveform-similarity benchmark ####

    sys_param["L_waveform"] = 256

    sys_param["rho_waveform"] = 0.05

    sys_param["lambda_bisect_iter"] = 60


    #### Fig. 5 candidate ####

    sys_param["N_fig5"] = 16

    sys_param["Gamma_fig5_dB_cand"] = np.arange(10, 21, 2)

    sys_param["Num_MC_fig5"] = 100


    #### Fig. 6 candidate ####

    sys_param["N_fig6"] = 16

    sys_param["Gamma_fig6_low_dB"] = 15

    sys_param["Gamma_fig6_high_dB"] = 35

    sys_param["Lambda_fig6_dB"] = 8

    sys_param["Lambda_fig6"] = sys_param["P0"] * myf_db2lin(sys_param["Lambda_fig6_dB"])

    sys_param["Num_iter_fig6"] = 12

    sys_param["seed_fig6"] = 7

    #### Fig. 7 candidate ####

    sys_param["N_fig7"] = 16

    sys_param["K_fig7"] = 3

    sys_param["Gamma_fig7_dB"] = 20

    sys_param["Gamma_fig7"] = myf_db2lin(sys_param["Gamma_fig7_dB"])

    sys_param["kappa_fig7_cand"] = [0, 0.03, 0.06, 0.09]


    #### Fig. 2 candidate ####

    sys_param["N_fig2"] = 16

    sys_param["K_fig2"] = 1

    sys_param["Gamma_fig2_dB_cand"] = np.arange(0, 36, 2)

    return sys_param
