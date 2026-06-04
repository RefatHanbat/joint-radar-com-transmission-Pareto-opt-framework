import numpy as np

from fParam import *

from fChannel import *

from fAlgorithm_Impairment import *

from fBeampattern import *

from fPlot import *


def main():

    sys_param = myf_sys_param()

    results_fig7 = {}

    N = sys_param["N_fig7"]

    K = sys_param["K_fig7"]

    sys_param["Gamma"] = sys_param["Gamma_fig7"]

    kappa_cand = sys_param["kappa_fig7_cand"]

    fixed_channel = myf_fixed_channel_fig3()

    H = fixed_channel[N][0:K, :]

    print(f"\nRunning Fig. 7 with N = {N}, K = {K}")

    for kappa in kappa_cand:

        print(f"  Solving kappa = {kappa}")

        sol_impairment = myf_solve_joint_impairment_P11(
            sys_param,
            H,
            K,
            N,
            kappa
        )

        if sol_impairment["Q"] is None:

            raise RuntimeError(
                f"Impairment optimization failed for kappa = {kappa}: "
                f"{sol_impairment['status']}"
            )

        # Important:
        # use Q = R + D for impaired beampattern
        theta_plot, pattern_dB = myf_compute_beampattern(
            sys_param,
            sol_impairment["Q"],
            N
        )

        results_fig7[kappa] = {

            "solution": sol_impairment,

            "theta": theta_plot,

            "pattern_dB": pattern_dB
        }

    myf_plot_fig7_impairment(results_fig7)

    return results_fig7


if __name__ == "__main__":

    results_fig7 = main()