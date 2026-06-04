import numpy as np

from fParam import *

from fChannel import *

from fAlgorithm_Fig6 import *

from fPlot import *


def main():

    sys_param = myf_sys_param()

    N = sys_param["N_fig6"]

    results_fig6 = {}

    # ==================================================
    # Use one common channel realization for all K
    # ==================================================
    # Option 1: use fixed channel from paper Fig. 3
    # This makes K=1,2,3 comparable.
    # ==================================================

    fixed_channel = myf_fixed_channel_fig3()

    H_all = fixed_channel[N][0:3, :]

    # ==================================================
    # Alternative option:
    # If you want random channel, use this instead.
    #
    # np.random.seed(sys_param["seed_fig6"])
    # H_all = myf_generate_rayleigh_channel(3, N)
    # ==================================================

    for K in sys_param["K_cand"]:

        print(f"\nRunning Fig. 6 for K = {K}")

        H = H_all[0:K, :]

        result_trace = myf_bisection_sinr_trace_fig6(
            sys_param,
            H,
            K,
            N
        )

        results_fig6[K] = result_trace

    myf_plot_fig6_convergence(results_fig6)

    return results_fig6


if __name__ == "__main__":

    results_fig6 = main()