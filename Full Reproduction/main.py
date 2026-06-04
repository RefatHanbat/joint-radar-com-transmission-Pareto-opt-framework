from fParam import *

from fChannel import *

from fAlgorithm_RadarOnly import *

from fBeampattern import *

from fPlot import *

from fAlgorihtm_joint import * 

'''
Main.py 

author : Refat Khan

Date : 26/05/2026

'''

def main():

    sys_param = myf_sys_param()

    fixed_channel = myf_fixed_channel_fig3()

    results = {}

    for N in sys_param["N_fig3_cand"]:

        print(f"\n Solving N = {N}")

        results[N] = {}

        ## Radar only Case : P22 ###

        print("Radar only")

        sol_radar = myf_solve_radar_only_P22(sys_param, N)

        if sol_radar["S0"] is None:

            raise RuntimeError(f"Radar-only optimization failed for N = {N}: {sol_radar['status']}")

        theta_plot, pattern_dB = myf_compute_beampattern(sys_param, sol_radar["S0"], N)

        results[N]["Radar-only"] = {

            "solution": sol_radar,

            "theta" : theta_plot,

            "pattern_dB" : pattern_dB
        }

        ### Joint K = 1 , 2 , 3 ###

        for K in sys_param["K_cand"]:

            print(f"Joint-K = {K}")

            H = fixed_channel[N][0: K, :]

            # print(f"H : {H.shape}")

            sol_joint = myf_solve_joint_P11(sys_param, H, K, N)

            theta_plot, pattern_dB = myf_compute_beampattern(sys_param, sol_joint["S_sum"], N)

            results[N][f"Joint-K = {K}"] = {

                "solution" : sol_joint,

                "theta" : theta_plot,

                "pattern_dB" : pattern_dB
            }

    myf_plot_fig3(results)

    return results

        



if __name__ == "__main__":

    result = main()
