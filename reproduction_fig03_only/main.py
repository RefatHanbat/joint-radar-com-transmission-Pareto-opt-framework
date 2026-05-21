import os

import argparse

import numpy as np

from fParam import *

from fChannel import *

from fAlgorithms import *

from fCalculations import *

from fPlot import *

'''
main.py
'''


def myf_run_fig3(sys_param, fixed_channel, output_path):

    ### Functions ###

    print("Run Fig. 3")

    fig3_result = []


    for ind1 in range(0, len(sys_param["N_fig3_cand"])):

        N = sys_param["N_fig3_cand"][ind1]

        result_temp = {}

        result_temp["N"] = N

        result_temp["theta_grid_plot"] = np.copy(sys_param["theta_grid_plot"])

        result_temp["P_dBi"] = []

        param_channel = {}

        param_channel["N"] = N

        param_channel["K"] = 0

        channel = {}

        channel["H"] = np.zeros((0, N), dtype=complex)

        print(f"channel : {channel}")

        param_algorithm = {}

        param_algorithm["N"] = N

        param_algorithm["K"] = 0


        param_algorithm["Gamma"] = 0
        
        param_algorithm["kappa"] = 0

        solutions = myf_algorithm_Pareto(sys_param, channel, param_algorithm)

        if (solutions["Q"] is None):

            raise RuntimeError("Radar-only optimization failed for N = %d, status = %s"%(N, solutions["status"]))
        
        result_temp["P_dBi"].append(myf_beampattern_dBi(sys_param, solutions["Q"], sys_param["theta_grid_plot"]))

        print("  N = %d, Radar-only DPSL = %.2f dB"%(N, myf_DPSL_dB(sys_param, solutions["Q"])))

        for K in sys_param["K_cand"]:

            param_channel["K"] = K

            channel = myf_channel_fig3(fixed_channel, param_channel)

            param_algorithm["K"] = K

            param_algorithm["Gamma"] = myf_db2lin(sys_param["Gamma_fig3_dB"])

            solutions = myf_algorithm_Pareto(sys_param, channel, param_algorithm)

            if (solutions["Q"] is None):

                raise RuntimeError("Joint optimization failed for N = %d, K = %d, status = %s"%(N, K, solutions["status"]))
            
            result_temp["P_dBi"].append(myf_beampattern_dBi(sys_param, solutions["Q"], sys_param["theta_grid_plot"]))

            print("  N = %d, K = %d, DPSL = %.2f dB, SINR = "%(N, K, myf_DPSL_dB(sys_param, solutions["Q"])),
                  
                  np.round(myf_SINR_dB(sys_param, channel, solutions), 2))

        fig3_result.append(result_temp)

    myf_plot_fig3(sys_param, fig3_result, output_path)


def myf_parse_args():

    ### Functions ###

    parser = argparse.ArgumentParser()

    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")

    parser.add_argument("--max-iter", type=int, default=3000)

    parser.add_argument("--output", default="figures/paper_fig3.png")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    return args


### System parameters ###

args = myf_parse_args()

sys_param = myf_sys_param()

sys_param["solver_name"] = args.solver

sys_param["solver_max_iter"] = args.max_iter

sys_param["solver_verbose"] = args.verbose

fixed_channel = myf_fixed_channel_fig3()


output_dir = os.path.dirname(args.output)

if (output_dir != "") and not os.path.exists(output_dir):

    os.makedirs(output_dir)


### Function ###

myf_run_fig3(sys_param, fixed_channel, args.output)

print("Finished")
