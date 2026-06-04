import numpy as np

from fParam import *
from fChannel import *
from fAlgorihtm_joint import *
from fPlot import *


def main():

    sys_param = myf_sys_param()

    results_fig2 = {}

    N = sys_param["N_fig2"]

    K = sys_param["K_fig2"]

    Gamma_dB_cand = sys_param["Gamma_fig2_dB_cand"]

    # ==================================================
    # Use one fixed channel realization
    # ==================================================

    fixed_channel = myf_fixed_channel_fig3()

    H = fixed_channel[N][0:K, :]

    eta_list = []

    feasible_list = []

    print(f"\nRunning Fig. 2 Pareto-boundary illustration")
    
    print(f"N = {N}, K = {K}")

    for Gamma_dB in Gamma_dB_cand:

        print(f"  Solving Gamma = {Gamma_dB} dB")

        sys_param["Gamma"] = myf_db2lin(Gamma_dB)

        try:

            sol_joint = myf_solve_joint_P11(
                sys_param,
                H,
                K,
                N
            )

        except Exception as e:

            print(f"    Solver crashed at Gamma = {Gamma_dB} dB")
            print("    Error:", e)

            eta_list.append(np.nan)

            feasible_list.append(False)

            continue

        if sol_joint["status"] in sys_param["solver_accept_status"]:

            eta_list.append(sol_joint["eta"])

            feasible_list.append(True)

            eta_dB = myf_lin2db(sol_joint["eta"] / sys_param["P0"])

            print(f"    feasible, eta = {eta_dB:.3f} dB")

        else:

            eta_list.append(np.nan)

            feasible_list.append(False)

            print(f"    infeasible, status = {sol_joint['status']}")

    eta_arr = np.array(eta_list)

    feasible_arr = np.array(feasible_list)

    eta_dB_arr = np.full_like(eta_arr, np.nan, dtype=float)

    valid_idx = np.logical_and(feasible_arr, eta_arr > 0)

    eta_dB_arr[valid_idx] = myf_lin2db(
        eta_arr[valid_idx] / sys_param["P0"]
    )

    results_fig2["Gamma_dB"] = Gamma_dB_cand

    results_fig2["eta"] = eta_arr

    results_fig2["eta_dB"] = eta_dB_arr

    results_fig2["feasible"] = feasible_arr

    myf_plot_fig2_pareto_boundary(results_fig2)

    return results_fig2


if __name__ == "__main__":

    results_fig2 = main()