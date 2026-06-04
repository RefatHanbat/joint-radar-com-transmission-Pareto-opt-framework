import numpy as np

from fParam import *

from fChannel import *

from fAlgorihtm_joint import *

from fAlgorithm_Similarity import *

from fAlgorithm_RadarOnly import *

from fBeampattern import *

from fAlgorithm_WaveformP22 import *

from fPlot import *


def main():

    sys_param = myf_sys_param()

    results_fig4 = {}

    Num_MC = sys_param["Num_MC_fig4"]

    N_cand = sys_param["N_fig4_cand"]

    K_cand = sys_param["K_cand"]

    # ==================================================
    # Radar-only benchmark curve
    # ==================================================

    results_fig4["Radar-only"] = {}

    eta_radar_only_list = []

    S0_ref_dict = {}

    for N in N_cand:

        print(f"\nSolving Radar-only benchmark for N = {N}")

        sol_radar = myf_solve_radar_only_P22(sys_param, N)

        if sol_radar["S0"] is None:

            raise RuntimeError(
                f"Radar-only optimization failed for N = {N}: "
                f"{sol_radar['status']}"
            )

        S0_ref = sol_radar["S0"]

        S0_ref_dict[N] = S0_ref

        eta_radar_only = myf_compute_min_DPSL_numeric(
            sys_param,
            S0_ref,
            N
        )

        eta_radar_only_list.append(eta_radar_only)

    eta_radar_only_arr = np.array(eta_radar_only_list)

    results_fig4["Radar-only"]["N"] = N_cand

    results_fig4["Radar-only"]["eta_avg"] = eta_radar_only_arr

    results_fig4["Radar-only"]["eta_avg_dB"] = myf_lin2db(
        eta_radar_only_arr / sys_param["P0"]
    )

    # ==================================================
    # Joint schemes: Proposed, P21, P22
    # ==================================================

    for K in K_cand:

        print(f"\nRunning Fig. 4 for K = {K}")

        results_fig4[K] = {}

        eta_proposed_avg_list = []

        eta_similarity_avg_list = []

        eta_waveform22_avg_list = []

        for N in N_cand:

            print(f"  Solving N = {N}")

            eta_proposed_list = []

            eta_similarity_list = []

            eta_waveform22_list = []

            # Use radar-only covariance as reference
            S0_ref = S0_ref_dict[N]

            for mc in range(Num_MC):

                H = myf_generate_rayleigh_channel(K, N)

                # ==================================================
                # Proposed Pareto method
                # ==================================================

                try:

                    sol_joint = myf_solve_joint_P11(sys_param, H, K, N)

                    if sol_joint["status"] in sys_param["solver_accept_status"]:

                        eta_proposed_list.append(sol_joint["eta"])

                    else:

                        print(
                            f"    MC {mc}: Proposed solver failed, "
                            f"status = {sol_joint['status']}"
                        )

                except Exception as e:

                    print(f"    MC {mc}: Proposed solver crashed for K = {K}, N = {N}")
                    print("    Error:", e)

                # ==================================================
                # Similarity benchmark from Paper [21]
                # ==================================================

                try:

                    sol_similarity = myf_solve_similarity_P21(
                        sys_param,
                        H,
                        K,
                        N,
                        S0_ref
                    )

                    if sol_similarity["status"] in sys_param["solver_accept_status"]:

                        eta_similarity = myf_compute_min_DPSL_numeric(
                            sys_param,
                            sol_similarity["S_sum"],
                            N
                        )

                        eta_similarity_list.append(eta_similarity)

                    else:

                        print(
                            f"    MC {mc}: Similarity P21 solver failed, "
                            f"status = {sol_similarity['status']}"
                        )

                except Exception as e:

                    print(f"    MC {mc}: Similarity P21 solver crashed for K = {K}, N = {N}")
                    print("    Error:", e)

                # ==================================================
                # Waveform similarity benchmark from Paper [22]
                # Count only positive DPSL values
                # ==================================================

                try:

                    S_symbol = myf_generate_qpsk_symbols(
                        K,
                        sys_param["L_waveform"]
                    )

                    sol_waveform22 = myf_solve_waveform_similarity_P22(
                        sys_param,
                        H,
                        K,
                        N,
                        S0_ref,
                        S_symbol
                    )

                    eta_waveform22 = myf_compute_min_DPSL_numeric(
                        sys_param,
                        sol_waveform22["R_X"],
                        N
                    )

                    if eta_waveform22 > 0:

                        eta_waveform22_list.append(eta_waveform22)

                    else:

                        print(
                            f"    MC {mc}: Waveform P22 negative DPSL "
                            f"for K={K}, N={N}, eta={eta_waveform22}"
                        )

                except Exception as e:

                    print(f"    MC {mc}: Waveform P22 solver crashed for K = {K}, N = {N}")
                    print("    Error:", e)

            # ==================================================
            # Average over Monte Carlo
            # ==================================================

            eta_proposed_avg = (
                np.mean(eta_proposed_list)
                if len(eta_proposed_list) > 0
                else np.nan
            )

            eta_similarity_avg = (
                np.mean(eta_similarity_list)
                if len(eta_similarity_list) > 0
                else np.nan
            )

            eta_waveform22_avg = (
                np.mean(eta_waveform22_list)
                if len(eta_waveform22_list) > 0
                else np.nan
            )

            eta_proposed_avg_list.append(eta_proposed_avg)

            eta_similarity_avg_list.append(eta_similarity_avg)

            eta_waveform22_avg_list.append(eta_waveform22_avg)

            print(
                f"    Proposed successful MC       = {len(eta_proposed_list)} / {Num_MC}"
            )

            print(
                f"    Similarity P21 successful MC = {len(eta_similarity_list)} / {Num_MC}"
            )

            print(
                f"    Waveform P22 positive DPSL   = {len(eta_waveform22_list)} / {Num_MC}"
            )

        # ==================================================
        # Store results for this K
        # ==================================================

        eta_proposed_arr = np.array(eta_proposed_avg_list)

        eta_similarity_arr = np.array(eta_similarity_avg_list)

        eta_waveform22_arr = np.array(eta_waveform22_avg_list)

        results_fig4[K]["N"] = N_cand

        results_fig4[K]["eta_proposed_avg"] = eta_proposed_arr

        results_fig4[K]["eta_similarity_avg"] = eta_similarity_arr

        results_fig4[K]["eta_waveform22_avg"] = eta_waveform22_arr

        results_fig4[K]["eta_proposed_avg_dB"] = myf_lin2db(
            eta_proposed_arr / sys_param["P0"]
        )

        results_fig4[K]["eta_similarity_avg_dB"] = myf_lin2db(
            eta_similarity_arr / sys_param["P0"]
        )

        eta_waveform22_ratio = eta_waveform22_arr / sys_param["P0"]

        eta_waveform22_dB = np.full_like(
            eta_waveform22_ratio,
            np.nan,
            dtype=float
        )

        valid_idx = eta_waveform22_ratio > 0

        eta_waveform22_dB[valid_idx] = myf_lin2db(
            eta_waveform22_ratio[valid_idx]
        )

        results_fig4[K]["eta_waveform22_avg_dB"] = eta_waveform22_dB

    # ==================================================
    # Plot all methods
    # ==================================================

    myf_plot_fig4_all_methods(results_fig4)

    return results_fig4


if __name__ == "__main__":

    results_fig4 = main()