import numpy as np
import cvxpy as cp

from fConstraints import *


def myf_check_feasible_P2_fig6(sys_param, H, K, N, Gamma_trial, Lambda_trial):

    P0 = sys_param["P0"]

    sigma2 = sys_param["sigma2"]

    theta_sidelobe = sys_param["theta_sidelobe"]

    S_var = [
        cp.Variable((N, N), hermitian=True)
        for _ in range(K)
    ]

    S_sum = sum(S_var)

    constraints = []

    # ==================================================
    # PSD constraints
    # ==================================================

    for k in range(K):

        constraints.append(S_var[k] >> 0)

    # ==================================================
    # Total transmit power constraint
    # ==================================================

    constraints.append(cp.real(cp.trace(S_sum)) <= P0)

    # ==================================================
    # Main-beam width constraints
    #
    # C_a = 0.5 A_0 - A_a
    # C_b = 0.5 A_0 - A_b
    #
    # tr(C_a S_sum) >= 0
    # tr(C_b S_sum) >= 0
    # ==================================================

    C_a, C_b, _ = myf_C_matrices(
        sys_param,
        theta_sidelobe[0],
        N
    )

    constraints.append(myf_trace(C_a, S_sum) >= 0)

    constraints.append(myf_trace(C_b, S_sum) >= 0)

    # ==================================================
    # DPSL constraints
    #
    # tr(C_m S_sum) >= Lambda_trial
    # ==================================================

    for theta_m in theta_sidelobe:

        _, _, C_m = myf_C_matrices(sys_param, theta_m, N)

        constraints.append(myf_trace(C_m, S_sum) >= Lambda_trial)

    # ==================================================
    # SINR constraints
    #
    # signal/Gamma - interference >= sigma2
    # ==================================================

    for k in range(K):

        h_k = H[k, :].reshape(-1, 1)

        H_k = h_k @ h_k.conj().T

        signal_term = myf_trace(H_k, S_var[k]) / Gamma_trial

        interference_term = 0

        for i in range(K):

            if i != k:

                interference_term += myf_trace(H_k, S_var[i])

        constraints.append(
            signal_term - interference_term >= sigma2
        )

    # ==================================================
    # Stable feasibility check
    #
    # Instead of Minimize(0), minimize transmit power.
    # This helps the SDP solver behave more consistently
    # during bisection.
    # ==================================================

    problem = cp.Problem(
        cp.Minimize(cp.real(cp.trace(S_sum))),
        constraints
    )

    try:

        problem.solve(
            solver=cp.MOSEK,
            verbose=sys_param["solver_verbose"]
        )

    except Exception as e:

        print("MOSEK failed in Fig. 6 feasibility check:", e)

        try:

            problem.solve(
                solver=cp.SCS,
                max_iters=5000,
                eps=1e-4,
                verbose=False
            )

        except Exception as e2:

            print("SCS also failed in Fig. 6 feasibility check:", e2)

            return False

    # For Fig. 6 bisection, avoid accepting inaccurate status
    return problem.status == "optimal"


def myf_bisection_sinr_trace_fig6(sys_param, H, K, N):

    Gamma_low_dB = sys_param["Gamma_fig6_low_dB"]

    Gamma_high_dB = sys_param["Gamma_fig6_high_dB"]

    Lambda_trial = sys_param["Lambda_fig6"]

    Num_iter = sys_param["Num_iter_fig6"]

    sinr_trace_dB = []

    feasible_trace = []

    low_dB = Gamma_low_dB

    high_dB = Gamma_high_dB

    for iter_idx in range(Num_iter):

        Gamma_trial_dB = (low_dB + high_dB) / 2

        Gamma_trial = 10 ** (Gamma_trial_dB / 10)

        is_feasible = myf_check_feasible_P2_fig6(
            sys_param,
            H,
            K,
            N,
            Gamma_trial,
            Lambda_trial
        )

        sinr_trace_dB.append(Gamma_trial_dB)

        feasible_trace.append(is_feasible)

        if is_feasible:

            low_dB = Gamma_trial_dB

        else:

            high_dB = Gamma_trial_dB

        print(
            f"    Iter {iter_idx + 1}: "
            f"Gamma = {Gamma_trial_dB:.3f} dB, "
            f"feasible = {is_feasible}"
        )

    result = {}

    result["iteration"] = np.arange(1, Num_iter + 1)

    result["sinr_trace_dB"] = np.array(sinr_trace_dB)

    result["feasible_trace"] = feasible_trace

    return result