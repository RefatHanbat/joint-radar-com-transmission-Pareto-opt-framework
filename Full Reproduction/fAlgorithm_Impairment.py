import numpy as np

import cvxpy as cp

from fConstraints import *


def myf_solve_joint_impairment_P11(sys_param, H, K, N, kappa):

    P0 = sys_param["P0"]

    sigma2 = sys_param["sigma2"]

    Gamma = sys_param["Gamma"]

    theta_sidelobe = sys_param["theta_sidelobe"]

    # ==================================================
    # Variables: S_k
    # ==================================================

    S_var = [

        cp.Variable((N, N), hermitian=True)

        for _ in range(K)
    ]

    R_var = sum(S_var)

    # ==================================================
    # Transmitter impairment covariance
    #
    # D = kappa^2 * diag(diag(R))
    #
    # Q = R + D
    # ==================================================

    D_var = (kappa ** 2) * cp.diag(cp.real(cp.diag(R_var)))

    Q_var = R_var + D_var

    eta_var = cp.Variable()

    constraints = []

    # ==================================================
    # PSD constraints
    # ==================================================

    for k in range(K):

        constraints.append(S_var[k] >> 0)

    # ==================================================
    # Power constraint
    # ==================================================

    constraints.append(cp.real(cp.trace(Q_var)) <= P0)

    # ==================================================
    # Main beam width constraints
    # ==================================================

    C_a, C_b, _ = myf_C_matrices(sys_param, theta_sidelobe[0], N)

    constraints.append(myf_trace(C_a, Q_var) >= 0)

    constraints.append(myf_trace(C_b, Q_var) >= 0)

    # ==================================================
    # DPSL constraints
    #
    # eta_m >= eta_var
    # tr(C_m Q) >= eta_var
    # ==================================================

    for theta_m in theta_sidelobe:

        _, _, C_m = myf_C_matrices(sys_param, theta_m, N)

        constraints.append(myf_trace(C_m, Q_var) >= eta_var)

    # ==================================================
    # SINR constraints with transmitter distortion
    #
    # signal / Gamma - interference - distortion >= sigma2
    # ==================================================

    for k in range(K):

        h_k = H[k, :].reshape(-1, 1)

        H_k = h_k @ h_k.conj().T

        signal_term = myf_trace(H_k, S_var[k]) / Gamma

        interference_term = 0

        for i in range(K):

            if i != k:

                interference_term += myf_trace(H_k, S_var[i])

        distortion_term = myf_trace(H_k, D_var)

        constraints.append(
            signal_term - interference_term - distortion_term >= sigma2
        )

    # ==================================================
    # Objective
    # ==================================================

    problem = cp.Problem(cp.Maximize(eta_var), constraints)

    try:

        problem.solve(
            solver=cp.MOSEK,
            verbose=sys_param["solver_verbose"]
        )

    except Exception as e:

        print("MOSEK failed in impairment solver:", e)

        try:

            problem.solve(
                solver=cp.SCS,
                max_iters=5000,
                eps=1e-4,
                verbose=False
            )

        except Exception as e2:

            print("SCS also failed:", e2)

    solution = {}

    solution["status"] = problem.status

    solution["S_list"] = None

    solution["R"] = None

    solution["D"] = None

    solution["Q"] = None

    solution["eta"] = np.nan

    if problem.status in sys_param["solver_accept_status"]:

        S_list = [

            np.array(S_var[k].value)

            for k in range(K)
        ]

        R_val = np.sum(S_list, axis=0)

        D_val = (kappa ** 2) * np.diag(np.real(np.diag(R_val)))

        Q_val = R_val + D_val

        solution["S_list"] = S_list

        solution["R"] = R_val

        solution["D"] = D_val

        solution["Q"] = Q_val

        solution["eta"] = np.real(eta_var.value)

    return solution