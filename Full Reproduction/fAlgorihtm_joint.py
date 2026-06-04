import numpy as np
import cvxpy as cp

from fConstraints import myf_trace, myf_C_matrices


def myf_solve_joint_P11(sys_param, H, K, N):

    P0 = sys_param["P0"]

    sigma2 = sys_param["sigma2"]

    Gamma = sys_param["Gamma"]

    theta_sidelobe = sys_param["theta_sidelobe"]

    S_var = [ cp.Variable((N, N), hermitian=True) for _ in range(K) ]

    eta_var = cp.Variable()

    S_sum = sum(S_var)

    constraints = []

    for k in range(K):

        constraints.append(S_var[k] >> 0)

    constraints.append(cp.real(cp.trace(S_sum)) <= P0)

    ########### Main-beam width constraints #################

    C_a, C_b, _ = myf_C_matrices(sys_param, theta_sidelobe[0], N)

    constraints.append(myf_trace(C_a, S_sum) >= 0)

    constraints.append(myf_trace(C_b, S_sum) >= 0)

    ##################### DPSL constraints ######################

    for theta_m in theta_sidelobe:

        _, _, C_m = myf_C_matrices(sys_param, theta_m, N)

        constraints.append(myf_trace(C_m, S_sum) >= eta_var)

    ######################## SINR constraints ################################

    for k in range(K):

        h_k = H[k, :].reshape(-1, 1)

        H_k = h_k @ h_k.conj().T

        signal_term = myf_trace(H_k, S_var[k]) / Gamma

        interference_term = 0

        for i in range(K):

            if i != k:

                interference_term += myf_trace(H_k, S_var[i])

        constraints.append( signal_term - interference_term >= sigma2 )

    problem = cp.Problem(cp.Maximize(eta_var), constraints)

    try:

        problem.solve(

            solver=sys_param["solver_name"],

            verbose=sys_param["solver_verbose"],
        )
    
    except Exception as e :

        print("Mosek failed . Trying SCS fallback...")

        print("Error: ", e)

        problem.solve(

            solver = cp.SCS,

            max_iters = 5000,

            eps = 1e-4,

            verbose = False
        )

    solution = {}

    solution["status"] = problem.status

    solution["S_list"] = None

    solution["S_sum"] = None

    solution["eta"] = np.nan

    if problem.status in sys_param["solver_accept_status"]:

        S_list = [ np.array(S_var[k].value) for k in range(K) ]

        solution["S_list"] = S_list

        solution["S_sum"] = np.sum(S_list, axis=0)

        solution["eta"] = np.real(eta_var.value)

    return solution