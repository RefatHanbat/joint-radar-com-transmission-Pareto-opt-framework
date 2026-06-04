import numpy as np

import cvxpy as cp

from fConstraints import *


def myf_solve_similarity_P21(sys_param, H, K, N, S0_ref):

    P0 = sys_param["P0"]

    sigma2 = sys_param["sigma2"]

    Gamma = sys_param["Gamma"]

    S_var = [ cp.Variable((N, N), hermitian=True) for _ in range(K) ]

    S_sum = sum(S_var)

    constraints = []

    #####  S_k >= 0   ###

    for k in range(K):

        constraints.append(S_var[k] >> 0)

    

    ##### diag(sum_k S_k) = P0/N * 1 ############

    constraints.append( cp.real(cp.diag(S_sum)) == (P0 / N) * np.ones(N))

    ######## SINR constraints ###########

    for k in range(K):

        h_k = H[k, :].reshape(-1, 1)

        H_k = h_k @ h_k.conj().T

        signal_term = myf_trace(H_k, S_var[k]) / Gamma

        interference_term = 0

        for i in range(K):

            if i != k:

                interference_term += myf_trace(H_k, S_var[i])

        constraints.append(

            signal_term - interference_term >= sigma2
        )

    ###### Similarity objective: 
    ########## min || sum_k S_k - S0_ref ||_F^2 #########

    objective = cp.Minimize(

        cp.sum_squares(cp.abs(S_sum - S0_ref))
    )

    problem = cp.Problem(objective, constraints)

    try:

        problem.solve(

            solver=cp.MOSEK,

            verbose=sys_param["solver_verbose"]
        )

    except Exception as e:

        print("MOSEK failed in similarity benchmark:", e)

        problem.solve(

            solver=cp.SCS,

            max_iters=5000,

            eps=1e-4,

            verbose=False
        )

    solution = {}

    solution["status"] = problem.status

    solution["S_list"] = None

    solution["S_sum"] = None

    solution["similarity_error"] = np.nan

    if problem.status in sys_param["solver_accept_status"]:

        S_list = [

            np.array(S_var[k].value)

            for k in range(K)
            
        ]

        solution["S_list"] = S_list

        solution["S_sum"] = np.sum(S_list, axis=0)

        solution["similarity_error"] = np.real(problem.value)

    return solution