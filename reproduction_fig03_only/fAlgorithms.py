import numpy as np
import cvxpy as cp
from fCalculations import *

'''
fAlgorithms.py
'''


def myf_trace_quad(A_mat, X_mat):

    y = cp.real(cp.trace(A_mat@X_mat))

    return y


def myf_outer(x_vec):

    X_mat = x_vec@x_vec.conj().T

    return X_mat


def myf_solve_status(sys_param, problem):

    ### Parameters ###

    solver_name = sys_param["solver_name"]

    solver_max_iter = sys_param["solver_max_iter"]

    solver_verbose = sys_param["solver_verbose"]


    ### Functions ###

    solve_param = {}

    solve_param["solver"] = solver_name

    solve_param["verbose"] = solver_verbose

    if (solver_name == "CLARABEL"):

        solve_param["max_iter"] = solver_max_iter

    elif (solver_name == "SCS"):

        solve_param["max_iters"] = solver_max_iter

    try:

        problem.solve(**solve_param)

    except Exception:

        pass

    if not (problem.status in sys_param["solver_accept_status"]):

        try:

            problem.solve(solver="SCS", max_iters=5000, verbose=False, eps=10**(-4))

        except Exception:

            pass

    return problem.status


def myf_algorithm_Pareto(sys_param, channel, param_algorithm):

    ### Parameters ###

    N = param_algorithm["N"]

    K = param_algorithm["K"]

    Gamma = param_algorithm["Gamma"]

    kappa = param_algorithm["kappa"]

    H = channel["H"]

    P0 = sys_param["P0"]

    sigma2 = sys_param["sigma2"]

    theta0 = sys_param["theta0"]

    theta_a = sys_param["theta_a"]

    theta_b = sys_param["theta_b"]

    theta_sidelobe = sys_param["theta_sidelobe"]


    ### Functions ###

    Num_cov = max(K, 1)

    S_var = [cp.Variable((N, N), hermitian=True) for _ in range(0, Num_cov)]

    d_var = cp.Variable(N, nonneg=True)

    D_var = cp.diag(d_var)

    R_var = sum(S_var)

    Q_var = R_var + D_var

    eta_var = cp.Variable()

    a_0 = myf_steering_vector(sys_param, theta0, N)

    a_a = myf_steering_vector(sys_param, theta_a, N)

    a_b = myf_steering_vector(sys_param, theta_b, N)

    A_0 = myf_outer(a_0)

    A_a = myf_outer(a_a)

    A_b = myf_outer(a_b)

    peak = myf_trace_quad(A_0, Q_var)

    constraints = []

    for ind in range(0, Num_cov):

        constraints.append(S_var[ind] >> 0)

    constraints.append(cp.real(cp.trace(Q_var)) <= P0)

    constraints.append(0.5*peak <= myf_trace_quad(A_a, Q_var))

    constraints.append(0.5*peak <= myf_trace_quad(A_b, Q_var))

    for ind in range(0, N):

        constraints.append((kappa**2)*cp.real(R_var[ind, ind]) <= d_var[ind])


    for ind in range(0, np.size(theta_sidelobe)):

        a_m = myf_steering_vector(sys_param, theta_sidelobe[ind], N)

        A_m = myf_outer(a_m)

        constraints.append(peak - myf_trace_quad(A_m, Q_var) >= eta_var)

    for ind1 in range(0, K):

        h_vec = H[ind1, :].reshape(-1, 1)

        H_mat = myf_outer(h_vec)

        signal = myf_trace_quad(H_mat, S_var[ind1])

        interference = 0

        for ind2 in range(0, K):

            if (ind2 != ind1):

                interference = interference + myf_trace_quad(H_mat, S_var[ind2])

        distortion = myf_trace_quad(H_mat, D_var)

        constraints.append(signal >= Gamma*(interference + distortion + sigma2))

    problem = cp.Problem(cp.Maximize(eta_var), constraints)

    status = myf_solve_status(sys_param, problem)

    solutions = {}

    solutions["status"] = status

    solutions["R"] = None

    solutions["Q"] = None

    solutions["D"] = None

    solutions["S"] = None

    solutions["eta"] = np.nan

    if (status in sys_param["solver_accept_status"]):

        S_val = [np.array(S_var[ind].value) for ind in range(0, Num_cov)]

        R_val = np.sum(S_val, axis=0)

        D_val = np.array(D_var.value)

        Q_val = R_val + D_val

        solutions["R"] = np.copy(R_val)

        solutions["Q"] = np.copy(Q_val)

        solutions["D"] = np.copy(D_val)

        solutions["S"] = S_val[0:K] if (K > 0) else [S_val[0]]
        
        solutions["eta"] = np.real(eta_var.value)

    return solutions
