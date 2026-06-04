import numpy as np

import cvxpy as cp

from fConstraints import *



# def myf_solve_radar_only_P22(sys_param, N):

#     P0 = sys_param["P0"]

#     theta_sidelobe = sys_param["theta_sidelobe"]

#     S0_var = cp.Variable((N,N), hermitian = True)

#     eta_var = cp.Variable()

#     constraints = []

#     constraints.append(S0_var >> 0 )

#     constraints.append(cp.real(cp.trace(S0_var)) <= P0)

#     # print(f"theta_sidelobe : {theta_sidelobe[0]}")


#     ########### Main beam width constrains ######

#     C_a, C_b, _ = myf_C_matrices(sys_param, theta_sidelobe[0],N)

#     constraints.append(myf_trace(C_a, S0_var) >= 0)

#     constraints.append(myf_trace(C_b, S0_var) >= 0 )

#     #### DPSL constraints #################

#     for theta_m in  theta_sidelobe:

#         # print(f"theta_m : {theta_m}")

#         _,_, C_m = myf_C_matrices(sys_param, theta_m, N)

#         constraints.append(myf_trace(C_m, S0_var) >= eta_var)

#     problem = cp.Problem(cp.Maximize(eta_var), constraints)

#     problem.solve(

#         solver = sys_param["solver_name"],

#         verbose = sys_param["solver_verbose"]
#     )

#     solution = {}

#     solution["status"] = problem.status

#     solution["S0"] = None
    
#     solution["eta"] = np.nan

#     if problem.status in sys_param["solver_accept_status"]:

#         solution["S0"] = np.array(S0_var.value)

#         solution["eta"] = np.real(eta_var.value)

#     return solution

def myf_solve_radar_only_P22(sys_param, N):

    P0 = sys_param["P0"]

    theta_sidelobe = sys_param["theta_sidelobe"]

    S0_var = cp.Variable((N, N), hermitian=True)

    eta_var = cp.Variable()

    constraints = []

    ### S0 >= 0 ###

    constraints.append(S0_var >> 0)

    #### transmit power constraint ###

    constraints.append(cp.real(cp.trace(S0_var)) <= P0)

    # ==================================================
    # Main beam width constraints
    #
    # tr(C_a S0) >= 0
    # tr(C_b S0) >= 0
    #
    # where:
    # C_a = 0.5 A_0 - A_a
    # C_b = 0.5 A_0 - A_b
    # ==================================================

    C_a, C_b, _ = myf_C_matrices(
        sys_param,
        theta_sidelobe[0],
        N
    )

    constraints.append(myf_trace(C_a, S0_var) >= 0)

    constraints.append(myf_trace(C_b, S0_var) >= 0)

    # ==================================================
    # DPSL constraints
    #
    # tr(C_m S0) >= eta
    #
    # where:
    # C_m = A_0 - A_m
    # ==================================================

    for theta_m in theta_sidelobe:

        _, _, C_m = myf_C_matrices(sys_param, theta_m, N)

        constraints.append(myf_trace(C_m, S0_var) >= eta_var)

    ######### maximize minimum DPSL #######

    problem = cp.Problem(cp.Maximize(eta_var), constraints)

    problem.solve(
        
        solver=sys_param["solver_name"],

        verbose=sys_param["solver_verbose"]
    )

    solution = {}

    solution["status"] = problem.status

    solution["S0"] = None

    solution["eta"] = np.nan

    if problem.status in sys_param["solver_accept_status"]:

        solution["S0"] = np.array(S0_var.value)

        solution["eta"] = np.real(eta_var.value)

    return solution