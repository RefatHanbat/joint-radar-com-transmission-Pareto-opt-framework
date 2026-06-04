import cvxpy as cp

from fSteering import myf_outer_a

'''
fConstraints.py

'''
def myf_trace(A, S):

    return cp.real(cp.trace(A @ S))


# def myf_C_matrices(sys_param, theta_m, N):

#     A_0 = myf_outer_a(sys_param, sys_param["theta0"], N)

#     A_a = myf_outer_a(sys_param, sys_param["theta_a"], N)

#     A_b = myf_outer_a(sys_param, sys_param["theta_b"], N)

#     A_m = myf_outer_a(sys_param, theta_m, N)

#     C_a = A_a - 0.5 * A_0

#     C_b = A_b - 0.5 * A_0

#     C_m = A_0 - A_m

#     return C_a, C_b, C_m


def myf_C_matrices(sys_param, theta_m, N):

    A_0 = myf_outer_a(sys_param, sys_param["theta0"], N)

    A_a = myf_outer_a(sys_param, sys_param["theta_a"], N)

    A_b = myf_outer_a(sys_param, sys_param["theta_b"], N)

    A_m = myf_outer_a(sys_param, theta_m, N)

    C_a = 0.5 * A_0 - A_a

    C_b = 0.5 * A_0 - A_b

    C_m = A_0 - A_m

    return C_a, C_b, C_m