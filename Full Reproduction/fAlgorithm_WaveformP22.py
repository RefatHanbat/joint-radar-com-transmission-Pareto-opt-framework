import numpy as np


def myf_matrix_sqrt_psd(R):

    eigval, eigvec = np.linalg.eigh((R + R.conj().T) / 2)

    eigval = np.maximum(eigval, 0)

    R_sqrt = eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.conj().T

    return R_sqrt


def myf_make_reference_waveform_from_covariance(S0_ref, L):

    N = S0_ref.shape[0]

    S0_sqrt = myf_matrix_sqrt_psd(S0_ref)

    U = np.zeros((N, L), dtype=complex)

    U[:, 0:N] = np.eye(N)

    X0 = np.sqrt(L) * S0_sqrt @ U

    return X0


def myf_solve_waveform_similarity_P22(sys_param, H, K, N, S0_ref, S_symbol):

    P0 = sys_param["P0"]

    L = sys_param["L_waveform"]

    rho = sys_param["rho_waveform"]

    Num_iter = sys_param["lambda_bisect_iter"]

    # Reference waveform X0 such that (1/L) X0 X0^H = S0_ref

    X0 = myf_make_reference_waveform_from_covariance(S0_ref, L)

    R_X0 = (X0 @ X0.conj().T) / L

    err_cov = np.linalg.norm(R_X0 - S0_ref, "fro")

    # print("X0 covariance error =", err_cov)

    # Paper [22] compact LS form:
    # min ||A X - B||_F^2
    # Q = A^H A, G = A^H B

    Q = rho * (H.conj().T @ H) + (1 - rho) * np.eye(N)

    G = rho * (H.conj().T @ S_symbol) + (1 - rho) * X0

    # EVD of Q
    eigval, eigvec = np.linalg.eigh((Q + Q.conj().T) / 2)

    lambda_min = np.min(eigval)

    G_tilde = eigvec.conj().T @ G

    # Need ||X(lambda)||_F^2 = L P0
    target_power = L * P0

    def power_for_lambda(lam):

        denom = eigval + lam

        X_tilde = G_tilde / denom[:, None]

        power = np.linalg.norm(X_tilde, "fro") ** 2

        return power

    # lower bound must make Q + lambda I positive definite
    lam_low = -lambda_min + 1e-10

    lam_high = 1.0

    while power_for_lambda(lam_high) > target_power:

        lam_high = 2 * lam_high

    for _ in range(Num_iter):

        lam_mid = (lam_low + lam_high) / 2

        if power_for_lambda(lam_mid) > target_power:

            lam_low = lam_mid

        else:

            lam_high = lam_mid

    lambda_opt = lam_high

    X_tilde_opt = G_tilde / (eigval + lambda_opt)[:, None]

    X_opt = eigvec @ X_tilde_opt

    R_X = (X_opt @ X_opt.conj().T) / L

    solution = {}

    solution["status"] = "optimal"

    solution["X"] = X_opt

    solution["R_X"] = R_X

    solution["lambda_opt"] = lambda_opt

    solution["objective"] = (
        rho * np.linalg.norm(H @ X_opt - S_symbol, "fro") ** 2
        + (1 - rho) * np.linalg.norm(X_opt - X0, "fro") ** 2
    )

    return solution