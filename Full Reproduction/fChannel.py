import numpy as np
'''
fChannel.py

'''

def myf_fixed_channel_fig3():

    fixed_channel = {}

    h1_12 = np.array([

        -0.21 - 0.75j, 0.58 - 0.021j, -0.15 - 0.06j,

        -0.73 + 0.029j, 0.096 + 0.16j, -0.67 - 0.17j,

        -0.38 + 1.6j, -0.34 - 1.2j, -0.14 + 0.3j,

        -0.18 - 0.86j, 0.87 + 0.23j, -0.31 - 0.46j

    ], dtype=complex)

    h2_12 = np.array([

        0.016 + 0.66j, 1.1 + 0.13j, 0.44 + 1.1j,

        0.67 - 0.52j, 0.36 + 0.3j, -0.11 + 1.4j,

        1.2 + 0.24j, -0.5 - 0.42j, -0.19 - 1.2j,

        -0.75 + 0.047j, -0.16 + 0.77j, -0.11 + 0.18j

    ], dtype=complex)

    h3_12 = np.array([

        0.036 + 0.25j, 0.33 - 1.1j, 0.13 + 0.07j,

        0.22 - 0.022j, 0.18 - 0.26j, -0.1 - 1.6j,

        -0.62 + 0.71j, -0.83 - 0.2j, 1.1 + 0.33j,

        1.1 + 0.46j, -1.1 + 0.71j, 0.2 - 0.67j

    ], dtype=complex)

    h1_16 = np.concatenate((

        h1_12,

        np.array([

            -0.18 - 0.93j, -0.88 - 0.039j,

            -0.36 + 0.25j, -2.1 + 0.17j

        ], dtype=complex)
    ))

    h2_16 = np.concatenate((
        h2_12,
        np.array([
            0.31 + 0.65j, -0.67 + 0.64j,
            -0.23 + 0.88j, -0.32 - 0.49j
        ], dtype=complex)
    ))

    h3_16 = np.concatenate((
        h3_12,
        np.array([
            0.28 + 1.0j, -0.52 + 0.42j,
            0.0 + 0.66j, 0.88 - 0.46j
        ], dtype=complex)
    ))

    fixed_channel[12] = np.array([h1_12, h2_12, h3_12])

    fixed_channel[16] = np.array([h1_16, h2_16, h3_16])

    return fixed_channel


def myf_generate_rayleigh_channel(K, N):

        H = (
            np.random.randn(K, N)
            + 1j * np.random.randn(K, N)
        ) / np.sqrt(2)

        return H


def myf_generate_qpsk_symbols(K, L):

    bits_real = 2 * np.random.randint(0, 2, size=(K, L)) - 1

    bits_imag = 2 * np.random.randint(0, 2, size=(K, L)) - 1

    S = (bits_real + 1j * bits_imag) / np.sqrt(2)

    return S
