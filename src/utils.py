from scipy.linalg import eigh
import numpy as np
from src.structural_models.structural_causal_model import StructuralCausalModel
from sympy import symbols, Eq
import torch


def is_psd(matrix: np.array):
    """
    Check if a matrix is positive semi-definite
    :param matrix: a matrix (numpy.array)
    :return: True/False if matrix is PSD
    """
    # Compute eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # Check if all eigenvalues are non-negative
    if np.all(eigenvalues >= 0):
        return True
    else:
        return False


def get_near_psd(matrix: np.array):
    """
    Find nearest PSD matrix if matrix if not PSD, else return original matrix
    :param matrix: a matrix (numpy.array)
    :return: A PSD matrix (numpy.array)
    """
    # Get the symmetric part of the distance matrix
    sym_dist_matrix = 0.5 * (matrix + matrix.T)

    # Compute the eigenvalues and eigenvectors of the symmetric distance matrix
    eig_vals, eig_vecs = eigh(sym_dist_matrix)

    # Set negative eigenvalues to zero
    eig_vals[eig_vals < 0] = 0

    # Construct the nearest semi-positive definite matrix
    nearest_spd_matrix = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T

    # Ensure that the matrix does not contain complex numbers
    nearest_spd_matrix = np.real_if_close(nearest_spd_matrix)

    return nearest_spd_matrix


def gen_toy_data(N):
    # Start with SCM
    x1, x2, x3, u1, u2, u3 = symbols("x1 x2 x3 u1 u2 u3")

    # Define the equations
    eq1 = Eq(x1, u1)
    eq2 = Eq(x2, u2 + 0.5 * x1)
    eq3 = Eq(x3, u3 + 0.2 * x1 + 0.3 * x2)

    equations = [eq1, eq2, eq3]

    endog_vars = [x1, x2, x3]
    exog_vars = [u1, u2, u3]

    distribution_list = [
        {
            "name": "u1",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 1},
        },
        {
            "name": "u2",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 1},
        },
        {
            "name": "u3",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 0.5},
        },
    ]

    outcome_weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

    scm = StructuralCausalModel(endog_vars, exog_vars, equations)
    X = scm.generate_data(
        N, distribution_list=distribution_list, outcome_weights=outcome_weights
    )[0]

    return X, scm
