from scipy.linalg import eigh
import numpy as np


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


def vprint(string: str, verbose: bool = False):
    """
    Print a string if verbose is set to True
    :param verbose:
    :param string: string to be printed
    :return: None
    """
    if verbose:
        print(string)
