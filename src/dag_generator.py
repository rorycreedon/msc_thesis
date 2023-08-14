from numba import jit
import numpy as np
from itertools import product
from typing import Generator


@jit(nopython=True)
def has_cycle_optimized(
    matrix: np.ndarray, node: int, visited: np.ndarray, current_path: np.ndarray
) -> bool:
    """
    Check for cycle in the adjacency matrix.

    :param matrix: The adjacency matrix.
    :param node: The current node being checked.
    :param visited: An array indicating which nodes have been visited.
    :param current_path: An array indicating nodes in the current path.
    :return: True if a cycle is detected, otherwise False.
    """
    if current_path[node]:
        return True

    if visited[node]:
        return False

    visited[node] = True
    current_path[node] = True

    for neighbor in range(len(matrix)):
        if matrix[node][neighbor] and has_cycle_optimized(
            matrix, neighbor, visited, current_path
        ):
            return True

    current_path[node] = False
    return False


def add_edges_generator_fixed(
    matrix: np.ndarray, row: int, N: int
) -> Generator[np.ndarray, None, None]:
    """
    A generator version of the add_edges function.

    :param matrix: The adjacency matrix.
    :param row: The current row being populated in the matrix.
    :param N: The dimension of the matrix (N x N).
    :yield: Yield valid DAG matrices one-by-one.
    """
    if row == N:
        yield matrix.copy()
        return

    for values in product([0, 1], repeat=N):
        new_matrix = matrix.copy()
        new_matrix[row, :] = values
        if not any(
            has_cycle_optimized(
                new_matrix,
                node,
                np.zeros(N, dtype=np.bool_),
                np.zeros(N, dtype=np.bool_),
            )
            for node in range(N)
        ):
            yield from add_edges_generator_fixed(new_matrix, row + 1, N)


def dag_generator(N: int) -> Generator[np.ndarray, None, None]:
    """
    Generate DAGs for a given number of nodes.

    :param N: Number of nodes.
    :return: A generator yielding valid DAG matrices for the given number of nodes.
    """
    return add_edges_generator_fixed(np.zeros((N, N), dtype=np.int8), 0, N)


if __name__ == "__main__":
    # Usage
    all_dags = dag_generator(5)
    print(len(list(all_dags)))
