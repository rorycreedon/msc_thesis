from numba import jit
import numpy as np
from itertools import product
import time


@jit(nopython=True)
def has_cycle_optimized(matrix, node, visited, current_path):
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


def add_edges_generator_fixed(matrix, row, N):
    """A generator version of the add_edges function."""
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


def dag_generator(N):
    return add_edges_generator_fixed(np.zeros((N, N), dtype=np.int8), 0, N)


if __name__ == "__main__":
    # Usage
    all_dags = dag_generator(5)
    print(len(list(all_dags)))
