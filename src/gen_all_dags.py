from numba import jit
import numpy as np

from itertools import product
import time

from tqdm import tqdm


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


def add_edges(matrix, row, N):
    results = []
    if row == N:
        results.append(matrix.copy())
        return results

    for values in product([0, 1], repeat=N):
        matrix[row, :] = values
        if not any(
            has_cycle_optimized(
                matrix, node, np.zeros(N, dtype=np.bool_), np.zeros(N, dtype=np.bool_)
            )
            for node in range(N)
        ):
            results.extend(add_edges(matrix, row + 1, N))
    return results


def generate_dags(N):
    matrix = np.zeros((N, N), dtype=np.int8)
    return add_edges(matrix, 0, N)


if __name__ == "__main__":
    # Usage
    start = time.time()
    dags_3_optimized = generate_dags(15)
    end = time.time()
    print(f"Took {end - start} seconds")
    print(len(dags_3_optimized))
