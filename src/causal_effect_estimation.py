import collections

import numpy as np
from dowhy import CausalModel
from typing import List
import pandas as pd


def indices_generator(W: np.ndarray):
    """
    Creates a generator that yields the indices of the nonzero elements of W.
    :param W: Weighted adjacency matrix (np.ndarray)
    :return: Generator
    """
    # Loop through treatment variables
    for i in range(W.shape[0] - 1):
        # Loop through outcome variables
        for j in range(W.shape[1] - 1):
            # If there is an edge from i to j
            if not np.isclose(W[i, j], 0):
                # Return the causal effect
                yield i, j


def causal_effect_estimation(
    graph_name: str, df: pd.DataFrame, W: np.ndarray, labels: List[str]
):
    # Create empty array
    causal_effects = np.zeros((df.shape[1] - 1, df.shape[1] - 1))

    # Loop through all edges
    for i, j in indices_generator(W.T):
        # Create a causal model
        model = CausalModel(
            data=df,
            treatment=labels[i],
            outcome=labels[j],
            graph=f"graphs/{graph_name}.gml",
            proceed_when_unidentifiable=True,
        )

        # Estimate the causal effect
        identified_estimand = model.identify_effect()
        causal_estimate = model.estimate_effect(
            identified_estimand, method_name="backdoor.linear_regression"
        )

        # TODO: implement GLM where variables are not continuous

        # Print the causal effect
        causal_effects[i, j] = causal_estimate.value

    print(causal_effects.T)
