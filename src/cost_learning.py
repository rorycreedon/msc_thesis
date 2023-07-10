import numpy as np
import pandas as pd
import cvxpy as cp


class CostLearn:
    def __init__(
        self,
        X: pd.DataFrame,
        weights: np.ndarray,
        bias: float,
        n_rounds: int,
        M: np.ndarray = None,
    ):
        self.X = X
        self.weights = weights
        self.bias = bias
        self.n_rounds = n_rounds
        self.pairwise_comparisons = np.empty(
            shape=(n_rounds, 2, X.shape[0], X.shape[1])
        )
        if M is None:
            self.M = np.linalg.inv(X.cov())
        else:
            self.M = M
        self.outcomes = np.empty(shape=(n_rounds, X.shape[0]))

    def gen_pairwise_comparisons(self):
        """
        Generate pairwise comparisons for cost learning.
        :return: Pairwise comparisons, of shape (n_rounds, 2, n_samples, n_features)
        """
        # Add perturbations to X
        perturbations = np.random.uniform(
            0, 1, size=(self.n_rounds * 2, self.X.shape[0], self.X.shape[1])
        )
        X_perturbed = self.X.values + perturbations

        # Evaluate the model on the perturbed data
        logits = np.sum((X_perturbed * self.weights) + self.bias, axis=-1)

        # Change one variable (randomly selected) in X_perturbed such that h(X)=0.5
        def random_pick_index(arr):
            return np.random.choice(len(arr))

        idx = np.apply_along_axis(random_pick_index, 2, X_perturbed)
        values_to_change = np.take_along_axis(
            X_perturbed, idx[..., np.newaxis], axis=2
        ).squeeze()
        weights_to_change = self.weights[idx]
        updated_values = values_to_change - logits / weights_to_change
        idx0, idx1 = np.ogrid[: X_perturbed.shape[0], : X_perturbed.shape[1]]
        X_perturbed[idx0, idx1, idx] = updated_values

        # Tuples of pairwise comparisons
        self.pairwise_comparisons = np.empty(
            shape=(self.n_rounds, 2, self.X.shape[0], self.X.shape[1])
        )
        for i in range(self.n_rounds * 2):
            self.pairwise_comparisons[i // 2, i % 2] = X_perturbed[i]

        return self.pairwise_comparisons

    def eval_comparisons(self, cost_func, **args):
        """
        Evaluate the cost function on the pairwise comparisons.
        :param cost_func: Cost function to evaluate
        :param args: Arguments to pass to the cost function
        :return: Outcomes of the cost function, of shape (n_rounds, n_samples) where -1 means the first option is better and +1 means the second option is better
        """
        for i in range(self.n_rounds):
            cost_0 = cost_func(
                pd.DataFrame(
                    self.pairwise_comparisons[i, 0],
                    columns=self.X.columns,
                    index=self.X.index,
                ),
                **args
            )
            cost_1 = cost_func(
                pd.DataFrame(
                    self.pairwise_comparisons[i, 1],
                    columns=self.X.columns,
                    index=self.X.index,
                ),
                **args
            )
            self.outcomes[i] = np.where(cost_0 - cost_1 < 0, -1, 1)
        return self.outcomes

    def loss(self, backend="cvxpy"):
        """
        Compute the loss of the model.
        :return: Loss
        """
        if backend == "cvxpy":
            # Loop through all the comparisons
            loss = 0
            for i in range(self.n_rounds):
                for j in range(self.X.shape[0]):
                    # Compute the loss
                    loss += cp.pos(
                        1
                        - (
                            self.outcomes[i, j]
                            * (  # y_i
                                cp.quad_form(
                                    (
                                        self.X.iloc[j]
                                        - self.pairwise_comparisons[i, 0, j]
                                    ),
                                    self.M,
                                )  # x_i A x_i
                                - cp.quad_form(
                                    (
                                        self.X.iloc[j]
                                        - self.pairwise_comparisons[i, 1, j]
                                    ),
                                    self.M,
                                )  # x_j A x_j
                            )
                        )
                    )

        elif backend == "gurobi":
            raise NotImplementedError("Gurobi backend not implemented yet.")

        else:
            raise ValueError("Invalid backend.")

        return loss

    def solve(self, backend="cvxpy", verbose=True):
        if backend == "cvxpy":
            # Convert M to a cvxpy variable starting with the current value
            self.M = cp.Variable(self.M.shape, symmetric=False)

            # Setup constraints
            constraints = [
                self.M >> 0,
                # cp.norm(self.M, "fro") <= self.M.shape[0],
            ]
            # for k in range(self.V.shape[0]):
            #     constraints.append(cp.norm(self.V[k], 2) <= 4)
            # constraints.append(cp.norm(self.V, "fro") <= 10 * self.V.shape[0])

            # Setup problem
            loss = self.loss() / (self.n_rounds * self.X.shape[0])
            problem = cp.Problem(cp.Minimize(loss), constraints)

            # Solve problem
            problem.solve(verbose=verbose)

            # Return solution
            return self.M.value
