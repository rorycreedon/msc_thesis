# Importing packages
import pandas as pd
import numpy as np
import cvxpy as cp
import sklearn

# Local imports
from src.utils import is_psd, get_near_psd


class LearnedCostsRecourse:
    """
    A class for calculating recourse with learned costs.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        M_ground_truth: np.ndarray,
        M: np.ndarray = None,
        n_rounds: int = 10,
    ):
        """
        Initialize the class with the data and the model coefficients.
        :param X: the negatively classified data (pd.DataFrame)
        :param M: PSD matrix for Mahalanobis distance (np.ndarray)
        :param M_ground_truth: ground truth PSD matrix for Mahalanobis distance (np.ndarray)
        :param n_rounds: number of rounds for cost learning (int)
        """
        self.X = X
        self.M_ground_truth = M_ground_truth
        if M is None:
            self.M = np.eye(X.shape[1])
        else:
            if not is_psd(M):
                M = get_near_psd(M)
            self.M = M

        # Added in the update_classifier method
        self.clf = None
        self.weights = None
        self.bias = None

        # Recourse dataframe
        self.recourse = None

        # Cost Learning
        self.pairwise_comparisons = None
        self.outcomes = None
        self.M_cvxpy = cp.Variable(self.M.shape, value=self.M, symmetric=True)
        self.n_rounds = n_rounds

    def update_data(self, X: pd.DataFrame) -> None:
        """
        Update the data.
        :param X: the updated negatively classified data (pd.DataFrame)
        :return: None
        """
        self.X = X

    def update_classifier(self, clf: sklearn.linear_model.LogisticRegression) -> None:
        """
        Get the negatively classified data.
        :param clf: the updated classifier (sklearn.linear_model.LogisticRegression)
        :return: None
        """
        self.clf = clf
        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]

    @staticmethod
    def ground_truth_costs(
        X: pd.DataFrame,
        X_prime: pd.DataFrame,
        M: np.ndarray = None,
        form: str = "mahalanobis",
    ) -> np.ndarray:
        """
        Calculate the ground truth costs for a given change in X.
        :param X: original features (pd.DataFrame)
        :param X_prime: features after recourse (pd.DataFrame)
        :param M: PSD matrix for Mahalanobis distance (np.ndarray)
        :param form: form of cost function (str)
        :return: cost values (np.ndarray)
        """
        if form == "mahalanobis":
            if M is None:
                M = np.eye(X.shape[1])
            return np.einsum(
                "ji,jk,ki->i",
                (X - X_prime).T,
                M,
                (X - X_prime).T,
            )

    def opt_logistic(self, cost_function: str = "mahalanobis") -> np.ndarray:
        """
        Calculate the optimal recourse for a given classifier and cost function M.
        :param cost_function: form of cost function (str)
        :return: Recourse values (np.ndarray)
        """
        x = cp.Variable(self.X.shape)

        if (cost_function == "quadratic") or np.all(self.M == np.eye(self.X.shape[1])):
            expr = cp.sum_squares(x - self.X)

        elif cost_function == "mahalanobis":
            quad_forms = [
                cp.quad_form(x[i] - self.X.iloc[i], self.M)
                for i in range(self.X.shape[0])
            ]
            expr = cp.sum(quad_forms)

        else:
            raise ValueError(f"{cost_function} not recognised")

        # Optimisation
        objective = cp.Minimize(expr)
        constraints = [cp.matmul(x, self.weights) + self.bias >= 0]

        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)

        return x.value

    def gen_pairwise_comparisons(self) -> None:
        """
        Generate pairwise comparisons for cost learning. Append on to existing comparisons if they exist.
        :return: None
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
        X_perturbed = np.reshape(
            X_perturbed, (self.n_rounds, 2, X_perturbed.shape[1], X_perturbed.shape[2])
        )
        # Concat on self.X
        X_to_concat = np.expand_dims(self.X.values, axis=(0, 1))
        X_to_concat = np.repeat(X_to_concat, self.n_rounds, axis=0)
        X_perturbed = np.concatenate((X_to_concat, X_perturbed), axis=1)

        # Create pairwise comparisons
        if self.pairwise_comparisons is None:
            self.pairwise_comparisons = X_perturbed
        else:
            self.pairwise_comparisons = np.concatenate(
                (self.pairwise_comparisons, X_perturbed), axis=-2
            )

    def eval_comparisons(self) -> None:
        """
        Evaluate the pairwise comparisons.
        :return: None
        """
        self.outcomes = np.empty(
            shape=(self.n_rounds, self.pairwise_comparisons.shape[2])
        )
        for i in range(self.n_rounds):
            cost_0 = self.ground_truth_costs(
                self.pairwise_comparisons[i, 0],
                self.pairwise_comparisons[i, 2],
                M=self.M_ground_truth,
                form="mahalanobis",
            )
            if np.min(cost_0) < 0:
                raise AssertionError(
                    f"Cost function must be non-negative, min is {np.min(cost_0)}"
                )
            cost_1 = self.ground_truth_costs(
                self.pairwise_comparisons[i, 1],
                self.pairwise_comparisons[i, 2],
                M=self.M_ground_truth,
                form="mahalanobis",
            )
            if np.min(cost_1) < 0:
                raise AssertionError(
                    f"Cost function must be non-negative, min is {np.min(cost_1)}"
                )
            self.outcomes[i] = np.where(cost_0 - cost_1 < 0, -1, 1)

    def cost_loss(
        self, loss_function: str = "hinge", margin: float = 0
    ) -> cp.Expression:
        """
        Compute loss function for cost learning.
        :param loss_function: the loss function to use (str)
        :param margin: If loss function is max_margin, the margin to use (float)
        :return: The loss value (cp.Expression)
        """
        loss = 0

        # Define loss function
        if loss_function == "hinge":
            loss_func = lambda x: cp.pos(1 - x)
        elif loss_function == "logistic":
            loss_func = lambda x: cp.logistic(-x)
        elif loss_function == "max_margin":
            loss_func = lambda x: cp.pos(1 - x + margin)
        else:
            raise ValueError(f"{loss_function} not recognised")

        # Loop over the rounds
        for i in range(self.n_rounds):
            # Create a matrix for each round of pairwise comparisons
            diff_0 = self.pairwise_comparisons[i, 2] - self.pairwise_comparisons[i, 0]
            diff_1 = self.pairwise_comparisons[i, 2] - self.pairwise_comparisons[i, 1]

            # Compute the squared Mahalanobis distance for each round and row
            quad_0 = cp.diag(diff_0 @ self.M_cvxpy @ diff_0.T)
            quad_1 = cp.diag(diff_1 @ self.M_cvxpy @ diff_1.T)

            # Compute the loss for this round
            loss += cp.sum(loss_func(cp.multiply(self.outcomes[i], quad_0 - quad_1)))

        # Normalise the loss
        loss /= self.n_rounds * self.X.shape[0]

        return loss

    def learn_costs(
        self, verbose: bool = False, loss_function: str = "hinge", margin: float = 0
    ) -> None:
        """
        Learn costs.
        :param verbose: whether to detailed information of the convex optimisation (bool)
        :param loss_function: the loss function to use (str)
        :param margin: If loss function is max_margin, the margin to use (float)
        :return: None
        """
        # Calculate/update the pairwise comparisons
        self.gen_pairwise_comparisons()
        self.eval_comparisons()

        # Define the problem
        objective = cp.Minimize(
            self.cost_loss(loss_function=loss_function, margin=margin)
        )
        constraints = [self.M_cvxpy >> 0]
        prob = cp.Problem(objective, constraints)

        # Solve the problem
        prob.solve(verbose=verbose, solver="MOSEK")

        # Store the results
        self.M = self.M_cvxpy.value

    def compute_recourse(
        self,
        cost_function: str = "mahalanobis",
        C: float = np.inf,
        learn_costs: bool = True,
        verbose: bool = False,
        loss_function: str = "hinge",
        margin: float = 0,
    ) -> None:
        """
        Compute the recourse for a given classifier and cost function.
        :param cost_function: form of cost function (str)
        :param C: Cost threshold for recourse (float)
        :param learn_costs: Whether to learn costs (bool)
        :param verbose: whether to detailed information of the convex optimisation (bool)
        :param loss_function: the loss function to use (str)
        :param margin: If loss function is max_margin, the margin to use (float)
        :return: None
        """
        # Get the optimal recourse
        recourse_x = self.opt_logistic(cost_function=cost_function)

        # Setup dataframe to store results
        self.recourse = pd.DataFrame(recourse_x, columns=self.X.columns)

        # Learn costs
        if learn_costs:
            self.learn_costs(
                verbose=verbose,
                loss_function=loss_function,
                margin=margin,
            )

        # Add in costs to dataframe
        self.recourse.index = self.X.index
        self.recourse["learned_cost"] = self.ground_truth_costs(
            X=self.X,
            X_prime=self.recourse[self.X.columns],
            M=self.M,
            form="mahalanobis",
        )
        self.recourse["ground_truth_cost"] = self.ground_truth_costs(
            X=self.X,
            X_prime=self.recourse[self.X.columns],
            M=self.M_ground_truth,
            form="mahalanobis",
        )

        # Returning original values of X is cost < C
        self.recourse.update(self.X[self.recourse["ground_truth_cost"] > C])
        self.recourse["prob"] = self.clf.predict_proba(
            self.recourse[self.X.columns].values
        )[:, 1]
        eps = 1e-6
        self.recourse["Y"] = (self.recourse["prob"] >= 0.5 - eps).astype(float)
