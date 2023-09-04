import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from typing import Union
import argparse

from src.cost_learning.softsort import SoftSort
from src._old.kde import GaussianKDE


class CausalRecourseGenerator:
    """
    Class that implements causal recourse generation.
    """

    def __init__(
        self,
        learn_ordering: bool = False,
    ) -> None:
        """
        Initialize the class
        :param learn_ordering: Whether to learn the ordering of the features
        """
        super(CausalRecourseGenerator, self).__init__()
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learn_ordering = learn_ordering
        if self.learn_ordering is True:
            self.sorter = SoftSort(tau=0.1, hard=True, power=1.0).to(self.device)

        # Data
        self.X = None
        self.W_adjacency = None
        self.W_classifier = None
        self.b_classifier = None

        # Ordering
        self.fixed_ordering = None

        # Beta
        self.beta = None

        # KDE
        self.kde = None

    def add_data(
        self,
        X: torch.Tensor,
        W_adjacency: torch.Tensor,
        W_classifier: torch.Tensor,
        b_classifier: float = 0,
    ) -> None:
        """
        Add data to the class
        :param X: The original values of each feature
        :param W_adjacency: The weighted adjacency matrix
        :param W_classifier: The weights of the classifier
        :param b_classifier: The bias of the classifier
        """
        if X.shape[1] != W_adjacency.shape[0]:
            raise ValueError("X and W_adjacency must have the same number of features")
        if X.shape[1] != W_classifier.shape[0]:
            raise ValueError("X and W_classifier must have the same number of features")
        if W_classifier.shape[0] != W_adjacency.shape[1]:
            raise ValueError(
                "W_classifier and W_adjacency must have the same number of features"
            )
        self.X = X.to(self.device)
        self.W_adjacency = (W_adjacency + torch.eye(W_adjacency.shape[0])).to(
            self.device
        )
        self.W_classifier = W_classifier.to(self.device)
        self.b_classifier = torch.tensor(b_classifier).to(self.device)

        # Update default values based on this
        self.beta = torch.ones(self.X.shape[1]).to(self.device)
        self.fixed_ordering = (
            torch.arange(self.X.shape[1]).repeat(self.X.shape[0], 1).to(self.device)
        )

    def set_ordering(self, ordering: torch.Tensor) -> None:
        """
        Set the ordering of the features
        :param ordering: The ordering of the features
        """
        if ordering.shape != self.X.shape:
            raise ValueError("Ordering must have the same shape as X")
        self.fixed_ordering = ordering.to(self.device)

    def set_beta(self, beta: torch.Tensor) -> None:
        """
        Set the beta values
        :param beta: The beta values
        """
        # if beta.shape[0] != self.X.shape[1]:
        #     raise ValueError("Beta must have the same number of features as X")
        self.beta = beta.to(self.device)

    def set_sorter(self, tau: float, power: float = 1.0) -> None:
        """
        Update the sorter
        :param tau: The temperature parameter
        :param hard: Whether to use soft or hard sorting
        :param power: The power to use in the semi-metric d
        """
        self.sorter = SoftSort(tau=tau, hard=True, power=power, device=self.device)

    def define_kde(self) -> None:
        """
        Define the KDE
        :return: None
        """
        self.kde = GaussianKDE(data=self.X, device=self.device)

    def log_kde_shift(self, X_prime: torch.Tensor, A: torch.tensor, eps: float = 1e-8):
        """
        Compute the KDE shift cost function
        :param X_prime: X_prime, which is being optimised
        :param A: Actions to take for each variable
        :param eps: Epsilon to add to percentiles to avoid log(0)
        :return: The log percentile shift cost function
        """
        original_percentiles = self.kde.integrate_neg_inf(X_prime)
        new_percentiles = self.kde.integrate_neg_inf(X_prime + A)
        # return torch.square(original_percentiles - new_percentiles)
        return torch.square(
            torch.log((1 - new_percentiles + eps) / (1 - original_percentiles + eps))
        )

    def loss_differentiable(
        self, A: torch.Tensor, O: torch.Tensor, cost_function: str = "l2_norm"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable loss function to measure cost (uses SoftSort)
        :param cost_function: Cost function to use
        :param A: Actions to take for each variable
        :param O: The initial ordering of the features
        :return: (X_prime, cost) where X_prime is the sorted values and cost total cost of applying A with ordering O.
        """
        # Init result tensors
        X_prime = self.X.clone()
        S = self.sorter(O)
        cost = torch.zeros(self.X.shape[0]).to(self.device)

        for i in range(self.W_adjacency.shape[0]):
            if cost_function == "kde_shift":
                cost += torch.sum(
                    self.log_kde_shift(X_prime, A * S[:, i]) * self.beta, dim=1
                )
            elif cost_function == "l2_norm":
                cost += torch.sum(A**2 * S[:, i] * self.beta, dim=1)
            else:
                raise ValueError("Cost function not recognised")
            assert (cost >= 0).all(), "Cost should be positive"
            X_prime += (
                (self.W_adjacency.T * S[:, i].unsqueeze(-1)) @ A.unsqueeze(-1)
            ).squeeze(-1)

        return X_prime, cost

    def loss_non_differentiable(
        self, A: torch.Tensor, cost_function: str = "l2_norm"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Non-differentiable, batched loss function to measure cost (requires fixed ordering)
        :param A: Actions to take for each variable
        :param cost_funtion: The cost function to use
        :return: (X_prime, cost) where X_prime is the sorted values and cost total cost of applying A with ordering fixed_ordering.
        """
        # Initialize result tensors
        X_prime = self.X.clone()
        S = torch.argsort(self.fixed_ordering)
        cost = torch.zeros(self.X.shape[0]).to(self.device)
        A_ordered = torch.gather(A, 1, S)

        if self.beta.dim() == 1:
            for i in range(self.W_adjacency.shape[0]):
                if cost_function == "kde_shift":
                    cost += torch.sum(
                        self.log_kde_shift(X_prime, A * (S == i)) * self.beta,
                        dim=1,
                    )
                elif cost_function == "l2_norm":
                    cost += A_ordered[:, i] ** 2 * self.beta[S[:, i]]
                else:
                    raise ValueError("Cost function not recognised")
                X_prime += self.W_adjacency[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
        elif self.beta.dim() == 2:
            beta_ordered = torch.gather(self.beta, 1, S)
            for i in range(self.W_adjacency.shape[0]):
                if cost_function == "percentile_shift":
                    cost += torch.sum(
                        self.log_percentile_shift(X_prime, A_ordered[:, i] * S[:, i])
                        * beta_ordered[:, i],
                        dim=1,
                    )
                elif cost_function == "l2_norm":
                    cost += A_ordered[:, i] ** 2 * beta_ordered[:, i]
                else:
                    raise ValueError("Cost function not recognised")
                assert (cost >= 0).all(), "Cost should be positive"
                X_prime += self.W_adjacency[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
        else:
            raise ValueError("Beta must be a 1D or 2D tensor")

        return X_prime, cost

    def recover_interventions(self, A: torch.Tensor, O: torch.Tensor):
        """
        Recover the interventions from A and O
        THIS WORKS ON THE ASSUMPTION THAT WE HAVE PERFECT KNOWLEDGE OF THE ADJACENCY MATRIX.s
        :param A: The actions to take for each variable (after downstream interventions)
        :param O: The ordering of the features
        :return:
        """
        # Init result tensors
        if self.learn_ordering:
            S = torch.max(self.sorter(O), dim=1)[1]
        else:
            S = torch.argsort(self.fixed_ordering)

        A_ordered = torch.gather(A, 1, S)
        X_prime = self.X.clone()
        actions = torch.zeros_like(self.X)

        for i in range(self.W_adjacency.shape[0]):
            X_prime += self.W_adjacency[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
            # set actions as what they need to get to
            actions[torch.arange(self.X.shape[0]), S[:, i]] = torch.gather(
                X_prime, 1, S
            )[:, i]

        return actions

    @staticmethod
    def clip_tensor_gradients_by_row(tensor, max_norm):
        """Vectorized gradient clipping by row for a single tensor."""
        if tensor.grad is not None:
            with torch.no_grad():
                grad_norms = tensor.grad.norm(p=2, dim=1, keepdim=True)
                clip_coef = torch.clamp(max_norm / (grad_norms + 1e-6), max=1)
                tensor.grad.mul_(clip_coef)

    def gen_recourse(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
        verbose: bool = False,
        format_as_df: bool = True,
        cost_function: str = "l2_norm",
    ) -> Union[
        pd.DataFrame,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Generate recourse actions (and orderings)
        :param classifier_margin: The margin of the classifier
        :param max_epochs: The maximum number of epochs to run
        :param lr: learning rate
        :param verbose: Whether to print progress
        :param format_as_df: Whether to format the output as a dataframe
        :param cost_function: The cost function to use
        :return: X_prime: the updated feature values after recourse, O: the ordering of actions, A: the actions, cost: the cost of recourse, prob: the probability of positive classification
        """
        vprint = print if verbose else lambda *a, **k: None
        vprint(f"Using: {self.device}")

        # Check positive classifier margin
        assert (
            classifier_margin >= 0
        ), "Classifier margin must be greater than or equal to 0"

        # Initialise parameters
        lambda1 = torch.rand(self.X.shape[0], dtype=torch.float64).to(self.device)
        lambda1.requires_grad = True
        A = torch.zeros(self.X.shape, dtype=torch.float64).to(self.device)
        A.requires_grad = True
        O = torch.rand(self.X.shape, dtype=torch.float64).to(self.device)
        O.requires_grad = True

        # Handle optimisers
        max_optimiser = optim.SGD([lambda1], lr=lr)
        param_list = [{"params": [A], "lr": lr}]
        if self.learn_ordering:
            param_list.append({"params": [O], "lr": lr})
        min_optimiser = optim.SGD(param_list)

        objective_list = []
        constraint_list = []

        # Sort X if percentile shift is used
        if cost_function == "percentile_shift":
            self.sort_X()
        elif cost_function == "kde_shift":
            self.define_kde()

        for epoch in range(max_epochs):
            # Maximise wrt C
            if self.learn_ordering:
                X_prime, cost = self.loss_differentiable(
                    A=A, O=O, cost_function=cost_function
                )
            else:
                X_prime, cost = self.loss_non_differentiable(
                    A=A, cost_function=cost_function
                )
            constraint = (
                X_prime @ self.W_classifier + self.b_classifier - classifier_margin
            )
            max_loss = torch.sum((lambda1 * constraint) - cost)

            max_optimiser.zero_grad()
            max_loss.backward()
            max_optimiser.step()

            # Minimise wrt A, O, beta
            if self.learn_ordering:
                X_prime, cost = self.loss_differentiable(
                    A=A, O=O, cost_function=cost_function
                )
            else:
                X_prime, cost = self.loss_non_differentiable(
                    A=A, cost_function=cost_function
                )
            constraint = (
                X_prime @ self.W_classifier + self.b_classifier - classifier_margin
            )  # TODO: see if this works for any (differentiable) classifier
            min_loss = torch.sum(cost - (lambda1 * constraint))

            min_optimiser.zero_grad()
            self.clip_tensor_gradients_by_row(A, 0.5)
            self.clip_tensor_gradients_by_row(O, 0.5)
            min_loss.backward()
            min_optimiser.step()

            # Track objective and constraints
            objective_list.append(cost.mean().item())
            constraint_list.append(constraint.mean().item())

            # log metrics to wandb
            # wandb.log(
            #     {
            #         "objective": cost.mean().item(),
            #         "constraint": constraint.mean().item(),
            #     }
            # )

            # Convergence criteria
            if (
                epoch > 100
                and np.std(objective_list[-10:]) < 1e-5
                and np.std(constraint_list[-10:]) < 1e-5
            ):
                break

            if epoch % 100 == 0 and verbose:
                vprint(
                    f"Epoch {epoch} | Objective: {objective_list[-1]} | Constraint: {constraint_list[-1]}"
                )

        if format_as_df:
            # Clean for dataframe
            X_recourse = X_prime.detach().to("cpu")
            if self.learn_ordering:
                action_order = torch.max(self.sorter(O), dim=2)[1].detach().to("cpu")
            else:
                action_order = self.fixed_ordering.to("cpu")
            actions = self.recover_interventions(A, O)
            actions = actions.detach().to("cpu")
            costs = cost.detach().to("cpu")
            probs = torch.sigmoid(constraint + classifier_margin).detach().to("cpu")

            data = {f"X{i+1}": X_recourse[:, i] for i in range(self.X.shape[1])}
            data.update({f"a{i+1}": actions[:, i] for i in range(self.X.shape[1])})
            data.update(
                {f"order{i+1}": action_order[:, i] for i in range(self.X.shape[1])}
            )
            data.update({"cost": costs, "prob": probs})

            df = pd.DataFrame(data)

            return df

        else:
            if self.learn_ordering:
                action_order = torch.max(self.sorter(O), dim=1)[1].detach().to("cpu")
            else:
                action_order = self.fixed_ordering.to("cpu")
            actions = self.recover_interventions(A, O)
            actions = actions.detach().to("cpu")
            return (
                X_prime.detach().to("cpu"),
                action_order,
                actions,
                cost.detach().to("cpu"),
                torch.sigmoid(constraint + classifier_margin).detach().to("cpu"),
            )


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--cost_function", type=str, default="l2_norm")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--learn_ordering", type=bool, default=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=4000)
    args = parser.parse_args()

    N = args.N

    # FIXED PARAMETERS
    X = torch.rand(N, 4, dtype=torch.float64)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float64,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # GEN RECOURSE
    recourse_gen = CausalRecourseGenerator(learn_ordering=args.learn_ordering)
    recourse_gen.add_data(
        X=X, W_adjacency=W_adjacency, W_classifier=W_classifier, b_classifier=0.5
    )
    recourse_gen.set_beta(beta)
    recourse_gen.set_ordering(torch.arange(4).repeat(N, 1))
    recourse_gen.set_sorter(tau=args.tau)

    start = time.time()
    df = recourse_gen.gen_recourse(
        classifier_margin=0.02,
        max_epochs=args.max_epochs,
        verbose=True,
        cost_function=args.cost_function,
        lr=args.lr,
        format_as_df=True,
    )
    print(f"Time taken: {time.time() - start} seconds")
    print(f"Average cost: {df['cost'].mean()}")

    # Gen recourse
    print(df[["a1", "a2", "a3", "a4", "cost", "prob"]].head())
    print(f"Number of indivuduals negatively classified: {np.sum(df.prob < 0.5)}")
