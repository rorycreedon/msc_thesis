import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.multiprocessing as mp
from functools import partial
from tqdm import tqdm


class SoftSort(nn.Module):
    """
    Code used from paper "SoftSort: A Continuous Relaxation for the argsort Operator" (Prillo and Eisenschlos, 2020)
    Source: https://github.com/sprillo/softsort/blob/master/pytorch/softsort.py
    """

    def __init__(self, tau: float = 1.0, hard: bool = False, power: float = 1.0):
        """
        Initialize the class
        :param tau: temperature parameter
        :param hard: whether to use soft or hard sorting
        :param power: power to use in the semi-metric d
        """
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.power = power

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the class
        :param scores: The scores to be sorted
        :return: The softmax of sorted scores (descending sort)
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=False, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(
            self.power
        ).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=P_hat.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat


class CausalRecourseGeneration:
    """
    Class that implements causal recourse generation.
    """

    def __init__(
        self,
        X: torch.Tensor,
        W_adjacency: torch.Tensor,
        beta: torch.Tensor,
        sorter: SoftSort,
        W_classifier: torch.Tensor,
        b_classifier: torch.Tensor = 0,
        fixed_ordering: torch.Tensor = None,
    ) -> None:
        """
        Initialize the class
        :param X: The original values of each feature
        :param W_adjacency: The weighted adjacency matrix
        :param beta: The relative mutability of each feature
        :param sorter: The softsort-ing function
        """
        self.X = X
        self.W_adjacency = W_adjacency + torch.eye(W_adjacency.shape[0])
        self.beta = beta
        self.sorter = sorter
        self.W_classifier = W_classifier
        self.b_classifier = b_classifier
        self.fixed_ordering = fixed_ordering

    def loss_differentiable(
        self,
        A: torch.Tensor,
        O: torch.Tensor,
        index: int,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Differentiable loss function to measure cost (uses SoftSort)
        :param A: Actions to take for each variable
        :param O: The initial ordering of the features
        :param index: The index of the instance to be explained
        :return: (X_bar, cost) where X_bar is the sorted values and cost total cost of applying A with ordering O.
        """
        # Initialize result tensors
        X_bar = self.X[index].clone()
        if O.dim() == 1:
            S = self.sorter(O.unsqueeze(0))[0]
        else:
            S = self.sorter(O)[0]
        cost = 0

        for i in range(self.W_adjacency.shape[0]):
            X_bar += (self.W_adjacency * S[i]) @ A
            cost += torch.sum(A**2 * S[i] * self.beta)

        return X_bar, cost

    def loss_non_differentiable(
        self,
        A: torch.Tensor,
        index: int,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Non-differentiable loss function to measure cost (requires fixed ordering)
        :param A: Actions to take for each variable
        :param index: The index of the instance to be explained
        :return: (X_bar, cost) where X_bar is the sorted values and cost total cost of applying A with ordering fixed_ordering.
        """
        # Initialize result tensors
        X_bar = self.X[index].clone()
        S = torch.argsort(self.fixed_ordering)
        cost = 0

        for i in range(self.W_adjacency.shape[0]):
            X_bar += (self.W_adjacency * S[i]) @ A
            cost += torch.sum(A**2 * S[i] * self.beta)

        return X_bar, cost

    def optimisation(
        self,
        C: torch.Tensor,
        A: torch.Tensor,
        O: torch.Tensor,
        max_optimiser: torch.optim,
        min_optimiser: torch.optim,
        index: int,
        classifier_margin: float = 0.02,
        max_epochs: int = 2_000,
        optimise_ordering: bool = True,
    ):
        """
        Optimisation
        :param C:
        :param A:
        :param O:
        :param max_optimiser:
        :param min_optimiser:
        :param classifier_margin:
        :param max_epochs:
        :return:
        """
        # Create tracking lists
        objective_list = []
        constraint_list = []

        for i in range(max_epochs):
            # Maximise wrt C
            if optimise_ordering:
                X_bar, cost = self.loss_differentiable(A=A, O=O, index=index)
            else:
                X_bar, cost = self.loss_non_differentiable(A=A, index=index)
            constraint = X_bar @ self.W_classifier - classifier_margin
            max_loss = (C * constraint) - cost

            max_optimiser.zero_grad()
            max_loss.backward()
            max_optimiser.step()

            # Minimise wrt A, O, beta
            if optimise_ordering:
                X_bar, cost = self.loss_differentiable(A=A, O=O, index=index)
            else:
                X_bar, cost = self.loss_non_differentiable(A=A, index=index)
            constraint = X_bar @ self.W_classifier - classifier_margin
            min_loss = cost - (C * constraint)

            min_optimiser.zero_grad()
            min_loss.backward()
            min_optimiser.step()

            # Track objective and constraints
            objective_list.append(cost.item())
            constraint_list.append(constraint.item())

            # Early stopping
            if (
                i > 100
                and np.std(objective_list[-10:]) < 1e-4
                and np.std(constraint_list[-10:]) < 1e-4
            ):
                break

        # Print final ordering
        ordering = torch.max(self.sorter(O.unsqueeze(0)), dim=1)[1].squeeze(0)

        # Probability
        prob = torch.sigmoid(constraint + classifier_margin)

        # Return results
        return X_bar, ordering, A, cost, prob

    def fit(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 2_000,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """

        :param max_optimiser_params:
        :param min_optimiser_params:
        :param classifier_margin:
        :param max_epochs:
        :return:
        """
        # Check positive classifier margin
        assert (
            classifier_margin >= 0
        ), "Classifier margin must be greater than or equal to 0"

        # Create lists of A and O
        A_list = [
            torch.zeros(self.X.shape[1], dtype=torch.float32, requires_grad=True)
            for _ in range(self.X.shape[0])
        ]
        if self.fixed_ordering is None:
            O_list = [
                torch.rand(self.X.shape[1], dtype=torch.float32, requires_grad=True)
                for _ in range(self.X.shape[0])
            ]
        else:
            O_list = [
                torch.tensor(
                    self.fixed_ordering, dtype=torch.float32, requires_grad=False
                )
                for _ in range(self.X.shape[0])
            ]

        # Create tensors to collect results
        X_recourse = torch.zeros(self.X.shape)
        action_order = torch.zeros(self.X.shape)
        actions = torch.zeros(self.X.shape)
        costs = torch.zeros(self.X.shape[0])
        probs = torch.zeros(self.X.shape[0])

        # Loop through each row
        for i in tqdm(range(self.X.shape[0])):
            # Initialise parameters
            C = torch.rand(
                1, dtype=torch.float32, requires_grad=True
            )  # lagrangian multiplier
            A = A_list[i]
            O = O_list[i]

            # Handle optimisers
            max_optimiser = optim.SGD([C], lr=1e-2)
            if self.fixed_ordering is None:
                min_optimiser = optim.SGD(
                    [
                        {"params": [A], "lr": 1e-2},
                        {"params": [O], "lr": 1e-2},
                    ]
                )
            else:
                min_optimiser = optim.SGD(
                    [
                        {"params": [A], "lr": 1e-2},
                    ]
                )

            X_bar, ordering, action, cost, prob = self.optimisation(
                C=C,
                A=A,
                O=O,
                max_optimiser=max_optimiser,
                min_optimiser=min_optimiser,
                index=i,
                classifier_margin=classifier_margin,
                max_epochs=max_epochs,
                optimise_ordering=(self.fixed_ordering is None),
            )

            # Collect results
            X_recourse[i] = X_bar.detach()
            action_order[i] = ordering.detach()
            actions[i] = action.detach()
            costs[i] = cost.detach()
            probs[i] = prob.detach()

        # Put into a dataframe
        data = (
            [f"X{i}" for i in range(1, self.X.shape[1] + 1)]
            + [f"a{i}" for i in range(1, self.X.shape[1] + 1)]
            + [f"order{i}" for i in range(1, self.X.shape[1] + 1)]
            + ["cost", "prob"]
        )

        data = {f"X{i+1}": X_recourse[:, i] for i in range(self.X.shape[1])}
        data.update({f"a{i+1}": actions[:, i] for i in range(self.X.shape[1])})
        data.update({f"order{i+1}": action_order[:, i] for i in range(self.X.shape[1])})
        data.update({"cost": costs, "prob": probs})

        df = pd.DataFrame(data)

        return df


class CausalRecourseGenerationParallel(CausalRecourseGeneration):
    @staticmethod
    def worker(index, instance, classifier_margin, max_epochs, optimise_ordering):
        """Modified worker function to accept the instance as an argument."""
        C = torch.rand(1, dtype=torch.float32, requires_grad=True)
        A = torch.zeros(instance.X.shape[1], dtype=torch.float32, requires_grad=True)
        if optimise_ordering:
            O = torch.rand(instance.X.shape[1], dtype=torch.float32, requires_grad=True)
        else:
            O = (
                torch.ones(
                    instance.X.shape[1], dtype=torch.float32, requires_grad=False
                )
                * instance.fixed_ordering
            )

        max_optimiser = optim.SGD([C], lr=1e-2)
        if optimise_ordering:
            min_optimiser = optim.SGD(
                [
                    {"params": [A], "lr": 1e-2},
                    {"params": [O], "lr": 1e-2},
                ]
            )
        else:
            min_optimiser = optim.SGD(
                [
                    {"params": [A], "lr": 1e-2},
                ]
            )

        X_bar, ordering, A, cost, prob = instance.optimisation(
            C=C,
            A=A,
            O=O,
            max_optimiser=max_optimiser,
            min_optimiser=min_optimiser,
            index=index,
            classifier_margin=classifier_margin,
            max_epochs=max_epochs,
            optimise_ordering=optimise_ordering,
        )

        # Detaching tensors from the computation graph before returning them
        return (
            X_bar.detach(),
            ordering.detach(),
            A.detach(),
            cost.detach(),
            prob.detach(),
        )

    def fit_parallel(
        self,
        classifier_margin=0.02,
        max_epochs=2000,
        num_processes=None,
    ):
        """Parallelized version of the fit function using Python's native multiprocessing."""
        assert (
            classifier_margin >= 0
        ), "Classifier margin must be greater than or equal to 0"

        # Using partial to set the constant arguments for the worker_func
        func = partial(
            self.worker,
            instance=self,
            classifier_margin=classifier_margin,
            max_epochs=max_epochs,
            optimise_ordering=(self.fixed_ordering is None),
        )

        with mp.Pool(processes=num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(func, range(self.X.shape[0])),
                    total=self.X.shape[0],
                )
            )

        # Reformatting the results to match the original structure
        X_recourse, action_order, actions, costs, probs = zip(*results)

        # Convert the results into a DataFrame
        data = {
            f"X{i+1}": torch.stack(list(X_recourse))[:, i]
            for i in range(self.X.shape[1])
        }
        data.update(
            {
                f"a{i+1}": torch.stack(list(actions))[:, i]
                for i in range(self.X.shape[1])
            }
        )
        data.update(
            {
                f"order{i+1}": torch.stack(list(action_order))[:, i]
                for i in range(self.X.shape[1])
            }
        )
        data.update(
            {"cost": torch.stack(list(costs)), "prob": torch.stack(list(probs))}
        )

        df = pd.DataFrame(data)
        return df


if __name__ == "__main__":
    # FIXED PARAMETERS
    beta = torch.tensor([1, 1, 1, 1], dtype=torch.float32, requires_grad=True)
    # X = torch.tensor([[-1, -2, -5, -1], [-3, -6, 3, -6]], dtype=torch.float32)
    X = torch.rand(100, 4, dtype=torch.float32)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float32,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float32)

    recourse_gen = CausalRecourseGenerationParallel(
        X=X,
        W_adjacency=W_adjacency,
        W_classifier=W_classifier,
        beta=beta,
        sorter=SoftSort(tau=0.1),
        fixed_ordering=torch.Tensor([0, 1, 2, 3]),
    )

    # Gen recourse
    df = recourse_gen.fit_parallel(num_processes=4)
    print(df.head())
