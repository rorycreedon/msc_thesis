import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd


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
        learn_beta: bool = False,
        learn_ordering: bool = False,
    ) -> None:
        """
        Initialize the class
        :param learn_beta: Whether to learn the beta values
        :param learn_ordering: Whether to learn the ordering of the features
        """
        super(CausalRecourseGeneration, self).__init__()
        self.learn_beta = learn_beta
        self.learn_ordering = learn_ordering
        if self.learn_ordering is True:
            self.sorter = SoftSort(tau=0.1, hard=True, power=1.0)

        # Data
        self.X = None
        self.W_adjacency = None
        self.W_classifier = None
        self.b_classifier = None

        # Ordering
        self.fixed_ordering = None

        # Beta
        self.beta = None

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
        self.X = X
        self.W_adjacency = W_adjacency + torch.eye(W_adjacency.shape[0])
        self.W_classifier = W_classifier
        self.b_classifier = b_classifier

    def set_ordering(self, ordering: torch.Tensor) -> None:
        """
        Set the ordering of the features
        :param ordering: The ordering of the features
        """
        if ordering.shape != self.X.shape:
            raise ValueError("Ordering must have the same shape as X")
        self.fixed_ordering = ordering

    def set_beta(self, beta: torch.Tensor) -> None:
        """
        Set the beta values
        :param beta: The beta values
        """
        if beta.shape[0] != self.X.shape[1]:
            raise ValueError("Beta must have the same number of features as X")
        self.beta = beta

    def set_sorter(self, tau: float, hard: bool = True, power: float = 1.0) -> None:
        """
        Update the sorter
        :param tau: The temperature parameter
        :param hard: Whether to use soft or hard sorting
        :param power: The power to use in the semi-metric d
        """
        self.sorter = SoftSort(tau=tau, hard=hard, power=power)

    def loss_differentiable(
        self, A: torch.Tensor, O: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Differentiable, batched loss function to measure cost (uses SoftSort)
        :param A: Actions to take for each variable
        :param O: The initial ordering of the features
        :return: (X_bar, cost) where X_bar is the sorted values and cost total cost of applying A with ordering O.
        """
        # Init result tensors
        X_bar = self.X.clone()
        S = self.sorter(O)
        cost = torch.zeros(X.shape[0])

        for i in range(self.W_adjacency.shape[0]):
            X_bar += (
                (self.W_adjacency.T * S[:, 0].unsqueeze(-1)) @ A.unsqueeze(-1)
            ).squeeze(-1)
            cost += torch.sum(A**2 * S[:, i] * self.beta, dim=1)

        return X_bar, cost

    def loss_non_differentiable(self, A: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Non-differentiable, batched loss function to measure cost (requires fixed ordering)
        :param A: Actions to take for each variable
        :return: (X_bar, cost) where X_bar is the sorted values and cost total cost of applying A with ordering fixed_ordering.
        """
        # Initialize result tensors
        X_bar = self.X.clone()
        S = torch.argsort(self.fixed_ordering)
        cost = torch.zeros(self.X.shape[0])
        A_ordered = torch.gather(A, 1, S)

        for i in range(self.W_adjacency.shape[0]):
            X_bar += self.W_adjacency[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
            cost += A_ordered[:, i] ** 2 * self.beta[S[:, i]]

        return X_bar, cost

    def gen_recourse(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
        verbose: bool = False,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        :param classifier_margin: The margin of the classifier
        :param max_epochs: The maximum number of epochs to run
        :param lr: learning rate
        :param verbose: Whether to print progress
        :return: X_bar: the updated feature values after recourse, O: the ordering of actions, A: the actions, cost: the cost of recourse, prob: the probability of positive classification
        """
        # Check positive classifier margin
        assert (
            classifier_margin >= 0
        ), "Classifier margin must be greater than or equal to 0"

        # Initialise parameters
        C = torch.rand(self.X.shape[0], dtype=torch.float64, requires_grad=True)
        A = torch.zeros(self.X.shape, dtype=torch.float64, requires_grad=True)
        O = torch.rand(self.X.shape, dtype=torch.float64, requires_grad=True)

        # Handle optimisers
        max_optimiser = optim.SGD([C], lr=lr)
        if self.learn_ordering:
            min_optimiser = optim.SGD(
                [
                    {"params": [A], "lr": lr},
                    {"params": [O], "lr": lr},
                ]
            )
        else:
            min_optimiser = optim.SGD(
                [
                    {"params": [A], "lr": lr},
                ]
            )

        objective_list = []
        constraint_list = []

        for epoch in range(max_epochs):
            # Maximise wrt C
            if self.learn_ordering:
                X_bar, cost = self.loss_differentiable(A=A, O=O)
            else:
                X_bar, cost = self.loss_non_differentiable(A=A)
            constraint = (
                X_bar @ self.W_classifier + self.b_classifier - classifier_margin
            )
            max_loss = torch.sum((C * constraint) - cost)

            max_optimiser.zero_grad()
            max_loss.backward()
            max_optimiser.step()

            # Minimise wrt A, O, beta
            if self.learn_ordering:
                X_bar, cost = self.loss_differentiable(A=A, O=O)
            else:
                X_bar, cost = self.loss_non_differentiable(A=A)
            constraint = (
                X_bar @ self.W_classifier + self.b_classifier - classifier_margin
            )
            min_loss = torch.sum(cost - (C * constraint))

            min_optimiser.zero_grad()
            min_loss.backward()
            min_optimiser.step()

            # Track objective and constraints
            objective_list.append(cost.mean().item())
            constraint_list.append(constraint.mean().item())

            # Early stopping
            if (
                epoch > 100
                and np.std(objective_list[-10:]) < 1e-5
                and np.std(constraint_list[-10:]) < 1e-5
            ):
                break

            if epoch % 500 == 0 and verbose:
                print(
                    f"Epoch {epoch}: Objective: {objective_list[-1]}, Constraint: {constraint_list[-1]}"
                )

        # Clean for dataframe
        X_recourse = X_bar.detach()
        if self.learn_ordering:
            action_order = torch.max(self.sorter(O), dim=1)[1].detach()
        else:
            action_order = self.fixed_ordering
        actions = A.detach()
        costs = cost.detach()
        probs = torch.sigmoid(constraint + classifier_margin).detach()

        data = {f"X{i+1}": X_recourse[:, i] for i in range(self.X.shape[1])}
        data.update({f"a{i+1}": actions[:, i] for i in range(self.X.shape[1])})
        data.update({f"order{i+1}": action_order[:, i] for i in range(self.X.shape[1])})
        data.update({"cost": costs, "prob": probs})

        df = pd.DataFrame(data)

        return df


if __name__ == "__main__":
    # FIXED PARAMETERS
    X = torch.rand(1000, 4, dtype=torch.float64)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float64,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # GEN RECOURSE
    recourse_gen = CausalRecourseGeneration(learn_beta=False, learn_ordering=True)
    recourse_gen.add_data(
        X=X, W_adjacency=W_adjacency, W_classifier=W_classifier, b_classifier=0.5
    )
    recourse_gen.set_beta(beta)
    recourse_gen.set_ordering(torch.arange(4).repeat(1000, 1))
    recourse_gen.set_sorter(tau=0.1, hard=True)
    df = recourse_gen.gen_recourse(
        classifier_margin=0.02, max_epochs=5_000, verbose=True
    )

    # Gen recourse
    print(df.head())
    print(f"Number of indivuduals negatively classified: {np.sum(df.prob < 0.5)}")
