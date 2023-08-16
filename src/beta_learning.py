import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm

from causal_recourse_gen import CausalRecourseGenerator


class BetaLearner(CausalRecourseGenerator):
    def __init__(
        self,
        n_comparisons: int,
        ground_truth_beta: torch.Tensor,
        learn_ordering: bool = False,
    ):
        super(BetaLearner, self).__init__(
            learn_beta=False,
            learn_ordering=learn_ordering,
        )
        self.n_comparisons = n_comparisons
        self.ground_truth_beta = ground_truth_beta

        # Sampled betas
        self.sampled_betas = None

        # Outcomes, ordering and actions
        self.outcomes = None
        self.ordering = None
        self.actions = None

    def sample_betas(self, xrange: tuple[float, float]) -> None:
        """
        Sample betas from a uniform distribution
        :param xrange: Range of the uniform distribution
        :return: None
        """
        self.sampled_betas = (xrange[0] - xrange[1]) * torch.rand(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        ) + xrange[1]

    @staticmethod
    def eval_actions_non_differentiable(
        X: torch.Tensor,
        W_adjacency: torch.Tensor,
        ordering: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a set of actions and orderings (non-differentiable).
        :param X: The original values of each feature
        :param W_adjacency: A weighted adjacency matrix
        :param ordering: The ordering of the actions
        :param beta: The beta values for each feature
        :param A: The actions to be evaluated
        :return: (X_bar, cost) where X_bar is the new values of each feature and cost is the cost of each action
        """
        X_bar = X.clone()
        S = torch.argsort(ordering)
        cost = torch.zeros(X.shape[0])
        A_ordered = torch.gather(A, 1, S)
        W_temp = W_adjacency + torch.eye(W_adjacency.shape[0])

        if beta.dim() == 1:
            for i in range(X.shape[1]):
                X_bar += W_temp[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
                cost += A_ordered[:, i] ** 2 * beta[S[:, i]]
        elif beta.dim() == 2:
            beta_ordered = torch.gather(beta, 1, S)
            for i in range(X.shape[1]):
                X_bar += W_temp[S[:, i]] * A_ordered[:, i].unsqueeze(-1)
                cost += A_ordered[:, i] ** 2 * beta_ordered[:, i]
        else:
            raise ValueError("Beta must be a 1D or 2D tensor")

        return X_bar, cost

    @staticmethod
    def eval_actions_differentiable(
        X: torch.Tensor,
        W_adjacency: torch.Tensor,
        ordering: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        sorter,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate a set of actions and orderings (differentiable).
        :param X: The original values of each feature
        :param W_adjacency: A weighted adjacency matrix
        :param ordering: The ordering of the actions
        :param beta: The beta values for each feature
        :param A: The actions to be evaluated
        :param sorter: A function that sorts the ordering
        :return: (X_bar, cost) where X_bar is the new values of each feature and cost is the cost of each action
        """
        # Init result tensors
        X_bar = X.clone()
        S = sorter(ordering)
        cost = torch.zeros(X.shape[0])

        for i in range(W_adjacency.shape[0]):
            X_bar += (
                (W_adjacency.T * S[:, 0].unsqueeze(-1)) @ A.unsqueeze(-1)
            ).squeeze(-1)
            cost += torch.sum(A**2 * S[:, i] * beta, dim=1)

        return X_bar, cost

    def eval_sampled_betas(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
    ) -> None:
        """
        Generate recourse actions (and orders) for the sampled betas and evaluate which of the actions and orders have lower ground truth costs
        :param classifier_margin: Margin for the classifier
        :param max_epochs: Maximum number of epochs for the recourse generator
        :param lr: Learning rate for the recourse generator
        :return: None
        """
        self.ordering = torch.zeros(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        )
        self.actions = torch.zeros(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        )
        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])

        print("Calculating alternative recourse actions and orders...")

        with tqdm(
            total=self.sampled_betas.shape[0] * self.sampled_betas.shape[1]
        ) as pbar:
            for i in range(self.sampled_betas.shape[0]):
                for j in range(self.sampled_betas.shape[1]):
                    self.set_beta(self.sampled_betas[i, j])
                    _, order, action, cost, _ = self.gen_recourse(
                        classifier_margin=classifier_margin,
                        max_epochs=max_epochs,
                        lr=lr,
                        format_as_df=False,
                    )
                    self.ordering[i, j] = order
                    self.actions[i, j] = action
                    costs[i, j] = self.eval_actions_non_differentiable(
                        X=self.X,
                        W_adjacency=self.W_adjacency,
                        ordering=order,
                        beta=self.ground_truth_beta,
                        A=action,
                    )[1]
                    pbar.update(1)

        self.outcomes = torch.where(costs[0] - costs[1] < 0, -1, 1)

    def eval_worker(
        self, i: int, j: int, classifier_margin: float, max_epochs: int, lr: float
    ) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Worker function for parallel evaluation of sampled betas
        :param i: Index of the first dimension of the sampled betas
        :param j: Index of the second dimension of the sampled betas
        :param classifier_margin: Classifier margin for the recourse generator
        :param max_epochs: Maximum number of epochs for the recourse generator
        :param lr: Learning rate for the recourse generator
        :return:
        """
        self.set_beta(self.sampled_betas[i, j])
        _, order, action, cost, _ = self.gen_recourse(
            classifier_margin=classifier_margin,
            max_epochs=max_epochs,
            lr=lr,
            format_as_df=False,
        )
        return i, j, order, action, cost

    def eval_sampled_betas_parallel(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
    ) -> None:
        """
        Parallelised function to generate recourse actions (and orders) for the sampled betas and evaluate which of the actions and orders have lower ground truth costs
        :param classifier_margin: Margin for the classifier
        :param max_epochs: Maximum number of epochs for the recourse generator
        :param lr: Learning rate for the recourse generator
        :return: None
        """
        self.ordering = torch.zeros(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        )
        self.actions = torch.zeros(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        )
        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])

        print("Calculating alternative recourse actions and orders...")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.eval_worker, i, j, classifier_margin, max_epochs, lr
                )
                for i in range(self.sampled_betas.shape[0])
                for j in range(self.sampled_betas.shape[1])
            ]

            with tqdm(
                total=self.sampled_betas.shape[0] * self.sampled_betas.shape[1]
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i, j, order, action, cost = future.result()
                    self.ordering[i, j] = order
                    self.actions[i, j] = action
                    costs[i, j] = self.eval_actions_non_differentiable(
                        X=self.X,
                        W_adjacency=self.W_adjacency,
                        ordering=order,
                        beta=self.ground_truth_beta,
                        A=action,
                    )[1]
                    pbar.update(1)

        self.outcomes = torch.where(costs[0] - costs[1] < 0, -1, 1)

    @staticmethod
    def hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Hinge loss
        :param y_pred: predicted values of y
        :param y_true: true values of y
        :return: hinge loss
        """
        return torch.clamp(1 - (y_true * y_pred), min=0)

    def beta_loss(self, beta: torch.Tensor, tanh_param: float = 50) -> torch.Tensor:
        """
        Loss function for the beta learner
        :param beta: Beta to be evaluated
        :param tanh_param: Parameter for the tanh function
        :return: Loss
        """
        costs_0 = torch.stack(
            [
                self.eval_actions_non_differentiable(
                    X=self.X,
                    W_adjacency=self.W_adjacency,
                    ordering=self.ordering[0][i],
                    beta=beta,
                    A=self.actions[0][i],
                )[1]
                for i in range(self.n_comparisons)
            ]
        )

        costs_1 = torch.stack(
            [
                self.eval_actions_non_differentiable(
                    X=self.X,
                    W_adjacency=self.W_adjacency,
                    ordering=self.ordering[1][i],
                    beta=beta,
                    A=self.actions[1][i],
                )[1]
                for i in range(self.n_comparisons)
            ]
        )

        # pred_outcomes = torch.where(costs_0 - costs_1 < 0, -1, 1)
        pred_outcomes = torch.tanh(tanh_param * (costs_0 - costs_1))

        return torch.sum(self.hinge_loss(pred_outcomes, self.outcomes)) / (
            self.outcomes.shape[0] * self.outcomes.shape[1]
        )

    def learn(
        self,
        max_epochs: int = 5_000,
        lr: float = 2e-2,
        l2_reg: float = 0.1,
        tanh_param: float = 50,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Learn beta
        :param max_epochs: Max number of epochs
        :param lr: learning rate
        :param l2_reg: L2 regularisation
        :param tanh_param: Parameter for the tanh function
        :param verbose: Whether to print progress
        :return: Learned beta
        """
        # Initialise parameters
        learned_beta = torch.ones(
            self.X.shape[1], dtype=torch.float64, requires_grad=True
        )

        # Optimiser
        optimizer = torch.optim.AdamW([learned_beta], lr=lr, weight_decay=l2_reg)

        loss_list = []

        print("Learning beta...")

        # Training loop
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = self.beta_loss(learned_beta, tanh_param=tanh_param)
            loss.backward()
            optimizer.step()
            # Ensure beta is positive
            learned_beta.data = torch.clamp(learned_beta.data, min=0)
            loss_list.append(loss.item())
            if verbose and epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss {loss.item()}")
            # Early stopping
            if np.std(loss_list[-10:]) < 1e-6 and epoch > 100:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        if verbose:
            print(f"Learned beta: {learned_beta.detach()}")
            print(f"Ground truth beta: {self.ground_truth_beta}")

        return learned_beta.detach()


if __name__ == "__main__":
    N = 500

    # FIXED PARAMETERS
    X = torch.rand(N, 4, dtype=torch.float64)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float64,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # GENERATE ALTERNATIVE ACTIONS AND ORDERINGS
    beta_learner = BetaLearner(
        n_comparisons=5,
        learn_ordering=False,
        ground_truth_beta=torch.tensor([2, 1, 7, 0.2]),
    )
    beta_learner.add_data(
        X=X, W_adjacency=W_adjacency, W_classifier=W_classifier, b_classifier=0.5
    )
    beta_learner.set_beta(beta)
    beta_learner.set_ordering(torch.arange(4).repeat(N, 1))
    # beta_learner.set_sorter(tau=0.1)
    beta_learner.sample_betas((0, 3))
    # beta_learner.eval_sampled_betas_parallel()
    beta_learner.eval_sampled_betas()

    # LEARN BETA
    beta = beta_learner.learn(
        max_epochs=5_000, lr=2e-2, l2_reg=0.1, tanh_param=20, verbose=True
    )
