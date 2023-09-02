import torch
import numpy as np
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt

from causal_recourse_gen import CausalRecourseGenerator


class BetaLearner(CausalRecourseGenerator):
    def __init__(
        self,
        n_comparisons: int,
        ground_truth_beta: torch.Tensor,
        ground_truth_W: torch.Tensor,
        learn_ordering: bool = False,
        W_known: bool = False,
    ):
        super(BetaLearner, self).__init__(
            learn_ordering=learn_ordering,
        )
        self.n_comparisons = n_comparisons
        self.ground_truth_beta = ground_truth_beta / torch.sum(ground_truth_beta)
        self.ground_truth_W = ground_truth_W

        # Whether or not the weighted adjacency matrix is known
        self.W_known = W_known

        # Sampled betas
        self.sampled_betas = None

        # Outcomes, ordering and actions
        self.true_outcomes = None
        self.outcomes = None
        self.ordering = None
        self.actions = None

    def eval_random_actions(self, eval_noise: float = 0, W_noise: float = 0) -> None:
        """
        Generate random actions for the pairwise comparisons
        :return: None
        """
        self.actions = torch.rand(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1], dtype=torch.float64
        )
        N, D = self.X.shape
        self.ordering = torch.empty(2, self.n_comparisons, N, D, dtype=torch.long)
        # Fill the tensor with random permutations
        for i in range(2):
            for j in range(self.n_comparisons):
                self.ordering[i, j] = torch.multinomial(
                    torch.ones(N, D) / D, D, replacement=False
                )

        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])

        for i in range(self.actions.shape[0]):
            for j in range(self.actions.shape[1]):
                costs[i, j] = self.eval_actions_new_non_diff(
                    X=self.X.to("cpu"),
                    W_adjacency=self.ground_truth_W,
                    ordering=self.ordering[i, j],
                    beta=self.ground_truth_beta,
                    actions=self.actions[i, j],
                    W_noise=W_noise,
                )

        ratio = costs[0] / costs[1]
        ratio += torch.normal(0, eval_noise, size=ratio.shape)
        self.outcomes = torch.where(ratio < 1, -1, 1)
        self.true_outcomes = torch.where(costs[0] - costs[1] < 0, -1, 1)

    def gen_pairwise_comparisons(self) -> None:
        """
        Generate pairwise comparisons for cost learning. Append on to existing comparisons if they exist.
        :return: None
        """
        # Add perturbations to X
        perturbations = torch.rand(
            size=(self.n_comparisons * 2, self.X.shape[0], self.X.shape[1])
        )  # TODO: update this
        X_perturbed = self.X + perturbations

        # Evaluate the model on the perturbed data
        logits = X_perturbed @ self.W_classifier + self.b_classifier

        # Change one variable (randomly selected) in X_perturbed such that h(X)=0.5
        def random_pick_index(arr):
            return np.random.choice(len(arr))

        idx = np.apply_along_axis(random_pick_index, 2, X_perturbed)
        values_to_change = np.take_along_axis(
            X_perturbed, idx[..., np.newaxis], axis=2
        ).squeeze()
        weights_to_change = self.W_classifier[torch.tensor(idx)]
        updated_values = values_to_change - logits / weights_to_change
        idx0, idx1 = np.ogrid[: X_perturbed.shape[0], : X_perturbed.shape[1]]
        X_perturbed[idx0, idx1, idx] = updated_values
        X_perturbed = np.reshape(
            X_perturbed,
            (self.n_comparisons, 2, X_perturbed.shape[1], X_perturbed.shape[2]),
        )

        # Create pairwise comparisons
        self.pairwise_comparisons = X_perturbed

    @staticmethod
    def eval_actions_new_non_diff(
        X, actions, ordering, W_adjacency, beta, W_noise=0
    ) -> torch.Tensor:
        """
        Evaluate the actions in the pairwise comparisons
        :return: None
        """
        X_bar = X.clone()
        if (
            ordering.shape[-1] == ordering.shape[-2]
        ):  # this is the cases where we have a permutation matrix
            S = torch.argsort(ordering)
        else:
            S = ordering.to(int)
        actions_ordered = torch.gather(actions, 1, S)
        costs = torch.zeros(X.shape[0])

        # Add noise to W
        N, D = X.shape
        W_repeated = (W_adjacency + torch.eye(D)).repeat(N, 1, 1)
        added_noise = W_noise * torch.randn(N, D, D)
        mask = torch.eye(D).repeat(N, 1, 1)
        W_noisy = W_repeated + (
            added_noise * (1 - mask)
        )  # ensuring no noise on diagonals
        # W_temp = W_adjacency + torch.eye(W_adjacency.shape[0])

        for i in range(W_adjacency.shape[0]):
            costs += (
                beta[ordering.to(int)[:, i]]
                * (
                    actions[torch.arange(X.shape[0]), S[:, i]]
                    - X_bar[torch.arange(X.shape[0]), S[:, i]]
                )
                ** 2
            )
            X_bar += W_noisy[torch.arange(N), S[:, i]] * actions_ordered[
                :, i
            ].unsqueeze(-1)

        return costs

    @staticmethod
    def eval_actions_new_diff(X, actions, ordering, W_adjacency, sorter, beta):
        # Init result tensors
        X_bar = X.clone()
        S = sorter(ordering)
        costs = torch.zeros(X.shape[0])
        W_temp = W_adjacency + torch.eye(W_adjacency.shape[0])

        for i in range(W_adjacency.shape[0]):
            costs += (S @ beta)[:, i] * (
                torch.bmm(S, X.unsqueeze(2)).squeeze(2)[:, i]
                - torch.bmm(S, actions.unsqueeze(2)).squeeze(2)[:, i]
            ) ** 2
            X_bar += (
                (W_temp.T * S[:, i].unsqueeze(-1)) @ actions.unsqueeze(-1)
            ).squeeze(-1)

        return costs

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
                (W_adjacency.T * S[:, i].unsqueeze(-1)) @ A.unsqueeze(-1)
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
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1], dtype=torch.float64
        )
        self.actions = torch.zeros(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1], dtype=torch.float64
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
                    self.actions[i, j] = action.detach()
                    costs[i, j] = self.eval_actions_new_non_diff(
                        X=self.X.to("cpu"),
                        W_adjacency=self.ground_truth_W,
                        ordering=order,
                        beta=self.ground_truth_beta,
                        actions=action,
                    )
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
        W_adjacency_ground_truth: torch.Tensor = None,
    ) -> None:
        """
        Parallelised function to generate recourse actions (and orders) for the sampled betas and evaluate which of the actions and orders have lower ground truth costs
        :param classifier_margin: Margin for the classifier
        :param max_epochs: Maximum number of epochs for the recourse generator
        :param lr: Learning rate for the recourse generator
        :param W_adjacency_ground_truth: Ground truth adjacency matrix
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
                        W_adjacency=self.ground_truth_W,
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

    def beta_loss(
        self, beta: torch.Tensor, W_adjacency: torch.Tensor, tanh_param: float = 50
    ) -> torch.Tensor:
        """
        Loss function for the beta learner
        :param beta: Beta to be evaluated
        :param tanh_param: Parameter for the tanh function
        :return: Loss
        """
        costs_0 = torch.stack(
            [
                self.eval_actions_new_non_diff(
                    X=self.X,
                    W_adjacency=W_adjacency,
                    ordering=self.ordering[0][i],
                    beta=beta,
                    actions=self.actions[0][i],
                )
                for i in range(self.n_comparisons)
            ]
        )

        costs_1 = torch.stack(
            [
                self.eval_actions_new_non_diff(
                    X=self.X,
                    W_adjacency=W_adjacency,
                    ordering=self.ordering[1][i],
                    beta=beta,
                    actions=self.actions[1][i],
                )
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
        vprint = print if verbose else lambda *a, **k: None

        # Initialise parameters
        learned_beta = torch.ones(
            self.X.shape[1], dtype=torch.float64, requires_grad=True
        )

        # If W assumed to be known, set it to the ground truth
        if self.W_known:
            learned_W = self.ground_truth_W
        else:
            # require grad for W_adjacency
            temp = self.W_adjacency.clone().detach().numpy()
            temp = temp - np.eye(temp.shape[0])
            # set diagonals of temp to 0
            temp = temp - np.diag(np.diag(temp))
            learned_W = torch.tensor(temp, dtype=torch.float64, requires_grad=True)

        # Optimiser
        if self.W_known:
            optimizer = torch.optim.AdamW([learned_beta], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.AdamW(
                [learned_W, learned_beta], lr=lr, weight_decay=l2_reg
            )

        loss_list = []

        if verbose:
            # figure out how many costs were correctly predicted
            costs_0 = torch.stack(
                [
                    self.eval_actions_new_non_diff(
                        X=self.X,
                        W_adjacency=learned_W,
                        ordering=self.ordering[0][i],
                        beta=learned_beta,
                        actions=self.actions[0][i],
                    )
                    for i in range(self.n_comparisons)
                ]
            )

            costs_1 = torch.stack(
                [
                    self.eval_actions_new_non_diff(
                        X=self.X,
                        W_adjacency=learned_W,
                        ordering=self.ordering[1][i],
                        beta=learned_beta,
                        actions=self.actions[1][i],
                    )
                    for i in range(self.n_comparisons)
                ]
            )
            pred_outcomes = torch.where(costs_0 - costs_1 < 0, -1, 1)
            vprint(
                f"Initial Accuracy: {torch.sum(pred_outcomes == self.outcomes) / (self.outcomes.shape[0] * self.outcomes.shape[1])}"
            )

        vprint("Learning beta...")

        # Training loop
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = self.beta_loss(
                learned_beta, tanh_param=tanh_param, W_adjacency=learned_W
            )
            loss.backward()

            if self.W_known is False:
                # add a mask to gre gradients so gradients are fixed at 0 for diagonal elements
                mask = 1 - torch.eye(learned_W.shape[0])
                learned_W.grad *= mask

            optimizer.step()

            # Ensure beta is positive and normalise so it sums to 1
            learned_beta.data = torch.clamp(learned_beta.data, min=0)
            learned_beta.data = learned_beta.data / torch.sum(learned_beta.data)

            # Track losses
            loss_list.append(loss.item())
            if verbose and epoch % 500 == 0:
                vprint(f"Epoch {epoch} | Loss {loss.item()}")
            # Early stopping
            if np.std(loss_list[-15:]) < 1e-6 and epoch > 100:
                if verbose:
                    vprint(f"Converged at epoch {epoch}")
                break

        if verbose:
            # figure out how many costs were correctly predicted
            costs_0 = torch.stack(
                [
                    self.eval_actions_new_non_diff(
                        X=self.X,
                        W_adjacency=learned_W,
                        ordering=self.ordering[0][i],
                        beta=learned_beta,
                        actions=self.actions[0][i],
                    )
                    for i in range(self.n_comparisons)
                ]
            )

            costs_1 = torch.stack(
                [
                    self.eval_actions_new_non_diff(
                        X=self.X,
                        W_adjacency=learned_W,
                        ordering=self.ordering[1][i],
                        beta=learned_beta,
                        actions=self.actions[1][i],
                    )
                    for i in range(self.n_comparisons)
                ]
            )
            pred_outcomes = torch.where(costs_0 - costs_1 < 0, -1, 1)
            vprint(
                f"Accuracy: {torch.sum(pred_outcomes == self.outcomes) / (self.outcomes.shape[0] * self.outcomes.shape[1])}"
            )

            vprint(f"Learned beta: {learned_beta.detach()}")
            vprint(f"Ground truth beta: {self.ground_truth_beta}\n")

            # if self.W_known:
            vprint(f"Learned W: {learned_W.detach()}")
            vprint(f"Ground truth W: {self.ground_truth_W}\n")

        return learned_beta.detach(), learned_W.detach(), loss_list


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    N = 500

    # FIXED PARAMETERS
    X = torch.rand(N, 4, dtype=torch.float64)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    W_adjacency_ground_truth = torch.tensor(
        [[0, 0, 0, 0], [0.9, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.1, 1.2, 0]],
        dtype=torch.float64,
    )

    # GENERATE ALTERNATIVE ACTIONS AND ORDERINGS
    beta_learner = BetaLearner(
        n_comparisons=5,
        learn_ordering=True,
        ground_truth_beta=torch.tensor([2, 1, 7, 0.2]),
        ground_truth_W=W_adjacency_ground_truth,
    )
    beta_learner.add_data(
        X=X, W_adjacency=W_adjacency, W_classifier=W_classifier, b_classifier=0.5
    )
    # beta_learner.gen_pairwise_comparisons()
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
