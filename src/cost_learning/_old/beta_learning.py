import torch
import numpy as np

from src.cost_learning.causal_recourse import CausalRecourse
from src.structural_models.structural_causal_model import StructuralCausalModel
from src.cost_learning.softsort import SoftSort
from src.utils import gen_toy_data


class BetaLearner(CausalRecourse):
    def __init__(
        self,
        X: torch.Tensor,
        n_comparisons: int,
        ground_truth_beta: torch.Tensor,
        scm: StructuralCausalModel,
        W_classifier: torch.Tensor,
        b_classifier: float,
        learn_ordering: bool = False,
        scm_known: bool = False,
    ):
        super(BetaLearner, self).__init__(
            X=X,
            scm=scm,
            W_classifier=W_classifier,
            b_classifier=b_classifier,
            learn_ordering=learn_ordering,
        )
        self.n_comparisons = n_comparisons
        self.ground_truth_beta = ground_truth_beta / torch.sum(ground_truth_beta)

        # Whether the weighted adjacency matrix is known
        self.scm_known = scm_known

        # Outcomes, ordering and actions
        self.true_outcomes = None
        self.outcomes = None
        self.ordering = None
        self.actions = None

    def eval_random_actions(self, eval_noise: float = 0, scm_noise: float = 0) -> None:
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
                costs[i, j] = self.eval_actions(
                    X=self.X.to("cpu"),
                    scm=self.scm,
                    ordering=self.ordering[i, j],
                    beta=self.ground_truth_beta,
                    actions=self.actions[i, j],
                    scm_noise=scm_noise,
                    sorter=self.sorter,
                    learn_ordering=self.learn_ordering,
                )

        ratio = costs[0] / costs[1]
        ratio += torch.normal(0, eval_noise, size=ratio.shape)
        self.outcomes = torch.where(ratio < 1, -1, 1)
        self.true_outcomes = torch.where(costs[0] - costs[1] < 0, -1, 1)

    @staticmethod
    def eval_actions_linear(
        X, actions, ordering, W_adjacency, beta, sorter, W_noise=0, learn_ordering=False
    ) -> torch.Tensor:
        """
        Evaluate the actions in the pairwise comparisons
        :return: None
        """
        X_bar = X.clone()
        if learn_ordering:
            S = sorter(ordering)
        else:
            S = torch.eye(X.shape[1])[torch.argsort(ordering)].to(int)

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
    def eval_actions_scm(
        X: torch.Tensor,
        actions: torch.Tensor,
        ordering: torch.Tensor,
        scm: StructuralCausalModel,
        sorter: SoftSort,
        beta: torch.Tensor,
        scm_noise: float = 0,
        learn_ordering=False,
    ) -> torch.Tensor:
        """
        Evaluate an actions
        :param X: original features
        :param actions: actions
        :param ordering: ordering (tensor of ints if fixed or tensor of floats)
        :param scm: SCM to use for evaluation
        :param sorter: SoftSort object
        :param beta: beta - feature mutability
        :param learn_ordering: whether to use learned ordering
        :return: costs
        """
        # Init result tensors
        X_prime = X.clone()
        if learn_ordering:
            S = sorter(ordering)
        else:
            S = torch.eye(X.shape[1])[torch.argsort(ordering)].to(int)
        cost = torch.zeros(X.shape[0])
        U = scm.abduction(X_prime)

        for i in range(X.shape[1]):
            cost += torch.sum(actions**2 * S[:, i] * beta, dim=1)
            assert (cost >= 0).all(), "Cost should be positive"
            U += actions * S[:, i]
            X_prime = scm.prediction(U) + torch.normal(0, scm_noise, size=X_prime.shape)
            # Add noise to SCM
            # noise = torch.normal(0, scm_noise, size=X_prime.shape) * S[:, i]

        return cost

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
                self.eval_actions_non_diff(
                    X=self.X.to("cpu"),
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
                self.eval_actions_non_diff(
                    X=self.X.to("cpu"),
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
            temp = self.W_adjacency.cpu().clone().detach().numpy()
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
                    self.eval_actions_non_diff(
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
                    self.eval_actions_non_diff(
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
                    self.eval_actions_non_diff(
                        X=self.X.to("cpu"),
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
                    self.eval_actions_non_diff(
                        X=self.X.to("cpu"),
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
    N = 500

    # # FIXED PARAMETERS
    # X = torch.rand(N, 4, dtype=torch.float64)
    # W_adjacency = torch.tensor(
    #     [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
    #     dtype=torch.float64,
    #     requires_grad=True,
    # )
    # W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    # beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # W_adjacency_ground_truth = torch.tensor(
    #     [[0, 0, 0, 0], [0.9, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.1, 1.2, 0]],
    #     dtype=torch.float64,
    # )

    X, scm = gen_toy_data(100)

    W_classifier = torch.tensor([1, 4, 3], dtype=torch.float64)
    b_classifier = -1

    # GENERATE ALTERNATIVE ACTIONS AND ORDERINGS
    beta_learner = BetaLearner(
        X=X,
        n_comparisons=5,
        learn_ordering=False,
        ground_truth_beta=torch.tensor([2, 1, 7]),
        scm_known=False,
        scm=scm,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
    )

    # beta_learner.set_beta(beta)
    # beta_learner.set_ordering(torch.arange(4).repeat(N, 1))
    beta_learner.eval_random_actions()

    # LEARN BETA
    beta = beta_learner.learn(
        max_epochs=5_000, lr=2e-2, l2_reg=0.1, tanh_param=20, verbose=True
    )
