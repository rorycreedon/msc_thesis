import torch
import numpy as np

from src.structural_models.structural_causal_model import StructuralCausalModel
from src.structural_models.synthetic_data import SimpleSCM, NonLinearSCM


class CostLearner:
    def __init__(
        self,
        X: torch.Tensor,
        n_comparisons: int,
        ground_truth_beta: torch.Tensor,
        scm: StructuralCausalModel,
        scm_known: bool = False,
    ) -> None:
        """
        Initialise the class
        :param X: Data matrix (torch.Tensor)
        :param n_comparisons: Number of comparisons (int)
        :param ground_truth_beta: Ground truth beta (torch.Tensor)
        :param scm: Structural causal model (StructuralCausalModel)
        :param scm_known: SCM known (bool)
        """

        self.X = X
        self.n_comparisons = n_comparisons
        # self.ground_truth_beta = ground_truth_beta / torch.sum(ground_truth_beta)
        self.ground_truth_beta = ground_truth_beta / torch.sum(
            ground_truth_beta, axis=1, keepdims=True
        )
        self.scm = scm
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
        self.actions = self.X + torch.normal(
            0,
            1,
            size=(2, self.n_comparisons, self.X.shape[0], self.X.shape[1]),
            dtype=torch.float64,
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
                costs[i, j] = self.eval_actions_scm(
                    X=self.X.cpu(),
                    actions=self.actions[i, j],
                    ordering=self.ordering[i, j],
                    scm=self.scm,
                    beta=self.ground_truth_beta,
                    scm_noise=scm_noise,
                )

        ratio = costs[0] / costs[1]
        ratio += torch.normal(0, eval_noise, size=ratio.shape)
        self.outcomes = torch.where(ratio < 1, -1, 1)
        self.true_outcomes = torch.where(costs[0] - costs[1] < 0, -1, 1)

    @staticmethod
    def eval_actions_scm(
        X: torch.Tensor,
        actions: torch.Tensor,
        ordering: torch.Tensor,
        scm: StructuralCausalModel,
        beta: torch.Tensor,
        scm_noise: float = 0,
    ) -> torch.Tensor:
        """
        Evaluate an actions using the underlying SCM
        :param X: original features
        :param actions: actions
        :param ordering: ordering (tensor of ints if fixed or tensor of floats)
        :param scm: SCM to use for evaluation
        :param beta: beta - feature mutability
        :return: costs
        """
        # Init result tensors
        X_prime = X.clone()
        S = torch.eye(X.shape[1])[torch.argsort(ordering)].to(int)
        S = ordering.to(int)
        cost = torch.zeros(X.shape[0])
        U = scm.abduction(X_prime)

        for i in range(X.shape[1]):
            cost += torch.sum(((actions - X_prime) ** 2) * (S == i) * beta, dim=1)
            assert (cost >= 0).all(), "Cost should be positive"
            U += (actions - X_prime) * (S == i)
            X_prime = scm.prediction(U)
            # Add noise to SCM
            noise = torch.normal(0, scm_noise, size=X_prime.shape) * (S != i)
            X_prime += noise

        return cost

    @staticmethod
    def eval_actions_linear(
        X: torch.Tensor,
        actions: torch.Tensor,
        ordering: torch.Tensor,
        W_adjacency: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate the actions in the pairwise comparisons
        :param X: original features
        :param actions: actions to get to each value
        :param ordering: ordering or fixed ordering
        :param W_adjacency: adjacency matrix loss being calculated for
        :param beta: beta to calculate loss for
        :return: costs
        """
        X_bar = X.clone()
        S = ordering.to(int)
        costs = torch.zeros(X.shape[0])
        W_temp = W_adjacency + torch.eye(W_adjacency.shape[0])

        for i in range(W_adjacency.shape[0]):
            costs += torch.sum(((actions - X_bar) ** 2) * (S == i) * beta, dim=1)
            X_bar += ((actions - X_bar) * (S == i).to(torch.float64)) @ W_temp

        return costs

    @staticmethod
    def hinge_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Hinge loss
        :param y_pred: predicted values of y
        :param y_true: true values of y
        :return: hinge loss
        """
        return torch.clamp(1 - (y_true * y_pred), min=0)

    def loss(
        self,
        beta: torch.Tensor,
        W_adjacency: torch.Tensor = None,
        tanh_param: float = 50,
        return_outcomes=False,
    ) -> torch.Tensor:
        """
        Loss function for the param learner
        :param beta: Beta to be evaluated
        :param tanh_param: Parameter for the tanh function
        :param return_outcomes: Whether to return outcomes
        :return: Loss
        """
        if self.scm_known:
            costs_0 = torch.stack(
                [
                    self.eval_actions_scm(
                        X=self.X.to("cpu"),
                        actions=self.actions[0][i],
                        ordering=self.ordering[0][i],
                        beta=beta,
                        scm=self.scm,
                        scm_noise=0,
                    )
                    for i in range(self.n_comparisons)
                ]
            )

            costs_1 = torch.stack(
                [
                    self.eval_actions_scm(
                        X=self.X.to("cpu"),
                        actions=self.actions[1][i],
                        ordering=self.ordering[1][i],
                        beta=beta,
                        scm=self.scm,
                        scm_noise=0,
                    )
                    for i in range(self.n_comparisons)
                ]
            )

        else:
            costs_0 = torch.stack(
                [
                    self.eval_actions_linear(
                        X=self.X.to("cpu"),
                        actions=self.actions[0][i],
                        ordering=self.ordering[0][i],
                        W_adjacency=W_adjacency,
                        beta=beta,
                    )
                    for i in range(self.n_comparisons)
                ]
            )

            costs_1 = torch.stack(
                [
                    self.eval_actions_linear(
                        X=self.X.to("cpu"),
                        actions=self.actions[1][i],
                        ordering=self.ordering[1][i],
                        W_adjacency=W_adjacency,
                        beta=beta,
                    )
                    for i in range(self.n_comparisons)
                ]
            )

        # pred_outcomes = torch.where(costs_0 - costs_1 < 0, -1, 1)
        pred_outcomes = torch.tanh(tanh_param * (costs_0 - costs_1))

        if return_outcomes:
            return torch.where(costs_0 - costs_1 < 0, -1, 1)

        else:
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
        learned_beta = torch.ones(self.X.shape, dtype=torch.float64, requires_grad=True)

        # If W assumed to be known, set it to the ground truth
        if self.scm_known is False:
            # require grad for W_adjacency
            temp = np.random.rand(self.X.shape[1], self.X.shape[1])
            # set diagonals of temp to 0
            temp = temp - np.diag(np.diag(temp))
            learned_W = torch.tensor(temp, dtype=torch.float64, requires_grad=True)
        else:
            learned_W = None

        # Optimiser
        if self.scm_known:
            optimizer = torch.optim.AdamW([learned_beta], lr=lr, weight_decay=l2_reg)
        else:
            optimizer = torch.optim.AdamW(
                [learned_W, learned_beta], lr=lr, weight_decay=l2_reg
            )

        loss_list = []

        vprint("Learning beta...")

        # pred_outcomes = self.loss(
        #     beta=learned_beta,
        #     tanh_param=tanh_param,
        #     W_adjacency=learned_W,
        #     return_outcomes=True,
        # )
        # vprint(
        #     f"Initial Accuracy: {torch.sum(pred_outcomes == self.outcomes) / (self.outcomes.shape[0] * self.outcomes.shape[1])}"
        # )

        # Training loop
        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = self.loss(learned_beta, tanh_param=tanh_param, W_adjacency=learned_W)
            loss.backward()

            if self.scm_known is False:
                # add a mask to gre gradients so gradients are fixed at 0 for diagonal elements
                mask = 1 - torch.eye(learned_W.shape[0])
                learned_W.grad *= mask

            optimizer.step()

            # Ensure beta is positive and normalise so it sums to 1
            learned_beta.data = torch.clamp(learned_beta.data, min=0)
            learned_beta.data = learned_beta.data / torch.sum(
                learned_beta.data, axis=1, keepdims=True
            )

            # Track losses
            loss_list.append(loss.item())
            if verbose and epoch % 100 == 0:
                pred_outcomes = self.loss(
                    learned_beta,
                    tanh_param=tanh_param,
                    W_adjacency=learned_W,
                    return_outcomes=True,
                )
                vprint(
                    f"Epoch {epoch} | Loss {loss.item()} | Accuracy {torch.sum(pred_outcomes == self.outcomes) / (self.outcomes.shape[0] * self.outcomes.shape[1])}"
                )
            # Early stopping
            if np.std(loss_list[-15:]) < 1e-6 and epoch > 100:
                if verbose:
                    vprint(f"Converged at epoch {epoch}")
                break

        # figure out how many costs were correctly predicted
        pred_outcomes = self.loss(
            learned_beta,
            tanh_param=tanh_param,
            W_adjacency=learned_W,
            return_outcomes=True,
        )
        vprint(
            f"Learned Accuracy: {torch.sum(pred_outcomes == self.outcomes) / (self.outcomes.shape[0] * self.outcomes.shape[1])}"
        )

        vprint(f"Learned beta: {learned_beta.detach()}")
        vprint(f"Ground truth beta: {self.ground_truth_beta}\n")

        # if self.W_known:
        vprint(f"Learned W: {learned_W.detach()}")

        return learned_beta.detach(), learned_W.detach(), loss_list


if __name__ == "__main__":
    # X, scm = gen_toy_data(10_000)
    N = 1_000
    SCM = SimpleSCM(N)
    SCM.simulate_data()
    y_pred, X_neg, clf = SCM.classify_data()
    X_neg = torch.tensor(X_neg, dtype=torch.float64)
    N_neg = X_neg.shape[0]

    # Ground truth beta
    beta_ground_truth = torch.rand(X_neg.shape[1], dtype=torch.float64).repeat(
        X_neg.shape[0], 1
    )
    beta_ground_truth += (
        torch.rand(X_neg.shape[0], X_neg.shape[1], dtype=torch.float64) / 5
    )
    beta_ground_truth = beta_ground_truth / torch.sum(beta_ground_truth, dim=1)[:, None]

    cost_learner = CostLearner(
        X=X_neg,
        n_comparisons=50,
        ground_truth_beta=beta_ground_truth,
        scm=SCM.scm,
        scm_known=True,
    )
    cost_learner.eval_random_actions(scm_noise=0, eval_noise=0)
    cost_learner.learn(verbose=True, lr=1e-3, l2_reg=0.1, max_epochs=2_000)
