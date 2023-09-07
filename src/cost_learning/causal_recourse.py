import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import time
from typing import Union
import matplotlib.pyplot as plt

from src.cost_learning.softsort import SoftSort
from src.structural_models.structural_causal_model import StructuralCausalModel
from src.structural_models.synthetic_data import SimpleSCM, NonLinearSCM
from src.utils import gen_toy_data


class CausalRecourse:
    """
    Class to implement and generate causal recourse.
    """

    def __init__(
        self,
        X: torch.Tensor,
        W_classifier: torch.Tensor,
        b_classifier: float,
        beta: torch.Tensor,
        use_scm: bool = True,
        scm: StructuralCausalModel = None,
        W_adjacency: torch.Tensor = None,
        learn_ordering: bool = False,
    ) -> None:
        """
        Initialize the class.
        :param scm: (assumed) structural causal model
        :param X: the data
        :param W_classifier: the classifier weights
        :param b_classifier: the classifier bias
        :param beta: beta - relative mutability
        :param learn_ordering: whether to learn the ordering of the variables

        """
        super(CausalRecourse, self).__init__()
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.learn_ordering = learn_ordering
        if self.learn_ordering is True:
            self.sorter = SoftSort(tau=0.1, hard=True, power=1.0).to(self.device)

        # Data
        self.X = X.to(self.device)

        # SCM/adjacency matrix
        self.use_scm = use_scm
        if self.use_scm:
            self.scm = scm
            # Initial value of U
            self.U = self.scm.abduction(self.X)
        else:
            self.W_adjacency = W_adjacency.to(self.device)

        # Classifier
        self.W_classifier = W_classifier.to(self.device)
        self.b_classifier = b_classifier

        # Beta
        if beta is None:
            self.beta = torch.ones(self.X.shape[1]).to(self.device)
        else:
            self.beta = beta.to(self.device)
        assert torch.min(self.beta) >= 0, "Beta must be greater than 0"
        # normalise beta
        self.beta = self.beta / torch.sum(self.beta, dim=1).unsqueeze(-1)

        # Ordering
        if self.learn_ordering is False:
            self.fixed_ordering = (
                torch.arange(self.X.shape[1]).repeat(self.X.shape[0], 1).to(self.device)
            )

    def set_ordering(self, ordering: torch.Tensor) -> torch.Tensor:
        """
        Fix the ordering of the variables.
        """
        if ordering.shape != self.X.shape:
            raise ValueError("Ordering must have the same shape as X")
        self.fixed_ordering = ordering.to(self.device)

    def set_beta(self, beta: torch.Tensor) -> None:
        """
        Set the beta values
        :param beta: The beta values
        """
        self.beta = beta.to(self.device)

    def set_sorter(self, sorter: SoftSort) -> None:
        """
        Set the sorter
        :param sorter: The sorter
        """
        self.sorter = sorter
        self.sorter.device = self.device

    def loss(
        self, A: torch.Tensor, O: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable loss function to measure cost (uses SoftSort)
        :param A: Actions to take for each variable
        :param O: The initial ordering of the features
        :return: (X_prime, cost) where X_prime is the sorted values and cost total cost of applying A with ordering O.
        """
        # Init result tensors
        X_prime = self.X.clone()
        if self.learn_ordering:
            S = self.sorter(O)
        else:
            S = torch.eye(self.X.shape[1])[torch.argsort(self.fixed_ordering)].to(int)

        cost = torch.zeros(self.X.shape[0]).to(self.device)

        if self.use_scm:
            U = self.scm.abduction(X_prime)

            for i in range(self.X.shape[1]):
                cost += torch.sum(A**2 * S[:, i] * self.beta, dim=1)
                assert (cost >= 0).all(), "Cost should be positive"
                U += A * S[:, i]
                X_prime = self.scm.prediction(U)

        else:
            W_temp = self.W_adjacency + torch.eye(self.W_adjacency.shape[0]).to(
                self.device
            )

            for i in range(self.X.shape[1]):
                cost += torch.sum(A**2 * S[:, i] * self.beta, dim=1)
                assert (cost >= 0).all(), "Cost should be positive"
                X_prime += ((W_temp * S[:, i].unsqueeze(-1)) @ A.unsqueeze(-1)).squeeze(
                    -1
                )

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
        X_prime = self.X.clone()
        if self.learn_ordering:
            S = torch.max(self.sorter(O), dim=1)[1]
        else:
            S = torch.argsort(self.fixed_ordering)

        actions = torch.zeros_like(self.X).to(self.device)

        if self.use_scm:
            for i in range(self.X.shape[1]):
                U = self.scm.abduction(X_prime)
                X_prime = self.scm.prediction(U + (A * (S == i)).to(torch.float64))
                # set actions as what they need to get to
                actions[torch.arange(self.X.shape[0]), S[:, i]] = torch.gather(
                    X_prime, 1, S
                )[:, i].to(torch.float64)

        else:
            A_ordered = torch.gather(A, 1, S)
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

        for epoch in range(max_epochs):
            # Maximise wrt lambda1
            X_prime, cost = self.loss(A=A, O=O)
            constraint = (
                X_prime @ self.W_classifier + self.b_classifier - classifier_margin
            )
            max_loss = torch.sum((lambda1 * constraint) - cost)
            max_optimiser.zero_grad()
            max_loss.backward()
            max_optimiser.step()

            # Minimise wrt A, O
            X_prime, cost = self.loss(A=A, O=O)
            constraint = (
                X_prime @ self.W_classifier + self.b_classifier - classifier_margin
            )  # TODO: see if this works for any (differentiable) classifier
            min_loss = torch.sum(cost - (lambda1 * constraint))

            min_optimiser.zero_grad()
            # self.clip_tensor_gradients_by_row(A, 0.5)
            # self.clip_tensor_gradients_by_row(O, 0.5)
            min_loss.backward()
            min_optimiser.step()

            # Track objective and constraints
            objective_list.append(cost.mean().item())
            constraint_list.append(constraint.mean().item())

            # Convergence criteria
            if (
                epoch > 100
                and np.std(objective_list[-20:]) < 1e-7
                and np.std(constraint_list[-20:]) < 1e-7
            ):
                break

            if epoch % 500 == 0 and verbose:
                vprint(
                    f"Epoch {epoch} | Objective: {objective_list[-1]} | Constraint: {constraint_list[-1]}"
                )

        # plt.plot(objective_list, label="Objective")
        # # second y axis
        # ax2 = plt.twinx()
        # ax2.plot(constraint_list, label="Constraint", color="orange")
        # plt.show()

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

            data = {f"X{i + 1}": X_recourse[:, i] for i in range(self.X.shape[1])}
            data.update({f"a{i + 1}": actions[:, i] for i in range(self.X.shape[1])})
            data.update(
                {f"order{i + 1}": action_order[:, i] for i in range(self.X.shape[1])}
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
    # X, scm = gen_toy_data(1000)

    # # Classifier
    # W_classifier = torch.tensor([-2, -3, -4], dtype=torch.float64)
    # b_classifier = 0.0

    # # Beta
    # beta = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)

    # W_adjacency = torch.tensor(
    #     [
    #         [0, 0.5, 0.35],
    #         [0, 0, 0.2],
    #         [0, 0, 0],
    #     ],
    #     dtype=torch.float64,
    # )

    N = 1000

    # Generate data from SCM
    SCM = NonLinearSCM(N)
    SCM.simulate_data()

    # Classify data
    y_pred, X_neg, clf = SCM.classify_data()
    N_neg = X_neg.shape[0]
    X_neg = torch.tensor(X_neg, dtype=torch.float64)

    print(f"{N_neg} negatively classified individuals")

    # Classification weights
    W_classifier = np.squeeze(clf.coef_)
    b_classifier = clf.intercept_[0]

    W_classifier = torch.tensor(W_classifier, dtype=torch.float64)
    print(W_classifier)
    print(b_classifier)

    # Ground truth beta
    beta = np.random.uniform(0, 1, size=(X_neg.shape))
    beta = beta / np.sum(beta, axis=1)[:, None]
    beta = torch.tensor(beta, dtype=torch.float64)

    W_adjacency = torch.tensor(
        [
            [0, 0.5, 0.35],
            [0, 0, 0.2],
            [0, 0, 0],
        ],
        dtype=torch.float64,
    )

    # Instantiate CausalRecourse
    cr = CausalRecourse(
        X=X_neg,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta=beta,
        use_scm=True,
        scm=SCM.scm,
        learn_ordering=True,
        W_adjacency=W_adjacency,
    )

    start = time.time()
    df = cr.gen_recourse(
        classifier_margin=0.02,
        max_epochs=4_000,
        verbose=True,
        lr=1e-2,
        format_as_df=True,
    )
    print(f"Time taken: {time.time() - start} seconds")
    print(f"Average cost: {df['cost'].mean()}")

    # Gen recourse
    print(df[["a1", "a2", "a3", "cost", "prob"]].head())
    print(f"Number of indivuduals negatively classified: {np.sum(df.prob < 0.5)}")
