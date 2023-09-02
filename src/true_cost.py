import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from softsort import SoftSort

matplotlib.use("TkAgg")


class TrueCost:
    def __init__(
        self,
        X: torch.Tensor,
        X_final: torch.Tensor,
        W_adjacency: torch.Tensor,
        beta: torch.Tensor,
        learn_ordering: bool = False,
        sorter: SoftSort = None,
    ):
        self.X = X
        self.X_final = X_final
        self.W_adjacency = W_adjacency + torch.eye(W_adjacency.shape[0])
        self.beta = beta / torch.sum(beta)
        self.learn_ordering = learn_ordering
        self.sorter = sorter

        self.objective = (
            lambda A, O: self.loss_diff(A, O)
            if self.learn_ordering
            else self.loss_non_diff(A, O)
        )

    def loss_diff(self, A: torch.Tensor, O: torch.Tensor):
        # Init result tensors
        X_prime = self.X.clone()
        S = self.sorter(O)
        cost = torch.zeros(self.X.shape[0])

        for i in range(self.W_adjacency.shape[0]):
            cost += torch.sum(A**2 * S[:, i] * self.beta, dim=1)
            assert (cost >= 0).all(), "Cost should be positive"
            X_prime += (
                (self.W_adjacency.T * S[:, i].unsqueeze(-1)) @ A.unsqueeze(-1)
            ).squeeze(-1)

        return X_prime, cost

    def loss_non_diff(self, A: torch.Tensor, O: torch.Tensor):
        # Init result tensors
        X_prime = self.X.clone()
        S = torch.argsort(O)
        cost = torch.zeros(self.X.shape[0])
        A_ordered = torch.gather(A, 1, S)

        for i in range(self.W_adjacency.shape[0]):
            cost += A_ordered[:, i] ** 2 * self.beta[S[:, i]]
            assert (cost >= 0).all(), "Cost should be positive"
            X_prime += self.W_adjacency[S[:, i]] * A_ordered[:, i].unsqueeze(-1)

        return X_prime, cost

    def constraint(self, A: torch.Tensor, O: torch.Tensor):
        if self.learn_ordering:
            X_prime, cost = self.loss_diff(A, O)
        else:
            X_prime, cost = self.loss_non_diff(A, O)
        return torch.mean((X_prime - self.X_final) ** 2, dim=1)

    @staticmethod
    def clip_tensor_gradients_by_row(tensor, max_norm):
        """Vectorized gradient clipping by row for a single tensor."""
        if tensor.grad is not None:
            with torch.no_grad():
                grad_norms = tensor.grad.norm(p=2, dim=1, keepdim=True)
                clip_coef = torch.clamp(max_norm / (grad_norms + 1e-6), max=1)
                tensor.grad.mul_(clip_coef)

    def eval_true_cost_constrained(self, lr, max_epochs, verbose=False):
        # Random initialisation of A and O
        A = torch.rand(size=self.X.shape, requires_grad=True, dtype=torch.float64)
        O = torch.rand(size=self.X.shape, requires_grad=True, dtype=torch.float64)

        # Initialise lambda
        lambda1 = torch.rand(self.X.shape[0], requires_grad=True, dtype=torch.float64)
        lambda2 = torch.rand(self.X.shape[0], requires_grad=True, dtype=torch.float64)

        # Init optimisers
        min_optimiser = optim.AdamW([A, O], lr=lr)
        max_optimiser = optim.AdamW([lambda1], lr=lr)

        # Init lists
        objective_list = []
        constraint_list = []

        # Loop through epochs
        for epoch in range(max_epochs):
            max_loss = torch.sum(
                (lambda1 * self.constraint(A, O)) - self.objective(A, O)[1]
            )
            max_optimiser.zero_grad()
            max_loss.backward()
            max_optimiser.step()

            min_loss = torch.sum(
                self.objective(A, O)[1] - (lambda1 * self.constraint(A, O))
            )
            min_optimiser.zero_grad()
            self.clip_tensor_gradients_by_row(A, 0.5)
            self.clip_tensor_gradients_by_row(O, 0.5)
            min_loss.backward()
            min_optimiser.step()

            objective_list.append(self.objective(A, O)[1].mean().detach())
            constraint_list.append(self.constraint(A, O).mean().detach())

            if (
                epoch > 100
                and np.std(objective_list[-10:]) < 1e-6
                and np.std(constraint_list[-10:]) < 1e-6
            ):
                break

            if verbose and epoch % 100 == 0:
                print(
                    f"Epoch: {epoch} | Objective: {objective_list[-1].mean()} | Constraint: {constraint_list[-1].mean()}"
                )

        print(f"Final actions: {A}")
        print(f"Final ordering: {O}")

        print(f"Beginning X: {self.X}")

        print(f"Final X: {self.X_final}")
        print(f"Learned X: {self.loss_diff(A, O)[0]}")

        print(f"Weights: {self.W_adjacency}")

        print(f"Final cost: {self.loss_diff(A, O)[1]}")

        return objective_list[-1]

    def eval_true_cost(self, lr, max_epochs, verbose=False):
        # lambda function to print only if verbose is True
        vprint = print if verbose else lambda *a, **k: None

        # Random initialisation of A and O
        A = torch.rand(size=self.X.shape, requires_grad=True, dtype=torch.float64)
        O = torch.rand(size=self.X.shape, requires_grad=True, dtype=torch.float64)

        # Init optimisers
        optimiser = optim.SGD([A, O], lr=lr)

        # Init lists
        objective_list = []

        # Loop through epochs
        for epoch in range(max_epochs):
            loss = self.constraint(A, O).sum()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            objective_list.append(loss.detach().item())

            if epoch > 100 and np.std(objective_list[-20:]) < 1e-6:
                vprint(f"Converged at epoch {epoch}")
                break

            if epoch % 100 == 0:
                vprint(f"Epoch: {epoch} | Objective: {objective_list[-1]}")

        # vprint(f"Final actions: {A}")
        # vprint(f"Final ordering: {O}")

        # vprint(f"Beginning X: {self.X}")

        vprint(f"Final X: {self.X_final}")
        vprint(f"Learned X: {self.loss_diff(A, O)[0]}")

        # vprint(f"Weights: {self.W_adjacency}")

        # vprint(f"Final cost: {self.loss_diff(A, O)[1]}")

        return self.loss_diff(A, O)[1]


if __name__ == "__main__":
    X = torch.rand(100, 2, dtype=torch.float64)
    X_final = torch.rand(100, 2, dtype=torch.float64) + 3
    W_adjacency = torch.rand(2, 2, dtype=torch.float64)
    # set diagonal to 0
    W_adjacency = W_adjacency * (
        1 - torch.eye(W_adjacency.shape[0], dtype=torch.float64)
    )
    # set lower triangle to 0
    W_adjacency = torch.triu(W_adjacency)
    beta = torch.rand(2, dtype=torch.float64)
    sorter = SoftSort(tau=0.1, hard=True)

    true_cost = TrueCost(
        X, X_final, W_adjacency, beta, learn_ordering=False, sorter=sorter
    )

    # ATTEMPT 1
    cost = true_cost.eval_true_cost_constrained(lr=0.01, max_epochs=1_000, verbose=True)

    # ATTEMPT 2
    cost = true_cost.eval_true_cost(lr=0.01, max_epochs=1_000, verbose=True)
