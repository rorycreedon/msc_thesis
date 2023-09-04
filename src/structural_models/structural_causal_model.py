import numpy as np
import pandas as pd
from sympy import symbols, Eq, solve
import sympytorch
import torch
import torch.distributions as dists
from typing import List


class StructuralCausalModel:
    def __init__(self, endog_vars: List, exog_vars: List, equations: List) -> None:
        """
        Initialize the class.
        :param endog_vars: Endogenous variables (X)
        :param exog_vars: Exogenous variables (U)
        :param equations: List of sympy equations relating X and U
        """
        self.equations = equations
        self.endog_vars = endog_vars
        self.exog_vars = exog_vars

        # Variables values
        self.U = None
        self.X = None

        # Solve for abduction equations
        solution = solve(self.equations, self.exog_vars, dict=True)
        assert len(solution) == 1
        self.abduction_expressions = []
        for key, item in solution[0].items():
            self.abduction_expressions.append(item)

        # Abduction mod
        self.abduction_mod = sympytorch.SymPyModule(
            expressions=self.abduction_expressions
        )

        # Solve for prediction equations
        solution = solve(self.equations, self.endog_vars, dict=True)
        self.prediction_equations = []
        for key, item in solution[0].items():
            self.prediction_equations.append(item)

        # Prediction mod
        self.prediction_mod = sympytorch.SymPyModule(
            expressions=self.prediction_equations
        )

    def generate_data(
        self, N: int, distribution_list: List, outcome_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate data from the SCM.
        :param N: the number of samples to generate
        :return: the generated data
        """
        samples = []

        for var in distribution_list:
            # Get the distribution class dynamically
            distribution_class = getattr(dists, var["distribution"])

            # Instantiate the distribution using the provided parameters
            distribution_instance = distribution_class(**var["params"])

            # Sample and append to the samples list
            samples_for_this_distribution = distribution_instance.sample((N,))
            samples.append(samples_for_this_distribution)

        self.U = torch.stack(samples, dim=1)
        self.X = self.prediction(self.U).to(torch.float64).detach()

        # Calculate y
        y = torch.bernoulli(torch.sigmoid(self.X @ outcome_weights))

        return self.X, y

    def abduction(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform abduction on the SCM.
        :param X: the observed data
        :return: the unobserved variables U
        """
        args = {}
        for i, var in enumerate(self.endog_vars):
            args[f"x{i+1}"] = X[:, i]
        return self.abduction_mod(**args)

    def prediction(self, U: torch.Tensor) -> torch.Tensor:
        """
        Perform prediction on the SCM.
        :param U: the unobserved variables
        :return: the predicted data X
        """
        args = {}
        for i, var in enumerate(self.exog_vars):
            args[f"u{i+1}"] = U[:, i]
        return self.prediction_mod(**args)


if __name__ == "__main__":
    # Start with SCM
    x1, x2, x3, u1, u2, u3 = symbols("x1 x2 x3 u1 u2 u3")

    # Define the equations
    eq1 = Eq(x1, u1)
    eq2 = Eq(x2, u2 + 0.5 * x1)
    eq3 = Eq(x3, u3 + 0.2 * x1 + 0.3 * x2)

    equations = [eq1, eq2, eq3]

    endog_vars = [x1, x2, x3]
    exog_vars = [u1, u2, u3]

    distribution_list = [
        {
            "name": "u1",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 1},
        },
        {
            "name": "u2",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 1},
        },
        {
            "name": "u3",
            "distribution": "Normal",
            "params": {"loc": 0, "scale": 0.5},
        },
    ]

    scm = StructuralCausalModel(endog_vars, exog_vars, equations)
    print(
        scm.generate_data(
            N=1000,
            distribution_list=distribution_list,
            outcome_weights=torch.tensor(
                [
                    0.1,
                    0.2,
                    0.3,
                ],
                dtype=torch.float64,
            ),
        )[1]
    )

    U_intervention = scm.U + torch.tensor([0, 0.2, 0])
    print(scm.prediction(U_intervention))
