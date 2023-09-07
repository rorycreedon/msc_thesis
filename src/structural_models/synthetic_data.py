import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from typing import Union
from sympy import *
import torch

from src.structural_models.structural_causal_model import StructuralCausalModel


class SimpleSCM:
    def __init__(self, N):
        self.N = N
        self.scm = None
        self.X = None
        self.y = None

    def simulate_data(self) -> None:
        """
        Simulate data from a structural causal model.
        :return: None
        """
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

        outcome_weights = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)

        self.scm = StructuralCausalModel(endog_vars, exog_vars, equations)
        self.X, self.y = self.scm.generate_data(
            N=self.N,
            distribution_list=distribution_list,
            outcome_weights=outcome_weights,
        )

    def classify_data(self):
        """
        Train a classifier on the data.
        :return: LogisticRegression object
        """
        clf = LogisticRegression().fit(self.X.numpy(), self.y.numpy())
        y_pred = clf.predict(self.X.numpy())
        X_neg = self.X.numpy()[y_pred == 0]
        return y_pred, X_neg, clf


class NonLinearSCM:
    def __init__(self, N):
        self.N = N
        self.scm = None
        self.X = None
        self.y = None

    def simulate_data(self) -> None:
        """
        Simulate data from a structural causal model.
        :return: None
        """
        # Start with SCM
        x1, x2, x3, x4, x5, u1, u2, u3, u4, u5 = symbols(
            "x1 x2 x3 x4 x5 u1 u2 u3 u4 u5"
        )

        # Define the equations
        eq1 = Eq(x1, u1)
        eq2 = Eq(x2, u2 - 2 / (1 + x1**2))
        eq3 = Eq(x3, u3 + 0.2 * x1 + 0.01 * x2**2)
        eq4 = Eq(x4, u4 + 2 * x3 + 3 * x1 * x2)
        eq5 = Eq(x5, u5 + 0.4 * x1)

        equations = [eq1, eq2, eq3, eq4, eq5]

        endog_vars = [x1, x2, x3, x4, x5]
        exog_vars = [u1, u2, u3, u4, u5]

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
            {
                "name": "u4",
                "distribution": "Normal",
                "params": {"loc": 0, "scale": 0.5},
            },
            {
                "name": "u5",
                "distribution": "Normal",
                "params": {"loc": 0, "scale": 0.5},
            },
        ]

        outcome_weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)

        self.scm = StructuralCausalModel(endog_vars, exog_vars, equations)
        self.X, self.y = self.scm.generate_data(
            N=self.N,
            distribution_list=distribution_list,
            outcome_weights=outcome_weights,
        )

    def classify_data(self):
        """
        Train a classifier on the data.
        :return: LogisticRegression object
        """
        clf = LogisticRegression().fit(self.X.numpy(), self.y.numpy())
        y_pred = clf.predict(self.X.numpy())
        X_neg = self.X.numpy()[y_pred == 0]
        return y_pred, X_neg, clf
