import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from typing import Union

from structural_models import StructuralCausalModel


class SimpleSCM:
    def __init__(self, N):
        self.N = N
        self.scm = None

    def simulate_data(self) -> None:
        """
        Simulate data from a structural causal model.
        :return: StructuralCausalModel object
        """
        # Define the SCM
        self.scm = StructuralCausalModel(self.N)

        # Fist variable is a normal distribution
        self.scm.add_variable(name="X1", distribution=norm, loc=0, scale=1)

        # Unobserved variable U2
        self.scm.add_variable(name="U2", distribution=norm, loc=0, scale=1)

        # X1 and U cause X2
        self.scm.add_relationship(
            causes={"X1": 0.5, "U2": 1}, effect="X2", noise_dist=norm, loc=0, scale=0
        )

        # Unobserved variable U3
        self.scm.add_variable(name="U3", distribution=norm, loc=0, scale=0.5)

        # X1, X2, and U3 cause X3
        self.scm.add_relationship(
            causes={"X1": 0.2, "X2": 0.3, "U3": 1},
            effect="X3",
            noise_dist=norm,
            loc=0,
            scale=0,
        )

        # X1 and X2 causes Y
        self.scm.add_binary_outcome(
            name="Y_true",
            weights={"X1": 0.1, "X2": 0.2, "X3": 0.3},
            noise_dist=norm,
            loc=0,
            scale=0,
        )

    def classify_data(self) -> Union[pd.Series, pd.DataFrame, LogisticRegression]:
        """
        Train a classifier on the data.
        :return: LogisticRegression object
        """
        clf = LogisticRegression(random_state=0).fit(
            self.scm.data[["X1", "X2", "X3"]], self.scm.data["Y_true"]
        )
        y_pred = clf.predict(self.scm.data[["X1", "X2", "X3"]])
        X_neg = self.scm.data.loc[y_pred == 0, ["X1", "X2", "X3"]]
        return y_pred, X_neg, clf

    def gen_weighted_adjacency_matrix(self) -> np.ndarray:
        """
        Generate a weighted adjacency matrix from the SCM.
        :return: W_adjacency
        """
        return self.scm.generate_adjacency_matrix(["X1", "X2", "X3"])


class FourVariableSCM:
    def __init__(self, N):
        self.N = N
        self.scm = None

    def simulate_data(self) -> None:
        """
        Simulate data from a structural causal model.
        :return: StructuralCausalModel object
        """
        # Define the SCM
        self.scm = StructuralCausalModel(self.N)

        # # Fist variable is a normal distribution
        self.scm.add_variable(name="X1", distribution=norm, loc=0, scale=2)

        # X1 causes X2
        self.scm.add_relationship(
            causes={"X1": 3}, effect="X2", noise_dist=norm, loc=1, scale=1
        )

        # There exists an unobserved variable U
        self.scm.add_variable(name="U", distribution=norm, loc=0, scale=1)

        # Y is caused by X2 and U
        self.scm.add_binary_outcome(
            name="Y_true",
            weights={"X2": 0.5, "U": -0.5},
            noise_dist=norm,
            loc=0,
            scale=0,
        )

        # X3 is caused by X1
        self.scm.add_relationship(
            causes={"X1": 2}, effect="X3", noise_dist=norm, loc=0, scale=0.25
        )

        # X4 is caused by Y
        self.scm.add_relationship(
            causes={"Y_true": 0.75}, effect="X4", noise_dist=norm, loc=0, scale=0.5
        )

    def classify_data(self) -> Union[pd.Series, pd.DataFrame, LogisticRegression]:
        """
        Train a classifier on the data.
        :return: LogisticRegression object
        """
        clf = LogisticRegression(random_state=0).fit(
            self.scm.data[["X1", "X2", "X3", "X4"]], self.scm.data["Y_true"]
        )
        y_pred = clf.predict(self.scm.data[["X1", "X2", "X3", "X4"]])
        X_neg = self.scm.data.loc[y_pred == 0, ["X1", "X2", "X3", "X4"]]
        return y_pred, X_neg, clf

    def gen_weighted_adjacency_matrix(self) -> np.ndarray:
        """
        Generate a weighted adjacency matrix from the SCM.
        :return: W_adjacency
        """
        return self.scm.generate_adjacency_matrix(["X1", "X2", "X3", "X4"])
