import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, bernoulli


class StructuralCausalModel:
    """
    A class for simulating data from a structural causal model and managing updates to the data.
    """

    def __init__(self, N: int):
        """
        Initialize the class with the sample size.
        :param N (int): sample size
        """
        self.N = N
        self.data = pd.DataFrame()
        self.relationships = {}
        self.distributions = {}
        self.outcomes = {}
        self.predictions = {}

    def add_variable(self, name: str, distribution: norm, **kwargs):
        """
        Add a variable to the model.
        :param name: (str) the name of the variable
        :param distribution: (scipy.stats distribution) the distribution to sample from
        :param kwargs: keyword arguments to pass to the distribution
        """
        self.data[name] = distribution.rvs(size=self.N, **kwargs)
        self.distributions[name] = {"distribution": distribution, "kwargs": kwargs}

    def add_relationship(self, causes: dict, effect: str, noise_dist=norm, **kwargs):
        """
        Add a relationship between two variables.
        :param causes : a dict of each cause and its weight
        :param effect: the effect variable
        :param noise_dist: the distribution to sample noise from (scipy.stats distribution)
        :param kwargs: keyword arguments to pass to the noise distribution
        :return: None
        """
        gen = (causes[cause] * self.data[cause] for cause in causes)
        self.data[effect] = sum(gen) + noise_dist.rvs(size=self.N, **kwargs)
        self.relationships[effect] = causes

    def add_binary_outcome(self, name: str, weights: dict, noise_dist=norm, **kwargs):
        """
        Add a binary outcome variable to the model.
        :param name: the name of the binary variable
        :param weights: dictionary of weights for each variable in the model
        :param noise_dist: the distribution to sample noise from (scipy.stats distribution)
        :param kwargs: keyword arguments to pass to the noise distribution
        """
        # Calculate probabilities
        noise = noise_dist.rvs(size=self.N, **kwargs)
        logit_p = (
            np.sum([weight * self.data[var] for var, weight in weights.items()], axis=0)
            + noise
        )
        p = expit(logit_p)

        # Sample from a Bernoulli distribution
        self.data[name] = bernoulli.rvs(p)
        self.outcomes[name] = {
            "noise_dist": noise_dist,
            "weights": weights,
            "kwargs": kwargs,
        }

    def append_data(self, df: pd.DataFrame, ids: pd.Series):
        """
        Append data to the model.
        :param df: the data to append
        :param ids: IDs to merge on
        """

        N_append = len(df)

        # Identify the columns that are not in the data and not the ID column
        cols_to_fill = [
            col
            for col in self.data.columns
            if (
                (col not in df.columns)
                and (col != "ID")
                and (col != "recourse_eligible")
            )
        ]

        # Append on data
        self.data = pd.concat([self.data, df], ignore_index=True, axis=0)

        # Loop through each column not in the appended data and generate data accordingly
        for col in cols_to_fill:
            # If the column is a binary outcome, generate data from a Bernoulli distribution
            if col in self.distributions:
                # Use the value of col based on the ID
                self.data[col].iloc[-N_append:] = self.data.loc[
                    self.data["ID"].isin(ids.values)
                ][col].values

            elif col in self.relationships:
                # If the column is a relationship, generate data from the relationship
                gen = (
                    self.relationships[col][cause] * self.data[cause].iloc[-N_append:]
                    for cause in self.relationships[col]
                )
                self.data[col].iloc[-N_append:] = sum(gen) + self.distributions[col][
                    "distribution"
                ].rvs(size=N_append, **self.distributions[col]["kwargs"])

            elif col in self.outcomes:
                # If the column is a binary outcome, generate data from a Bernoulli distribution
                noise = self.outcomes[col]["noise_dist"].rvs(
                    size=N_append, **self.outcomes[col]["kwargs"]
                )
                logit_p = (
                    np.sum(
                        [
                            weight * self.data[var].iloc[-N_append:]
                            for var, weight in self.outcomes[col]["weights"].items()
                        ],
                        axis=0,
                    )
                    + noise
                )
                p = expit(logit_p)
                assert min(p) > 0 and max(p) < 1, f"p is not between 0 and 1: {p}"
                self.data[col].iloc[-N_append:] = bernoulli.rvs(p)

            else:
                raise ValueError(f"Column {col} not found in the model.")

        # Old IDs now not eligible for recourse
        self.data["recourse_eligible"].loc[self.data["ID"].isin(ids.values)] = 0

        # Append on IDs
        self.data["ID"].iloc[-N_append:] = ids.values

        # Append on recourse eligibility
        self.data["recourse_eligible"].iloc[-N_append:] = 1

    def generate_data(self):
        """
        Return the generated data.
        :return data (pandas.DataFrame): the generated data
        """
        # Add an ID column
        self.data["ID"] = np.arange(self.N)
        self.data["recourse_eligible"] = 1

        # Sort columns alphabetically
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        return self.data

    def copy(self):
        """
        Return a copy of the model.
        :return: the model (object)
        """
        scm = StructuralCausalModel(self.N)
        scm.data = self.data.copy()
        scm.relationships = self.relationships.copy()
        scm.distributions = self.distributions.copy()
        scm.outcomes = self.outcomes.copy()
        return scm

    def get_parents(self, col, root_only=True, i=1):
        """
        Return the full relationship for a given column.
        :param col: column name
        :param root_only: whether to return only root nodes
        :param i: counter for weights (do not call)
        :return: dictionary of weights of root nodes
        """
        if root_only:
            weights = {}
            if col in self.distributions.keys():
                weights[col] = i
                return weights

            elif col in self.relationships.keys():
                for parent in self.relationships[col].keys():
                    weights.update(
                        self.get_parents(parent, i * self.relationships[col][parent])
                    )

            elif col in self.outcomes.keys():
                for parent in self.outcomes[col]["weights"].keys():
                    weights.update(
                        self.get_parents(
                            parent, i * self.outcomes[col]["weights"][parent]
                        )
                    )

            else:
                raise ValueError("Column not found in SCM")

            return weights

        else:
            if col in self.distributions.keys():
                return {col: 1}

            elif col in self.relationships.keys():
                return self.relationships[col]

            elif col in self.outcomes.keys():
                return self.outcomes[col]["weights"]

            else:
                raise ValueError(f"Column {col} not found in SCM")
