import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, bernoulli


class StructuralCausalModel:
    """
    A simple class for simulating data from a structural causal model.
    """

    def __init__(self, N: int):
        """
        Initialize the class with the sample size.

        :param N (int): the sample size
        """
        self.N = N
        self.data = pd.DataFrame()
        self.relationships = {}
        self.distributions = {}
        self.outcomes = {}

    def add_variable(self, name: str, distribution=norm, **kwargs):
        """
        Add a variable to the model.

        :param name: (str) the name of the variable
        :param distribution: (scipy.stats distribution) the distribution to sample from
        :param kwargs: keyword arguments to pass to the distribution
        """
        self.data[name] = distribution.rvs(size=self.N, **kwargs)
        self.distributions[name] = {'distribution': distribution, 'kwargs': kwargs}

    def add_relationship(self, causes: dict, effect: str, noise_dist=norm, **kwargs):
        """
        Add a relationship between two variables.

        Parameters:
        causes (dict) : a dict of each cause and its weight
        effect (str): the effect variable
        weight (List[float]): a list of weights for each cause variable
        noise_dist (scipy.stats distribution): the distribution to sample noise from
        kwargs: keyword arguments to pass to the noise distribution
        """
        gen = (causes[cause] * self.data[cause] for cause in causes)
        self.data[effect] = sum(gen) + noise_dist.rvs(size=self.N, **kwargs)
        self.relationships[effect] = causes

    def add_binary_outcome(self, name: str, weights: dict, noise_dist=norm, **kwargs):
        """
        Add a binary outcome variable to the model.

        Parameters:
        name (str): the name of the binary variable
        weights (dict): dictionary of weights for each variable in the model
        noise_dist (scipy.stats distribution): the distribution to sample noise from
        kwargs: keyword arguments to pass to the noise distribution
        """
        noise = noise_dist.rvs(size=self.N, **kwargs)
        logit_p = np.sum([weight * self.data[var] for var, weight in weights.items()], axis=0) + noise
        p = expit(logit_p)
        self.data[name] = bernoulli.rvs(p)
        self.outcomes[name] = {'noise_dist': noise_dist, 'weights': weights, 'kwargs': kwargs}

    def append_data(self, df: pd.DataFrame, ids: pd.Series):
        """
        Append data to the model.

        :param df: the data to append
        :param ids: IDs to merge on
        """

        N_append = len(df)

        # Identify the columns that are not in the data and not the ID column
        cols_to_fill = [col for col in self.data.columns if ((col not in df.columns) and (col != 'ID'))]

        # Append on data
        self.data = pd.concat([self.data, df], ignore_index=True, axis=0)

        # Loop through each column not in the appended data and generate data accordingly
        for col in cols_to_fill:
            # If the column is a binary outcome, generate data from a Bernoulli distribution
            if col in self.distributions:
                # Use the value of col based on the ID
                self.data[col].iloc[-N_append:] = self.data[col][ids.values].values

            elif col in self.relationships:
                # If the column is a relationship, generate data from the relationship
                gen = (self.relationships[col][cause] * self.data[cause].iloc[-N_append:] for cause in
                       self.relationships[col])
                self.data[col].iloc[-N_append:] = sum(gen) + self.distributions[col]['distribution'].rvs(size=N_append, **self.distributions[col]['kwargs'])

            elif col in self.outcomes:
                # If the column is a binary outcome, generate data from a Bernoulli distribution
                noise = self.outcomes[col]['noise_dist'].rvs(size=N_append, **self.outcomes[col]['kwargs'])
                logit_p = np.sum(
                    [weight * self.data[var].iloc[-N_append:] for var, weight in self.outcomes[col]['weights'].items()],
                    axis=0) + noise
                p = expit(logit_p)
                self.data[col].iloc[-N_append:] = bernoulli.rvs(p)
            else:
                raise ValueError(f'Column {col} not found in the model.')

        # Append on IDs
        self.data['ID'].iloc[-N_append:] = ids.values

    def generate_data(self):
        """
        Return the generated data.

        Returns:
        data (pandas.DataFrame): the generated data
        """
        # Add an ID column
        self.data['ID'] = np.arange(self.N)
        # Sort columns alphabetically
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        return self.data

    def copy(self):
        """
        Return a copy of the model.
        """
        scm = StructuralCausalModel(self.N)
        scm.data = self.data.copy()
        scm.relationships = self.relationships.copy()
        scm.distributions = self.distributions.copy()
        scm.outcomes = self.outcomes.copy()
        return scm
