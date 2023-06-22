import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from multiprocessing import Pool


class Recourse:
    """
    A class containing a simple algorithm for calculating recourse.
    """

    def __init__(self, X: pd.DataFrame, y_pred, clf):
        """
        Initialize the class with the data and the model coefficients.
        :param X (pd.DataFrame): the data
        :param y_pred (pd.Series): the predicted labels
        :param clf (sklearn classifier): the model
        """
        self.cdf_functions = []
        self.X = X[y_pred == 0]
        self.X_all = X
        self.y_pred = y_pred
        self.clf = clf
        self.original_prob = clf.predict_proba(X[y_pred == 0].values)[:, 0]
        self.partial_recourse = None
        self.original_prob_negative = None
        self.X_negative = None
        self.recourse = None

    @staticmethod
    def fit_kde_cdf(col):
        """
        Creates a lambda function to calculate CDF of Gaussian KDE for a given 1D array.

        :param col : the column to fit the KDE to
        :return: a lambda function to calculate the CDF of the KDE
        """
        return lambda x: gaussian_kde(col).integrate_box_1d(-np.inf, x)

    def cost(self, x_prime, row: int, type='quadratic'):
        """
        Cost function for the optimisation problem.
        :param type: type of cost function to use
        :param x_prime: Array of values for which cost is being calculated
        :param row: Index of individual for which cost is being calculated
        :return: Cost of moving to point x_prime
        """
        if type == 'kde':
            cost = 0
            for i in range(len(x_prime)):
                cost += np.abs(self.cdf_functions[i](x_prime[i]) - self.cdf_functions[i](self.X.iloc[row].values[i]))
            return cost

        elif type == 'quadratic':
            return np.sum((x_prime - self.X.iloc[row].values) ** 2)

    def constraint(self, x):
        """
        Constraint function for the optimisation problem.
        :param x: data point to be optimised
        :return: the value of the constraint
        """
        return self.clf.predict_proba(x.reshape(1, -1))[0][1] - 0.5

    def _scipy_opt_full(self, row, re_tries, recourse_x, recourse_cost, recourse_prob):
        x0 = self.X.iloc[row, :].values
        cons = {'type': 'eq', 'fun': self.constraint}
        res = minimize(self.cost, x0, args=row, constraints=cons)
        i = 0
        while (res.success is False) and (i <= re_tries):  # try 10 times and then give up
            res = minimize(self.cost, x0, args={'row': row, 'type': 'quadratic'}, constraints=cons)
            i += 1
        if res.success:
            recourse_x.append(res.x)
            recourse_cost.append(res.fun)
            recourse_prob.append(self.clf.predict_proba(res.x.reshape(1, -1))[0][1])

        else:
            recourse_x.append(self.X.iloc[row].values)
            recourse_cost.append(np.nan)
            recourse_prob.append(np.nan)

        return recourse_x, recourse_cost, recourse_prob

    def optimise_full_recourse(self, C: float, verbose: bool, re_tries: int):
        """
        Optimise the recourse problem.
        :param C: the cost threshold
        :param verbose: Whether to show progress bar
        :param re_tries: Number of times to retry if optimisation unsuccessful
        :return: None
        """

        # List to store recourse
        recourse_x = []
        recourse_cost = []
        recourse_prob = []

        if verbose:
            print("Calculating full recourse...")

        func_args = [(row, re_tries, recourse_x, recourse_cost, recourse_prob) for row in range(self.X.shape[0])]
        with Pool() as pool:
            results = pool.starmap(self._scipy_opt_full, func_args)

        for res in results:
            recourse_x.append(res[0][0])
            recourse_cost.append(res[1][0])
            recourse_prob.append(res[2][0])

        self.recourse = pd.DataFrame(recourse_x, columns=self.X.columns)
        self.recourse['cost'] = recourse_cost
        self.recourse['prob'] = recourse_prob
        self.recourse['Y'] = (self.recourse['cost'] <= C).astype(float)
        self.recourse.index = self.X.index

    def partial_objective(self, x_prime):
        """
        Objective function for the partial recourse optimisation problem - probability of being classified as 0.
        :param x_prime: data point to be optimised
        :return: Probability of being classified as 0
        """
        return self.clf.predict_proba(x_prime.reshape(1, -1))[0][0]

    def optimise_partial_recourse(self, C: float, verbose: bool, re_tries: int):

        # Limit to just the individuals with a cost greater than C or NaN
        self.X_negative = self.X[np.logical_or((self.recourse['cost'] > C).values, self.recourse['cost'].isna().values)]
        self.original_prob_negative = self.original_prob[
            np.logical_or((self.recourse['cost'] > C).values, self.recourse['cost'].isna().values)]

        if len(self.X_negative) == 0:
            if verbose:
                print("No individuals with cost greater than C")
            return None

        if verbose:
            print("Calculating partial recourse...")

        func_args = [(row, re_tries, C) for row in range(self.X_negative.shape[0])]
        with Pool() as pool:
            results = pool.starmap(self._scipy_opt_partial, func_args)

        # List to store recourse
        recourse_x = []
        recourse_cost = []
        recourse_prob = []

        for res in results:
            recourse_x.append(res[0][0])
            recourse_cost.append(res[1][0])
            recourse_prob.append(res[2][0])

        self.partial_recourse = pd.DataFrame(recourse_x, columns=self.X_negative.columns)
        self.partial_recourse['cost'] = recourse_cost
        self.partial_recourse['prob'] = recourse_prob
        eps = 1e-6
        self.partial_recourse['Y'] = (self.partial_recourse['prob'] >= (0.5 - eps)).astype(float)
        self.partial_recourse.index = self.X_negative.index

    def _scipy_opt_partial(self, row, re_tries, C):

        # List to store recourse
        recourse_x = []
        recourse_cost = []
        recourse_prob = []

        x0 = self.X.iloc[row, :].values
        cons = ({'type': 'ineq', 'fun': lambda x: C - self.cost(x, row, type='quadratic')},
                {'type': 'ineq', 'fun': lambda x: self.partial_objective(x) - 0.5},
                {'type': 'ineq', 'fun': lambda x: self.original_prob_negative[row] - self.partial_objective(x)})
        res = minimize(self.partial_objective, x0, constraints=cons)
        i = 0
        while (res.success is False) and (i <= re_tries):  # try 10 times and then give up
            i += 1
        if res.success:
            recourse_x.append(res.x)
            recourse_cost.append(self.cost(res.x, row))
            recourse_prob.append(self.clf.predict_proba(res.x.reshape(1, -1))[0][1])

        if not res.success:
            recourse_x.append(self.X_negative.iloc[row].values)
            recourse_cost.append(np.nan)
            recourse_prob.append(np.nan)

        return recourse_x, recourse_cost, recourse_prob

    def compute_recourse(self, C: float, partial_recourse: bool = False, verbose: bool = False, re_tries: int = 2):
        """
        Compute full recourse
        :param C: Cost threshold for partial recourse
        :param partial_recourse: Whether to compute partial recourse
        :param verbose: Whether to show progress bar
        :param re_tries: Number of times to retry if optimisation unsuccessful
        :return: None
        """
        # Fit KDE
        # for col in self.X.columns:
        #     self.fit_kde_cdf(self.X[col])

        # Make CDF functions
        # for col in self.X.columns:
        #     self.cdf_functions.append(self.fit_kde_cdf(self.X_all[col]))

        # Compute recourse
        self.optimise_full_recourse(C, verbose, re_tries)
        if partial_recourse:
            self.optimise_partial_recourse(C, verbose, re_tries)
            # Merge in partial recourse
            if len(self.X_negative) > 0:
                self.recourse.update(self.partial_recourse)
