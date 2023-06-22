import numpy as np
import pandas as pd
import numpy as np
from multiprocessing import Pool
import cvxpy as cp


class Recourse:
    """
    A class containing a simple algorithm for calculating recourse.
    """

    def __init__(self, X: pd.DataFrame, y_pred, clf, A):
        """
        Initialize the class with the data and the model coefficients.
        :param X (pd.DataFrame): the data
        :param y_pred (pd.Series): the predicted labels
        :param clf (sklearn classifier): the model
        """
        self.X = X[y_pred == 0]
        self.y_pred = y_pred
        self.clf = clf
        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]
        self.partial_recourse = None
        self.X_negative = None
        self.recourse = None
        self.A = A

    def _opt_full_logistic(self, cost_function: str = 'quadratic'):

        # Convex optimisation
        x = cp.Variable(self.X.shape)
        if cost_function == 'quadratic':
            expr = cp.sum_squares(x - self.X)
        elif cost_function == 'quad_form':
            expr = 0
            quad_forms = [cp.quad_form(x[i] - self.X.iloc[i], self.A) for i in range(self.X.shape[0])]
            expr = cp.sum(quad_forms)
        else:
            raise ValueError(f"{cost_function} not recognised")
        objective = cp.Minimize(expr)
        constraints = [cp.matmul(x, self.weights) + self.bias == 0]
        prob = cp.Problem(objective, constraints)
        prob.solve()

        if prob.status != "optimal":
            raise ValueError(f"Optimisation not optimal: {prob.status}")

        # Calculate costs
        if cost_function == 'quadratic':
            costs = np.sum((x.value - self.X.values) ** 2, axis=1)
        elif cost_function == 'quad_form':
            costs = np.array([cp.quad_form(x[i].value - self.X.iloc[i], self.A).value for i in range(self.X.shape[0])])
        recourse_x = x.value
        recourse_prob = self.clf.predict_proba(x.value)[:, 1]

        return costs, recourse_x, recourse_prob

    def _opt_partial_logistic(self, C, row, cost_function='quadratic'):

        def attempt_optimisation(C, row, cost_function, solver="ECOS", verbose=False):
            x = cp.Variable(self.X_negative.shape[1])
            objective = cp.Maximize(cp.matmul(x, self.weights) + self.bias)
            if cost_function == "quadratic":
                expr = cp.sum_squares(x - self.X_negative.iloc[row].values)
            elif cost_function == "quad_form":
                expr = cp.quad_form(x - self.X_negative.iloc[row].values, self.A)
            else:
                raise ValueError(f"{cost_function} not recognised")

            constraints = [expr <= C]
            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=verbose, solver=solver)

            return x, prob, expr

        x, prob, expr = attempt_optimisation(C, row, cost_function, verbose=False)

        # Try again with verbose=True and a different solver if not optimal
        if prob.status != "optimal":
            x, prob, expr = attempt_optimisation(C, row, cost_function, verbose=True, solver='SCS')
            if prob.status != "optimal":
                raise ValueError(f"Optimisation not optimal: {prob.status}")

        cost = expr.value

        return x.value, cost, self.clf.predict_proba(x.value.reshape(1, -1))[0][1]

    def optimise_full_recourse(self, C: float, cost_function: str = 'quadratic'):
        """
        Optimise the recourse problem.
        :param cost_function: type of cost function to use
        :param C: the cost threshold
        :return: None
        """

        recourse_cost, recourse_x, recourse_prob = self._opt_full_logistic(cost_function = cost_function)

        self.recourse = pd.DataFrame(recourse_x, columns=self.X.columns)
        self.recourse['cost'] = recourse_cost
        self.recourse['prob'] = recourse_prob
        self.recourse['Y'] = (self.recourse['cost'] <= C).astype(float)
        self.recourse.index = self.X.index

    def optimise_partial_recourse(self, C: float, cost_function: str = 'quadratic'):
        """
        Optimise the partial recourse problem.
        :param C: Cost threshold
        :param re_tries: Max number of re-tries
        :param cost_function: Type of cost function
        :return: None
        """

        # Limit to just the individuals with a cost greater than C or NaN
        self.X_negative = self.X[np.logical_or((self.recourse['cost'] > C).values, self.recourse['cost'].isna().values)]

        if len(self.X_negative) == 0:
            print("No individuals with cost greater than C")
            return None

        func_args = [(C, row, cost_function) for row in range(self.X_negative.shape[0])]
        with Pool() as pool:
            results = pool.starmap(self._opt_partial_logistic, func_args)

        # List to store recourse
        recourse_x = []
        recourse_cost = []
        recourse_prob = []

        for res in results:
            recourse_x.append(res[0])
            recourse_cost.append(res[1])
            recourse_prob.append(res[2])

        self.partial_recourse = pd.DataFrame(recourse_x, columns=self.X_negative.columns)
        self.partial_recourse['cost'] = recourse_cost
        self.partial_recourse['prob'] = recourse_prob
        eps = 1e-6
        self.partial_recourse['Y'] = (self.partial_recourse['prob'] >= (0.5 - eps)).astype(float)
        self.partial_recourse.index = self.X_negative.index

    def compute_recourse(self, C: float, partial_recourse: bool = False, cost_function: str = 'quadratic'):
        """
        Compute full recourse
        :param cost_function: Type of cost function to use
        :param C: Cost threshold for partial recourse
        :param partial_recourse: Whether to compute partial recourse
        :return: None
        """

        # Compute recourse
        self.optimise_full_recourse(C=C, cost_function=cost_function)
        if partial_recourse:
            self.optimise_partial_recourse(C=C, cost_function=cost_function)
            # Merge in partial recourse
            if len(self.X_negative) > 0:
                self.recourse.update(self.partial_recourse)
