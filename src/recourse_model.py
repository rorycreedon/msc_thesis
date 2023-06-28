import pandas as pd
import numpy as np
from multiprocessing import Pool
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB


class Recourse:
    """
    A class containing a simple algorithm for calculating recourse.
    """

    def __init__(self, X: pd.DataFrame, clf, A: np.array = None):
        """
        Initialize the class with the data and the model coefficients.
        :param X (pd.DataFrame): the negatively classified data
        :param clf (sklearn classifier): the model
        :param A (np.array): a PSD matrix
        """
        self.X = X
        self.clf = clf
        self.weights = clf.coef_[0]
        self.bias = clf.intercept_[0]
        self.partial_recourse = None
        self.X_negative = None
        self.recourse = None
        if A is None:
            self.A = np.linalg.inv(X.cov())
        else:
            self.A = A

    def _opt_full_logistic(
        self, cost_function: str = "quadratic", backend: str = "cvxpy"
    ):
        """
        Internal optimisation function for full recourse function
        :param cost_function: Cost function to use
        :param backend: Backend to use
        :return: List of costs for each row, recourse values, recourse probabilities
        """
        if backend == "cvxpy":
            x = cp.Variable(self.X.shape)

            # Define expression to minimise
            if cost_function == "quadratic":
                expr = cp.sum_squares(x - self.X)

            elif cost_function == "quad_form":
                quad_forms = [
                    cp.quad_form(x[i] - self.X.iloc[i], self.A)
                    for i in range(self.X.shape[0])
                ]
                expr = cp.sum(quad_forms)

            else:
                raise ValueError(f"{cost_function} not recognised")

            # Optimisation
            objective = cp.Minimize(expr)
            constraints = [cp.matmul(x, self.weights) + self.bias >= 0]

            prob = cp.Problem(objective, constraints)
            prob.solve(verbose=False)

            recourse_x = x.value

            if prob.status != "optimal":
                raise ValueError(f"Optimisation not optimal: {prob.status}")

        elif backend == "gurobi":
            # Turn off output
            gp.setParam("OutputFlag", 0)
            gp.setParam("NonConvex", 0)

            # Create model
            model = gp.Model("full_recourse")

            # Define variables
            X = model.addMVar(
                shape=self.X.shape,
                vtype=GRB.CONTINUOUS,
                name="x0",
                lb=self.X,
                ub=self.X,
            )
            X_prime = model.addMVar(
                shape=self.X.shape, name="x", vtype=GRB.CONTINUOUS, lb=-50, ub=50
            )

            # Define objective function
            if cost_function == "quadratic":
                obj = ((X - X_prime) * (X - X_prime)).sum()

            elif cost_function == "quad_form":
                obj = gp.quicksum(
                    (X_prime[i] - X[i]) @ self.A @ (X_prime[i] - X[i])
                    for i in range(self.X.shape[0])
                )

            else:
                raise ValueError(f"Cost function {cost_function} not recognised")

            model.setObjective(obj, GRB.MINIMIZE)

            # Define constraints
            model.addConstr(X_prime @ self.weights + self.bias >= 0, name="constraint")

            # Solve model
            model.optimize()

            # Check if optimal
            if model.status != GRB.OPTIMAL:
                raise ValueError(
                    f"Full recourse optimisation with cost function {cost_function} not optimal: {model.status}"
                )

            # Get optimal values
            recourse_x = X_prime.x

        else:
            raise ValueError(f"Backend {backend} not recognised")

        # Calculate cost of recourse under true model
        if cost_function == "quadratic":
            costs = np.sum((recourse_x - self.X.values) ** 2, axis=1)

        elif cost_function == "quad_form":
            # costs = np.array(
            #     [cp.quad_form(recourse_x[i] - self.X.iloc[i], self.A).value for i in range(self.X.shape[0])])
            costs = np.einsum(
                "ji,jk,ki->i", (self.X - recourse_x).T, self.A, (self.X - recourse_x).T
            )

        else:
            raise ValueError(f"Cost function {cost_function} not recognised")

        return costs, recourse_x

    def _opt_partial_logistic(
        self,
        C: float,
        row: int,
        cost_function: str = "quadratic",
        backend: str = "cvxpy",
    ):
        """
        Internal optimisation function for partial recourse function for a single row.
        :param C: Cost threshold
        :param row: Row of the data to optimise
        :param cost_function: Cost function to use
        :param backend: Backend to use for optimisation. Currently, cvxpy and gurobi supported
        :return:
        """

        if backend == "cvxpy":

            def attempt_optimisation(
                C: float,
                row: int,
                cost_function: str = "quadratic",
                solver: str = "ECOS",
                verbose: bool = False,
            ):
                # Define variables and objective function
                x = cp.Variable(self.X_negative.shape[1])
                objective = cp.Maximize(cp.matmul(x, self.weights) + self.bias)

                # Define expression for constraint
                if cost_function == "quadratic":
                    expr = cp.sum_squares(x - self.X_negative.iloc[row].values)

                elif cost_function == "quad_form":
                    expr = cp.quad_form(x - self.X_negative.iloc[row].values, self.A)

                else:
                    raise ValueError(f"{cost_function} not recognised")

                constraints = [expr <= C]

                # Optimisation
                prob = cp.Problem(objective, constraints)
                prob.solve(verbose=verbose, solver=solver)

                return x, prob, expr

            x, problem, cost_expr = attempt_optimisation(
                C=C, row=row, cost_function=cost_function, solver="ECOS", verbose=False
            )

            # Try again with verbose=True and a different solver if not optimal
            if problem.status != "optimal":
                x, problem, cost_expr = attempt_optimisation(
                    C=C,
                    row=row,
                    cost_function=cost_function,
                    verbose=True,
                    solver="SCS",
                )

                if problem.status != "optimal":
                    raise ValueError(f"Optimisation not optimal: {problem.status}")

            return x.value, cost_expr.value

        elif backend == "gurobi":
            model = gp.Model("partial_recourse")

            # Create a 2D array of variables
            x = model.addMVar(
                shape=self.X_negative.iloc[row].shape,
                vtype=GRB.CONTINUOUS,
                name="x",
                lb=-100,
                ub=100,
            )
            x_prime = model.addMVar(
                shape=self.X_negative.iloc[row].shape,
                vtype=GRB.CONTINUOUS,
                name="x0",
                lb=self.X_negative.iloc[row],
                ub=self.X_negative.iloc[row],
            )

            # Define objective function
            obj = x @ self.weights + self.bias
            model.setObjective(obj, GRB.MAXIMIZE)

            # Define constraints
            if cost_function == "quadratic":
                cost_expr = ((x - x_prime) * (x - x_prime)).sum()

            elif cost_function == "quad_form":
                cost_expr = (x - x_prime) @ self.A @ (x - x_prime)

            else:
                raise ValueError(f"Cost function {cost_function} not recognised")

            # Solve model
            model.addConstr(cost_expr <= C, name="constraint")
            model.optimize()
            recourse_x = x_prime.x

            # Check if optimal
            if model.status != GRB.OPTIMAL:
                print(
                    f"Partial recourse optimisation of row {row} with cost function {cost_function} not optimal, assuming no partial recourse actions taken at all"
                )
                recourse_x = self.X_negative.iloc[row].values

            return recourse_x, float(cost_expr.getValue())

        else:
            raise ValueError(f"Backend {backend} not recognised")

    def optimise_full_recourse(
        self, C: float, cost_function: str = "quadratic", backend: str = "cvxpy"
    ):
        """
        Optimise the recourse problem.
        :param C: Cost threshold
        :param cost_function: type of cost function to use
        :param backend: Backend to use for optimisation. Currently, cvxpy and gurobi supported
        :return: None
        """
        recourse_cost, recourse_x = self._opt_full_logistic(
            cost_function=cost_function, backend=backend
        )

        self.recourse = pd.DataFrame(recourse_x, columns=self.X.columns)

        self.recourse.index = self.X.index
        self.recourse["cost"] = recourse_cost
        # Returning original values of X is cost < C
        self.recourse.update(self.X[self.recourse["cost"] > C])
        self.recourse["prob"] = self.clf.predict_proba(
            self.recourse[self.X.columns].values
        )[:, 1]
        eps = 1e-6
        self.recourse["Y"] = (self.recourse["prob"] >= 0.5 - eps).astype(float)

    def optimise_partial_recourse(
        self, C: float, cost_function: str = "quadratic", backend: str = "cvxpy"
    ):
        """
        Optimise the partial recourse problem.
        :param C: Cost threshold
        :param cost_function: Type of cost function
        :param backend: Backend to use for optimisation. Currently, cvxpy and gurobi supported
        :return: None
        """

        # Limit to just the individuals with a cost greater than C or NaN
        self.X_negative = self.X[
            np.logical_or(
                (self.recourse["cost"] > C).values, self.recourse["cost"].isna().values
            )
        ]

        if len(self.X_negative) == 0:
            print("No individuals with cost greater than C")
            return None

        # Multiprocessing to optimise partial recourse for each row
        func_args = [
            (C, row, cost_function, backend) for row in range(self.X_negative.shape[0])
        ]
        with Pool() as pool:
            results = pool.starmap(self._opt_partial_logistic, func_args)

        # Lists to store recourse information
        recourse_x = []
        recourse_cost = []

        # Appending recourse information to lists
        for res in results:
            recourse_x.append(res[0])
            recourse_cost.append(res[1])

        # Storing recourse information in dataframe
        self.partial_recourse = pd.DataFrame(
            recourse_x, columns=self.X_negative.columns
        )
        self.partial_recourse["cost"] = recourse_cost
        self.partial_recourse["prob"] = self.clf.predict_proba(
            self.partial_recourse[self.X.columns].values
        )[:, 1]
        eps = 1e-6
        self.partial_recourse["Y"] = (
            self.partial_recourse["prob"] >= (0.5 - eps)
        ).astype(
            float
        )  # eps used to make sure that 0.5 is not rounded down
        self.partial_recourse.index = self.X_negative.index

    def compute_recourse(
        self,
        C: float,
        partial_recourse: bool = False,
        cost_function: str = "quadratic",
        backend: str = "cvxpy",
    ):
        """
        Compute full and partial recourse
        :param cost_function: Type of cost function to use
        :param C: Cost threshold for partial recourse
        :param partial_recourse: Whether to compute partial recourse
        :param backend: Backend to use for optimisation. Currently, cvxpy and gurobi supported
        :return: None
        """

        # Compute recourse
        self.optimise_full_recourse(C=C, cost_function=cost_function, backend=backend)

        if partial_recourse:
            self.optimise_partial_recourse(
                C=C, cost_function=cost_function, backend=backend
            )

            # Merge in partial recourse
            if len(self.X_negative) > 0:
                self.recourse.update(self.partial_recourse)
