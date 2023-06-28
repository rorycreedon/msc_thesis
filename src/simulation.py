from scm import StructuralCausalModel
from recourse_model import Recourse
from utils import get_near_psd, is_psd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
import matplotlib

matplotlib.use("TkAgg")
pd.set_option("mode.chained_assignment", None)


def simulate_data(N: int):
    # Define the SCM
    scm = StructuralCausalModel(N)

    # Fist variable is a normal distribution
    scm.add_variable(name="X1", distribution=norm, loc=0, scale=2)

    # X1 causes X2
    scm.add_relationship(causes={"X1": 3}, effect="X2", noise_dist=norm, loc=1, scale=1)

    # There exists an unobserved variable U
    scm.add_variable(name="U", distribution=norm, loc=0, scale=1)

    # Y is caused by X2 and U
    scm.add_binary_outcome(
        name="Y_true", weights={"X2": 0.5, "U": -0.5}, noise_dist=norm, loc=0, scale=0
    )

    # X3 is caused by X1
    scm.add_relationship(
        causes={"X1": -2}, effect="X3", noise_dist=norm, loc=0, scale=0.25
    )

    # X4 is caused by Y
    scm.add_relationship(
        causes={"Y_true": 0.75}, effect="X4", noise_dist=norm, loc=0, scale=0.5
    )

    # Return object
    return scm


# Responses over time
def simulate_recourse(
    C: float,
    scm: StructuralCausalModel,
    iterations: int = 5,
    partial_recourse: bool = True,
    cost_function: str = "quadratic",
    backend: str = "cvxpy",
):
    # Generate data
    data = scm.generate_data()
    X = data[["X1", "X2", "X3", "X4"]]
    scm.data["Y"] = scm.data["Y_true"].copy()

    # Define lists to store results
    accuracy = []
    class_positive = []
    true_positives = []

    # Make matrix for quadratic form cost function
    A = np.linalg.inv(
        X.cov()
    )  # using inverse of covariance matrix as A (Mahalanobis distance)
    if not is_psd(A):
        print("A is not PSD, getting nearest PSD matrix")
        A = get_near_psd(A)

    # Define train and test sets by IDs
    train_ids = np.random.choice(X.index, size=int(0.5 * len(X)), replace=False)
    test_ids = np.array([i for i in X.index if i not in train_ids])

    # Iterate over each round
    for i in range(1, iterations + 1):
        print(f"Iteration {i} started")

        # Split data into train and test
        X_train = scm.data[["X1", "X2", "X3", "X4"]][scm.data["ID"].isin(train_ids)]
        y_train = scm.data["Y_true"][scm.data["ID"].isin(train_ids)]
        X_test = scm.data[["X1", "X2", "X3", "X4"]][scm.data["ID"].isin(test_ids)]

        # Train classifier and predict
        clf = LogisticRegression(penalty="l2", C=2).fit(X_train.values, y_train.values)
        y_pred = clf.predict(X_test.values)

        # Calculate accuracy
        y_true = scm.data[scm.data["ID"].isin(test_ids)]["Y_true"].values
        assert y_pred.shape == y_true.shape
        accuracy.append(np.sum(y_pred == y_true) / len(y_true))

        # Predict for all data (for recourse)
        y_pred = clf.predict(scm.data[["X1", "X2", "X3", "X4"]].values)

        # If all predicted 1, then finish iterating
        if np.min(y_pred) == 1:
            print("All predicted 1")
            return accuracy, class_positive, true_positives

        # Compute recourse
        X_neg = scm.data[["X1", "X2", "X3", "X4"]]
        X_neg.index = scm.data["ID"]
        X_neg = X_neg.loc[((y_pred == 0) & (scm.data["recourse_eligible"] == 1)).values]
        recourse_model = Recourse(X_neg, clf, A)
        recourse_model.compute_recourse(
            C,
            partial_recourse=partial_recourse,
            cost_function=cost_function,
            backend=backend,
        )

        if (recourse_model.recourse.Y == 1).all():
            print("finished")
            return accuracy, class_positive, true_positives

        # Calculate proportion of test set that is positively classified
        class_positive.append(np.sum(y_pred[test_ids] == 1) / len(test_ids))
        true_positives.append(np.sum(y_true == 1) / len(y_true))

        # Update the SCM with the new data
        scm.append_data(
            recourse_model.recourse[["X1", "X2", "X3", "X4", "Y"]],
            ids=recourse_model.recourse.index.to_series(),
        )
        scm.data.drop_duplicates(subset=["ID"], inplace=True, keep="last")

        assert max(scm.data["ID"]) <= scm.N

    return accuracy, class_positive, true_positives


def plot(accuracy, class_positive, true_positives, C: float, cost_function: str):
    # Setup subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot accuracy
    ax[0].plot(accuracy, color="tab:blue")
    ax[0].set_title("Accuracy")

    # Plot proportion of positively classified and true positives
    ax[1].plot(class_positive, color="tab:green", label="Positively classified")
    ax[1].plot(true_positives, color="tab:orange", label="True positive")

    # Set titles and legend
    ax[1].set_title("% Positive Class")
    ax[1].legend()
    fig.suptitle(f"C = {C}, cost function = {cost_function}")

    # Show plot
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


if __name__ == "__main__":
    scm = simulate_data(2500)
    C = 0.1

    accuracy, class_positive, true_positives = simulate_recourse(
        C=C,
        scm=scm.copy(),
        iterations=10,
        partial_recourse=False,
        cost_function="quadratic",
        backend="cvxpy",
    )
    plot(accuracy, class_positive, true_positives, C=C, cost_function="quadratic")

    accuracy, class_positive, true_positives = simulate_recourse(
        C=C,
        scm=scm.copy(),
        iterations=10,
        partial_recourse=False,
        cost_function="quad_form",
        backend="cvxpy",
    )
    plot(accuracy, class_positive, true_positives, C=C, cost_function="quad_form")
