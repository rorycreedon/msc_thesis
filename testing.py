from sklearn.linear_model import LogisticRegression
import numpy as np

from src.recourse_model import LearnedCostsRecourse
from src.utils import get_near_psd, is_psd
from simulation import simulate_data


def sim(N):
    # Generate data
    scm = simulate_data(N)
    data = scm.generate_data()
    X = data[["X1", "X2", "X3", "X4"]]
    scm.data["Y"] = scm.data["Y_true"].copy()

    # Define lists to store results
    accuracy = []
    class_positive = []
    true_positives = []

    # Make matrix for quadratic form cost function
    M = X.cov().values
    if not is_psd(M):
        print("A is not PSD, getting nearest PSD matrix")
        M = get_near_psd(M)

    # Define train and test sets by IDs
    train_ids = np.random.choice(X.index, size=int(0.5 * len(X)), replace=False)
    test_ids = np.array([i for i in X.index if i not in train_ids])

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

    # Compute recourse
    X_neg = scm.data[["X1", "X2", "X3", "X4"]]
    X_neg.index = scm.data["ID"]
    X_neg = X_neg.loc[((y_pred == 0) & (scm.data["recourse_eligible"] == 1)).values]
    recourse_model = LearnedCostsRecourse(X_neg, M_ground_truth=X.cov().values)
    recourse_model.update_classifier(clf)
    recourse_model.compute_recourse(
        C=np.inf,
        verbose=False,
        loss_function="hinge",
    )

    return recourse_model, X_neg, X.cov().values


if __name__ == "__main__":
    recourse_model, X_neg, ground_truth = sim(2500)
    # cost_learner = CostLearn(X_neg, recourse_model.weights, recourse_model.bias, 10)
    # cost_learner.gen_pairwise_comparisons()
    #
    # ground_truth = X_neg.cov().values
    #
    # cost_learner.eval_comparisons(
    #     recourse_model.ground_truth_costs_static,
    #     causal=False,
    #     M=ground_truth,
    # )
    # cost_learner.solve(verbose=False, loss_function="max_margin")

    print(f"Is PSD?: {is_psd(recourse_model.M)}")
    print(recourse_model.M)

    print("ground truth M")
    print(ground_truth)

    print("ratio of learned M to ground truth")
    print(recourse_model.M / ground_truth)
