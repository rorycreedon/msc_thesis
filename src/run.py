import numpy as np
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

from structural_models import SimpleSCM, FourVariableSCM
from causal_recourse_gen import CausalRecourseGenerator
from beta_learning import BetaLearner


def gen_recourse(
    beta, args, X_neg, W_adjacency, W_classifier, b_classifier
) -> pd.DataFrame:
    """
    Generate recourse from a given beta.
    :param beta: Beta to use
    :param args: arguments from the script
    :param X_neg: negatively classified samples
    :param W_adjacency: Weighted adjacency matrix to use
    :param W_classifier: classifier weights
    :param b_classifier: classifier bias
    :return: Dataframe of recourse, actions, and costs
    """
    # Generate recourse
    recourse_gen = CausalRecourseGenerator(learn_ordering=args.learn_ordering)
    recourse_gen.add_data(
        X=torch.tensor(X_neg.values, dtype=torch.float64),
        W_adjacency=torch.tensor(W_adjacency, dtype=torch.float64),
        W_classifier=torch.tensor(W_classifier, dtype=torch.float64),
        b_classifier=b_classifier,
    )
    recourse_gen.set_beta(beta)
    recourse_gen.set_ordering(torch.arange(X_neg.shape[1]).repeat(X_neg.shape[0], 1))
    recourse_gen.set_sorter(tau=args.tau)

    df = recourse_gen.gen_recourse(
        classifier_margin=0.02,
        max_epochs=args.max_epochs,
        verbose=True,
        cost_function=args.cost_function,
        lr=args.lr,
        format_as_df=True,
    )
    return df


def learn_beta(
    X_neg: pd.DataFrame,
    ground_truth_beta: np.ndarray,
    ground_truth_W: np.ndarray,
    W_classifier: np.ndarray,
    b_classifier: float,
    n_comparisons: int = 5,
    W_known: bool = True,
    learn_ordering: bool = False,
    max_epochs: int = 5_000,
    lr: float = 1e-2,
    l2_reg: float = 0.1,
    tanh_param: float = 20,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Learn beta from data.
    :param X_neg: negatively classified data
    :param ground_truth_beta: ground truth beta
    :param ground_truth_W: ground truth weighted adjacency matrix
    :param W_classifier: classifier weights
    :param b_classifier: classifier bias
    :param n_comparisons: number of paired comparisons to generate
    :param W_known: whether or not the weighted adjacency matrix is known
    :param learn_ordering: whether or not to learn the ordering #TODO: check is worth removing this
    :param max_epochs: max number of epochs
    :param lr: learning rate
    :param l2_reg: L2 regularisation
    :param tanh_param: tanh parameter
    :param verbose: whether or not to print loss
    :return:
    """

    # Convert to torch tensors
    X_neg = torch.tensor(X_neg.values, dtype=torch.float64)
    ground_truth_beta = torch.tensor(ground_truth_beta, dtype=torch.float64)
    ground_truth_W = torch.tensor(ground_truth_W, dtype=torch.float64)
    W_classifier = torch.tensor(W_classifier, dtype=torch.float64)

    # GENERATE ALTERNATIVE ACTIONS AND ORDERINGS
    beta_learner = BetaLearner(
        n_comparisons=n_comparisons,
        learn_ordering=learn_ordering,
        ground_truth_beta=ground_truth_beta,
        ground_truth_W=ground_truth_W,
        W_known=W_known,
    )
    beta_learner.add_data(
        X=X_neg,
        W_adjacency=torch.rand(X_neg.shape[1], X_neg.shape[1], dtype=torch.float64),
        W_classifier=W_classifier,
        b_classifier=b_classifier,
    )
    # beta_learner.gen_pairwise_comparisons()
    beta_learner.set_ordering(torch.arange(X_neg.shape[1]).repeat(X_neg.shape[0], 1))
    beta_learner.sample_betas((0, 3))
    beta_learner.eval_sampled_betas()

    # LEARN BETA
    beta, loss_list = beta_learner.learn(
        max_epochs=max_epochs,
        lr=lr,
        l2_reg=l2_reg,
        tanh_param=tanh_param,
        verbose=verbose,
    )

    return beta, loss_list


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2500)
    parser.add_argument("--cost_function", type=str, default="l2_norm")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--learn_ordering", type=bool, default=False)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=2_500)
    args = parser.parse_args()

    N = args.N

    # Generate data from SCM
    scm = FourVariableSCM(N)
    scm.simulate_data()

    # Classify data
    y_pred, X_neg, clf = scm.classify_data()
    N_neg = X_neg.shape[0]

    # Classification weights
    W_classifier = np.squeeze(clf.coef_)
    b_classifier = clf.intercept_[0]

    # Ground truth weighted adjacency matrix
    W_adjacency_ground_truth = scm.gen_weighted_adjacency_matrix()

    # Ground truth beta
    beta_ground_truth = np.random.rand(X_neg.shape[1])

    # Learn beta
    beta, loss_list = learn_beta(
        X_neg=X_neg,
        ground_truth_beta=beta_ground_truth,
        ground_truth_W=W_adjacency_ground_truth,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        n_comparisons=25,
        W_known=False,
        learn_ordering=args.learn_ordering,
        max_epochs=args.max_epochs,
        lr=args.lr,
        l2_reg=0.1,
        tanh_param=20,
        verbose=True,
    )

    plt.plot(loss_list)
    plt.show()
