import numpy as np
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from typing import Union, List

matplotlib.use("TkAgg")

from structural_models import SimpleSCM, FourVariableSCM
from causal_recourse_gen import CausalRecourseGenerator
from beta_learning import BetaLearner
from true_cost import TrueCost
from softsort import SoftSort


def gen_recourse(
    beta: np.ndarray,
    learn_ordering: bool,
    X_neg: pd.DataFrame,
    W_adjacency: np.ndarray,
    W_classifier: np.ndarray,
    b_classifier: float,
    lr: float = 1e-2,
    tau: float = 0.1,
    max_epochs: int = 2_500,
    cost_function: str = "l2_norm",
) -> torch.Tensor:
    """
    Generate recourse from a given beta.
    :param beta: Beta to use
    :param learn_ordering: whether or not to learn the ordering
    :param X_neg: negatively classified samples
    :param W_adjacency: Weighted adjacency matrix to use
    :param W_classifier: classifier weights
    :param b_classifier: classifier bias
    :param lr: learning rate
    :param tau: tau parameter
    :param max_epochs: max number of epochs
    :param cost_function: cost function to use
    :return: Dataframe of recourse, actions, and costs
    """
    if type(X_neg) == pd.DataFrame:
        X_neg = torch.tensor(X_neg.values, dtype=torch.float64)
    if type(W_adjacency) == np.ndarray:
        W_adjacency = torch.tensor(W_adjacency, dtype=torch.float64)
    if type(W_classifier) == np.ndarray:
        W_classifier = torch.tensor(W_classifier, dtype=torch.float64)
    if type(beta) == np.ndarray:
        beta = torch.tensor(beta, dtype=torch.float64)

    # Generate recourse
    recourse_gen = CausalRecourseGenerator(learn_ordering=learn_ordering)
    recourse_gen.add_data(
        X=X_neg,
        W_adjacency=W_adjacency,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
    )
    recourse_gen.set_beta(beta)
    recourse_gen.set_ordering(torch.arange(X_neg.shape[1]).repeat(X_neg.shape[0], 1))
    recourse_gen.set_sorter(tau=tau)

    X_prime = recourse_gen.gen_recourse(
        classifier_margin=0.02,
        max_epochs=max_epochs,
        verbose=False,
        cost_function=cost_function,
        lr=lr,
        format_as_df=False,
    )[0]

    return X_prime


def learn_beta(
    X_neg: pd.DataFrame,
    ground_truth_beta: np.ndarray,
    ground_truth_W: np.ndarray,
    W_classifier: np.ndarray,
    b_classifier: float,
    eval_noise: float = 0,
    W_noise: float = 0,
    n_comparisons: int = 5,
    W_known: bool = True,
    learn_ordering: bool = False,
    max_epochs: int = 5_000,
    lr: float = 1e-2,
    l2_reg: float = 0.1,
    tanh_param: float = 20,
    verbose: bool = True,
) -> Union[torch.Tensor, torch.Tensor, List]:
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

    beta_learner.set_ordering(torch.arange(X_neg.shape[1]).repeat(X_neg.shape[0], 1))
    beta_learner.eval_random_actions(eval_noise=eval_noise, W_noise=W_noise)

    # LEARN BETA
    learned_beta, learned_W, loss_list = beta_learner.learn(
        max_epochs=max_epochs,
        lr=lr,
        l2_reg=l2_reg,
        tanh_param=tanh_param,
        verbose=verbose,
    )

    return learned_beta, learned_W, loss_list


def eval_true_cost(
    X_neg: pd.DataFrame,
    W_adjacency: np.ndarray,
    beta: np.ndarray,
    W_ground_truth: np.ndarray,
    beta_ground_truth: np.ndarray,
    learn_ordering: bool,
    tau: float,
    lr: float,
    max_epochs: int,
    verbose: bool,
) -> float:
    """
    Evaluate the true cost of a given recourse.
    :param X_neg: negatively classified data
    :param X_prime: recourse
    :param W_adjacency: weighted adjacency matrix
    :param beta: beta
    :param learn_ordering: whether to learn the ordering
    :param tau: tau parameter
    :param lr: learning rate
    :param max_epochs: max number of epochs
    :param verbose: whether to print loss
    :return: cost
    """
    # Generate recourse
    X_prime = gen_recourse(
        learn_ordering=args.learn_ordering,
        X_neg=X_neg,
        W_adjacency=W_adjacency,
        beta=beta,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        lr=args.lr,
        tau=args.tau,
        max_epochs=args.max_epochs * 2,
        cost_function=args.cost_function,
    )

    # Convert to torch tensors
    X_neg = torch.tensor(X_neg.values, dtype=torch.float64)
    W_ground_truth = torch.tensor(W_ground_truth, dtype=torch.float64)
    beta_ground_truth = torch.tensor(beta_ground_truth, dtype=torch.float64)

    # Calculate true cost
    true_cost = TrueCost(
        X=X_neg,
        X_final=X_prime,
        W_adjacency=W_ground_truth,
        beta=beta_ground_truth,
        learn_ordering=learn_ordering,
        sorter=SoftSort(tau=tau, hard=True),
    )
    cost = true_cost.eval_true_cost(lr=lr, max_epochs=max_epochs, verbose=verbose)

    return torch.mean(cost).detach().item()


def plot(W_range, e_range, results, title):
    results_mean = np.mean(results, axis=0)

    plt.imshow(results_mean, cmap="RdBu_r")
    plt.colorbar(
        label="Percentage Increase in Cost compared to Ground Truth SCM",
        format=matplotlib.ticker.PercentFormatter(1),
    )

    plt.xticks(np.arange(results_mean.shape[1]), W_range)
    plt.yticks(np.arange(results_mean.shape[0]), e_range)

    plt.xlabel("W noise (std)")
    plt.ylabel("Eval noise (std)")

    plt.title("Inc in cost at different noise levels")

    plt.show()
    plt.gca().figure.set_size_inches(6.3, 4)
    plt.savefig(f"plots/{title}.png")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2_000)
    parser.add_argument("--cost_function", type=str, default="l2_norm")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--learn_ordering", type=bool, default=False)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=2_500)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--num_levels", type=int, default=10)
    parser.add_argument("--max_noise", type=float, default=10)
    args = parser.parse_args()

    # Matplotlib setup
    plt.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "cmr10"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = "10"

    N = args.N

    # Generate data from SCM
    scm = SimpleSCM(N)
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
    beta_ground_truth = np.array([3, 2, 1])

    # Ground truth cost
    cost_ground_truth = eval_true_cost(
        X_neg=X_neg,
        W_adjacency=W_adjacency_ground_truth,
        beta=beta_ground_truth,
        W_ground_truth=W_adjacency_ground_truth,
        beta_ground_truth=beta_ground_truth,
        learn_ordering=args.learn_ordering,
        tau=args.tau,
        lr=args.lr,
        max_epochs=5_000,
        verbose=False,
    )
    print(f"Ground truth cost: {cost_ground_truth}")

    results = np.empty((args.num_trials, args.num_levels, args.num_levels))

    # LOOP THROUGH DIFFERENT noise levels
    W_noise_levels = np.linspace(0, args.max_noise, args.num_levels)
    eval_noise_levels = np.linspace(0, args.max_noise, args.num_levels)

    for n in range(2):
        for i, w_noise in enumerate(W_noise_levels):
            for j, e_noise in enumerate(eval_noise_levels):
                # Learn beta
                learned_beta, learned_W, loss_list = learn_beta(
                    X_neg=X_neg,
                    ground_truth_beta=beta_ground_truth,
                    ground_truth_W=W_adjacency_ground_truth,
                    W_classifier=W_classifier,
                    b_classifier=b_classifier,
                    eval_noise=e_noise,
                    W_noise=w_noise,
                    n_comparisons=5,
                    W_known=False,
                    learn_ordering=args.learn_ordering,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    l2_reg=0.1,
                    tanh_param=20,
                    verbose=False,
                )

                # Generate recourse
                cost = eval_true_cost(
                    X_neg=X_neg,
                    W_adjacency=learned_W,
                    beta=learned_beta,
                    W_ground_truth=W_adjacency_ground_truth,
                    beta_ground_truth=beta_ground_truth,
                    learn_ordering=args.learn_ordering,
                    tau=args.tau,
                    lr=args.lr,
                    max_epochs=5_000,
                    verbose=False,
                )
                print(f"W noise : {w_noise} | Eval noise: {e_noise} | Trial: {n}")
                print(
                    f"{round(100 * ((cost / cost_ground_truth) - 1), 2)}% increase in cost"
                )
                results[n, i, j] = (cost / cost_ground_truth) - 1

    # Plot
    plot(W_noise_levels, eval_noise_levels, results, title="SimpleSCM_noise_plot")
