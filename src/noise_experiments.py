import numpy as np
import pandas as pd
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
from typing import Union, List
import logging
import os

from structural_models import SimpleSCM, NonLinearSCM, StructuralCausalModel
from cost_learning import CausalRecourse, TrueCost, SoftSort, CostLearner


matplotlib.use("TkAgg")


# Ensure the 'logs' directory exists
if not os.path.exists("logs"):
    os.makedirs("logs")

# Set up logging
logging.basicConfig(
    filename="logs/noise_experiments.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Add a stream handler to also log to the console
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(name)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(console_handler)


def gen_recourse(
    beta: np.ndarray,
    learn_ordering: bool,
    X_neg: pd.DataFrame,
    W_classifier: np.ndarray,
    b_classifier: float,
    sorter: SoftSort,
    use_scm: bool,
    lr: float = 1e-2,
    max_epochs: int = 2_500,
    W_adjacency: np.ndarray = None,
    scm: StructuralCausalModel = None,
) -> torch.Tensor:
    """
    Generate recourse from a given beta.
    :param beta: Beta to use
    :param learn_ordering: whether to learn the ordering
    :param X_neg: negatively classified samples
    :param W_adjacency: Weighted adjacency matrix to use
    :param W_classifier: classifier weights
    :param b_classifier: classifier bias
    :param lr: learning rate
    :param tau: tau parameter
    :param max_epochs: max number of epochs
    :return: Dataframe of recourse, actions, and costs
    """
    if type(X_neg) == np.ndarray:
        X_neg = torch.tensor(X_neg, dtype=torch.float64)
    if use_scm is False:
        if type(W_adjacency) == np.ndarray:
            W_adjacency = torch.tensor(W_adjacency, dtype=torch.float64)
    if type(W_classifier) == np.ndarray:
        W_classifier = torch.tensor(W_classifier, dtype=torch.float64)
    if type(beta) == np.ndarray:
        beta = torch.tensor(beta, dtype=torch.float64)

    # Generate recourse
    recourse_gen = CausalRecourse(
        X=X_neg,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta=beta,
        use_scm=use_scm,
        W_adjacency=W_adjacency,
        scm=scm,
        learn_ordering=learn_ordering,
    )
    recourse_gen.set_sorter(sorter=sorter)

    X_prime = recourse_gen.gen_recourse(
        classifier_margin=0.02,
        max_epochs=max_epochs,
        verbose=False,
        lr=lr,
        format_as_df=False,
    )[0]

    return X_prime


def learn_beta(
    X_neg: np.ndarray,
    ground_truth_beta: np.ndarray,
    scm: StructuralCausalModel,
    eval_noise: float = 0,
    scm_noise: float = 0,
    n_comparisons: int = 5,
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
    :param W_classifier: classifier weights
    :param b_classifier: classifier bias
    :param n_comparisons: number of paired comparisons to generate
    :param max_epochs: max number of epochs
    :param lr: learning rate
    :param l2_reg: L2 regularisation
    :param tanh_param: tanh parameter
    :param verbose: whether or not to print loss
    :return:
    """

    # Convert to torch tensors
    X_neg = torch.tensor(X_neg, dtype=torch.float64)
    ground_truth_beta = torch.tensor(ground_truth_beta, dtype=torch.float64)

    # GENERATE ALTERNATIVE ACTIONS AND ORDERINGS
    beta_learner = CostLearner(
        scm=scm,
        X=X_neg,
        n_comparisons=n_comparisons,
        ground_truth_beta=ground_truth_beta,
        scm_known=False,
    )
    beta_learner.eval_random_actions(eval_noise=eval_noise, scm_noise=scm_noise)

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
    beta: np.ndarray,
    beta_ground_truth: np.ndarray,
    W_classifier: np.ndarray,
    b_classifier: float,
    learn_ordering: bool,
    lr: float,
    max_epochs: int,
    verbose: bool,
    sorter: SoftSort,
    use_scm: bool,
    scm: StructuralCausalModel = None,
    W_adjacency: np.ndarray = None,
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
        beta=beta,
        learn_ordering=args.learn_ordering,
        X_neg=X_neg,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        lr=args.lr,
        sorter=sorter,
        max_epochs=args.max_epochs * 2,
        use_scm=use_scm,
        W_adjacency=W_adjacency,
        scm=scm,
    )

    # Convert to torch tensors
    X_neg = torch.tensor(X_neg, dtype=torch.float64)
    beta_ground_truth = torch.tensor(beta_ground_truth, dtype=torch.float64)

    # Calculate true cost
    true_cost = TrueCost(
        X=X_neg,
        X_final=X_prime,
        scm=scm,
        beta=beta_ground_truth,
        learn_ordering=learn_ordering,
        sorter=sorter,
    )
    cost = true_cost.eval_true_cost(lr=lr, max_epochs=max_epochs, verbose=verbose)

    return torch.mean(cost).detach().item()


def plot(W_range, e_range, results, title):
    # Matplotlib setup
    plt.rcParams["savefig.dpi"] = 300
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "cmr10"
    matplotlib.rcParams["mathtext.fontset"] = "cm"
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = "10"

    results_mean = np.mean(results, axis=0) * 100  # Convert to percentages

    # Create meshgrid for the surface plot
    W_mesh, e_mesh = np.meshgrid(W_range, e_range)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(
        W_mesh,
        e_mesh,
        results_mean,
        cmap="RdBu_r",
        linewidth=0,
        antialiased=True,
        shade=True,
    )

    ax.set_ylabel(r"$W$ noise (std dev)", labelpad=5)  # Adjust space for y-axis label
    ax.set_xlabel(
        "Evaluation noise (std dev)", labelpad=5
    )  # Adjust space for x-axis label
    ax.set_zlabel("Cost Increase (%)", labelpad=5)  # Adjust space for z-axis label
    ax.set_title("Percentage Increase in Average Cost vs Ground Truth SCM")

    # Add percentage signs to the z-axis ticks
    ax.zaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%1.0f%%"))

    fig.set_size_inches(6.3, 4)
    fig.show()
    # save plot
    fig.savefig(f"plots/{title}.png")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--learn_ordering", type=bool, default=False)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_epochs", type=int, default=2_500)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--num_levels", type=int, default=2)
    parser.add_argument("--max_noise", type=float, default=2)
    parser.add_argument("--scm", type=str, default="nonlinear")
    args = parser.parse_args()

    N = args.N

    # Generate data from SCM
    if args.scm == "simple":
        SCM = SimpleSCM(N)
    elif args.scm == "nonlinear":
        SCM = NonLinearSCM(N)
    else:
        raise ValueError("SCM name not recognised.")
    SCM.simulate_data()

    # Classify data
    y_pred, X_neg, clf = SCM.classify_data()
    N_neg = X_neg.shape[0]

    # Classification weights
    W_classifier = np.squeeze(clf.coef_)
    b_classifier = clf.intercept_[0]

    # Ground truth beta
    beta_ground_truth = np.array([3, 2, 1, 4, 5])
    beta_ground_truth = beta_ground_truth / np.sum(beta_ground_truth)

    # Ground truth cost
    cost_ground_truth = eval_true_cost(
        X_neg=X_neg,
        use_scm=True,
        beta=beta_ground_truth,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta_ground_truth=beta_ground_truth,
        learn_ordering=args.learn_ordering,
        sorter=SoftSort(hard=True, tau=args.tau),
        lr=args.lr,
        max_epochs=20_000,
        verbose=False,
        scm=SCM.scm,
    )

    logging.info(f"Ground truth cost: {cost_ground_truth}")

    results = np.empty((args.num_trials, args.num_levels, args.num_levels))

    # LOOP THROUGH DIFFERENT noise levels
    scm_noise_levels = np.linspace(0, args.max_noise, args.num_levels)
    eval_noise_levels = np.linspace(0, args.max_noise, args.num_levels)

    for n in range(args.num_trials):
        for i, scm_noise in enumerate(scm_noise_levels):
            for j, e_noise in enumerate(eval_noise_levels):
                # Learn beta
                learned_beta, learned_W, loss_list = learn_beta(
                    X_neg=X_neg,
                    ground_truth_beta=beta_ground_truth,
                    scm=SCM.scm,
                    eval_noise=e_noise,
                    scm_noise=scm_noise,
                    n_comparisons=5,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    l2_reg=0.1,
                    tanh_param=20,
                    verbose=False,
                )
                # Generate recourse
                cost = eval_true_cost(
                    X_neg=X_neg,
                    use_scm=False,
                    W_adjacency=learned_W,
                    W_classifier=W_classifier,
                    b_classifier=b_classifier,
                    beta=learned_beta,
                    beta_ground_truth=beta_ground_truth,
                    learn_ordering=args.learn_ordering,
                    sorter=SoftSort(hard=True, tau=args.tau),
                    lr=args.lr,
                    max_epochs=20_000,
                    verbose=False,
                    scm=SCM.scm,
                )

                logging.info(
                    f"scm_noise : {scm_noise} | Eval noise: {e_noise} | Trial: {n}"
                )
                logging.info(
                    f"{round(100 * ((cost / cost_ground_truth) - 1), 2)}% increase in cost"
                )

                results[n, i, j] = (cost / cost_ground_truth) - 1

    # Save file
    np.save(f"results/{args.scm}SCM_noise_results", results)

    # Plot
    plot(
        scm_noise_levels, eval_noise_levels, results, title=f"{args.scm}SCM_noise_plot"
    )
