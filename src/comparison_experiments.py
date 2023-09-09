import numpy as np
import pandas as pd
import argparse
import torch
from typing import Union, List
import logging
import os

from structural_models import SimpleSCM, NonLinearSCM, StructuralCausalModel
from cost_learning import CausalRecourse, TrueCost, SoftSort, CostLearner


def setup_logging(args) -> None:
    # Ensure the 'logs' directory exists
    if not os.path.exists("logs"):
        os.makedirs("logs")

    # Set up logging
    logging.basicConfig(
        filename=f"logs/{args.scm_name}SCM_comparison_results.log",
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

    X_prime, order, actions, cost, pred = recourse_gen.gen_recourse(
        classifier_margin=0.02,
        max_epochs=max_epochs,
        verbose=False,
        lr=lr,
        format_as_df=False,
    )

    # Check to see that constraint is satistfied
    logging.info(
        f"negative classifications after recourse - {torch.sum((X_prime @ W_classifier + b_classifier) < 0)}"
    )

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
    if type(ground_truth_beta) == np.ndarray:
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
        max_epochs=args.max_epochs,
        use_scm=use_scm,
        W_adjacency=W_adjacency,
        scm=scm,
    )

    # Convert to torch tensors
    X_neg = torch.tensor(X_neg, dtype=torch.float64)
    if type(beta_ground_truth) == np.ndarray:
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


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5_000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--learn_ordering", type=bool, default=True)
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--max_epochs", type=int, default=10_000)
    parser.add_argument("--scm_noise", type=float, default=0)
    parser.add_argument("--eval_noise", type=float, default=0)
    parser.add_argument("--scm_name", type=str, default="simple")
    args = parser.parse_args()

    setup_logging(args)

    # Set seed
    np.random.seed(0)
    torch.manual_seed(0)

    N = args.N

    # Generate data from SCM
    if args.scm_name == "simple":
        SCM = SimpleSCM(N)
    elif args.scm_name == "nonlinear":
        SCM = NonLinearSCM(N)
    else:
        raise ValueError("SCM name not recognised.")
    SCM.simulate_data()

    # Classify data
    y_pred, X_neg, clf = SCM.classify_data()
    N_neg = X_neg.shape[0]

    logging.info(f"{N} individuals")
    logging.info(f"{N_neg} negatively classified individuals")

    # Classification weights
    W_classifier = np.squeeze(clf.coef_)
    b_classifier = clf.intercept_[0]

    # Ground truth beta
    if args.scm_name == "simple":
        beta_ground_truth = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64).repeat(X_neg.shape[0], 1)
    elif args.scm_name == "nonlinear":
        beta_ground_truth = torch.tensor([0.1, 0.2, 0.7, 0.4, 0.6], dtype=torch.float64).repeat(
            X_neg.shape[0], 1
        )
    beta_ground_truth += torch.rand(X_neg.shape[0], X_neg.shape[1], dtype=torch.float64)
    # beta_ground_truth = torch.rand(X_neg.shape, dtype=torch.float64)
    beta_ground_truth = beta_ground_truth / torch.sum(beta_ground_truth, dim=1)[:, None]

    # Comparison list
    comparison_list = [5, 10, 20, 50]

    # Results array
    results = np.zeros((len(comparison_list),))

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ground truth cost
    cost_ground_truth = eval_true_cost(
        X_neg=X_neg,
        use_scm=True,
        beta=beta_ground_truth,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta_ground_truth=beta_ground_truth,
        learn_ordering=args.learn_ordering,
        sorter=SoftSort(hard=True, tau=args.tau, device=device),
        lr=args.lr,
        max_epochs=20_000,
        verbose=False,
        scm=SCM.scm,
    )

    logging.info(f"Ground truth cost: {cost_ground_truth}")

    # IDENTITY W and uniform beta
    beta_random = torch.ones(X_neg.shape, dtype=torch.float64)
    beta_random = beta_random / torch.sum(beta_random, dim=1)[:, None]

    W_identity = torch.eye(X_neg.shape[1], dtype=torch.float64)

    cost_identity = eval_true_cost(
        X_neg=X_neg,
        use_scm=False,
        beta=beta_random,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta_ground_truth=beta_ground_truth,
        learn_ordering=args.learn_ordering,
        sorter=SoftSort(hard=True, tau=args.tau, device=device),
        lr=args.lr,
        max_epochs=20_000,
        verbose=False,
        scm=SCM.scm,
        W_adjacency=W_identity,
    )

    logging.info(
        f"Cost increase with random beta and identity W: {100*((cost_identity/cost_ground_truth) -1)}%"
    )

    # Random W and random beta
    W_random = torch.normal(
        0, 1, size=(X_neg.shape[1], X_neg.shape[1]), dtype=torch.float64
    )
    # Set diagonal to 0
    W_random = W_random - torch.diag(torch.diag(W_random))

    cost_random = eval_true_cost(
        X_neg=X_neg,
        use_scm=False,
        beta=beta_random,
        W_classifier=W_classifier,
        b_classifier=b_classifier,
        beta_ground_truth=beta_ground_truth,
        learn_ordering=args.learn_ordering,
        sorter=SoftSort(hard=True, tau=args.tau, device=device),
        lr=args.lr,
        max_epochs=20_000,
        verbose=False,
        scm=SCM.scm,
        W_adjacency=W_random,
    )

    logging.info(
        f"Cost increase with random beta and random W: {100*((cost_random/cost_ground_truth) -1)}%"
    )

    scm_noise_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    eval_noise_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5]

    results_array = np.empty(
        (len(scm_noise_list), len(eval_noise_list), len(comparison_list))
    )

    for i, scm_noise in enumerate(scm_noise_list):
        for j, eval_noise in enumerate(eval_noise_list):
            for k, n_comparisons in enumerate(comparison_list):
                # Learn beta
                learned_beta, learned_W, loss_list = learn_beta(
                    X_neg=X_neg,
                    ground_truth_beta=beta_ground_truth,
                    scm=SCM.scm,
                    eval_noise=eval_noise,
                    scm_noise=scm_noise,
                    n_comparisons=n_comparisons,
                    max_epochs=args.max_epochs,
                    lr=args.lr,
                    l2_reg=0.01,
                    tanh_param=20,
                    verbose=False,
                )

                # Calculate cost
                cost = eval_true_cost(
                    X_neg=X_neg,
                    use_scm=False,
                    beta=learned_beta,
                    W_classifier=W_classifier,
                    b_classifier=b_classifier,
                    beta_ground_truth=beta_ground_truth,
                    learn_ordering=args.learn_ordering,
                    sorter=SoftSort(hard=True, tau=args.tau, device=device),
                    lr=args.lr,
                    max_epochs=20_000,
                    verbose=False,
                    scm=SCM.scm,
                    W_adjacency=learned_W,
                )

                logging.info(
                    f"Comparisons: {n_comparisons} | SCM noise: {scm_noise}| Eval noise: {eval_noise}| Cost increasL {100*((cost/cost_ground_truth) -1)}%"
                )
                results_array[i, j, k] = (cost / cost_ground_truth) - 1

    # Save results
    np.save(f"results/{args.scm_name}SCM_comparison_results.npy", results_array)
