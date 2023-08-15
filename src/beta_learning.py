import torch
import concurrent.futures
import torch.multiprocessing as mp
import time

from src.causal_recourse_gen import CausalRecourseGenerator


class BetaLearning(CausalRecourseGenerator):
    def __init__(self, n_comparisons: int, learn_ordering: bool = False):
        super(BetaLearning, self).__init__(
            learn_beta=False,
            learn_ordering=learn_ordering,
        )
        self.n_comparisons = n_comparisons

        # Sampled betas
        self.sampled_betas = None

    def sample_betas(self, xrange: tuple[float, float]) -> None:
        self.sampled_betas = (xrange[0] - xrange[1]) * torch.rand(
            2, self.n_comparisons, self.X.shape[0], self.X.shape[1]
        ) + xrange[1]

    def eval_sampled_betas(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
    ):
        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])
        for i in range(self.sampled_betas.shape[0]):
            for j in range(self.sampled_betas.shape[1]):
                self.set_beta(self.sampled_betas[i, j])
                df = self.gen_recourse(
                    classifier_margin=classifier_margin, max_epochs=max_epochs, lr=lr
                )
                costs[i, j] = torch.tensor(df["cost"].values)

        return costs

    def _eval_beta(self, i, j, classifier_margin, max_epochs, lr):
        self.set_beta(self.sampled_betas[i, j])
        df = self.gen_recourse(
            classifier_margin=classifier_margin, max_epochs=max_epochs, lr=lr
        )
        return i, j, torch.tensor(df["cost"].values)

    def eval_sampled_betas_parallel(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
    ):
        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._eval_beta, i, j, classifier_margin, max_epochs, lr
                )
                for i in range(self.sampled_betas.shape[0])
                for j in range(self.sampled_betas.shape[1])
            ]

            for future in concurrent.futures.as_completed(futures):
                i, j, cost = future.result()
                costs[i, j] = cost

        return costs

    def _eval_beta_mp(self, args):
        i, j, classifier_margin, max_epochs, lr = args
        self.set_beta(self.sampled_betas[i, j])
        df = self.gen_recourse(
            classifier_margin=classifier_margin, max_epochs=max_epochs, lr=lr
        )
        return i, j, torch.tensor(df["cost"].values)

    def eval_sampled_betas_parallel_mp(
        self,
        classifier_margin: float = 0.02,
        max_epochs: int = 5_000,
        lr: float = 1e-2,
    ):
        costs = torch.zeros(2, self.n_comparisons, self.X.shape[0])

        # Create arguments for _eval_beta
        args = [
            (i, j, classifier_margin, max_epochs, lr)
            for i in range(self.sampled_betas.shape[0])
            for j in range(self.sampled_betas.shape[1])
        ]

        # Create a Pool and map _eval_beta over args
        with mp.Pool() as pool:
            results = pool.map(self._eval_beta_mp, args)

        # Populate costs tensor with results
        for i, j, cost in results:
            costs[i, j] = cost

        return costs


if __name__ == "__main__":
    # FIXED PARAMETERS
    X = torch.rand(1000, 4, dtype=torch.float64)
    W_adjacency = torch.tensor(
        [[0, 0, 0, 0], [0.3, 0, 0, 0], [0.2, 0, 0, 0], [0, 0.2, 0.3, 0]],
        dtype=torch.float64,
    )
    W_classifier = torch.tensor([-2, -3, -1, -4], dtype=torch.float64)
    beta = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)

    # LEARN BETA
    beta_learner = BetaLearning(n_comparisons=5, learn_ordering=False)
    beta_learner.add_data(
        X=X, W_adjacency=W_adjacency, W_classifier=W_classifier, b_classifier=0.5
    )
    beta_learner.set_beta(beta)
    beta_learner.set_ordering(torch.arange(4).repeat(1000, 1))
    beta_learner.sample_betas((0, 1))

    start = time.time()
    costs = beta_learner.eval_sampled_betas_parallel_mp()
    end = time.time()
    print(f"Multiprocessing: {end - start}")

    start = time.time()
    costs = beta_learner.eval_sampled_betas_parallel()
    end = time.time()
    print(f"Concurrent futures: {end - start}")
