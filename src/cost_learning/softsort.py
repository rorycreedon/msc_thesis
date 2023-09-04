import torch
import torch.nn as nn


class SoftSort(nn.Module):
    """
    Code used from paper "SoftSort: A Continuous Relaxation for the argsort Operator" (Prillo and Eisenschlos, 2020)
    Source: https://github.com/sprillo/softsort/blob/master/pytorch/softsort.py
    """

    def __init__(
        self,
        tau: float = 1.0,
        hard: bool = False,
        power: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize the class
        :param tau: temperature parameter
        :param hard: whether to use soft or hard sorting
        :param power: power to use in the semi-metric d
        :param device: device to use
        """
        super(SoftSort, self).__init__()
        self.hard = hard
        self.tau = tau
        self.power = power
        self.device = device

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the class
        :param scores: The scores to be sorted
        :return: The softmax of sorted scores (descending sort)
        """
        scores = scores.unsqueeze(-1)
        sorted = scores.sort(descending=False, dim=1)[0]
        pairwise_diff = (scores.transpose(1, 2) - sorted).abs().pow(
            self.power
        ).neg() / self.tau
        P_hat = pairwise_diff.softmax(-1)

        if self.hard:
            P = torch.zeros_like(P_hat, device=self.device)
            P.scatter_(-1, P_hat.topk(1, -1)[1], value=1)
            P_hat = (P - P_hat).detach() + P_hat
        return P_hat
