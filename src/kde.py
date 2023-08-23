import torch


class GaussianKDE:
    """Vectorised PyTorch port of scipy.stats.gaussian_kde.integrate_box_1d over multiple dimensions."""

    def __init__(self, data: torch.Tensor, device: str = "cpu"):
        """
        Initialise the KDE.
        :param data: A tensor of shape (N, D) where N is the number of samples and D is the number of dimensions.
        """
        N, D = data.shape
        self.device = device
        self.data = data.to(self.device)
        bandwidth = N ** (-1 / (1 + 4))  # Scott's rule
        covariance = torch.cov(self.data.T)
        self.stdev = torch.sqrt(bandwidth**2 * covariance.diagonal())

    def integrate_neg_inf(self, high: torch.Tensor):
        """
        Integrate from -inf to high.
        :param high: A tensor of high values to integrate to. Must be the same shape as indices.
        :param indices: A tensor of indices which indicate which index of the data's KDE to use
        :return: Values of the integral
        """
        high = high.to(self.device)
        # indices = indices.to(self.device)
        #
        # # Using advanced indexing to pick the relevant KDE dimension for each low and high value
        # selected_data = self.data[:, indices]
        # selected_stdev = self.stdev[indices]

        # normalised_high = (high - selected_data) / selected_stdev

        # Expand data and stdev for broadcasting
        expanded_data = self.data[:, None, :]
        expanded_stdev = self.stdev[None, :]

        normalised_high = (high - expanded_data) / expanded_stdev

        return torch.mean(torch.special.ndtr(normalised_high), dim=0)


if __name__ == "__main__":
    # gen data
    N = 10000
    D = 5
    means = torch.arange(0, D * 0.2, 0.2)
    data = torch.randn(N, D) + means

    # fit KDE
    kde = GaussianKDE(data)
    highs = torch.tensor([[0, 0.2, 0.4, 0.6, 0.8], [0, -0.2, -0.4, -0.6, -0.8]])
    print(kde.integrate_neg_inf(highs))
