from scipy.stats import norm

from structural_models.scm import StructuralCausalModel
from src.structure_learning import dagma_linear, process_df, plot_graph
from src.causal_effect_estimation import causal_effect_estimation


def simulate_data(N):
    # Define the SCM
    scm = StructuralCausalModel(N)
    # Fist variable is a normal distribution
    scm.add_variable(name="X1", distribution=norm, loc=0, scale=1)

    # Unobserved variable U1
    scm.add_variable(name="U1", distribution=norm, loc=0, scale=0.5)

    # X1 and U cause X2
    scm.add_relationship(
        causes={"X1": 0.5, "U1": 1}, effect="X2", noise_dist=norm, loc=0, scale=0
    )

    # Unobserved variable U2
    scm.add_variable(name="U2", distribution=norm, loc=0, scale=0.5)

    # X1 and U cause X2
    scm.add_relationship(
        causes={"X2": 0.5, "U2": 1}, effect="X3", noise_dist=norm, loc=0, scale=0
    )

    # X2 causes Y
    scm.add_binary_outcome(
        name="Y", weights={"X1": 2, "X2": 3, "X3": 2.5}, noise_dist=norm, loc=0, scale=0
    )
    # scm.add_relationship(
    #     causes={"X1": 2, "X2": 3, "X3": 2.5},
    #     effect="Y",
    #     noise_dist=norm,
    #     loc=0.3,
    #     scale=4,
    # )

    # Return object
    df = scm.generate_data()

    return df[["X1", "X2", "X3", "Y"]]


if __name__ == "__main__":
    data = simulate_data(50000)

    # Process the dataframe
    data_numpy, labels = process_df(data)
    # Learn and plot graph
    G, W = dagma_linear(data_numpy, labels, loss_type="l2", mask=None, lambda1=0)
    plot_graph(G, fig_size=(5, 5), file_name="dagma_linear.png")

    # Estimate causal effects
    causal_effects = causal_effect_estimation(
        graph_name="dagma_linear",
        df=data,
        W=W,
        labels=labels,
    )
