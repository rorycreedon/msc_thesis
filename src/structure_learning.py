import networkx as nx
import numpy as np
import pandas as pd
import torch
from typing import List
import matplotlib.pyplot as plt

from src.dagma.linear import DagmaLinear
from src.dagma.nonlinear import DagmaMLP, DagmaNonlinear


def process_df(df: pd.DataFrame) -> (np.ndarray, List):
    """
    Process the dataframe to be used in the causal model
    :param df: dataframe to be processed
    :return: Dataframe converted to numpy, and the labels
    """
    labels = list(df.columns)
    df = df.to_numpy().astype(np.float32)
    return df, labels


def dagma_linear(
    X: np.ndarray, labels: List, loss_type: str, mask: np.ndarray = None, **kwargs
) -> (nx.DiGraph, np.ndarray):
    """
    Learn a DAG using DAGMA linear
    :param X: np array of data
    :param labels: list of labels
    :param loss_type: loss type to use, either "l2" (much faster) or "logistic"
    :param mask: mask to use for the gradients (must be binary, and same shape as X). If mask[i,j] = 1, then variable with index j cannot be a parent of variable with index i
    :param kwargs: kwargs to pass to the DagmaLinear model (for the .fit() method)
    :return: networkx graph
    """
    # Fit DAGMA linear model to learn a DAG
    model = DagmaLinear(loss_type=loss_type, mask=mask)
    W = model.fit(X, **kwargs)
    print(W)

    # Convert to networkx graph
    G = nx.DiGraph(W.T)

    # Fix labels
    label_mapping = {node: labels[node] for node in G.nodes()}
    G = nx.relabel_nodes(G, label_mapping)

    # Write the graph to a file
    nx.write_gml(G, "graphs/dagma_linear.gml")

    return G, W


def dagma_mlp(
    X: np.ndarray, labels: List, dims: List, mask: np.ndarray = None, **kwargs
) -> (nx.DiGraph, np.ndarray):
    if mask is not None:
        mask = torch.Tensor(mask)

    # Fit DAGMA linear model to learn a DAG
    eq_model = DagmaMLP(dims=dims, bias=True)
    model = DagmaNonlinear(eq_model)
    W = model.fit(X, mask=mask, **kwargs)

    # Convert to networkx graph
    G = nx.DiGraph(W.T)

    # Fix labels
    label_mapping = {node: labels[node] for node in G.nodes()}
    G = nx.relabel_nodes(G, label_mapping)

    # Write the graph to a file
    nx.write_gml(G, "graphs/dagma_MLP.gml")

    return G, W


def plot_graph(
    G: nx.DiGraph,
    fig_size: tuple = (6, 6),
    file_name: str = "dagma_linear.png",
) -> None:
    # Save the graph as a png
    fig = plt.figure(1, figsize=fig_size, dpi=300)
    fig.suptitle("Learned DAG", fontsize=16)
    fig.tight_layout()
    nx.draw_shell(G, with_labels=True, font_weight="normal")
    fig.savefig(f"graphs/{file_name}", bbox_inches="tight")
