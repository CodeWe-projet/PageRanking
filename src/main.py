# file: src/main.py
# authors: Theo Technicguy, Alexandre Dewilde
# version: 0.0.1
#
# Main file
#

import numpy as np

# Calculate the transition probabilities matrix from a graph A (an adjacent matrix), p_ij = w_ij / w_i
def p_matrix(A :np.matrix) -> np.matrix:
    return A / A.sum(axis=1)


def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """Linear implementation of the PageRank algoritm.

    :param A: Regular and even directed graph
    :param alpha: teleportation parameter (0.9) default
    :param v: Personalisation vector
    """
    A_b = (np.identity(len(A)) - alpha * p_matrix(A)).T
    b = (1 - alpha) * v
    res = np.linalg.solve(A_b, b)
    return res / sum(res)


def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """Ex implementation of the PageRank algoritm.

    :param A: Regular and even directed graph
    :param alpha: teleportation parameter (0.9) default
    :param v: Personalisation vector
    """
    P = p_matrix(A)
    G = alpha * P + (1 - alpha) * v.T

    X = oldX = np.ones((len(A), 1)) /len(A)

    while ((X := G.T @ X) != oldX).all():
        oldX = X

    return np.ravel(X.T)