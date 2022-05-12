# file: src/main.py
# authors: Theo Technicguy, Alexandre Dewilde
# version: 0.0.1
#
# Main file
#

import csv
from pathlib import Path

import numpy as np

CWD = Path(".").absolute()
vector_csv = CWD / "VecteurPersonnalisation_Groupe20.csv"

with vector_csv.open("r") as f:
    reader = csv.reader(f)
    personnalisation_vector = np.array(list(reader)[0], dtype=np.float64)


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

import data

# Some debug lines
matrix = np.matrix(data.data)
res = pageRankLinear(matrix, 0.9, personnalisation_vector)
print(res)
res = (pageRankPower(matrix, 0.9, personnalisation_vector))
print(res)