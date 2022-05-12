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

# A Function to calculate the google matrix from P, alpha and v.
#  G = alpha * P  + (1 - alpha) * e * v.T (where e is a column vector full of 1)
def compute_g(alpha: float, P :np.array, v: np.array):
    return alpha * P + (1 - alpha) * np.ones((len(P),)) * v.T


# Calculate the transition probabilities matrix from a graph A (an adjacent matrix), p_ij = w_ij / w_i
def p_matrix(A :np.matrix) -> np.array:
    return A / A.sum(axis=1)



# A function for the power method


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
    G = compute_g(alpha, P, v)
    M = G.T
    
    # https://en.wikipedia.org/wiki/Power_iteration

    b_k = np.random.rand(M.shape[1]).reshape(-1, 1)

    for _ in range(30):
        b_k1 = np.dot(M, b_k)
        b_k = b_k1 / np.linalg.norm(b_k1)
    return np.ravel((b_k / sum(b_k)))

import data

# Some debug lines
matrix = np.matrix(data.data)
res = pageRankLinear(matrix, 0.9, personnalisation_vector)
print(res)
res = (pageRankPower(matrix, 0.9, personnalisation_vector))
print(res)