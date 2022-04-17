# file: src/main.py
# authors: Theo Technicguy
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
    personnalisation_vector = list(reader)[0]

# A Function to calculate the google matrix from P, alpha and v.
#  G = alpha * P  + (1 - alpha) * e * v.T (where e is a column vector full of 1)


# Calculate the transition probabilities matrix from a graph A (an adjacent matrix), p_ij = w_ij / w_i



# A function for the power method


def pageRankLinear(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """Linear implementation of the PageRank algoritm.

    :param A: Regular and even directed graph
    :param alpha: teleportation parameter (0.9) default
    :param v: Personalisation vector
    """

    return

def pageRankPower(A: np.matrix, alpha: float, v: np.array) -> np.array:
    """Ex implementation of the PageRank algoritm.

    :param A: Regular and even directed graph
    :param alpha: teleportation parameter (0.9) default
    :param v: Personalisation vector
    """

    return