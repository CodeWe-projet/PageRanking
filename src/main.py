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
