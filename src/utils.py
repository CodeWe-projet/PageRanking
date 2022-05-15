import csv
from pathlib import Path

import numpy as np

def get_personnalisation_vector():
    CWD = Path(".").absolute()
    vector_csv = CWD / "VecteurPersonnalisation_Groupe20.csv"

    with vector_csv.open("r") as f:
        reader = csv.reader(f)
        return np.array(list(reader)[0], dtype=np.float64)
