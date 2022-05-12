import unittest
import networkx as nx
import numpy as np
import csv

import data
import main

class TestPageRank(unittest.TestCase):
    def setUp(self):
        self.M = np.matrix(data.data)
        self.result = list(nx.pagerank(
            nx.from_numpy_array(np.array(data.data), create_using=nx.DiGraph),
            alpha=0.9,
            personalization={i: n for i, n in enumerate(main.personnalisation_vector)}
        ).values())

    def test_linear(self):
        np.testing.assert_almost_equal(
            list(main.pageRankLinear(self.M, 0.9, main.personnalisation_vector)),
            self.result,
            decimal=4
        )

    def test_power(self):
        np.testing.assert_almost_equal(
            list(main.pageRankPower(self.M, 0.9, main.personnalisation_vector)),
            self.result,
            decimal=4
        )


if __name__ == '__main__':
    unittest.main()

# # Some debug lines
# matrix = np.matrix(data.data)
# res = pageRankLinear(matrix, 0.9, personnalisation_vector)
# print(res)
# res = (pageRankPower(matrix, 0.9, personnalisation_vector))
# print(res)