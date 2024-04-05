import unittest
import json

import numpy as np

from pheval_elder.prepare.core.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    def test_calculate_average_embedding_weighted(self):
        # Mocking data
        hps_with_frequencies = {
            "HP:0000001": 0.5,
            "HP:0000002": 0.5,
            "HP:0000003": 0.25,
            "HP:0000004": 0.1
        }

        embeddings_dict = {
            "HP:0000001": {"embeddings": [1, 2, 3]},
            "HP:0000002": {"embeddings": [4, 5, 6]},
            "HP:0000003": {"embeddings": [7, 8, 9]},
            "HP:0000004": {"embeddings": [10, 11, 12]}
        }

        embeddings_dict = DataProcessor.convert_embeddings_to_numpy(embeddings_dict)
        expected_output = np.array([3.88888889, 4.88888889, 5.88888889])
        result = DataProcessor.calculate_average_embedding_weighted(hps_with_frequencies, embeddings_dict)
        np.testing.assert_array_almost_equal(result, expected_output)

