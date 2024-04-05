import unittest
import json

import numpy as np

from pheval_elder.prepare.core.OMIMHPOExtractor import OMIMHPOExtractor
from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor


class TestDataProcessor(unittest.TestCase):

    # def test_create_hpo_id_to_embedding(self):
    #     dbmanager = ChromaDBManager()
    #     processor = DataProcessor(db_manager=dbmanager)
    #     ont = dbmanager.get_collection("ont_hp")
    #     obj = processor.create_hpo_id_to_embedding(ont)
    #     OMIMHPOExtractor.save_results_as_pretty_json_string(obj, "embeddings_json.json")

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

    def test_calculate_average_embedding_weighted_first5(self):
        # Mocking data
        hps_with_frequencies = {
            "OMIM:609153": {
                            "HP:0000006": 0.5,
                            "HP:0002153": 0.5,
                            "HP:0002378": 0.5,
                            "HP:0003324": 0.5,
                            "HP:0003394": 0.5,
                            "HP:0003768": 0.5
                             },
            "OMIM:610370": {
                            "HP:0000007": 0.5,
                            "HP:0001508": 0.5,
                            "HP:0001944": 0.5,
                            "HP:0002013": 0.5,
                            "HP:0002014": 0.5,
                            "HP:0003623": 0.5,
                            "HP:0004918": 0.5
                             },
        }

        embeddings_dict = {
            "HP:0000001": {"embeddings": [1, 2, 3]},
            "HP:0000002": {"embeddings": [4, 5, 6]},
            "HP:0000003": {"embeddings": [7, 8, 9]},
            "HP:0000004": {"embeddings": [10, 11, 12]}
        }

        embeddings_dict = DataProcessor.convert_embeddings_to_numpy(embeddings_dict)
        expected_output = np.array([3.88888889, 4.88888889, 5.88888889])
        result = DataProcessor.calculate_weighted_llm_embeddings(hps_with_frequencies, embeddings_dict)
        np.testing.assert_array_almost_equal(result, expected_output)

    # def integration_test(self):
    #     dbmanager = ChromaDBManager()
    #     processor = DataProcessor(db_manager=dbmanager)
    #     ont = dbmanager.get_collection("ont_hp")
    #     hp_to_embedding = processor.create_hpo_id_to_embedding(ont)
    #




