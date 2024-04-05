import json
import os
import unittest
import numpy as np

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class TestDataProcessorIntegration(unittest.TestCase):
    def setUp(self):
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        self.data_processor = DataProcessor(self.db_manager)
        self.disease_to_hps = self.data_processor.disease_to_hps
        self.embeddings_dict = self.data_processor.hp_embeddings
        self.disease_to_hps_with_frequencies = self.data_processor.disease_to_hps_with_frequencies

    def test_embeddings_dict_is_not_empty(self):
        first_5 = list(self.embeddings_dict.items())[:5]
        for k,v in first_5:
            print(f"{k}: {v}")

        self.assertTrue(self.embeddings_dict)
        path = "/Users/carlo/Downloads/pheval.exomiser/output"
        result_path = os.path.join(path, "embeddings_dict")
        try:
            with open(result_path, "w") as f:
                for k, v in first_5:
                    f.write(f"{k}: {v}\n")
        except IOError as e:
            print(f"Error writing to file: {e}")

    def test_disease_to_hps_with_frequencies_is_not_empty(self):
        first_5 = list(self.disease_to_hps_with_frequencies.items())[:5]
        for k,v in first_5:
            print(f"{k}: {v}")
        self.assertTrue(self.disease_to_hps_with_frequencies)

    def test_disease_to_hps_is_not_empty(self):
        first_5 = list(self.disease_to_hps.items())[:5]
        for k,v in first_5:
            print(f"{k}: {v}")
        self.assertTrue(self.disease_to_hps)

    def test_convert_embeddings_to_numpy(self):
        numpy_embeddings_dict = self.data_processor.convert_embeddings_to_numpy(self.embeddings_dict)
        first_5 = list(numpy_embeddings_dict.items())[:5]
        for k, v in first_5:
            print(f"{k}: {v}")
        for hp_id, data in numpy_embeddings_dict.items():
            self.assertIsInstance(data['embeddings'], np.ndarray)

    def test_process_data_for_avg_service(self):

        all_data = []

        for disease, hps in self.disease_to_hps.items():
            average_embedding = self.data_processor.calculate_average_embedding(
                hps=hps,
                embeddings_dict=self.embeddings_dict
            )
            all_data.append((disease, average_embedding.tolist()))

        self.assertTrue(all_data)  # should be non-empty
        self.assertLessEqual(len(all_data), 12468)  # ize should not exceed the defined batch_size


if __name__ == '__main__':
    unittest.main()
