"""Testing one Service"""


import time
import unittest

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor
from pheval_elder.prepare.core.graph_embedding_extractor import GraphEmbeddingExtractor
from pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class TestDiseaseWeightedAvgEmbeddingService(unittest.TestCase):
    def setUp(self):
        start = time.time()
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        self.data_processor = DataProcessor(self.db_manager)
        self.extractor = GraphEmbeddingExtractor()
        self.graph_data_processor = GraphDataProcessor(self.extractor, self.db_manager)
        self.hp_service = HPEmbeddingService(self.data_processor, self.graph_data_processor)
        self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(self.data_processor, self.graph_data_processor)
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps = self.data_processor.disease_to_hps_with_frequencies
        end = time.time()
        print(end-start)
    def test_process_data(self):
        start = time.time()
        self.disease_weighted_service.process_data()
        end = time.time()
        print(end-start)



if __name__ == '__main__':
    unittest.main()
