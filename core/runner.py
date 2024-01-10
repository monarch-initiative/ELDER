# runner.py
from core.chromadb_manager import ChromaDBManager
from core.data_processor import DataProcessor
from core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from core.hp_embedding_service import HPEmbeddingService
from core.query_service import QueryService
from utils.similarity_measures import SimilarityMeasures
import time


class Runner:
    def __init__(self, similarity_measure=SimilarityMeasures.COSINE):
        self.db_manager = ChromaDBManager(similarity=similarity_measure)
        self.data_processor = DataProcessor(self.db_manager)
        self.hp_service = HPEmbeddingService(self.data_processor)
        self.disease_service = DiseaseAvgEmbeddingService(self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps

    def setup_collections(self):
        self.hp_service.process_data()
        self.disease_service.process_data()

    def run_analysis(self, input_hpos): # sim strategy can be going in later
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            disease_service=self.disease_service,
        )
        return query_service.query_diseases_by_hpo_terms_using_inbuild_distance_functions(input_hpos)
