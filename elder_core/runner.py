# runner.py
from elder_core.chromadb_manager import ChromaDBManager
from elder_core.data_processor import DataProcessor
from elder_core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from elder_core.disease_clustered_embedding_service import DiseaseClusteredEmbeddingService
from elder_core.hp_embedding_service import HPEmbeddingService
from elder_core.hpo_clustering import HPOClustering
from elder_core.query_service import QueryService
from utils.similarity_measures import SimilarityMeasures
import time


class Runner:
    def __init__(self, similarity_measure=SimilarityMeasures.COSINE):
        self.db_manager = ChromaDBManager(similarity=similarity_measure)
        self.data_processor = DataProcessor(self.db_manager)
        self.hp_service = HPEmbeddingService(self.data_processor)
        self.disease_service = DiseaseAvgEmbeddingService(self.data_processor)
        self.hpo_clustering = HPOClustering()
        self.disease_organ_service = DiseaseClusteredEmbeddingService(self.data_processor, hpo_clustering=self.hpo_clustering)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps

    def setup_collections(self):
        self.hp_service.process_data()
        self.disease_service.process_data()
        self.disease_organ_service.process_data()

    def run_analysis(self, input_hpos):  # sim strategy can be going in later
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            disease_service=self.disease_service,
            disease_organ_service=self.disease_organ_service
        )
        return query_service.query_diseases_using_organ_syst_embeddings(input_hpos)
