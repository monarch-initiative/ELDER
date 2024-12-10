# pheval_elder.py

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.query_service import QueryService
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures

#
"""
 this class is supposed to use the collectons that are there already without a setup and only the initializing 
 phase of the data

"""
class CustomElderRunner:
    def __init__(self, similarity_measure=SimilarityMeasures.COSINE):
        self.db_manager = ChromaDBManager(similarity=similarity_measure)
        self.data_processor = DataProcessor(self.db_manager)
        self.disease_service = DiseaseAvgEmbeddingService(self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps

    def setup_collections(self):
        self.disease_service.process_data()

    def run_analysis(self, input_hpos):
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
        )
        return query_service.query_for_average_llm_embeddings_collection_top10_only(input_hpos)
