from dataclasses import dataclass

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from pheval_elder.prepare.core.query_service import (QueryService)
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


@dataclass
class ElderRunner:
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE
    def __post_init__(self):
        """
        Data Resources
        """
        self.db_manager = ChromaDBManager(similarity=self.similarity_measure)
        self.data_processor = DataProcessor(db_manager=self.db_manager)
        #
        # self.hp_service = HPEmbeddingService(data_processor=self.data_processor)
        self.disease_service = DiseaseAvgEmbeddingService(data_processor=self.data_processor)
        self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(data_processor=self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        _ = self.data_processor.disease_to_hps_with_frequencies

    def setup_collections(self):
        # self.hp_service.process_data()
        self.disease_service.process_data()

    def run_analysis(self, input_hpos):
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
            weighted_average_llm_embedding_service=self.disease_weighted_service,
        )
        return query_service.query_for_average_llm_embeddings_collection_top10_only(input_hpos)
