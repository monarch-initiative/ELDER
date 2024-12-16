from dataclasses import dataclass

from pheval_elder.prepare.core.store.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor
from pheval_elder.prepare.core.collections.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.collections.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.query.query_service import (QueryService)
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures

@dataclass
class ElderRunner:
    collection_name: str
    embedding_model: str
    strategy: str
    nr_of_phenopackets: str
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE
    results_dir_name: str = None
    results_sub_dir: str = None

    def __post_init__(self):
        """
        Data Resources
        """
        # self.strategy = self.strategy.lower()
        self.embedding_model = self.embedding_model.lower()
        self.results_dir_name = self.embedding_model + "_" + self.strategy + "_" + "pheval_disease_results"
        self.results_sub_dir = self.embedding_model + "_" + self.strategy + "_" + self.collection_name
        self.db_manager = ChromaDBManager(
            similarity=self.similarity_measure,
            collection_name=self.collection_name,
            model_shorthand=self.embedding_model,
            strategy=self.strategy,
            nr_of_phenopackets=self.nr_of_phenopackets
        )

        self.data_processor = DataProcessor(db_manager=self.db_manager)
        self.disease_service = DiseaseAvgEmbeddingService(data_processor=self.data_processor)
        self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(data_processor=self.data_processor)

    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        _ = self.data_processor.disease_to_hps_with_frequencies

    def setup_collections(self):
        self.disease_service.process_data()

    def avg_analysis(self, input_hpos):

        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
        )
        avg_strategy_result = query_service.query_for_average_llm_embeddings_collection_top10_only(input_hpos)
        return avg_strategy_result

    def wgt_avg_analysis(self, input_hpos):
        query_service = QueryService(
            data_processor=self.data_processor,
            db_manager=self.db_manager,
            weighted_average_llm_embedding_service=self.disease_weighted_service,
        )
        weighted_avg_strategy_result = query_service.query_for_weighted_average_llm_embeddings_collection_top10_only(input_hpos)
        return weighted_avg_strategy_result

