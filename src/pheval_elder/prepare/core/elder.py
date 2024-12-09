from dataclasses import dataclass

from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.query_service import (QueryService)
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures

# path="/Users/carlo/chromadb/ada-002-hp"
# [Collection(id=2e270046-a9e6-4760-918c-547dab2da7c9, name=definition_hpo),
#  Collection(id=3be236f1-0d0e-4e28-8958-df670d73947f, name=relationships_hpo),
#  Collection(id=3fa252d3-89ff-4fea-83bb-c56fea815910, name=label_hpo),
#  Collection(id=6b589641-3e05-4b01-92ce-163665f06b72, name=lrd_hpo),
#  Collection(id=6ff54e05-10b4-45ab-83ad-8ca9b1c42558, name=definitions_hpo)]
@dataclass
class ElderRunner:
    collection_name: str
    embedding_model: str
    strategy: str
    similarity_measure: SimilarityMeasures = SimilarityMeasures.COSINE
    results_dir_name: str = None
    results_sub_dir: str = None
    def __post_init__(self):
        """
        Data Resources
        """
        self.strategy = self.strategy.lower()
        self.embedding_model = self.embedding_model.lower()
        self.results_dir_name = self.embedding_model + "_" + self.strategy + "_" + "pheval_disease_results"
        self.results_sub_dir = self.embedding_model + "_" + self.strategy + "_" + self.collection_name
        self.db_manager = ChromaDBManager(
            similarity=self.similarity_measure,
            collection_name=self.collection_name,
            model_shorthand=self.embedding_model,
            avg_collection_name=self.strategy,
        )
        self.data_processor = DataProcessor(db_manager=self.db_manager)
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

