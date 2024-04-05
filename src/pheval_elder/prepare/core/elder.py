# pheval_elder.py
from src.pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from src.pheval_elder.prepare.core.data_processor import DataProcessor
from src.pheval_elder.prepare.core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
from src.pheval_elder.prepare.core.disease_clustered_emb_service import DiseaseClusteredEmbeddingService
from src.pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from src.pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor
from src.pheval_elder.prepare.core.graph_deepwalk_avg_service import GraphDeepwalkAverageEmbeddingService
from src.pheval_elder.prepare.core.graph_deepwalk_weighted_service import GraphDeepwalkWeightedEmbeddingService
from src.pheval_elder.prepare.core.graph_embedding_extractor import GraphEmbeddingExtractor
from src.pheval_elder.prepare.core.graph_line_avg_service import GraphLineAverageEmbeddingService
from src.pheval_elder.prepare.core.graph_line_weighted_service import GraphLineWeightedEmbeddingService
from src.pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from src.pheval_elder.prepare.core.hpo_clustering import HPOClustering
from src.pheval_elder.prepare.core.query_service import QueryService
from src.pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


# from pheval_elder.prepare.elder_core.chromadb_manager import ChromaDBManager
# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
# from pheval_elder.prepare.elder_core.disease_avg_embedding_service import DiseaseAvgEmbeddingService
# from pheval_elder.prepare.elder_core.disease_clustered_emb_service import DiseaseClusteredEmbeddingService
# from pheval_elder.prepare.elder_core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor
# from pheval_elder.prepare.elder_core.graph_deepwalk_avg_service import GraphDeepwalkAverageEmbeddingService
# from pheval_elder.prepare.elder_core.graph_deepwalk_weighted_service import GraphDeepwalkWeightedEmbeddingService
# from pheval_elder.prepare.elder_core.graph_embedding_extractor import GraphEmbeddingExtractor
# from pheval_elder.prepare.elder_core.graph_line_avg_service import GraphLineAverageEmbeddingService
# from pheval_elder.prepare.elder_core.graph_line_weighted_service import GraphLineWeightedEmbeddingService
# from pheval_elder.prepare.elder_core.hp_embedding_service import HPEmbeddingService
# from pheval_elder.prepare.elder_core.hpo_clustering import HPOClustering
# from pheval_elder.prepare.elder_core.query_service import QueryService
# from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class ElderRunner:
    def __init__(self, similarity_measure=SimilarityMeasures.COSINE):
        """
        Data Resources
        """
        self.db_manager = ChromaDBManager(
            similarity=similarity_measure
        )
        self.extractor = GraphEmbeddingExtractor()
        self.graph_data_processor = GraphDataProcessor(
            extractor=self.extractor,
            manager=self.db_manager
        )
        self.data_processor = DataProcessor(db_manager=self.db_manager)

        """
        LLM Embedding services
        """
        self.hp_service = HPEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )
        self.disease_service = DiseaseAvgEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )
        self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )

        """
        Organ vector context
        """
        self.hpo_clustering = HPOClustering()
        self.disease_organ_service = DiseaseClusteredEmbeddingService(
            data_processor=self.data_processor,
            hpo_clustering=self.hpo_clustering,
            graph_data_processor=self.graph_data_processor,
        )

        """
        Graph Embedding services
        """
        self.graph_deepwalk_average_service = GraphDeepwalkAverageEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )
        self.graph_deepwalk_weighted_service = GraphDeepwalkWeightedEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )
        self.graph_line_average_service = GraphLineAverageEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )
        self.graph_line_weighted_service = GraphLineWeightedEmbeddingService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor
        )



    def initialize_data(self):
        _ = self.data_processor.hp_embeddings
        _ = self.data_processor.disease_to_hps
        _ = self.data_processor.disease_to_hps_with_frequencies
        _ = self.graph_data_processor.line_graph_embeddings
        _ = self.graph_data_processor.deepwalk_graph_embeddings

    def setup_collections(self):
        self.hp_service.process_data()
        self.disease_service.process_data()
        # self.disease_organ_service.process_data()
        # self.disease_weighted_service.process_data()
        # self.graph_line_average_service.process_data()
        # self.graph_line_weighted_service.process_data()
        # self.graph_deepwalk_average_service.process_data()
        # self.graph_deepwalk_weighted_service.process_data()

    def run_analysis(self, input_hpos):  # sim strategy can be going in later
        query_service = QueryService(
            data_processor=self.data_processor,
            graph_data_processor=self.graph_data_processor,
            db_manager=self.db_manager,
            average_llm_embedding_service=self.disease_service,
            organ_vector_service=self.disease_organ_service,
            weighted_average_llm_embedding_service=self.disease_weighted_service,
            average_deepwalk_graph_embedding_service=self.graph_deepwalk_average_service,
            average_line_graph_embedding_service=self.graph_line_average_service,
            weighted_average_deepwalk_graph_embedding_service=self.graph_deepwalk_weighted_service,
            weighted_average_line_graph_embedding_service=self.graph_line_weighted_service


        )
        return query_service.query_for_average_llm_embeddings_collection_top10_only(input_hpos)
