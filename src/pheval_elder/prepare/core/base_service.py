from abc import ABC, abstractmethod

from chromadb.types import Collection

from src.pheval_elder.prepare.core.data_processor import DataProcessor
from src.pheval_elder.prepare.core.graph_data_processor import GraphDataProcessor

# from pheval_elder.prepare.elder_core.data_processor import DataProcessor
# from pheval_elder.prepare.elder_core.graph_data_processor import GraphDataProcessor

"""
    Interface for HPEmbeddingsService & DiseaseAvgEmbeddingsService
    All methods must be implemented by subclasses.
"""


class BaseService(ABC):

    # TODO: GraphDataprocessor is basically having same func as DataProcessor but HP embeddings  and calc functions
    # TODO: are differing. DiseaseToHP Dicts are same however, so better to keep that in a different one and make others utils

    def __init__(self, data_processor: DataProcessor, graph_data_processor: GraphDataProcessor):
        # Data and util LLM Embeddings
        self.data_processor = data_processor

        # Data and util Graph Embeddings
        self.graph_data_processor = graph_data_processor

        # Hp to Embedding Dict
        self.hp_embeddings = data_processor.hp_embeddings
        self.line_graph_embeddings = graph_data_processor.line_graph_embeddings()
        self.deepwalk_graph_emebddings = graph_data_processor.deepwalk_graph_embeddings()
        # Hp to Embedding collection (not needed tbh as that HPEmbeddingservice is deprecated)
        self.hp_embeddings_collection = data_processor.db_manager.hp_embeddings_collection

        # Disease to HPs
        self.disease_to_hps = data_processor.disease_to_hps

        # Disease to HPs with frequencies
        self.disease_to_hps_with_frequencies_dp = data_processor.disease_to_hps_with_frequencies
        self.disease_to_hps_with_frequencies_gpd = graph_data_processor.disease_to_hp_with_frequencies()

        # Average LLM Embedding
        self.disease_new_avg_embeddings_collection = data_processor.db_manager.disease_new_avg_embeddings_collection

        # Organ Vector LLM Embeddings
        self.clustered_new_embeddings_collection = data_processor.db_manager.clustered_new_embeddings_collection

        # Weighted Average LLM Embeddings
        self.disease_weighted_avg_embeddings_collection = data_processor.db_manager.disease_weighted_avg_embeddings_collection

        # Weighted Average Line-Embeddings
        self.disease_weighted_avg_line_graph_embeddings_collection = data_processor.db_manager.disease_weighted_avg_line_graph_embeddings_collection
        self.disease_weighted_avg_deepwalk_graph_embeddings_collection = data_processor.db_manager.disease_weighted_avg_deepwalk_graph_embeddings_collection

        # Average Deepwalk-Embeddings
        self.disease_avg_line_graph_embeddings_collection = data_processor.db_manager.disease_avg_line_graph_embeddings_collection
        self.disease_avg_deepwalk_graph_embeddings_collection = data_processor.db_manager.disease_avg_deepwalk_graph_embeddings_collection

    @abstractmethod
    def process_data(self) -> Collection:
        pass

    @abstractmethod
    def upsert_batch(self, *args, **kwargs):
        pass
