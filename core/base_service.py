from abc import ABC, abstractmethod

from chromadb.types import Collection

from core.data_processor import DataProcessor

"""
    Interface for HPEmbeddingsService & DiseaseAvgEmbeddingsService
    All methods must be implemented by subclasses.
"""


class BaseService(ABC):
    def __init__(self, data_processor: DataProcessor):
        # just that
        self.data_processor = data_processor
        # put into __init__ hpService
        self.hp_embeddings = data_processor.hp_embeddings
        self.hp_embeddings_collection = data_processor.db_manager.hp_embeddings_collection
        # put into init diseaseAvgService
        self.disease_to_hps = data_processor.disease_to_hps
        self.disease_avg_embeddings_collection = data_processor.db_manager.disease_avg_embeddings_collection

    @abstractmethod
    def process_data(self) -> Collection:
        pass
    @abstractmethod
    def upsert_batch(self, batch):
        pass
