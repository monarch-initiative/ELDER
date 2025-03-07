from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict

from chromadb.types import Collection

from pheval_elder.prepare.core.data_processing.data_processor import DataProcessor


@dataclass
class BaseService(ABC):

    data_processor: Optional[DataProcessor]
    disease_to_hps: Dict = None
    disease_to_hps_with_frequencies_dp: Dict = None
    disease_new_avg_embeddings_collection: Collection = None
    disease_weighted_avg_embeddings_collection: Collection = None

    def __post_init__(self):
        self.hp_embeddings = self.data_processor.hp_embeddings
        self.disease_to_hps = self.data_processor.disease_to_hps
        self.disease_to_hps_with_frequencies_dp = self.data_processor.disease_to_hps_with_frequencies
        self.disease_new_avg_embeddings_collection = self.data_processor.db_manager.disease_avg_embeddings_collection
        self.disease_weighted_avg_embeddings_collection = self.data_processor.db_manager.disease_weighted_avg_embeddings_collection


    @abstractmethod
    def process_data(self) -> Collection:
        pass

    @abstractmethod
    def upsert_batch(self, *args, **kwargs):
        pass
