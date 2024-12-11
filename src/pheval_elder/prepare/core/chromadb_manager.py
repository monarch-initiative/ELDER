import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Optional, Dict

from chromadb import ClientAPI as API, ClientAPI
import chromadb
from dataclasses import dataclass, field
from chromadb.api.models.Collection import Collection
from chromadb.types import Collection
from sqlalchemy.orm.collections import collection

from pheval_elder.prepare.config import config_loader
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


logger = logging.getLogger(__name__)

@dataclass
class ChromaDBManager:
    collection_name: str = None
    client: API = None
    path: str = None
    ontology: str = field(default="hp")
    strategy: str = field(default="avg") # wgt_avg
    model_shorthand: str = field(default="ada")
    ont_hp: Collection = None
    similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE
    nr_of_phenopackets: str = None

    def __post_init__(self):
        if self.path is None:
            config = config_loader.load_config()
            self.path = config["chroma_db_path"]
            self.client = chromadb.PersistentClient(path=self.path)
        else:
            self.client = chromadb.PersistentClient(path=self.path)
        if self.ont_hp is None and self.collection_name:
            self.ont_hp = self.client.get_collection(self.collection_name)

    @property
    def disease_weighted_avg_embeddings_collection(self) -> Collection:
        return self._get_disease_weighted_avg_embeddings_collection(
            self.collection_name,
            self.strategy,
            self.model_shorthand,
            self.nr_of_phenopackets
        )

    @property
    def disease_avg_embeddings_collection(self) -> Collection:
        return self._get_disease_avg_embeddings_collection(
            self.collection_name,
            self.strategy,
            self.model_shorthand,
            self.nr_of_phenopackets
        )

    def _get_disease_avg_embeddings_collection(
            self,
            collection_name: str,
            avg_strategy: str,
            model_shorthand: str,
            nr_of_phenopackets: str
    ) -> Collection:
        avg_collection = self.get_or_create_collection(
            name=f"{model_shorthand}_{avg_strategy}_{nr_of_phenopackets}_{collection_name}",
        )
        return avg_collection

    def _get_disease_weighted_avg_embeddings_collection(
            self,
            collection_name: str,
            avg_weighted_strategy: str,
            model_shorthand: str,
            nr_of_phenopackets: str
    ) -> Collection:
        avg_wgt_collection = self.get_or_create_collection(
            name=f"{model_shorthand}_{avg_weighted_strategy}_{nr_of_phenopackets}_{collection_name}"        )
        return avg_wgt_collection

    def get_or_create_collection(self, name: str) -> Collection:
        try:
            similarity_str_value = self.similarity.value if self.similarity else SimilarityMeasures.COSINE.value
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={
                    "hnsw:space": similarity_str_value,
                    "hnsw:search_ef": 800,
                    "hnsw:construction_ef": 200,
                    "hnsw:M": 16
                })
            return collection
        except Exception as e:
            raise ValueError(f"Error getting/creating collection {name}: {str(e)}")

    def list_collections(self) -> Sequence[Collection]:
        return self.client.list_collections()
