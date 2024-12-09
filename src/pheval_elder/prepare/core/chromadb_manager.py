import logging
from typing import Optional, Dict

from chromadb import ClientAPI as API
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
    collection_name: str
    client: API = None
    path: str = None
    ontology: str = field(default="hp")
    avg_collection_name: str = field(default="avg")
    avg_weighted_collection_name: str = field(default="wgt_avg")
    model_shorthand: str = field(default="ada")
    ont_hp: Collection = None
    similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE
    hp_embeddings_collection: Collection = None
    disease_avg_embeddings_collection: Collection = None
    disease_weighted_avg_embeddings_collection: Collection = None

    def __post_init__(self):
        if self.path is None:
            config = config_loader.load_config()
            self.path = config["chroma_db_path"]
            self.client = chromadb.PersistentClient(path=self.path)
        if self.ont_hp is None:
            self.ont_hp = self.client.get_collection(self.collection_name)


        self.hp_embeddings_collection = self.get_or_create_collection(
            f"{self.model_shorthand}_{self.ontology}_{self.collection_name}"
        )
        self.disease_avg_embeddings_collection = self.get_or_create_collection(
            f"{self.model_shorthand}_{self.avg_collection_name}_{self.collection_name}"
        )
        self.disease_weighted_avg_embeddings_collection = self.get_or_create_collection(
            f"{self.model_shorthand}_{self.avg_weighted_collection_name}_{self.collection_name}"
        )

    def get_or_create_collection(self, name: str):
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

    def list_collections(self):
        return self.client.list_collections()
