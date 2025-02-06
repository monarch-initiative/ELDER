import logging
from collections.abc import Sequence
from typing import Optional, ClassVar, Iterable

import numpy as np
from chromadb import ClientAPI as API
import chromadb
from dataclasses import dataclass, field

from chromadb.types import Collection
from oaklib.utilities.iterator_utils import chunk
from pydantic import ValidationError
from venomx.model.venomx import Index

from pheval_elder.metadata.metadata import Metadata
from pheval_elder.prepare.config import config_loader
from pheval_elder.prepare.core.utils.utils import populate_venomx, normalize_metadata
from pheval_elder.prepare.core.utils.similarity_measures import SimilarityMeasures


logger = logging.getLogger(__name__)

@dataclass
class ChromaDBManager:
    name: ClassVar[str] = "chromadb"
    collection_name: str = None
    client: API = None
    path: str = None
    ontology: str = field(default="hp")
    strategy: str = field(default="avg") # wgt_avg
    model_shorthand: str = field(default="ada") # should no default
    ont_hp: Collection = None
    similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE
    nr_of_phenopackets: str = None
    nr_of_results: int = None
    auto_create: bool = False

    def __post_init__(self):
        if not self.auto_create:
            if self.collection_name is None:
                raise RuntimeError(f"Collection name of embedded HP data (curateGPT output) must be provided.")  # lrd_hpo, label_hpo, defintion_hpo, relationships_hpo
            if self.path is None:
                config = config_loader.load_config()
                self.path = config["chroma_db_path"]
                self.client = chromadb.PersistentClient(path=self.path)
            else:
                self.client = chromadb.PersistentClient(path=self.path)
            if self.ont_hp is None and self.collection_name and self.auto_create == False:
                self.ont_hp = self.client.get_collection(self.collection_name)
        else:
            self.handle_auto_create()
        # if self.auto_create:
        #     self.ont_hp = self.client.get_or_create_collection(self.collection_name)

    @property
    def disease_weighted_avg_embeddings_collection(self) -> Collection:
        return self._get_disease_weighted_avg_embeddings_collection(
            self.collection_name,
            self.strategy,
            self.model_shorthand,
            self.nr_of_phenopackets,
            self.nr_of_results
        )

    @property
    def disease_avg_embeddings_collection(self) -> Collection:
        return self._get_disease_avg_embeddings_collection(
            self.collection_name,
            self.strategy,
            self.model_shorthand,
            self.nr_of_phenopackets,
            self.nr_of_results
        )

    def _get_disease_avg_embeddings_collection(
            self,
            collection_name: str,
            avg_strategy: str,
            model_shorthand: str,
            nr_of_phenopackets: str,
            nr_of_results: int

    ) -> Collection:
        avg_collection = self.get_or_create_collection(
            name=f"{model_shorthand}_{avg_strategy}_{nr_of_phenopackets}_{collection_name}_top_{nr_of_results}_results",
        )
        return avg_collection

    def _get_disease_weighted_avg_embeddings_collection(
            self,
            collection_name: str,
            avg_weighted_strategy: str,
            model_shorthand: str,
            nr_of_phenopackets: str,
            nr_of_results: int
    ) -> Collection:
        avg_wgt_collection = self.get_or_create_collection(
            name=f"{model_shorthand}_{avg_weighted_strategy}_{nr_of_phenopackets}_{collection_name}_top_{nr_of_results}_results")
        return avg_wgt_collection

    def get_or_create_collection(self, name: str) -> Collection:
        try:
            similarity_str_value = self.similarity.value if self.similarity else SimilarityMeasures.COSINE.value
            collection = self.client.get_or_create_collection(
                name=name,
                # TODO: this is for HPO, adjust for others (maybe a bool flag in params, to adjust)
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

    """Huggingface insertion"""
    def insert_from_huggingface(
        self,
        objs: Iterable[dict],
        collection: str = None,
        batch_size: int = None,
        venomx: Optional[Metadata] = None,
        method_name = "add",
        **kwargs,
    ):
        client = self.client
        model = None

        try:
            if venomx:
                hf_metadata_model = venomx.venomx.embedding_model.name
                if hf_metadata_model:
                    model = hf_metadata_model
        except Exception as e:
            raise KeyError(f"Metadata from {collection} is not compatible with the current version of CurateGPT") from e

        venomx = populate_venomx(collection, model, venomx.venomx)
        cm = self.update_collection_metadata(
            collection_name=collection,
            venomx=venomx
        )
        adapter_metadata = cm.serialize_venomx_metadata_for_adapter(self.name)
        self.ont_hp = client.get_or_create_collection(
            name=collection,
            metadata=adapter_metadata,
        )
        if batch_size is None:
            batch_size = 1000

        for next_objs in chunk(objs, batch_size):
            next_objs = list(next_objs)
            ids = [item['metadata']['id'] for item in next_objs]
            metadatas = [normalize_metadata(o) for o in next_objs]
            documents = [item['document'] for item in next_objs]
            embeddings = [item['embeddings'].tolist() if isinstance(item['embeddings'], np.ndarray)
                          else item['embeddings'] for item in next_objs]
            method = getattr(self.ont_hp, method_name)
            method(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

    def update_collection_metadata(self, collection_name: str, **kwargs) -> Metadata:
        """
        set the metadata for a collection downloaded from hf.
        """
        # self.ont_hp = self.client.get_or_create_collection(
        #     name=self.collection_name,
        # )

        metadata = Metadata(
            venomx=kwargs.get("venomx"),
            hnsw_space=kwargs.get("hnsw_space", "cosine"),
            object_type=kwargs.get("object_type"),
        )

        chromadb_metadata = metadata.serialize_venomx_metadata_for_adapter(self.name)
        self.client.get_or_create_collection(
            name=collection_name,
            metadata=chromadb_metadata
        )
        return metadata


    def collection_metadata(
        self, collection_name: Optional[str] = None, include_derived=False, **kwargs
    ) -> Optional[Metadata]:
        """
        Get the metadata for a collection.

        :param collection_name:
        :return:

        Parameters
        ----------
        """
        try:
            collection_obj = self.client.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            return None
        metadata_data = {**collection_obj.metadata, **kwargs}
        try:
            cm = Metadata.deserialize_venomx_metadata_from_adapter(metadata_data, self.name)
        except ValidationError as ve:
            logger.error(f"Deserializing failed. Creating clean and empty venomx object for insertion. Metadata validation error: {ve}")
            cm = Metadata(venomx=Index())

        if include_derived:
            try:
                logger.info(f"Getting object count for {collection_name}")
                cm.object_count = collection_obj.count()
            except Exception as e:
                logger.error(f"Failed to get object count: {e}")
        return cm


    def handle_auto_create(self):
        if self.path is None:
            config = config_loader.load_config()
            self.path = config["chroma_db_path"]
            self.client = chromadb.PersistentClient(path=self.path)
        else:
            self.client = chromadb.PersistentClient(path=self.path)
        self.ont_hp = self.client.get_or_create_collection(self.collection_name)