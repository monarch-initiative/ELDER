import logging
from typing import Optional
import chromadb

from pheval_elder.prepare.config import config_loader
from pheval_elder.prepare.config.config_loader import ElderConfig
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


logger = logging.getLogger(__name__)


class ChromaDBManager:
    def __init__(self, config: ElderConfig, similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE):
        # config = config_loader.load_config()
        path = config.chroma_db_path
        self.client = chromadb.PersistentClient(path=path)
        self.ont_hp = self.get_collection("ont_hp")
        self.hpoa = self.get_collection("small_hpoa3233")

        self.hp_embeddings_collection = self.get_collection("small_hpo") or self.create_collection(
            "small_hpo", similarity
        )
        self.disease_avg_embeddings_collection = self.get_collection("small_average") or self.create_collection(
            "small_average", similarity
        )
        self.clustered_embeddings_collection = self.get_collection("small_DiseaseOrganEmbeddings") or self.create_collection(
            "small_DiseaseOrganEmbeddings", similarity
        )

        self.organ_embeddings_collection = self.get_collection("small_Organ_Emb_28_02_24") or self.create_collection("small_Organ_Emb_28_02_24", similarity)
        # TODO: THIS 2 NEW ONES CREATED FOR USING THE OMIM DICT FROM phenotype.hpoa instead hpoa collection
        self.disease_new_avg_embeddings_collection = self.get_collection(
            "small_DiseaseNewAvgEmbeddingsNew") or self.create_collection(
            "small_DiseaseNewAvgEmbeddingsNew", similarity
        )
        self.clustered_new_embeddings_collection = self.get_collection(
            "small_Organ_Emb_0.05") or self.create_collection(
            "small_Organ_Emb_0.05", similarity  # THIS FOR TIMES - 0.1*
        )
        self.disease_weighted_avg_embeddings_collection = self.get_collection(
            "small_model_weighted_average") or self.create_collection(
            "small_model_weighted_average", similarity)

        # The following two are for graph embeddings on WEIGHTED AVERAGE calculations
        self.disease_weighted_avg_line_graph_embeddings_collection = self.get_collection(
            "small_disease_weighted_avg_line_graph_embeddings_collection") or self.create_collection(
            "small_disease_weighted_avg_line_graph_embeddings_collection", similarity)
        self.disease_weighted_avg_deepwalk_graph_embeddings_collection = self.get_collection(
            "small_disease_weighted_avg_deepwalk_graph_embeddings_collection") or self.create_collection(
            "small_disease_weighted_avg_deepwalk_graph_embeddings_collection", similarity)

        # The following two are for graph embeddings for normal AVERAGE calculations
        self.disease_avg_line_graph_embeddings_collection = self.get_collection(
            "small_disease_avg_line_graph_embeddings_collection") or self.create_collection(
            "small_disease_avg_line_graph_embeddings_collection", similarity)
        self.disease_avg_deepwalk_graph_embeddings_collection = self.get_collection(
            "small_disease_avg_deepwalk_graph_embeddings_collection") or self.create_collection(
            "small_disease_avg_deepwalk_graph_embeddings_collection", similarity)



    def create_collection(self, name: str, similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE):
        try:
            similarity_str_value = similarity.value if similarity else SimilarityMeasures.COSINE.value
            collection = self.client.create_collection(name=name, metadata={"hnsw:space": similarity_str_value})
            return collection
        except chromadb.db.base.UniqueConstraintError:
            logger.info(f"Collection {name} already exists")
            return None

    def get_collection(self, name: str):
        try:
            return self.client.get_collection(name)
        except Exception as e:
            logger.info(f"Error getting collection {name}: {str(e)}")
            return None

    def list_collections(self):
        return self.client.list_collections()
