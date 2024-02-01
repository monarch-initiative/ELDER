
import logging
from typing import Optional
import chromadb
from utils.similarity_measures import SimilarityMeasures
from config import config_loader
logger = logging.getLogger(__name__)


class ChromaDBManager:
    def __init__(self, similarity: Optional[SimilarityMeasures] = SimilarityMeasures.COSINE):
        config = config_loader.load_config()
        path = config['chroma_db_path']
        # if path is None:
        #     config = self.load_config()
        #     path = config['chroma_db_path']
        self.client = chromadb.PersistentClient(path=path)
        self.ont_hp = self.get_collection("ont_hp")
        self.hpoa = self.get_collection("hpoa")
        #DiseaseA : [avegraeVector of Disease]
        self.hp_embeddings_collection = self.get_collection("HPtoEmbeddings") or self.create_collection("HPtoEmbeddings", similarity)
        self.disease_avg_embeddings_collection = self.get_collection("DiseaseAvgEmbeddings") or self.create_collection("DiseaseAvgEmbeddings", similarity)
        self.clustered_embeddings_collection = self.get_collection("DiseaseOrganEmbeddings") or self.create_collection(
            "DiseaseOrganEmbeddings", similarity
        )

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