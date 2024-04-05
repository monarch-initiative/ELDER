from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_clustered_emb_service import DiseaseClusteredEmbeddingService
from pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from pheval_elder.prepare.core.hpo_clustering import HPOClustering
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class Main:
    def __init__(self):
        self.output_dir = "output"
        self.runner = None
        self.results = []

    def letsgohpo(self):
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        self.data_processor = DataProcessor(self.db_manager)
        self.hp_service = HPEmbeddingService(self.data_processor)
        self.hpo_clust = HPOClustering()
        self.disease_service = DiseaseClusteredEmbeddingService(self.data_processor, self.hpo_clust)
        self.hpo = self.disease_service.get_all_clusters()

def main():
    print("init_main")
    main_instance = Main()
    print("letsgohpo")
    main_instance.letsgohpo()


if __name__ == "__main__":
    main()