
import time
import os
from pheval_elder.main.constants import allfromomim619340
from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_clustered_emb_service import DiseaseClusteredEmbeddingService
from pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.elder import ElderRunner
from pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from pheval_elder.prepare.core.hpo_clustering import HPOClustering
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class Main:
    def __init__(self):
        self.output_dir = "output"
        self.runner = None
        self.results = []

    def prepare(self):
        start = time.time()
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        end = time.time()
        print(f"ChromaDBManager: {end-start}")

        start = time.time()
        self.data_processor = DataProcessor(self.db_manager)
        end = time.time()
        print(f"DataProcessor: {end-start}")

        # self.hp_embeddings = self.data_processor.hp_embeddings
        # self.disease_to_hps = self.data_processor.disease_to_hps_with_frequencies
        start = time.time()
        self.hpo_clustering = HPOClustering()
        end = time.time()
        print(f"HPOClustering: {end-start}")

        # start = time.time()
        # self.hp_service = HPEmbeddingService(self.data_processor)
        # end = time.time()
        # print(f"HPEmbeddingService: {end-start}")

        start = time.time()
        self.disease_organ_service = DiseaseClusteredEmbeddingService(self.data_processor, self.hpo_clustering)
        end = time.time()
        print(f"DiseaseClusteredEmbeddingService: {end-start}")

    def run(self):
        self.disease_organ_service.process_data()


def main():
    print("init_main")
    main_instance = Main()
    print("init_prepare")
    main_instance.prepare()
    print("init_run")
    main_instance.run()

if __name__ == "__main__":
    main()