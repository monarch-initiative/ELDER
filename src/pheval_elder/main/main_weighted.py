
import time
import os
from pheval_elder.main.constants import allfromomim619340
from pheval_elder.prepare.core.chromadb_manager import ChromaDBManager
from pheval_elder.prepare.core.data_processor import DataProcessor
from pheval_elder.prepare.core.disease_weighted_avg_embedding_service import DiseaseWeightedAvgEmbeddingService
from pheval_elder.prepare.core.elder import ElderRunner
from pheval_elder.prepare.core.hp_embedding_service import HPEmbeddingService
from pheval_elder.prepare.utils.similarity_measures import SimilarityMeasures


class Main:
    def __init__(self):
        self.output_dir = "output"
        self.runner = None
        self.results = []

    def prepare(self):
        start = time.time()
        self.db_manager = ChromaDBManager(similarity=SimilarityMeasures.COSINE)
        self.data_processor = DataProcessor(self.db_manager)
        # self.hp_embeddings = self.data_processor.hp_embeddings
        # self.disease_to_hps = self.data_processor.disease_to_hps_with_frequencies
        self.hp_service = HPEmbeddingService(self.data_processor)
        self.disease_weighted_service = DiseaseWeightedAvgEmbeddingService(self.data_processor)

        end = time.time()
        print(end - start)

    def run(self):
        self.disease_weighted_service.process_data()


def main():
    print("init_main")
    main_instance = Main()
    print("init_prepare")
    main_instance.prepare()
    print("init_run")
    main_instance.run()

if __name__ == "__main__":
    main()